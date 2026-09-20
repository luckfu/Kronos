from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from webui import app as web_app
from webui.auth import hash_apr1, verify_credentials, verify_htpasswd_hash


TEST_SECRET = 'unit-test-only-secret-not-for-production'
TEST_USERNAME = 'analyst'
TEST_PASSWORD = 'correct-horse'


def write_htpasswd(path, username, hashed):
    path.write_text(f'{username}:{hashed}\n', encoding='utf-8')
    return path


def enable_auth(monkeypatch, tmp_path, hashed=None, username=TEST_USERNAME):
    if hashed is None:
        hashed = hash_apr1(TEST_PASSWORD, 'saltsalt')
    htpath = write_htpasswd(tmp_path / 'htpasswd', username, hashed)
    monkeypatch.setenv('KRONOS_SECRET_KEY', TEST_SECRET)
    monkeypatch.setenv('KRONOS_HTPASSWD_PATH', str(htpath))
    monkeypatch.setenv('KRONOS_COOKIE_SECURE', '0')
    monkeypatch.setenv('KRONOS_COOKIE_PATH', '/')
    monkeypatch.setenv('KRONOS_SESSION_DAYS', '30')
    monkeypatch.delenv('KRONOS_AUTH_DISABLED', raising=False)
    web_app.configure_auth(web_app.app)
    return htpath


def cookie_header(response, name='kronos_session'):
    headers = response.headers.getlist('Set-Cookie')
    return next((item for item in headers if item.startswith(f'{name}=')), '')


def test_apr1_matches_openssl_vector():
    hashed = hash_apr1('myPassword', 'r31.....')
    assert hashed == '$apr1$r31.....$HqJZimcKQFAMYayBlzkrA/'
    assert verify_htpasswd_hash('myPassword', hashed)
    assert not verify_htpasswd_hash('wrong-password', hashed)


def test_bcrypt_and_sha1_hashes_verify():
    sha_hash = '{SHA}W6ph5Mm5Pz8GgiULbPgzG37mj9g='
    assert verify_htpasswd_hash('password', sha_hash)
    assert not verify_htpasswd_hash('other', sha_hash)

    bcrypt = pytest_import_bcrypt()
    hashed = bcrypt.hashpw(TEST_PASSWORD.encode('utf-8'), bcrypt.gensalt(rounds=4)).decode('ascii')
    apache = '$2y$' + hashed[4:]
    assert verify_htpasswd_hash(TEST_PASSWORD, hashed)
    assert verify_htpasswd_hash(TEST_PASSWORD, apache)
    assert not verify_htpasswd_hash('wrong-password', apache)


def pytest_import_bcrypt():
    import bcrypt
    return bcrypt


def test_wrong_password_is_rejected(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    response = client.post('/login', data={
        'username': TEST_USERNAME,
        'password': 'wrong-password',
        'remember': '1',
    })

    assert response.status_code == 401
    assert 'kronos_session=' not in cookie_header(response)
    page = response.get_data(as_text=True)
    assert '用户名或密码不正确' in page
    assert TEST_PASSWORD not in page


def test_successful_login_sets_httponly_samesite_cookie(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    response = client.post('/login', data={
        'username': TEST_USERNAME,
        'password': TEST_PASSWORD,
        'remember': '1',
        'next': '/',
    })

    cookie = cookie_header(response)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/')
    assert cookie.startswith('kronos_session=')
    assert 'HttpOnly' in cookie
    assert 'SameSite=Lax' in cookie
    assert TEST_PASSWORD not in cookie
    assert TEST_PASSWORD not in response.get_data(as_text=True)


def test_remembered_session_allows_pages_and_apis(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    login = client.post('/login', data={
        'username': TEST_USERNAME,
        'password': TEST_PASSWORD,
        'remember': '1',
    })
    cookie = cookie_header(login)
    assert 'Expires=' in cookie
    expires = parsedate_to_datetime(cookie.split('Expires=', 1)[1].split(';', 1)[0])
    remaining = expires - datetime.now(timezone.utc)
    assert timedelta(days=29) <= remaining <= timedelta(days=31)

    home = client.get('/')
    dates = client.get('/api/daily-rankings/dates')

    assert home.status_code == 200
    assert '每日排名' in home.get_data(as_text=True)
    assert '退出' in home.get_data(as_text=True)
    assert dates.status_code == 200
    assert dates.get_json()['dates'] == []


def test_logout_clears_session_and_blocks_api(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()
    client.post('/login', data={
        'username': TEST_USERNAME,
        'password': TEST_PASSWORD,
        'remember': '1',
    })

    logout = client.get('/logout')
    api = client.get('/api/daily-rankings/dates')
    home = client.get('/', follow_redirects=False)

    assert logout.status_code in (302, 200)
    assert api.status_code == 401
    assert api.get_json()['error'] == 'authentication required'
    assert home.status_code == 302
    assert '/login' in home.headers['Location']


def test_api_without_cookie_returns_401(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    response = client.get('/api/daily-rankings/dates')

    assert response.status_code == 401
    assert response.get_json()['error'] == 'authentication required'


def test_html_without_cookie_redirects_to_login(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    browser = client.get('/', headers={'Accept': 'text/html'})
    generic = client.get('/')

    assert browser.status_code == 302
    assert '/login' in browser.headers['Location']
    assert generic.status_code == 302
    assert '/login' in generic.headers['Location']
    assert 'WWW-Authenticate' not in generic.headers


def test_health_and_login_page_are_public(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    health = client.get('/health')
    login = client.get('/login')

    assert health.status_code == 200
    assert health.get_json()['status'] == 'ok'
    assert login.status_code == 200
    assert 'name="password"' in login.get_data(as_text=True)
    assert '记住登录' in login.get_data(as_text=True)


def test_login_does_not_rewrite_htpasswd(monkeypatch, tmp_path):
    htpath = enable_auth(monkeypatch, tmp_path)
    before = htpath.read_bytes()
    mtime = htpath.stat().st_mtime
    client = web_app.app.test_client()

    client.post('/login', data={
        'username': TEST_USERNAME,
        'password': TEST_PASSWORD,
        'remember': '1',
    })
    client.post('/login', data={
        'username': TEST_USERNAME,
        'password': 'wrong-password',
    })

    assert htpath.read_bytes() == before
    assert htpath.stat().st_mtime == mtime
    assert verify_credentials(TEST_USERNAME, TEST_PASSWORD, path=str(htpath))
    assert not verify_credentials(TEST_USERNAME, 'wrong-password', path=str(htpath))


def test_auth_stays_off_without_htpasswd(monkeypatch):
    monkeypatch.delenv('KRONOS_HTPASSWD_PATH', raising=False)
    monkeypatch.delenv('KRONOS_REMOTE_ONLY', raising=False)
    monkeypatch.delenv('KRONOS_AUTH_REQUIRED', raising=False)
    monkeypatch.delenv('KRONOS_AUTH_DISABLED', raising=False)
    web_app.configure_auth(web_app.app)

    response = web_app.app.test_client().get('/api/daily-rankings/dates')

    assert response.status_code == 200


def test_nginx_location_no_longer_uses_basic_auth():
    text = Path('deploy/oracle-kronos/nginx-kronos-location.conf').read_text(encoding='utf-8')
    assert 'auth_basic' not in text
    assert 'proxy_set_header X-Forwarded-Prefix /kronos' in text


def test_prefixed_login_redirects_under_kronos(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()
    headers = {'X-Forwarded-Prefix': '/kronos'}

    login = client.get('/login', headers=headers)
    posted = client.post('/login', data={
        'username': TEST_USERNAME,
        'password': TEST_PASSWORD,
        'remember': '1',
        'next': '/kronos/',
    }, headers=headers)

    assert 'action="/kronos/login"' in login.get_data(as_text=True)
    assert posted.headers['Location'] == '/kronos/'
