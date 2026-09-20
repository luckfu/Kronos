import json
from pathlib import Path

from webui import app as web_app
from tests.test_web_auth import enable_auth


def _client():
    return web_app.app.test_client()


def test_manifest_and_sw_are_public_without_auth(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = _client()

    manifest = client.get('/manifest.webmanifest')
    worker = client.get('/sw.js')
    offline = client.get('/offline')
    static_manifest = client.get('/static/manifest.webmanifest')
    static_worker = client.get('/static/sw.js')
    icon = client.get('/static/icon-192.png')

    assert manifest.status_code == 200
    assert 'manifest+json' in (manifest.headers.get('Content-Type') or '')
    payload = json.loads(manifest.get_data(as_text=True))
    assert payload['name'] == 'Kronos'
    assert payload['short_name'] == 'Kronos预测台'
    assert payload['display'] == 'standalone'
    assert payload['start_url'] in ('./', '/kronos/', '/')
    assert payload['scope'] in ('./', '/kronos/', '/')
    assert payload['theme_color'] == '#0a0e14'
    assert payload['background_color'] == '#0a0e14'
    sizes = {icon['sizes'] for icon in payload['icons']}
    assert '192x192' in sizes
    assert '512x512' in sizes

    assert worker.status_code == 200
    assert 'javascript' in (worker.headers.get('Content-Type') or '')
    assert worker.headers.get('Service-Worker-Allowed') == '/'
    assert 'no-cache' in (worker.headers.get('Cache-Control') or '')
    script = worker.get_data(as_text=True)
    assert "CACHE_NAME = 'kronos-shell-" in script
    assert 'isApiPath' in script
    assert 'jsonOffline' in script
    assert 'networkOnlyApi' in script
    assert "status: 503" in script

    assert offline.status_code == 200
    assert '离线' in offline.get_data(as_text=True)
    assert static_manifest.status_code == 200
    assert static_worker.status_code == 200
    assert icon.status_code == 200
    assert icon.data[:8] == b'\x89PNG\r\n\x1a\n'


def test_prefixed_pwa_urls_stay_under_kronos(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = _client()
    headers = {'X-Forwarded-Prefix': '/kronos'}

    worker = client.get('/sw.js', headers=headers)
    manifest = client.get('/manifest.webmanifest', headers=headers)
    login = client.get('/login', headers=headers)
    home = client.get('/', headers=headers)

    assert worker.status_code == 200
    assert worker.headers.get('Service-Worker-Allowed') == '/kronos/'
    html = login.get_data(as_text=True)
    assert 'rel="manifest"' in html
    assert '/kronos/manifest.webmanifest' in html
    assert "serviceWorker" in html
    assert "'/sw.js'" in html or '"/sw.js"' in html
    assert 'apple-mobile-web-app-capable' in html
    assert 'theme-color' in html
    assert home.status_code == 302
    payload = json.loads(manifest.get_data(as_text=True))
    assert payload['start_url'] == './'


def test_login_daily_rankings_and_index_register_sw():
    login = Path('webui/templates/login.html').read_text(encoding='utf-8')
    rankings = Path('webui/templates/daily_rankings.html').read_text(encoding='utf-8')
    index = Path('webui/templates/index.html').read_text(encoding='utf-8')
    for source in (login, rankings, index):
        assert "_pwa_head.html" in source
        assert "_pwa_register.html" in source
        assert 'apple-touch-icon' in source


def test_api_still_requires_auth_after_pwa_routes(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = _client()

    response = client.get('/api/daily-rankings/dates')

    assert response.status_code == 401
    assert response.get_json()['error'] == 'authentication required'
