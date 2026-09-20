"""Cookie-session login for the Kronos web gateway.

Validates usernames and passwords against the existing Nginx htpasswd file
(read-only). Mobile browsers do not reliably persist HTTP Basic Auth, so
Kronos uses an HttpOnly session cookie instead.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from datetime import timedelta

from flask import (
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

logger = logging.getLogger(__name__)

SESSION_USER_KEY = 'user'
DEFAULT_HTPASSWD_PATH = '/etc/nginx/.htpasswd_clawd'
DEFAULT_SESSION_DAYS = 30
_ITOA64 = './0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_DUMMY_APR1 = '$apr1$xxxxxxxx$8n4XhvW4nYzK3cYH8sYq1.'


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def htpasswd_path():
    return os.getenv('KRONOS_HTPASSWD_PATH', DEFAULT_HTPASSWD_PATH)


def session_lifetime():
    raw = os.getenv('KRONOS_SESSION_DAYS', str(DEFAULT_SESSION_DAYS))
    try:
        days = int(raw)
    except (TypeError, ValueError):
        days = DEFAULT_SESSION_DAYS
    return timedelta(days=max(1, days))


def cookie_path():
    configured = os.getenv('KRONOS_COOKIE_PATH')
    if configured:
        return configured
    if _env_flag('KRONOS_REMOTE_ONLY'):
        return '/kronos'
    return '/'


def cookie_secure():
    if os.getenv('KRONOS_COOKIE_SECURE') is not None:
        return _env_flag('KRONOS_COOKIE_SECURE')
    return _env_flag('KRONOS_REMOTE_ONLY')


def auth_enabled():
    if _env_flag('KRONOS_AUTH_DISABLED'):
        return False
    if os.path.isfile(htpasswd_path()):
        return True
    # Remote-only Oracle deploys must never serve the UI without a login.
    return _env_flag('KRONOS_REMOTE_ONLY') or _env_flag('KRONOS_AUTH_REQUIRED')


def forwarded_prefix():
    return (request.headers.get('X-Forwarded-Prefix') or '').rstrip('/')


def public_url(path):
    if not path.startswith('/'):
        path = '/' + path
    return forwarded_prefix() + path


def safe_next_url(value):
    default = public_url('/')
    candidate = (value or '').strip()
    if not candidate:
        return default
    if not candidate.startswith('/') or candidate.startswith('//') or '\\' in candidate:
        return default
    return candidate


def wants_json():
    if request.path.startswith('/api/'):
        return True
    if request.is_json:
        return True
    accept = (request.headers.get('Accept') or '').lower()
    return 'application/json' in accept and 'text/html' not in accept


class PrefixMiddleware:
    """Honor X-Forwarded-Prefix so redirects and url_for stay under /kronos."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        prefix = environ.get('HTTP_X_FORWARDED_PREFIX')
        if prefix:
            environ['SCRIPT_NAME'] = prefix.rstrip('/')
        proto = environ.get('HTTP_X_FORWARDED_PROTO')
        if proto:
            environ['wsgi.url_scheme'] = proto.split(',', 1)[0].strip()
        return self.app(environ, start_response)


def _to64(value, count):
    chars = []
    for _ in range(count):
        chars.append(_ITOA64[value & 0x3F])
        value >>= 6
    return ''.join(chars)


def hash_apr1(password, salt):
    """Apache APR1 MD5 as produced by `htpasswd -m` / `openssl passwd -apr1`."""
    magic = '$apr1$'
    password_b = password.encode('utf-8')
    salt_b = salt.encode('ascii')[:8]

    digest = hashlib.md5()
    digest.update(password_b + magic.encode('ascii') + salt_b)

    alt = hashlib.md5(password_b + salt_b + password_b).digest()
    remaining = len(password_b)
    while remaining > 0:
        digest.update(alt[: min(16, remaining)])
        remaining -= 16

    length = len(password_b)
    while length:
        if length & 1:
            digest.update(b'\x00')
        else:
            digest.update(password_b[:1])
        length >>= 1
    final = digest.digest()

    for index in range(1000):
        digest = hashlib.md5()
        if index & 1:
            digest.update(password_b)
        else:
            digest.update(final)
        if index % 3:
            digest.update(salt_b)
        if index % 7:
            digest.update(password_b)
        if index & 1:
            digest.update(final)
        else:
            digest.update(password_b)
        final = digest.digest()

    encoded = (
        _to64((final[0] << 16) | (final[6] << 8) | final[12], 4)
        + _to64((final[1] << 16) | (final[7] << 8) | final[13], 4)
        + _to64((final[2] << 16) | (final[8] << 8) | final[14], 4)
        + _to64((final[3] << 16) | (final[9] << 8) | final[15], 4)
        + _to64((final[4] << 16) | (final[10] << 8) | final[5], 4)
        + _to64(final[11], 2)
    )
    return f'{magic}{salt_b.decode("ascii")}${encoded}'


def _verify_apr1(password, hashed):
    parts = hashed.split('$')
    if len(parts) != 4 or parts[1] != 'apr1':
        return False
    return hmac.compare_digest(hash_apr1(password, parts[2]), hashed)


def _verify_sha1(password, hashed):
    expected = hashed[5:]
    digest = base64.b64encode(hashlib.sha1(password.encode('utf-8')).digest()).decode('ascii')
    return hmac.compare_digest(digest, expected)


def _verify_bcrypt(password, hashed):
    try:
        import bcrypt
    except ImportError:
        return _verify_passlib(password, hashed)
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('ascii'))
    except (TypeError, ValueError):
        return False


def _verify_passlib(password, hashed):
    try:
        from passlib.context import CryptContext
    except ImportError:
        return False
    context = CryptContext(
        schemes=(
            'bcrypt',
            'apr_md5_crypt',
            'sha512_crypt',
            'sha256_crypt',
            'ldap_salted_sha1',
            'des_crypt',
        ),
        deprecated='auto',
    )
    try:
        return bool(context.verify(password, hashed))
    except (ValueError, TypeError):
        return False


def _verify_crypt(password, hashed):
    try:
        import crypt
    except ImportError:
        return False
    try:
        return hmac.compare_digest(crypt.crypt(password, hashed), hashed)
    except (TypeError, ValueError, OSError):
        return False


def verify_htpasswd_hash(password, hashed):
    hashed = (hashed or '').strip()
    if not password or not hashed:
        return False
    if hashed.startswith(('$2a$', '$2b$', '$2y$')):
        return _verify_bcrypt(password, hashed)
    if hashed.startswith('$apr1$'):
        return _verify_apr1(password, hashed)
    if hashed.startswith('{SHA}'):
        return _verify_sha1(password, hashed)
    if _verify_passlib(password, hashed):
        return True
    if hashed.startswith(('$1$', '$5$', '$6$')):
        return _verify_crypt(password, hashed)
    return _verify_crypt(password, hashed)


def lookup_htpasswd_hash(path, username):
    if not username or not path:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                name, separator, hashed = line.partition(':')
                if separator and name == username:
                    return hashed
    except OSError as exc:
        logger.error('Unable to read htpasswd file: %s', exc.__class__.__name__)
        return None
    return None


def verify_credentials(username, password, path=None):
    """Return True if username/password match a htpasswd entry.

    The password file is only read. Missing users still run a dummy hash
    check so timing does not advertise whether the account exists.
    """
    path = path or htpasswd_path()
    username = (username or '').strip()
    password = password or ''
    hashed = lookup_htpasswd_hash(path, username)
    if hashed is None:
        verify_htpasswd_hash(password, _DUMMY_APR1)
        return False
    return verify_htpasswd_hash(password, hashed)


def configure_auth(flask_app):
    secret = os.getenv('KRONOS_SECRET_KEY')
    if secret:
        flask_app.secret_key = secret
    flask_app.config.update(
        SESSION_COOKIE_NAME=os.getenv('KRONOS_COOKIE_NAME', 'kronos_session'),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=cookie_secure(),
        SESSION_COOKIE_SAMESITE=os.getenv('KRONOS_COOKIE_SAMESITE', 'Lax'),
        SESSION_COOKIE_PATH=cookie_path(),
        PERMANENT_SESSION_LIFETIME=session_lifetime(),
        SESSION_REFRESH_EACH_REQUEST=True,
    )
    return flask_app


def _login_payload():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return (
            str(payload.get('username') or ''),
            str(payload.get('password') or ''),
            bool(payload.get('remember', True)),
            payload.get('next'),
        )
    remember_raw = request.form.get('remember')
    remember = True if remember_raw is None else remember_raw not in ('', '0', 'false', 'off')
    if request.method == 'POST' and 'remember' not in request.form:
        remember = False
    return (
        request.form.get('username') or '',
        request.form.get('password') or '',
        remember,
        request.form.get('next') or request.args.get('next'),
    )


def login_view():
    if auth_enabled() and session.get(SESSION_USER_KEY):
        return redirect(safe_next_url(request.args.get('next')))
    error = None
    next_url = request.args.get('next') or request.form.get('next') or public_url('/')
    if request.method == 'POST':
        username, password, remember, next_url = _login_payload()
        if not current_app.secret_key:
            logger.error('KRONOS_SECRET_KEY is not set; login is unavailable')
            error = '服务器未配置登录密钥'
            status = 500
        elif verify_credentials(username, password):
            session.clear()
            session[SESSION_USER_KEY] = username.strip()
            session.permanent = bool(remember)
            current_app.permanent_session_lifetime = session_lifetime()
            target = safe_next_url(next_url)
            if wants_json():
                response = jsonify({'ok': True, 'user': username.strip(), 'remember': bool(remember)})
                return response, 200
            return redirect(target)
        else:
            error = '用户名或密码不正确'
            status = 401
            if wants_json():
                return jsonify({'error': error}), status
        return render_template(
            'login.html',
            error=error,
            next_url=safe_next_url(next_url),
            remember=remember if request.method == 'POST' else True,
            session_days=session_lifetime().days,
        ), status
    return render_template(
        'login.html',
        error=error,
        next_url=safe_next_url(next_url),
        remember=True,
        session_days=session_lifetime().days,
    )


def logout_view():
    session.clear()
    if wants_json():
        response = jsonify({'ok': True})
        response.delete_cookie(
            current_app.config.get('SESSION_COOKIE_NAME', 'kronos_session'),
            path=current_app.config.get('SESSION_COOKIE_PATH') or '/',
        )
        return response
    response = redirect(url_for('login') if auth_enabled() else public_url('/'))
    response.delete_cookie(
        current_app.config.get('SESSION_COOKIE_NAME', 'kronos_session'),
        path=current_app.config.get('SESSION_COOKIE_PATH') or '/',
    )
    return response


def require_login():
    path = request.path or '/'
    if path == '/health' or path == '/favicon.ico' or path.startswith('/static/'):
        return None
    if request.endpoint in ('login', 'logout', 'static', 'health', 'favicon'):
        return None
    if not auth_enabled():
        return None
    if not current_app.secret_key:
        logger.error('KRONOS_SECRET_KEY is not set; refusing protected routes')
        if wants_json():
            return jsonify({'error': 'server authentication is not configured'}), 500
        return ('KRONOS_SECRET_KEY is not configured', 500)
    if session.get(SESSION_USER_KEY):
        return None
    if wants_json():
        return jsonify({'error': 'authentication required'}), 401
    return redirect(url_for('login', next=public_url(path)))


def inject_auth():
    user = session.get(SESSION_USER_KEY)
    enabled = auth_enabled()
    return {
        'kronos_auth_enabled': enabled,
        'kronos_user': user,
        'kronos_logout_url': url_for('logout') if enabled else '',
        'kronos_login_url': url_for('login') if enabled else '',
    }


def init_auth(flask_app):
    configure_auth(flask_app)
    if not isinstance(flask_app.wsgi_app, PrefixMiddleware):
        flask_app.wsgi_app = PrefixMiddleware(flask_app.wsgi_app)
    flask_app.before_request(require_login)
    flask_app.context_processor(inject_auth)
    flask_app.add_url_rule('/login', endpoint='login', view_func=login_view, methods=['GET', 'POST'])
    flask_app.add_url_rule('/logout', endpoint='logout', view_func=logout_view, methods=['GET', 'POST'])
    return flask_app
