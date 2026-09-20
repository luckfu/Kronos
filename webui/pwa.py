"""Installable PWA shell for the Kronos web UI.

Serves the web app manifest and service worker at the application root so the
worker scope covers /kronos/ (or / when no reverse-proxy prefix is present).
These routes are public: browsers fetch them without a session cookie.
"""

from __future__ import annotations

from flask import current_app, render_template, send_from_directory

try:
    from webui.auth import forwarded_prefix
except ImportError:
    from auth import forwarded_prefix


def scope_path():
    prefix = forwarded_prefix()
    if not prefix:
        return '/'
    return prefix if prefix.endswith('/') else prefix + '/'


def service_worker():
    response = send_from_directory(
        current_app.static_folder,
        'sw.js',
        mimetype='application/javascript',
        max_age=0,
    )
    response.headers['Service-Worker-Allowed'] = scope_path()
    response.headers['Cache-Control'] = 'no-cache'
    return response


def pwa_manifest():
    response = send_from_directory(
        current_app.static_folder,
        'manifest.webmanifest',
        mimetype='application/manifest+json',
        max_age=3600,
    )
    response.headers['Content-Type'] = 'application/manifest+json'
    response.headers['Cache-Control'] = 'public, max-age=3600, must-revalidate'
    return response


def offline_view():
    return render_template('offline.html')


def init_pwa(flask_app):
    flask_app.add_url_rule(
        '/sw.js',
        endpoint='service_worker',
        view_func=service_worker,
        methods=['GET'],
    )
    flask_app.add_url_rule(
        '/manifest.webmanifest',
        endpoint='pwa_manifest',
        view_func=pwa_manifest,
        methods=['GET'],
    )
    flask_app.add_url_rule(
        '/offline',
        endpoint='offline',
        view_func=offline_view,
        methods=['GET'],
    )
    return flask_app
