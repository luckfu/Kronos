"""Kaggle REST auth for the relay scripts (no kaggle CLI in the loop).

Why: the 2.2.4 CLI client intermittently dies with ``SSLEOFError`` against
api.kaggle.com, so the relay talks to the REST API directly.  OAuth access
tokens only live 30 minutes, and the CLI is what normally refreshes them, so
this module exchanges the stored ``refresh_token`` itself and caches the new
access token in-process (never writes ~/.kaggle).

Resolution order for the token:
  1. ``KAGGLE_API_TOKEN`` (raw access token)
  2. ``~/.kaggle/credentials.json`` → ``access_token`` if not near expiry
  3. ``~/.kaggle/credentials.json`` → exchange ``refresh_token``
  4. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` (legacy basic auth)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

API_ROOT = os.environ.get("KRONOS_KAGGLE_API", "https://api.kaggle.com")
_REFRESH_RPC = f"{API_ROOT}/v1/users.AccountService/GenerateAccessToken"
_STATE: dict = {"token": None, "expires_at": 0.0, "session": None}


def credentials_path() -> Path:
    base = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return base / "credentials.json"


def read_credentials() -> dict:
    path = credentials_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _parse_expiry(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def refresh_access_token(refresh_token: str) -> str:
    """Exchange the refresh token for a fresh 30-min access token."""
    if requests is None:
        raise RuntimeError("requests is required to refresh a Kaggle token")
    response = requests.post(
        _REFRESH_RPC,
        json={"refreshToken": refresh_token, "apiVersion": "API_VERSION_V1"},
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    token = str(body.get("token") or "")
    if not token:
        raise RuntimeError(f"Refresh returned no token: {body}")
    _STATE["token"] = token
    _STATE["expires_at"] = time.time() + int(body.get("expiresIn", 1800))
    return token


def access_token(force_refresh: bool = False) -> str:
    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_token and not force_refresh:
        return env_token
    if (
        not force_refresh
        and _STATE["token"]
        and _STATE["expires_at"] - 120 > time.time()
    ):
        return _STATE["token"]
    credentials = read_credentials()
    refresh = str(credentials.get("refresh_token") or "").strip()
    if refresh:
        return refresh_access_token(refresh)
    token = str(credentials.get("access_token") or "").strip()
    if token and _parse_expiry(str(credentials.get("access_token_expiration") or "")) > time.time():
        return token
    raise RuntimeError(
        "No usable Kaggle credentials: set KAGGLE_API_TOKEN or run `kaggle auth login`"
    )


class _AuthSession(requests.Session):
    """Retry once after a refresh when the server answers 401."""

    def request(self, method, url, **kwargs):  # type: ignore[override]
        self.headers["Authorization"] = f"Bearer {access_token()}"
        response = super().request(method, url, **kwargs)
        if response.status_code == 401:
            self.headers["Authorization"] = f"Bearer {access_token(force_refresh=True)}"
            response = super().request(method, url, **kwargs)
        return response


def session():
    """Shared session with JSON headers and auto-refresh on 401."""
    if requests is None:
        raise RuntimeError("requests is required (pip install -r requirements-dev.txt)")
    if _STATE["session"] is None:
        client = _AuthSession()
        client.headers.update({"Content-Type": "application/json"})
        user = os.environ.get("KAGGLE_USERNAME", "").strip()
        key = os.environ.get("KAGGLE_KEY", "").strip()
        if user and key and not (os.environ.get("KAGGLE_API_TOKEN") or read_credentials()):
            client.auth = (user, key)
        _STATE["session"] = client
    return _STATE["session"]


def username() -> str:
    name = os.environ.get("KAGGLE_USERNAME", "").strip()
    if name:
        return name
    return str(read_credentials().get("username") or "").strip()
