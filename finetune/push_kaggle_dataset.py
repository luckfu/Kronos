#!/usr/bin/env python3
"""Create a Kaggle dataset from a local directory over REST (no kaggle CLI).

The 2.2.4 CLI dies with ``SSLEOFError`` behind this machine's proxy, so the
relay talks to the API directly.  Creating a dataset is a two-step handshake:

1. ``blobs.BlobApiService/StartBlobUpload`` → ``{token, createUrl}`` per file
2. ``PUT createUrl`` with the raw bytes (a pre-signed Google resumable upload)
3. ``POST /api/v1/datasets/create/new`` with the blob tokens

Tokens are only valid for a short window, so files are uploaded immediately
before the create call.  Nothing is written to ``~/.kaggle``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))

import relay_auth  # noqa: E402

START_BLOB = "https://api.kaggle.com/v1/blobs.BlobApiService/StartBlobUpload"
CREATE_DATASET = "https://www.kaggle.com/api/v1/datasets/create/new"
BLOB_TYPE = "DATASET"


def start_blob(session, path: Path, name: str | None = None) -> dict:
    body = {
        "type": BLOB_TYPE,
        "name": name or path.name,
        "contentLength": path.stat().st_size,
        "lastModifiedEpochSeconds": int(path.stat().st_mtime),
    }
    response = session.post(START_BLOB, json=body, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("token") or not payload.get("createUrl"):
        raise RuntimeError(f"StartBlobUpload returned no token/url for {path.name}: {payload}")
    return payload


def put_file(path: Path, create_url: str, retries: int = 5) -> None:
    """Upload bytes to the pre-signed resumable URL.

    A plain Session honours the machine's proxy env, which is what Google
    Storage needs here; only Kaggle's own endpoints had the TLS failure.
    """
    size = path.stat().st_size
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with path.open("rb") as handle:
                response = _put_stream(create_url, handle, size)
            if response.status_code in (200, 201):
                return
            if response.status_code == 503:
                last_error = RuntimeError("503 from upload endpoint")
            else:
                raise RuntimeError(
                    f"Upload failed for {path.name}: {response.status_code} {response.text[:300]}"
                )
        except Exception as exc:  # noqa: BLE001 - network flake
            last_error = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Upload failed for {path.name}: {last_error}")


def _put_stream(url: str, handle, size: int):
    import requests

    session = requests.Session()
    session.headers.update({"Content-Length": str(size)})
    return session.put(url, data=handle, timeout=(60, 1800))


def create_dataset(session, owner: str, slug: str, title: str, license_name: str,
                   private: bool, tokens: list[str]) -> dict:
    body = {
        "ownerSlug": owner,
        "slug": slug,
        "title": title,
        "licenseName": license_name,
        "isPrivate": bool(private),
        "files": [{"token": token} for token in tokens],
    }
    response = session.post(CREATE_DATASET, json=body, timeout=300)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("status") or "").lower() != "ok":
        raise RuntimeError(f"Dataset create failed: {payload.get('error') or payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="package directory to upload")
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME", "luckfu"))
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--license", default="CC0-1.0")
    parser.add_argument("--public", action="store_true", help="default is private")
    parser.add_argument("--dry-run", action="store_true", help="upload but skip the create call")
    args = parser.parse_args()

    files = sorted(
        (p for p in args.dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(args.dir).as_posix(),
    )
    if not files:
        raise SystemExit(f"no files in {args.dir}")

    session = relay_auth.session()
    tokens: list[str] = []
    for path in files:
        rel = path.relative_to(args.dir).as_posix()
        blob = start_blob(session, path, rel)
        print(f"uploading {rel} ({path.stat().st_size/1e6:.1f} MB)", flush=True)
        put_file(path, blob["createUrl"])
        tokens.append(blob["token"])
        print(f"  ok {path.name}", flush=True)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "files": [p.name for p in files]}, indent=2))
        return 0

    payload = create_dataset(
        session, args.owner, args.slug, args.title, args.license, not args.public, tokens
    )
    print(json.dumps({"ref": payload.get("ref"), "url": payload.get("url"),
                      "status": payload.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
