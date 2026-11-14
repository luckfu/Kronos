"""Download prediction shards from a finished sealed-OOS kernel (REST; no CLI).

`ListKernelSessionOutput` returns one signed URL per output file, so this pulls
only `base_clean_sealed_oos/shards/*.csv.gz` (+ summary.json) and never touches
the full output zip that truncates around 0.5-1.1 GB.

The kernel is private, so the credentials must belong to its owner
(KAGGLE_API_TOKEN + KAGGLE_USERNAME, or ~/.kaggle/credentials.json).

    python3 finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py \
        --kernel wynstonliu/kronos-base-pretrained-oos65-full \
        --out scratch/oos65/pretrained_base
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "base_clean_bootstrap_relay"))

import relay_auth  # noqa: E402

LIST_OUTPUT = "https://api.kaggle.com/v1/kernels.KernelsApiService/ListKernelSessionOutput"


def list_files(session, owner: str, slug: str) -> list[dict]:
    files, token = [], ""
    while True:
        body = {"userName": owner, "kernelSlug": slug, "pageSize": 200}
        if token:
            body["pageToken"] = token
        response = session.post(LIST_OUTPUT, json=body, timeout=120)
        response.raise_for_status()
        payload = response.json()
        files.extend(payload.get("files") or [])
        token = payload.get("nextPageToken") or ""
        if not token:
            return files


def download(session, url: str, target: Path, attempts: int = 4) -> None:
    for attempt in range(1, attempts + 1):
        try:
            # Signed kaggleusercontent URLs reject the API bearer header.
            with session.get(url, stream=True, timeout=(20, 300),
                             headers={"Authorization": None, "Content-Type": None}) as response:
                response.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                tmp.replace(target)
                return
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            if attempt == attempts:
                raise
            print(f"  retry {attempt} {target.name}: {type(exc).__name__}", flush=True)
            time.sleep(3 * attempt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, help="owner/slug")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pattern", action="append",
                        default=None, help="fnmatch on fileName; repeatable")
    args = parser.parse_args()
    patterns = args.pattern or ["*/shards/*.csv.gz", "*/summary.json"]

    owner, slug = args.kernel.split("/", 1)
    session = relay_auth.session()
    files = list_files(session, owner, slug)
    wanted = [f for f in files if any(fnmatch.fnmatch(f["fileName"], p) for p in patterns)]
    if not wanted:
        names = [f["fileName"] for f in files[:20]]
        raise SystemExit(f"No output matched {patterns}; first files: {names}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for entry in wanted:
        target = out / Path(entry["fileName"]).name
        if target.is_file() and target.stat().st_size > 0:
            continue
        download(session, entry["url"], target)
        print(f"  ok {target.name} {target.stat().st_size}", flush=True)
    shards = sorted(out.glob("*.csv.gz"))
    print(json.dumps({"kernel": args.kernel, "listed": len(files),
                      "downloaded_matching": len(wanted), "shards": len(shards),
                      "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
