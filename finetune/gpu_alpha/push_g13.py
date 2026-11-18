"""Push Experiment G13 (G2_FT_L120 16-path AR OOS eval) to Kaggle 2x T4 GPU."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))
import relay_auth  # noqa: E402

OWNER = os.environ.get("KAGGLE_OWNER", "smmt315")
SLUG = "kronos-alpha-g13-ft-l120-ar16"
TITLE = "Kronos Alpha G13 FT L120 AR16"
DATASETS = [
    os.environ.get("KAGGLE_FT_DATASET", f"{OWNER}/kronos-small-g2-ft-l120-merged"),
    os.environ.get("KAGGLE_OOS_DATASET", f"{OWNER}/a-share-120d-oos-20260717-20261023-v2"),
]


def source_bundle() -> bytes:
    entries = []
    for path in sorted((ROOT / "model").glob("*.py")):
        entries.append((Path("model") / path.name, path))
    entries.append((Path("finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"),
                    ROOT / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"))
    entries.append((Path("finetune/gpu_alpha/g13_ft_l120_ar_entry.py"), HERE / "g13_ft_l120_ar_entry.py"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, src in entries:
            info = zipfile.ZipInfo(arc.as_posix(), (2026, 11, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, src.read_bytes())
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--stride", default="2")
    args = parser.parse_args()
    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    code = f'''import base64, hashlib, io, os, runpy, shutil, sys, zipfile
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["G13_STRIDE"] = {args.stride!r}
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/g13_src")
if repo.exists():
    shutil.rmtree(repo)
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
os.environ["BASE_OOS_SOURCE"] = str(repo)
sys.path.insert(0, str(repo))
runpy.run_path(str(repo / "finetune/gpu_alpha/g13_ft_l120_ar_entry.py"), run_name="__main__")
'''
    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)
    (build / "script.py").write_text(code)
    compile(code, str(build / "script.py"), "exec")
    print(json.dumps({"slug": f"{OWNER}/{SLUG}", "sha256": digest, "bundle_bytes": len(payload)}, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")
        return
    session = relay_auth.session()
    resp = session.post(
        f"{relay_auth.API_ROOT}/v1/kernels.KernelsApiService/SaveKernel",
        json={
            "slug": f"{OWNER}/{SLUG}",
            "newTitle": TITLE,
            "text": code,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "datasetDataSources": DATASETS,
            "kernelDataSources": [],
            "machineShape": "NvidiaTeslaT4",
        },
        timeout=180,
    )
    print("HTTP", resp.status_code)
    print(resp.text[:500])


if __name__ == "__main__":
    main()
