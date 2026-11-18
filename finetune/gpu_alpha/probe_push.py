"""Push a fast Kaggle GPU probe checking GPU spec, C2 weights, and train/val/oos datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))
import relay_auth  # noqa: E402

OWNER = "luckfu"
SLUG = "kronos-alpha-gpu-probe"
TITLE = "Kronos Alpha GPU Probe"
DATASETS = [
    "luckfu/a-share-120d-temporal-symbol-holdout",
    "luckfu/kronos-small-cosine-c2-best-weights",
    "luckfu/a-share-120d-oos-20260717-20261023-v2",
]

PROBE_CODE = r'''import json, pickle, time
from pathlib import Path
import pandas as pd
import torch

print("CUDA:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // (1024**2), "MB")

inp = Path("/kaggle/input")
for p in sorted(inp.glob("**/train_data.pkl")) + sorted(inp.glob("**/val_data.pkl")):
    t0 = time.time()
    with p.open("rb") as f:
        data = pickle.load(f)
    syms = list(data.keys())
    all_min = min(df.index.min() for df in data.values())
    all_max = max(df.index.max() for df in data.values())
    lens = [len(df) for df in data.values()]
    print(f"PKL {p}: syms={len(syms)} min={all_min} max={all_max} total_rows={sum(lens)} med_len={sorted(lens)[len(lens)//2]} ({time.time()-t0:.1f}s)")

for p in sorted(inp.glob("**/kronos-small-cosine-c2-best-weights/*")):
    print("C2:", p.name, p.stat().st_size)
cfg_p = sorted(inp.glob("**/kronos-small-cosine-c2-best-weights/config.json"))
if cfg_p:
    print("C2 config:", cfg_p[0].read_text())

oos_raw = sorted(inp.glob("**/combined_raw.csv"))
for p in oos_raw:
    df = pd.read_csv(p, usecols=["date", "symbol"])
    print("OOS raw:", p, "rows=", len(df), "syms=", df["symbol"].nunique(), "min=", df["date"].min(), "max=", df["date"].max())
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    out_dir = Path(__file__).resolve().parent / "build" / SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "script.py").write_text(PROBE_CODE)
    if not args.push:
        print("Ready, pass --push to submit")
        return
    session = relay_auth.session()
    resp = session.post(
        f"{relay_auth.API_ROOT}/v1/kernels.KernelsApiService/SaveKernel",
        json={
            "slug": f"{OWNER}/{SLUG}",
            "newTitle": TITLE,
            "text": PROBE_CODE,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "datasetDataSources": DATASETS,
            "kernelDataSources": [],
            "machineShape": "NvidiaTeslaT4",
        },
        timeout=120,
    )
    print("HTTP", resp.status_code)
    print(resp.text[:500])


if __name__ == "__main__":
    main()
