"""TimesFM-3 SweetSpot (native merged, zero LoRA) on the 65-day temporal OOS package.

Differences from ``eval_native_merged.py`` (whose forward/inference code is reused as-is):

- weights are the private ModelScope repo's ``model.safetensors`` + ``sector_embedding.pt``
  (mounted from a private Kaggle dataset, verified by SHA256), not a parent kernel's ``.pt`` export;
- windows are rebuilt in the kernel from the package's ``combined_raw.csv`` and
  ``evaluation_samples.jsonl`` (no pickled panel, so no pandas-version coupling);
- every signal date is scored, and metrics are reported per pre-registered segment.

Frozen in ``finetune/reports/20261112_timesfm3_sweetspot_new_window_prereg_cn.md``.
No credentials are needed in the kernel unless EVAL_WEIGHTS_FROM_MODELSCOPE=1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = "luckfu/Kronos-TimesFM3-V11-SweetSpot"
EXPECTED_SHA256 = {
    "model.safetensors": "f914935e6413d705",       # 16-hex prefixes of the repo's SHA256
    "sector_embedding.pt": "11af58200e6c8a33",
}
INPUT_ROOT = Path(os.environ.get("EVAL_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("EVAL_OUTPUT", "/kaggle/working/eval_results"))
SEAL_AFTER = "2026-10-23"
SEGMENTS = {"A_selection_window": ("2026-08-03", "2026-09-17"), "B_new_window": ("2026-09-18", "2026-10-23"),
            "C_early": ("2026-07-17", "2026-07-31")}
FEATURES = ("open", "high", "low", "close", "volume", "amount", "size_percentile")
LOOKBACK, HORIZON = 120, 10
MAX_DOWNLOAD_RETRIES = 200


def log(event: str, **fields) -> None:
    print(json.dumps({"time": time.time(), "event": event, **fields}, ensure_ascii=False, default=float), flush=True)


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def locate_weights(root: Path) -> Path:
    """Directory holding both weight files, verified by SHA256.

    Kaggle reaches the ModelScope CDN at about 1 MB/min with constant resets, so the
    weights are mounted from a private Kaggle dataset.  EVAL_WEIGHTS_FROM_MODELSCOPE=1
    downloads them instead.
    """
    if os.environ.get("EVAL_WEIGHTS_FROM_MODELSCOPE", "") == "1":
        target = Path("/kaggle/working/sweetspot_weights")
        download_weights(target)
        return target
    hits = [p.parent for p in sorted(root.glob("**/model.safetensors")) if (p.parent / "sector_embedding.pt").is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one mounted directory with model.safetensors + sector_embedding.pt; found {hits}")
    for name, prefix in EXPECTED_SHA256.items():
        actual = sha256_prefix(hits[0] / name)
        if actual != prefix:
            raise RuntimeError(f"{name}: SHA256 {actual} != expected {prefix}")
        log("weights_verified", file=name, sha256=actual)
    return hits[0]


def download_weights(target: Path) -> None:
    """Fetch the private repo's files, resuming with Range requests when the CDN drops the connection."""
    import requests

    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("MODELSCOPE_API_TOKEN is required: the model repo is private")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    url = f"https://modelscope.cn/api/v1/models/{REPO}/repo"
    target.mkdir(parents=True, exist_ok=True)
    for name, prefix in EXPECTED_SHA256.items():
        path, total, attempts = target / name, None, 0
        path.write_bytes(b"")
        while total is None or path.stat().st_size < total:
            have = path.stat().st_size
            try:
                with session.get(url, params={"Revision": "master", "FilePath": name}, stream=True,
                                 headers={"Range": f"bytes={have}-"}, timeout=(30, 120)) as resp:
                    if resp.status_code != 206:           # the resume logic below assumes a ranged reply
                        raise RuntimeError(f"{name}: expected HTTP 206 for a ranged request, got {resp.status_code}")
                    total = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
                    with path.open("ab") as handle:
                        for chunk in resp.iter_content(1 << 20):
                            handle.write(chunk)
            except (requests.exceptions.RequestException, OSError) as error:
                attempts += 1
                log("download_interrupted", file=name, have=path.stat().st_size, attempt=attempts,
                    error=type(error).__name__)
                if attempts > MAX_DOWNLOAD_RETRIES:
                    raise
                time.sleep(min(60, 5 * attempts))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if not digest.hexdigest().startswith(prefix):
            raise RuntimeError(f"{name}: SHA256 {digest.hexdigest()[:16]} != expected {prefix}")
        log("weights_downloaded", file=name, bytes=path.stat().st_size, sha256=digest.hexdigest()[:16],
            interruptions=attempts)


def build_windows(root: Path, sector2id: dict[str, int], unknown_id: int):
    """Same normalisation and filters as eval_native_merged.extract_full_oos_set, from raw bars."""
    import pandas as pd

    raw = pd.read_csv(next(iter(sorted(root.glob("**/combined_raw.csv")))), dtype={"symbol": str})
    raw["date"] = raw["date"].astype(str)
    raw = raw[raw["volume"] > 0].sort_values(["symbol", "date"]).reset_index(drop=True)
    raw["size_percentile"] = raw.groupby("date")["market_cap"].rank(pct=True, method="average")
    samples = pd.read_json(next(iter(sorted(root.glob("**/evaluation_samples.jsonl")))), lines=True,
                           dtype={"symbol": str})
    samples["asof_date"] = samples["asof_date"].astype(str)
    if samples["asof_date"].max() > SEAL_AFTER:
        raise RuntimeError("package contains signal dates from the sealed acceptance window")

    values = raw[list(FEATURES)].to_numpy(dtype=np.float32)
    first_row = raw.groupby("symbol", sort=False).cumcount().to_numpy()
    row_of = {key: i for i, key in enumerate(zip(raw["symbol"], raw["date"]))}
    contexts, targets, sectors, dates, symbols = [], [], [], [], []
    dropped = {"no_bar": 0, "short_window": 0, "bad_values": 0}
    for sym, asof, sector in zip(samples["symbol"], samples["asof_date"], samples["sector"]):
        end = row_of.get((sym, asof))
        if end is None:
            dropped["no_bar"] += 1
            continue
        start, stop = end - (LOOKBACK - 1), end + HORIZON + 1
        if first_row[end] < LOOKBACK - 1 or stop > len(values) or raw["symbol"].iat[stop - 1] != sym:
            dropped["short_window"] += 1
            continue
        w = values[start:stop].copy()
        p0 = w[LOOKBACK - 1, 3]
        if not (np.isfinite(w).all() and (w[:, :4] > 0).all() and (w[:, 4] > 0).all()) or p0 <= 1e-4:
            dropped["bad_values"] += 1
            continue
        w[:, :4] /= p0
        w[:, 4] /= np.median(w[:LOOKBACK, 4]) + 1e-4
        w[:, 5] /= np.median(w[:LOOKBACK, 5]) + 1e-4
        w[:, 4:6] = np.clip(w[:, 4:6], 0.0, 50.0)
        w[:, :4] = np.clip(w[:, :4], 0.05, 10.0)
        contexts.append(w[:LOOKBACK])
        targets.append(w[LOOKBACK:, 3])
        sectors.append(sector2id.get(sector, unknown_id))
        dates.append(asof)
        symbols.append(sym)
    log("windows_ready", samples=len(samples), kept=len(contexts), dropped=dropped,
        unknown_sector_share=float(np.mean(np.asarray(sectors) == unknown_id)))
    return (np.asarray(contexts, dtype=np.float32), np.asarray(targets, dtype=np.float32),
            np.asarray(sectors, dtype=np.int64), dates, symbols)


def main() -> None:
    import pandas as pd
    from safetensors.torch import load_file

    import eval_native_merged as base

    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = locate_weights(INPUT_ROOT)

    split_csv = sorted(INPUT_ROOT.glob("**/symbol_split.csv"))[0]
    names = sorted(pd.read_csv(split_csv)["sector"].dropna().unique().tolist())   # train.py's id rule
    sector_state = torch.load(weights / "sector_embedding.pt", map_location="cpu", weights_only=False)
    if sector_state["weight"].shape[0] != len(names) + 5:
        raise RuntimeError(f"sector vocabulary mismatch: {len(names)} names vs embedding {tuple(sector_state['weight'].shape)}")
    contexts, targets, sectors, dates, symbols = build_windows(
        INPUT_ROOT, {name: i for i, name in enumerate(names)}, len(names))

    device = torch.device("cuda")
    from timesfm3 import TimesFM3Forecaster
    model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=512).model.to(device)
    model.load_state_dict(load_file(str(weights / "model.safetensors")), strict=True)   # every tensor must match
    sector_embedding = nn.Embedding(sector_state["weight"].shape[0], int(model.input_patch_len)).to(device)
    sector_embedding.load_state_dict(sector_state, strict=True)
    log("model_ready", sectors=len(names), gpus=torch.cuda.device_count())
    if torch.cuda.device_count() >= 2:
        attrs = {n: getattr(model, n) for n in ("input_patch_len", "output_patch_len", "use_stitching",
                                               "_stitching_extract_len", "rolls", "use_frozen_running_stats")
                 if hasattr(model, n)}
        model = nn.DataParallel(model)
        for name, value in attrs.items():
            setattr(model, name, value)

    base.MAX_SAFE_SECONDS = 10**9          # the reused loop truncates silently at its 40-minute guard
    pred, true = base.run_full_inference(model, contexts, targets, sectors, device, sector_embedding, batch_size=512)
    if len(pred) != len(dates) or not np.isfinite(pred).all():
        raise RuntimeError(f"incomplete or non-finite inference: {len(pred)} of {len(dates)}")
    np.savez_compressed(OUTPUT / "eval_predictions_sweetspot_window.npz", pred_ret=pred.astype(np.float32),
                        true_ret=true.astype(np.float32), dates=np.array(dates), symbols=np.array(symbols))

    date_array = np.array(dates)
    report = {"model": REPO, "weights_sha256": EXPECTED_SHA256, "sample_count": int(len(pred)), "segments": {}}
    for name, (start, end) in SEGMENTS.items():
        mask = (date_array >= start) & (date_array <= end)
        if mask.any():
            metrics = base.calculate_metrics(pred[mask], true[mask], asof_dates=list(date_array[mask]))
            report["segments"][name] = {"signal_start": start, "signal_end": end, "samples": int(mask.sum()),
                                        "signal_days": int(len(set(date_array[mask]))), "metrics": metrics}
    (OUTPUT / "report_sweetspot_window.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("finished", report=report)


if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "git+https://github.com/google-research/timesfm.git", "safetensors"])
    main()
