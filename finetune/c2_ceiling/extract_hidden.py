#!/usr/bin/env python3
"""Kernel entry point: dump the frozen C2 hidden state for every window (no training).

For each window C2 reads the 120 history bars exactly as the first step of the
evaluator's decode does, and the hidden state at the last position (the vector
the output head sees before the first forecast token) is saved.  A small head
fitted on these vectors answers "does C2's internal representation hold
ranking information that sampling 16 paths throws away?".

Two sources:
  oos  the 65-day package (signal dates 2026-07-17..2026-10-23), same
       preprocessing as ``evaluate_base_sealed.py`` (imported, not copied);
  val  the symbol-holdout validation panel (unseen symbols, 2025-07..2026-07),
       same preprocessing as training (``finetune/dataset.py``).

Outputs under /kaggle/working/c2_hidden/: ``oos_hidden_<date>.npz``,
``val_hidden_<YYYY-MM>.npz`` (float16 vectors + keys/labels) and ``summary.json``.
No signal date after 2026-10-23 is read.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "finetune", REPO / "finetune/oos_eval/base_clean_sealed_oos"):
    sys.path.insert(0, str(path))

LAST_SIGNAL_DATE = "2026-10-23"
OUTPUT = Path(os.environ.get("HIDDEN_OUTPUT", "/kaggle/working/c2_hidden"))
INPUT = Path(os.environ.get("HIDDEN_INPUT", "/kaggle/input"))
BATCH = int(os.environ.get("HIDDEN_BATCH", "512"))
LIMIT = int(os.environ.get("HIDDEN_LIMIT", "0") or "0")      # smoke runs only
WORKERS = int(os.environ.get("HIDDEN_WORKERS", "4"))
VAL_PACKAGE = os.environ.get("HIDDEN_VAL_PACKAGE", "a-share-120d-temporal-symbol-holdout")
OOS_PACKAGE = os.environ.get("HIDDEN_OOS_PACKAGE", "a-share-120d-oos-20260717-20261023-v2")


def phase(name, **values):
    print(json.dumps({"time": time.time(), "phase": name, **values}, ensure_ascii=False), flush=True)


@torch.inference_mode()
def hidden_states(model, tokenizer, x, stamp, sector, percentile, device):
    """Last-position context after the 120 history bars (pre-head, post-norm)."""
    x = torch.clip(x.to(device), -5, 5)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        tokens = tokenizer.encode(x, half=True)
        _, context = model.decode_s1(tokens[0], tokens[1], stamp.to(device),
                                     sector_id=sector.to(device), size_percentile=percentile.to(device))
    return context[:, -1, :].float().cpu().numpy().astype(np.float16)


def run_oos(model, tokenizer, device):
    import evaluate_base_sealed as ev
    # The training dataset ships its own evaluation package, so name the OOS one.
    roots = sorted(path.parent for path in INPUT.glob(f"**/{OOS_PACKAGE}/combined_raw.csv"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one {OOS_PACKAGE} package: {roots}")
    panel, samples = ev.load_dataset(roots[0])
    if samples.asof_date.max() > LAST_SIGNAL_DATE:
        raise RuntimeError("OOS package reaches past the allowed signal dates")
    store, counts = ev.WindowStore(panel), {}
    dates = sorted(samples.asof_date.unique())
    for date in dates[:LIMIT] if LIMIT else dates:
        records = samples[samples.asof_date == date].to_dict("records")
        records = records[:LIMIT * 64] if LIMIT else records
        identities, vectors = [], []
        for offset in range(0, len(records), BATCH):
            items = [store.prepare(r) for r in records[offset:offset + BATCH]]
            vectors.append(hidden_states(
                model, tokenizer,
                torch.as_tensor(np.stack([i["x"] for i in items])),
                torch.as_tensor(np.stack([i["stamp"][:120] for i in items])),
                torch.as_tensor([i["sector_id"] for i in items]),
                torch.as_tensor([i["percentile"] for i in items], dtype=torch.float32), device))
            identities.extend(i["identity"] for i in items)
        hidden = np.concatenate(vectors)
        if not np.isfinite(hidden).all():
            raise RuntimeError(f"Non-finite hidden state on {date}")
        np.savez_compressed(OUTPUT / f"oos_hidden_{date}.npz", identity=np.asarray(identities), hidden=hidden)
        counts[date] = len(identities)
        phase("oos_date_done", date=date, windows=len(identities))
    return counts


def run_val(model, tokenizer, device):
    val_path = sorted(INPUT.glob(f"**/{VAL_PACKAGE}/val_data.pkl"))
    meta_path = sorted(INPUT.glob(f"**/{VAL_PACKAGE}/asset_metadata.csv"))
    if len(val_path) != 1 or len(meta_path) != 1:
        raise RuntimeError(f"Expected one val panel and one metadata file: {val_path} {meta_path}")
    os.environ.update({
        "KRONOS_VAL_DATA_PATHS": str(val_path[0]), "KRONOS_TRAIN_DATA_PATHS": str(val_path[0]),
        "KRONOS_METADATA_PATH": str(meta_path[0]),
        "KRONOS_LOOKBACK_WINDOW": "120", "KRONOS_PREDICT_WINDOW": "10", "KRONOS_NUM_SECTORS": "86",
        "KRONOS_NUM_SIZE_BUCKETS": "0", "KRONOS_USE_SECTOR_FEATURES": "1", "KRONOS_USE_SIZE_FEATURES": "0",
        "KRONOS_USE_SIZE_PERCENTILE": "1", "KRONOS_VAL_SIGNAL_START": "2025-07-01",
        "KRONOS_VAL_SIGNAL_END": "2026-07-02", "KRONOS_VALIDATION_SAMPLES": "0",
        "KRONOS_VALIDATION_FULL_ONLY": "1", "KRONOS_VALIDATION_QUICK_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "0",
    })
    from dataset import QlibDataset
    import evaluate_base_sealed as ev
    dataset = QlibDataset("val")
    if dataset.asset_metadata.sector_map != ev.SECTOR_MAP:
        raise RuntimeError("Validation sector ids differ from the evaluator's")
    dataset.active_positions = np.arange(dataset.total_samples)      # idx == position in dataset.indices
    total = min(dataset.total_samples, LIMIT * 64) if LIMIT else dataset.total_samples
    symbols, asof, returns = [], [], []
    for symbol, start in dataset.indices[:total]:
        close = dataset.data[symbol]["close"].to_numpy()
        symbols.append(str(symbol))
        asof.append(str(dataset.timestamps_by_symbol[str(symbol)].iloc[start + 119].date()))
        returns.append(float(close[start + 129] / close[start + 119] - 1.0))   # same label as the evaluator
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(total)), batch_size=BATCH,
                                         shuffle=False, num_workers=WORKERS)
    vectors = [hidden_states(model, tokenizer, batch[0][:, :120], batch[1][:, :120], batch[2], batch[4], device)
               for batch in loader]
    hidden = np.concatenate(vectors)
    if len(hidden) != total or not np.isfinite(hidden).all():
        raise RuntimeError("Validation hidden states are incomplete or non-finite")
    frame = pd.DataFrame({"symbol": symbols, "asof_date": asof, "ret10": returns})
    for month, rows in frame.groupby(frame.asof_date.str[:7]):
        np.savez_compressed(OUTPUT / f"val_hidden_{month}.npz", symbol=rows.symbol.to_numpy(),
                            asof_date=rows.asof_date.to_numpy(), ret10=rows.ret10.to_numpy(),
                            hidden=hidden[rows.index.to_numpy()])
    phase("val_done", windows=total, dates=int(frame.asof_date.nunique()), symbols=int(frame.symbol.nunique()))
    return dict(windows=total, dates=int(frame.asof_date.nunique()), symbols=int(frame.symbol.nunique()),
                first=frame.asof_date.min(), last=frame.asof_date.max())


def main():
    from huggingface_hub import snapshot_download
    from model import Kronos, KronosTokenizer
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = sorted(INPUT.glob("**/kronos-small-cosine-c2-best-weights/model.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(f"Expected one C2 checkpoint: {weights}")
    model = Kronos.from_pretrained(str(weights[0].parent), num_sectors=86, num_size_buckets=0, context_layer=6,
                                   use_size_percentile=True, size_mlp_hidden_dim=64).to(device).eval()
    tokenizer_dir = os.environ.get("HIDDEN_TOKENIZER_DIR", "").strip() or snapshot_download(
        repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()
    phase("model_loaded", weights=str(weights[0]), device=str(device))
    summary = {"oos": run_oos(model, tokenizer, device)}
    if os.environ.get("HIDDEN_SKIP_VAL", "0") != "1":
        # The OOS vectors are already on disk; a failed kernel cannot be listed
        # reliably, so a validation-side failure is recorded instead of raised.
        try:
            summary["val"] = run_val(model, tokenizer, device)
        except Exception as error:  # noqa: BLE001
            summary["val_error"] = f"{type(error).__name__}: {error}"
            phase("val_failed", error=summary["val_error"])
    summary["oos_windows"] = int(sum(summary["oos"].values()))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    phase("finished", oos_dates=len(summary["oos"]), oos_windows=summary["oos_windows"],
          val=summary.get("val"), val_error=summary.get("val_error"))


if __name__ == "__main__":
    main()
