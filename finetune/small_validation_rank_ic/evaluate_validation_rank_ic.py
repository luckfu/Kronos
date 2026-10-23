"""Offline validation Rank IC for C2 vs the clean-line checkpoints.

This does not change training.  It builds one frozen subset of the 520-symbol
validation panel (20 signal dates x 200 symbols), generates one path per
window with each checkpoint, and computes the mean same-day Spearman IC of
predicted D10 return vs actual D10 return.  The point is to check whether
validation IC ranks the checkpoints in the same order as the sealed OOS.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "finetune") not in sys.path:
    sys.path.insert(0, str(ROOT / "finetune"))

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK, PREDICT, WINDOW = 120, 10, 130
NUM_SECTORS = 86
TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
SEED = 20261023
N_DATES, N_SYMBOLS = 20, 200
FIRST_DATE, LAST_DATE = "2025-07-03", "2026-07-02"
OUTPUT = Path("/kaggle/working/validation_rank_ic")
LABELS = ("c2_best", "c03_best", "epoch2_best", "e03b_best")
PATTERNS = {
    "c2_best": "**/kronos-small-c2-best-checkpoint/model.safetensors",
    "c03_best": "**/small_0.1_clean_wc_dual_t4_20261011/checkpoints/best_model/model.safetensors",
    "epoch2_best": "**/kronos-small-epoch2-best-checkpoint/model.safetensors",
    "e03b_best": "**/small_0.1_clean_bigbatch_20261022/checkpoints/best_model/model.safetensors",
}


def phase(name, **values):
    print(json.dumps(dict(time=time.time(), phase=name, **values)), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(root, pattern):
    paths = sorted(Path(root).glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern}; found {paths[:8]}")
    return paths[0]


def build_subset():
    from asset_metadata import AssetMetadata

    val_path = find_one("/kaggle/input", "**/a-share-120d-temporal-symbol-holdout/val_data.pkl")
    metadata_path = find_one("/kaggle/input", "**/a-share-120d-temporal-symbol-holdout/asset_metadata.csv")
    phase("val_inputs", val_data=str(val_path), metadata=str(metadata_path))
    import pickle
    with val_path.open("rb") as handle:
        panel = pickle.load(handle)
    metadata = AssetMetadata(str(metadata_path), NUM_SECTORS, 0)

    dates = sorted({ts for frame in panel.values() for ts in frame.index})
    eligible = [ts for ts in dates
                if pd.Timestamp(FIRST_DATE) <= ts <= pd.Timestamp(LAST_DATE)]
    picks = sorted(set(np.linspace(0, len(eligible) - 1, N_DATES).round().astype(int)))
    selected_dates = [eligible[i] for i in picks]

    rows = []
    for ts in selected_dates:
        candidates = []
        for symbol, frame in panel.items():
            if ts not in frame.index:
                continue
            index = int(frame.index.get_loc(ts))
            if index >= LOOKBACK - 1 and index + PREDICT < len(frame):
                candidates.append(symbol)
        candidates.sort()
        rng = random.Random(SEED + int(ts.value // 10**9))
        chosen = sorted(rng.sample(candidates, min(N_SYMBOLS, len(candidates))))
        for symbol in chosen:
            frame = panel[symbol]
            index = int(frame.index.get_loc(ts))
            start = index - (LOOKBACK - 1)
            close = frame["close"].to_numpy(dtype=np.float64)
            actual = float(close[index + PREDICT] / close[index] - 1.0)
            sector_id, _, _ = metadata.get_conditions(symbol, asof=ts)
            size = float(frame["size_percentile"].iloc[index])
            if not np.isfinite(actual) or not np.isfinite(size):
                continue
            rows.append(dict(
                symbol=symbol, asof_date=str(ts.date()),
                target_date=str(frame.index[index + PREDICT].date()),
                start_index=start, actual_return_d10=actual,
                sector_id=int(sector_id), size_percentile=float(np.clip(size, 0, 1)),
            ))
    subset = pd.DataFrame(rows)
    if len(subset) < 3000:
        raise RuntimeError(f"Validation subset too small: {len(subset)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUTPUT / "subset.csv", index=False)
    (OUTPUT / "subset_meta.json").write_text(json.dumps(dict(
        dates=[str(ts.date()) for ts in selected_dates],
        samples=int(len(subset)), dates_count=len(selected_dates),
        symbols_per_date=N_SYMBOLS, seed=SEED,
        val_data_sha=sha(val_path), metadata_sha=sha(metadata_path),
    ), indent=2))
    phase("subset_ready", samples=len(subset), dates=len(selected_dates))


def spearman(x, y):
    x = pd.Series(np.asarray(x, dtype=np.float64))
    y = pd.Series(np.asarray(y, dtype=np.float64))
    return float(x.rank().corr(y.rank()))


def evaluate_checkpoint(label, subset, panel, tokenizer, tokenizer_dir):
    import torch
    from model import Kronos

    weights = find_one("/kaggle/input", PATTERNS[label])
    phase("model_load", label=label, weights=str(weights), sha=sha(weights))
    device = torch.device("cuda:0")
    model = Kronos.from_pretrained(
        str(weights.parent), num_sectors=NUM_SECTORS, num_size_buckets=0,
        context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    from model.kronos import auto_regressive_inference

    predictions = []
    batch_size = 24
    for offset in range(0, len(subset), batch_size):
        batch = subset[offset:offset + batch_size]
        xs, stamps, sectors, sizes, actuals, last_closes = [], [], [], [], [], []
        for row in batch.itertuples(index=False):
            frame = panel[row.symbol]
            window = frame.iloc[int(row.start_index):int(row.start_index) + WINDOW]
            if len(window) != WINDOW:
                raise RuntimeError(f"Incomplete window: {row.symbol} {row.asof_date}")
            values = window[FEATURES].to_numpy(dtype=np.float32)
            raw_close = window["close"].to_numpy(dtype=np.float64)
            mean, std = values[:LOOKBACK].mean(0), values[:LOOKBACK].std(0)
            x = np.clip((values - mean) / (std + 1e-5), -5, 5)[:LOOKBACK]
            dates = pd.DatetimeIndex(window.index)
            stamp = np.column_stack([
                dates.minute, dates.hour, dates.weekday,
                dates.day, dates.month,
            ]).astype(np.float32)
            xs.append(x); stamps.append(stamp)
            sectors.append(int(row.sector_id)); sizes.append(float(row.size_percentile))
            actuals.append(float(row.actual_return_d10)); last_closes.append(float(raw_close[LOOKBACK - 1]))
        x = torch.as_tensor(np.stack(xs), device=device)
        stamp = torch.as_tensor(np.stack(stamps), device=device)
        sector = torch.as_tensor(sectors, device=device)
        size = torch.as_tensor(sizes, dtype=torch.float32, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            forecasts = auto_regressive_inference(
                tokenizer, model, x, stamp[:, :LOOKBACK], stamp[:, LOOKBACK:WINDOW],
                max_context=512, pred_len=PREDICT, clip=5, T=0.65, top_k=0,
                top_p=0.8, sample_count=1, verbose=False,
                sector_id=sector, size_percentile=size, return_samples=True,
            )
        for i, row in enumerate(batch.itertuples(index=False)):
            window = panel[row.symbol].iloc[int(row.start_index):int(row.start_index) + WINDOW]
            values = window[FEATURES].to_numpy(dtype=np.float32)
            mean, std = values[:LOOKBACK].mean(0), values[:LOOKBACK].std(0)
            predicted_close = (
                forecasts[i, :, -PREDICT:, 3].astype(np.float64)
                * (float(std[3]) + 1e-5) + float(mean[3])
            ).mean(0)
            predicted = float(predicted_close[-1] / last_closes[i] - 1.0)
            predictions.append((row.asof_date, predicted, actuals[i]))
        if offset % (batch_size * 20) == 0:
            phase("prediction", label=label, done=min(offset + batch_size, len(subset)),
                  total=len(subset))

    frame = pd.DataFrame(predictions, columns=["asof_date", "predicted", "actual"])
    per_date = {}
    for day, group in frame.groupby("asof_date"):
        if len(group) >= 30:
            per_date[day] = spearman(group["predicted"], group["actual"])
    values = list(per_date.values())
    result = dict(
        label=label, weights=str(weights), sha256=sha(weights),
        samples=int(len(frame)), dates=len(values),
        mean_rank_ic=float(np.mean(values)),
        median_rank_ic=float(np.median(values)),
        positive_days=int(sum(value > 0 for value in values)),
        icir=float(np.mean(values) / np.std(values, ddof=1)) if len(values) > 1 else None,
        per_date=per_date,
    )
    phase("checkpoint_done", **{k: result[k] for k in
                                ("label", "mean_rank_ic", "positive_days", "dates")})
    del model
    torch.cuda.empty_cache()
    return result


def worker(rank):
    import torch
    torch.set_num_threads(2)
    import pickle
    subset = pd.read_csv(OUTPUT / "subset.csv")
    with find_one("/kaggle/input", "**/a-share-120d-temporal-symbol-holdout/val_data.pkl").open("rb") as handle:
        panel = pickle.load(handle)
    tokenizer_dir = None
    for candidate in sorted(Path("/kaggle/input").glob("**/Kronos-Tokenizer-base/model.safetensors")):
        if sha(candidate) == TOKENIZER_SHA:
            tokenizer_dir = candidate.parent
            break
    if tokenizer_dir is None:
        raise RuntimeError("Tokenizer SHA not found")
    from model import KronosTokenizer
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir)).to(
        torch.device("cuda:0")).eval()
    assigned = [LABELS[0], LABELS[1]] if rank == 0 else [LABELS[2], LABELS[3]]
    results = {}
    for label in assigned:
        results[label] = evaluate_checkpoint(label, subset, panel, tokenizer, tokenizer_dir)
    (OUTPUT / "results").mkdir(exist_ok=True)
    (OUTPUT / "results" / f"rank{rank}.json").write_text(json.dumps(results, indent=2))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SMALL_VAL_RANK") is not None:
        worker(int(os.environ["SMALL_VAL_RANK"]))
        return
    build_subset()
    phase("spawn_workers")
    processes = []
    for rank in range(2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), SMALL_VAL_RANK=str(rank))
        processes.append(subprocess.Popen([sys.executable, "-u", __file__], env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Worker failures: {codes}")
    summary = {}
    for path in sorted((OUTPUT / "results").glob("rank*.json")):
        summary.update(json.loads(path.read_text()))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    phase("finished", summary={label: value["mean_rank_ic"] for label, value in summary.items()})


if __name__ == "__main__":
    main()
