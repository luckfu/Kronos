"""Zero-training validation generative-IC checkpoint reselection (Beta v2.1 C1).

Question: does the val forecast loss (WFL) ranking of forecast checkpoints agree
with their *generative-derived return10d Rank IC* ranking? Sealed OOS
(kronos_beta_v2_time_oos_through_20260903) is NOT read here.

Validation set: temporal_symbol_validation_v1 ``val_data.pkl`` (520 holdout
symbols, signal 2025-07-01..2026-07-02, 123,836 windows / 242 signal dates;
sha256 4cce31bc...). Windows follow QlibDataset('val') exactly:
window = lookback 120 + predict 10 + 1 = 131 rows, asof row index 119.

Subsample (fixed, deterministic, documented): ``VAL_SUBSAMPLE_DATES`` signal
dates picked by ``np.linspace(0, n_dates - 1, VAL_SUBSAMPLE_DATES).round()``
over the sorted 242 dates (spacing ~10.5 trading days, so 10d labels of
neighbouring picks barely overlap), ALL symbols on each picked date.

Score (same recipe as Baseline 2 production arm):
  predicted_return_d10 = mean over N=5 AR samples (T=0.65, top_p=0.8,
  top_k=0, clip=5, seed=20260906) of denorm_close[d10] / last_close - 1.
Label: return_10d = close[asof+10] / close[asof] - 1 (raw close).
Also per checkpoint, on the same subsample: teacher-forcing weighted forecast
loss (C1 horizon weights 1.364..0.455), per-sample then averaged — identical
to the batch formula used by training validation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import Kronos, KronosTokenizer

from dataset import build_beta_v21_labels
from evaluate_beta_v21_time_oos import (
    FEATURES,
    FORECAST_HORIZON_WEIGHTS,
    LOOKBACK,
    PREDICT,
    WINDOW,
    time_features,
)
from evaluate_beta_v21_generative_return_oos import ARMS, decode_records

VAL_SIGNAL_START = "2025-07-01"
VAL_SIGNAL_END = "2026-07-02"
EXPECTED_VAL_SHA256 = "4cce31bc3e70eab83d5b7ea05f19fce04aa57a87f3acf00b882ddfbac4219bf7"
EXPECTED_VAL_SAMPLES = 123836
EXPECTED_VAL_DATES = 242
EXPECTED_SECTORS = 86
VAL_SUBSAMPLE_DATES = 24
PROD_ARM = next(arm for arm in ARMS if arm["name"] == "prod_t065_p80_n5")
TF_BATCH = 64


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_val_root(input_root: Path) -> Path:
    """Locate temporal_symbol_validation_v1 data root (val_data.pkl + asset_metadata.csv)."""
    matches = sorted(
        {
            path.parent.parent.resolve()
            for path in Path(input_root).glob(
                "**/*temporal_symbol_validation_v1/processed_datasets/val_data.pkl"
            )
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one temporal_symbol_validation_v1 root, found {matches}")
    root = matches[0]
    if not (root / "asset_metadata.csv").is_file():
        raise RuntimeError(f"asset_metadata.csv missing under {root}")
    return root


def load_sector_labels(metadata_csv: Path) -> list[str]:
    """Sorted unique sector strings over the full metadata file (AssetMetadata contract)."""
    sectors: set[str] = set()
    for chunk in pd.read_csv(metadata_csv, usecols=["sector"], chunksize=2_000_000):
        sectors.update(chunk["sector"].dropna().astype(str).unique())
    labels = sorted(sectors)
    if len(labels) != EXPECTED_SECTORS:
        raise RuntimeError(f"Expected {EXPECTED_SECTORS} sectors, found {len(labels)}")
    return labels


def build_val_records(panel: dict) -> list[dict]:
    """Mirror QlibDataset('val') eligibility: every full 131-row window with asof in range."""
    start = np.datetime64(VAL_SIGNAL_START, "D")
    end = np.datetime64(VAL_SIGNAL_END, "D")
    records = []
    for symbol in panel:
        frame = panel[symbol]
        count = len(frame) - WINDOW + 1
        if count <= 0:
            continue
        dates = frame.index.to_numpy(dtype="datetime64[D]")
        for position in range(count):
            asof = dates[position + LOOKBACK - 1]
            if start <= asof <= end:
                records.append(
                    {
                        "symbol": str(symbol),
                        "start_index": int(position),
                        "asof_date": str(asof),
                        "target_date": str(dates[position + LOOKBACK - 1 + PREDICT]),
                    }
                )
    return records


def select_subsample_dates(all_dates: list[str], count: int = VAL_SUBSAMPLE_DATES) -> list[str]:
    ordered = sorted(set(all_dates))
    picks = np.unique(np.linspace(0, len(ordered) - 1, count).round().astype(int))
    return [ordered[index] for index in picks]


class ValWindowStore:
    """Window store for val_data.pkl (inline sector + size_percentile, raw-close label)."""

    def __init__(self, panel: dict, sector_labels: list[str]):
        self.panel = panel
        self.sector_map = {value: index for index, value in enumerate(sector_labels)}
        self.unknown_sector = EXPECTED_SECTORS

    def prepare(self, record: dict) -> dict:
        frame = self.panel[record["symbol"]]
        start = int(record["start_index"])
        window = frame.iloc[start : start + WINDOW]
        if len(window) != WINDOW:
            raise RuntimeError(f"Incomplete window: {record['symbol']}@{record['asof_date']}")
        if str(window.index[LOOKBACK - 1].date()) != record["asof_date"]:
            raise RuntimeError(f"Identity drift: {record['symbol']}@{record['asof_date']}")
        raw = window[FEATURES].to_numpy(dtype=np.float64)
        values = raw.astype(np.float32)
        mean = values[:LOOKBACK].mean(axis=0)
        std = values[:LOOKBACK].std(axis=0)
        normalized = np.clip((values - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        sector = window["sector"].iloc[LOOKBACK - 1]
        sector_id = (
            self.sector_map.get(str(sector), self.unknown_sector)
            if pd.notna(sector)
            else self.unknown_sector
        )
        percentile = float(window["size_percentile"].iloc[LOOKBACK - 1])
        if not math.isfinite(percentile):
            percentile = 0.5
        close = FEATURES.index("close")
        return_10d = float(raw[LOOKBACK - 1 + PREDICT, close] / raw[LOOKBACK - 1, close] - 1.0)
        labels = build_beta_v21_labels(raw, LOOKBACK, record["asof_date"])
        return {
            "symbol": record["symbol"],
            "asof_date": record["asof_date"],
            "target_date": record["target_date"],
            "direction": int(return_10d > 0),
            "return_10d": return_10d,
            "x": normalized,
            "stamp": time_features(window.index),
            "sector_id": int(sector_id),
            "size_percentile": float(np.clip(percentile, 0.0, 1.0)),
            "mean": mean,
            "std": std,
            "utility": float(labels["utility"].item()),
            "date_id": int(labels["date_id"].item()),
        }


def teacher_forcing_losses(model, tokenizer, records, store, device, use_amp=True) -> pd.DataFrame:
    """Per-sample forecast / weighted forecast CE (C1 horizon weights)."""
    weights = torch.as_tensor(FORECAST_HORIZON_WEIGHTS, device=device, dtype=torch.float32)
    weights = weights / weights.sum()
    forecast = slice(LOOKBACK - 1, LOOKBACK - 1 + PREDICT)
    rows = []
    for offset in range(0, len(records), TF_BATCH):
        items = [store.prepare(record) for record in records[offset : offset + TF_BATCH]]
        x = torch.as_tensor(np.stack([item["x"] for item in items]), device=device)
        stamp = torch.as_tensor(np.stack([item["stamp"] for item in items]), device=device)
        sector = torch.as_tensor([item["sector_id"] for item in items], device=device, dtype=torch.long)
        percentile = torch.as_tensor(
            [item["size_percentile"] for item in items], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            encoded = tokenizer.encode(x, half=True)
            token_in = [encoded[0][:, :-1], encoded[1][:, :-1]]
            token_out = [encoded[0][:, 1:], encoded[1][:, 1:]]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(
                    token_in[0], token_in[1], stamp[:, :-1, :],
                    sector_id=sector, size_percentile=percentile,
                    use_teacher_forcing=True, s1_targets=token_out[0],
                    return_auxiliary=False, asof_index=LOOKBACK - 1,
                )
            logits = [part.float() for part in logits]
            ce = []
            for part in range(2):
                ce.append(
                    F.cross_entropy(
                        logits[part][:, forecast].transpose(1, 2),
                        token_out[part][:, forecast],
                        reduction="none",
                    )
                )  # [B, PREDICT]
            weighted = ((ce[0] * weights).sum(1) + (ce[1] * weights).sum(1)) / 2
            plain = (ce[0].mean(1) + ce[1].mean(1)) / 2
        for index, item in enumerate(items):
            rows.append(
                {
                    "symbol": item["symbol"],
                    "asof_date": item["asof_date"],
                    "weighted_forecast_loss": float(weighted[index].item()),
                    "forecast_loss": float(plain[index].item()),
                }
            )
    return pd.DataFrame(rows)


def _rank_ic(frame: pd.DataFrame, score: str, label: str) -> float:
    if len(frame) < 3:
        return float("nan")
    return float(frame[score].rank().corr(frame[label].rank()))


def _top_bottom(frame: pd.DataFrame, score: str, label: str, buckets: int) -> float:
    if len(frame) < buckets * 2:
        return float("nan")
    groups = pd.qcut(frame[score].rank(method="first"), buckets, labels=False)
    means = frame[label].groupby(groups).mean()
    return float(means.iloc[-1] - means.iloc[0])


def score_val_checkpoint(frame: pd.DataFrame) -> dict:
    score = "predicted_return_d10"
    daily = []
    for asof, group in frame.groupby("asof_date"):
        daily.append(
            {
                "asof_date": asof,
                "samples": int(len(group)),
                "return10d_rank_ic": _rank_ic(group, score, "return_10d"),
                "utility_rank_ic": _rank_ic(group, score, "utility"),
                "top_bottom_decile_return10d": _top_bottom(group, score, "return_10d", 10),
                "top_bottom_quintile_return10d": _top_bottom(group, score, "return_10d", 5),
                "mkt_mean_return10d": float(group["return_10d"].mean()),
            }
        )
    daily_frame = pd.DataFrame(daily)
    ic = daily_frame["return10d_rank_ic"].dropna()
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    result = {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "score_kind": "predicted_return_d10_generative_ar_mean",
        "return10d_rank_ic_daily": float(ic.mean()),
        "return10d_rank_ic_pooled": _rank_ic(frame, score, "return_10d"),
        "return10d_rank_icir": float(ic.mean() / ic_std) if ic_std and ic_std > 0 else None,
        "return10d_rank_ic_pos_rate": float((ic > 0).mean()),
        "return10d_rank_ic_se": float(ic_std / math.sqrt(len(ic))) if len(ic) > 1 else None,
        "utility_rank_ic_daily": float(daily_frame["utility_rank_ic"].mean()),
        "top_bottom_decile_return10d": float(daily_frame["top_bottom_decile_return10d"].mean()),
        "top_bottom_quintile_return10d": float(daily_frame["top_bottom_quintile_return10d"].mean()),
        "by_signal_date": daily,
    }
    if "weighted_forecast_loss" in frame.columns:
        result["weighted_forecast_loss_subsample"] = float(frame["weighted_forecast_loss"].mean())
        result["forecast_loss_subsample"] = float(frame["forecast_loss"].mean())
    return result


def compare_rankings(summaries: list[dict]) -> dict:
    """Flag WFL-vs-generative-IC ranking divergence (lower WFL = better, higher IC = better)."""
    usable = [s for s in summaries if s.get("return10d_rank_ic_daily") is not None]
    by_wfl_sub = sorted(usable, key=lambda s: s.get("weighted_forecast_loss_subsample", 1e9))
    by_ic = sorted(usable, key=lambda s: -s["return10d_rank_ic_daily"])
    full = [s for s in usable if s.get("full_val_weighted_forecast_loss") is not None]
    by_wfl_full = sorted(full, key=lambda s: s["full_val_weighted_forecast_loss"])
    order = lambda items: [s["label"] for s in items]
    divergent_sub = order(by_wfl_sub) != order(by_ic)
    full_ic_order = [label for label in order(by_ic) if label in order(by_wfl_full)]
    return {
        "rank_by_wfl_subsample": order(by_wfl_sub),
        "rank_by_wfl_full_val_logged": order(by_wfl_full),
        "rank_by_gen_return10d_ic": order(by_ic),
        "divergence_wfl_subsample_vs_ic": divergent_sub,
        "divergence_wfl_full_vs_ic": order(by_wfl_full) != full_ic_order,
        "best_by_ic": order(by_ic)[0] if by_ic else None,
        "best_by_wfl_subsample": order(by_wfl_sub)[0] if by_wfl_sub else None,
    }


def worker_from_plan(plan_path: Path, rank: int, world_size: int) -> int:
    """(checkpoint, date) round-robin shard worker; writes per-shard CSV.gz."""
    import gc
    import pickle

    plan = json.loads(Path(plan_path).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError(f"worker {rank} has no GPU")
    device = torch.device("cuda:0")
    with Path(plan["val_data"]).open("rb") as handle:
        panel = pickle.load(handle)
    store = ValWindowStore(panel, plan["sector_labels"])
    records_by_date: dict[str, list[dict]] = {}
    for record in json.loads(Path(plan["records_file"]).read_text()):
        records_by_date.setdefault(record["asof_date"], []).append(record)
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer_dir"]).to(device).eval()
    checkpoints = {item["label"]: item for item in plan["checkpoints"]}
    shards = Path(plan["shards"])
    deadline = float(plan["deadline"])
    tasks = [task for index, task in enumerate(plan["tasks"]) if index % world_size == rank]
    print(json.dumps({"phase": "worker_started", "rank": rank, "tasks": len(tasks),
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    current_label, model = None, None
    for task in tasks:
        label = task["checkpoint"]
        shard = shards / f"{label}__{task['date']}.csv.gz"
        if shard.is_file():
            continue
        if time.time() > deadline:
            print(json.dumps({"phase": "worker_deadline", "rank": rank, "skipped_from": task}), flush=True)
            break
        if label != current_label:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            model = (
                Kronos.from_pretrained(checkpoints[label]["path"], local_files_only=True)
                .to(device)
                .eval()
            )
            current_label = label
        started = time.time()
        records = records_by_date[task["date"]]
        generated = decode_records(
            PROD_ARM, model, tokenizer, records, store, device,
            effective_batch=int(plan["effective_batch"]), use_amp=True, label=label,
        )
        losses = teacher_forcing_losses(model, tokenizer, records, store, device)
        merged = generated.merge(losses, on=["symbol", "asof_date"], how="left", validate="one_to_one")
        if merged["weighted_forecast_loss"].isna().any():
            raise RuntimeError(f"TF loss join failed for {label}@{task['date']}")
        staging = shard.with_name(shard.name + ".tmp")
        merged.to_csv(staging, index=False, compression="gzip")
        staging.replace(shard)
        day = score_val_checkpoint(merged)["by_signal_date"][0]
        print(json.dumps({"phase": "shard_done", "rank": rank, "checkpoint": label,
                          "date": task["date"], "rows": int(len(merged)),
                          "seconds": round(time.time() - started, 1),
                          "return10d_rank_ic": day["return10d_rank_ic"],
                          "top_bottom_decile": day["top_bottom_decile_return10d"],
                          "wfl": float(merged["weighted_forecast_loss"].mean())}), flush=True)
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"phase": "worker_finished", "rank": rank}), flush=True)
    return 0


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    if len(_sys.argv) >= 3 and _sys.argv[1] == "--worker-plan":
        raise SystemExit(
            worker_from_plan(
                Path(_sys.argv[2]),
                int(_os.environ["KRONOS_SWEEP_RANK"]),
                int(_os.environ.get("KRONOS_SWEEP_WORLD", "2")),
            )
        )
    raise SystemExit("usage: --worker-plan PLAN (launched by kaggle_beta_v21_c1_val_gen_ic.py)")
