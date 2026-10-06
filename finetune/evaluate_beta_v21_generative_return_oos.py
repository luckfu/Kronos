"""Baseline 2: zero-training generative OHLC → derived 10d return on sealed 18d OOS.

Hypothesis: C2's ~0.18 return10d Rank IC comes from generative AR decode
(auto_regressive_inference sample mean), not a trained return_head. This
module applies the same decode spirit to Beta Seg155 forecast-best
(forecast-only trunk; no aux heads) and scores:

  score = predicted_return_d10
        = mean_over_samples( close_path[d10] / last_close - 1 )

Exact beta-base recipe (mirrors Small C2 / c2_18d_alpha_oos.decode_date):
  - True AR sampling via model.kronos.auto_regressive_inference
    (NOT teacher-forcing CE; NOT Stage3 soft top-N stitch)
  - return_samples=True; per-path denormalize close; mean cumulative terminal
  - Locked decode arms (a priori, same as C2 production lock):
      ranking     T=0.60 / top_p=0.9 / N=16 / seed=20260906
      production  T=0.65 / top_p=0.8 / N=5  / seed=20260906
  - top_k=0, clip=5, max_context=512, pred_len=10, lookback=120
  - sector_id + size_percentile conditioning (Beta contract)

Never trains. Never touches val_data.pkl. Package:
  kronos_beta_v2_time_oos_through_20260903 (08-11→09-03, 92751).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference

from beta_v21 import mean_within_date_spearman, same_date_pairwise_accuracy
from dataset import build_beta_v21_labels
from evaluate_beta_v21_time_oos import (
    EXPECTED_EVALUATION_NAME,
    FEATURES,
    LOOKBACK,
    PREDICT,
    WINDOW,
    find_evaluation_root,
    load_incremental_records,
    time_features,
)


MINIMUM_UTILITY_GAP = 0.005
DECODE_SEED = 20260906

# C2 locked decode arms — same a-priori protocols, applied to Beta Seg155.
ARMS = (
    {
        "name": "prod_t065_p80_n5",
        "sample_count": 5,
        "temperature": 0.65,
        "top_p": 0.8,
        "seed": DECODE_SEED,
        "protocol": "production",
    },
    {
        "name": "rank_t060_p90_n16",
        "sample_count": 16,
        "temperature": 0.6,
        "top_p": 0.9,
        "seed": DECODE_SEED,
        "protocol": "ranking",
    },
)

# Sealed-18d reference numbers for the summary table (zero-GPU prior work).
REFERENCE = {
    "c2_prod_t065_p80_n5": {
        "return10d_rank_ic_daily": 0.1768718542617823,
        "utility_rank_ic_daily": 0.12083752234483813,
        "pairwise_accuracy": 0.565004451282918,
        "score_kind": "predicted_return_d10",
    },
    "c2_rank_t060_p90_n16": {
        "return10d_rank_ic_daily": 0.17962559658922883,
        "utility_rank_ic_daily": 0.12095893153611573,
        "pairwise_accuracy": 0.564927890049612,
        "score_kind": "predicted_return_d10",
    },
    "seg19_rank_frozen_best": {
        "return10d_rank_ic_daily": -0.007326050761061286,
        "utility_rank_ic_daily": 0.029951721395212755,
        "pairwise_accuracy": 0.5153679052244999,
        "score_kind": "expected_utility_score (return_head path)",
        "weighted_forecast_loss": 2.4360358587494653,
    },
    "seg155_forecast_best_teacher_forcing": {
        "weighted_forecast_loss": 2.4360358587494653,
        "score_kind": None,
        "note": "prior rank-OOS had no generative return score",
    },
}


class GenReturnWindowStore:
    """Window store with mean/std (for denorm) + Beta utility labels."""

    def __init__(self, panel: dict, sector_labels: list[str]):
        self.panel = panel
        self.sector_map = {value: index for index, value in enumerate(sector_labels)}

    def prepare(self, record: dict) -> dict:
        frame = self.panel[str(record["symbol"])]
        start = int(record["start_index"])
        window = frame.iloc[start : start + WINDOW]
        if len(window) != WINDOW:
            raise RuntimeError(
                f"Incomplete window: {record['symbol']}@{record['asof_date']}"
            )
        asof = window.index[LOOKBACK - 1]
        target = window.index[LOOKBACK - 1 + PREDICT]
        if str(asof.date()) != record["asof_date"] or str(target.date()) != record[
            "target_date"
        ]:
            raise RuntimeError(
                f"Sample identity drift: {record['symbol']}@{record['asof_date']}"
            )
        raw = window[FEATURES].to_numpy(dtype=np.float64)
        values = raw.astype(np.float32)
        mean = values[:LOOKBACK].mean(axis=0)
        std = values[:LOOKBACK].std(axis=0)
        normalized = np.clip((values - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        sector = str(window["sector"].iloc[LOOKBACK - 1])
        percentile = float(window["size_percentile"].iloc[LOOKBACK - 1])
        if not math.isfinite(percentile):
            percentile = 0.5
        labels = build_beta_v21_labels(raw, LOOKBACK, record["asof_date"])
        return {
            "symbol": str(record["symbol"]),
            "asof_date": record["asof_date"],
            "target_date": record["target_date"],
            "direction": record.get("direction"),
            "return_10d": float(record["return_10d"]),
            "x": normalized,
            "stamp": time_features(window.index),
            "sector_id": int(self.sector_map.get(sector, len(self.sector_map))),
            "size_percentile": float(np.clip(percentile, 0.0, 1.0)),
            "mean": mean,
            "std": std,
            "utility": float(labels["utility"].item()),
            "date_id": int(labels["date_id"].item()),
        }


def batches(records, store, batch_size):
    for offset in range(0, len(records), batch_size):
        yield [store.prepare(record) for record in records[offset : offset + batch_size]]


def stack_batch(items, device):
    return {
        "x": torch.as_tensor(np.stack([item["x"] for item in items]), device=device),
        "stamp": torch.as_tensor(
            np.stack([item["stamp"] for item in items]), device=device
        ),
        "sector": torch.as_tensor(
            [item["sector_id"] for item in items], device=device, dtype=torch.long
        ),
        "percentile": torch.as_tensor(
            [item["size_percentile"] for item in items],
            device=device,
            dtype=torch.float32,
        ),
    }


def to_daily(cumulative: np.ndarray) -> np.ndarray:
    previous = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], axis=-1
    )
    return (1.0 + cumulative) / (1.0 + previous) - 1.0


def decode_records(
    arm: dict,
    model,
    tokenizer,
    records: list[dict],
    store: GenReturnWindowStore,
    device: torch.device,
    effective_batch: int,
    use_amp: bool,
    label: str,
) -> pd.DataFrame:
    """C2-spirit generative decode → predicted_return_d10 (sample-mean terminal)."""
    sample_count = int(arm["sample_count"])
    batch_size = max(1, int(effective_batch) // sample_count)
    close_index = FEATURES.index("close")
    torch.manual_seed(int(arm["seed"]))
    np.random.seed(int(arm["seed"]))

    rows = []
    for items in batches(records, store, batch_size):
        batch = stack_batch(items, device)
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                forecast = auto_regressive_inference(
                    tokenizer,
                    model,
                    batch["x"][:, :LOOKBACK],
                    batch["stamp"][:, :LOOKBACK],
                    batch["stamp"][:, LOOKBACK : LOOKBACK + PREDICT],
                    max_context=512,
                    pred_len=PREDICT,
                    clip=5,
                    T=float(arm["temperature"]),
                    top_k=0,
                    top_p=float(arm["top_p"]),
                    sample_count=sample_count,
                    verbose=False,
                    sector_id=batch["sector"],
                    size_percentile=batch["percentile"],
                    return_samples=True,
                )
        # forecast: [B, N, lookback+pred, F] — take future close paths
        future = forecast[:, :, -PREDICT:, close_index]
        for index, item in enumerate(items):
            scale = float(item["std"][close_index]) + 1e-5
            shift = float(item["mean"][close_index])
            last_close = float(item["x"][LOOKBACK - 1, close_index]) * scale + shift
            paths = future[index].astype(np.float64) * scale + shift
            cumulative = paths / last_close - 1.0
            mean_cumulative = cumulative.mean(axis=0)
            actual = (
                item["x"][LOOKBACK : LOOKBACK + PREDICT, close_index].astype(np.float64)
                * scale
                + shift
            ) / last_close - 1.0
            predicted_daily = to_daily(cumulative)
            actual_daily = to_daily(actual[None, ...])[0]
            row = {
                "model": label,
                "arm": arm["name"],
                "protocol": arm["protocol"],
                "sample_count": sample_count,
                "temperature": float(arm["temperature"]),
                "top_p": float(arm["top_p"]),
                "seed": int(arm["seed"]),
                "symbol": item["symbol"],
                "asof_date": item["asof_date"],
                "target_date": item["target_date"],
                "direction": item["direction"],
                "return_10d": item["return_10d"],
                "utility": item["utility"],
                "date_id": item["date_id"],
                "predicted_return_d10": float(mean_cumulative[-1]),
                "predicted_path_vol": float(predicted_daily.std(axis=1).mean()),
                "predicted_terminal_dispersion": (
                    float(cumulative[:, -1].std(ddof=1)) if sample_count > 1 else None
                ),
                "realized_path_vol": float(actual_daily.std()),
                "score_kind": "predicted_return_d10_generative_ar_mean",
            }
            for horizon in range(PREDICT):
                row[f"predicted_return_d{horizon + 1}"] = float(mean_cumulative[horizon])
                row[f"actual_return_d{horizon + 1}"] = float(actual[horizon])
            rows.append(row)
    return pd.DataFrame(rows)


def score_predictions(frame: pd.DataFrame) -> dict:
    """Score generative predicted_return_d10 vs return_10d and utility."""
    if frame.empty:
        raise RuntimeError("empty predictions")
    scores = torch.as_tensor(frame["predicted_return_d10"].to_numpy(np.float32))
    utilities = torch.as_tensor(frame["utility"].to_numpy(np.float32))
    returns = torch.as_tensor(frame["return_10d"].to_numpy(np.float32))
    date_ids = torch.as_tensor(frame["date_id"].to_numpy(np.int64).copy())

    pairwise, pair_count = same_date_pairwise_accuracy(
        scores, utilities, date_ids, minimum_gap=MINIMUM_UTILITY_GAP
    )
    utility_rank_ic, utility_dates = mean_within_date_spearman(
        scores, utilities, date_ids
    )
    return10d_rank_ic, return_dates = mean_within_date_spearman(
        scores, returns, date_ids
    )
    score_rank = frame["predicted_return_d10"].rank()
    pooled_utility = float(score_rank.corr(frame["utility"].rank()))
    pooled_return = float(score_rank.corr(frame["return_10d"].rank()))

    daily = []
    for asof, group in frame.groupby("asof_date"):
        if len(group) < 2:
            continue
        g_scores = torch.as_tensor(group["predicted_return_d10"].to_numpy(np.float32))
        g_utils = torch.as_tensor(group["utility"].to_numpy(np.float32))
        g_returns = torch.as_tensor(group["return_10d"].to_numpy(np.float32))
        g_dates = torch.as_tensor(group["date_id"].to_numpy(np.int64).copy())
        day_util_ic, _ = mean_within_date_spearman(g_scores, g_utils, g_dates)
        day_ret_ic, _ = mean_within_date_spearman(g_scores, g_returns, g_dates)
        day_pw, day_pairs = same_date_pairwise_accuracy(
            g_scores, g_utils, g_dates, minimum_gap=MINIMUM_UTILITY_GAP
        )
        daily.append(
            {
                "asof_date": asof,
                "samples": int(len(group)),
                "return10d_rank_ic": day_ret_ic,
                "utility_rank_ic": day_util_ic,
                "pairwise_accuracy": day_pw,
                "pairwise_pairs": day_pairs,
            }
        )
    daily_frame = pd.DataFrame(daily)
    ret_series = daily_frame["return10d_rank_ic"].dropna()
    util_series = daily_frame["utility_rank_ic"].dropna()
    ret_std = float(ret_series.std(ddof=1)) if len(ret_series) > 1 else None
    util_std = float(util_series.std(ddof=1)) if len(util_series) > 1 else None

    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "score_kind": "predicted_return_d10_generative_ar_mean",
        "decode": {
            "arm": str(frame["arm"].iloc[0]),
            "protocol": str(frame["protocol"].iloc[0]),
            "temperature": float(frame["temperature"].iloc[0]),
            "top_p": float(frame["top_p"].iloc[0]),
            "sample_count": int(frame["sample_count"].iloc[0]),
            "seed": int(frame["seed"].iloc[0]),
            "teacher_forcing": False,
            "definition": (
                "mean_over_AR_samples(denorm_close[d10] / last_close - 1); "
                "same spirit as C2 predicted_return_d10"
            ),
        },
        "return10d_rank_ic_daily": float(ret_series.mean()) if len(ret_series) else None,
        "return10d_rank_ic_pooled": pooled_return,
        "return10d_rank_icir": (
            None
            if ret_std is None or ret_std == 0 or not len(ret_series)
            else float(ret_series.mean() / ret_std)
        ),
        "return10d_rank_ic_pos_rate": (
            float((ret_series > 0).mean()) if len(ret_series) else None
        ),
        "return10d_rank_ic_contract_dates": int(return_dates),
        "utility_rank_ic_daily": float(util_series.mean()) if len(util_series) else None,
        "utility_rank_ic_pooled": pooled_utility,
        "utility_rank_icir": (
            None
            if util_std is None or util_std == 0 or not len(util_series)
            else float(util_series.mean() / util_std)
        ),
        "utility_rank_ic_pos_rate": (
            float((util_series > 0).mean()) if len(util_series) else None
        ),
        "utility_rank_ic_contract_dates": int(utility_dates),
        "pairwise_accuracy": pairwise,
        "pairwise_pairs": int(pair_count),
        "pairwise_contract": (
            f"same-day |Δutility|>={MINIMUM_UTILITY_GAP}; "
            "score=predicted_return_d10; ties wrong"
        ),
        "by_signal_date": daily,
    }


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_path: Path, device: torch.device):
    model = (
        Kronos.from_pretrained(str(model_path), local_files_only=True)
        .to(device)
        .eval()
    )
    use_aux = bool(getattr(model, "use_beta_v21_auxiliary", False))
    return model, use_aux


def build_summary(
    label: str,
    model_path: Path,
    arm: dict,
    metrics: dict,
    seconds: float,
    use_aux: bool,
    model_sha: str | None,
) -> dict:
    return {
        "label": label,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "use_beta_v21_auxiliary": use_aux,
        "seconds": seconds,
        "evaluation_name": EXPECTED_EVALUATION_NAME,
        "baseline": "baseline_2_generative_return",
        "training": False,
        **metrics,
        "reference": REFERENCE,
        "vs_c2_rank_return10d_ic": (
            None
            if metrics.get("return10d_rank_ic_daily") is None
            else float(
                metrics["return10d_rank_ic_daily"]
                - REFERENCE["c2_rank_t060_p90_n16"]["return10d_rank_ic_daily"]
            )
        ),
        "vs_c2_same_arm_return10d_ic": (
            None
            if metrics.get("return10d_rank_ic_daily") is None
            or f"c2_{arm['name']}" not in REFERENCE
            else float(
                metrics["return10d_rank_ic_daily"]
                - REFERENCE[f"c2_{arm['name']}"]["return10d_rank_ic_daily"]
            )
        ),
        "vs_seg19_return_head_return10d_ic": (
            None
            if metrics.get("return10d_rank_ic_daily") is None
            else float(
                metrics["return10d_rank_ic_daily"]
                - REFERENCE["seg19_rank_frozen_best"]["return10d_rank_ic_daily"]
            )
        ),
    }


def worker_from_plan(plan_path: Path, rank: int, world_size: int) -> int:
    """Dual-GPU shard worker: (arm, date) round-robin; writes per-shard CSV.gz."""
    import gc
    import pickle

    plan = json.loads(Path(plan_path).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError(f"worker {rank} has no GPU")
    device = torch.device("cuda:0")
    evaluation_root = Path(plan["evaluation_root"])
    manifest = json.loads((evaluation_root / "evaluation_manifest.json").read_text())
    samples_path = evaluation_root / manifest["artifacts"]["samples_file"]
    panel_path = evaluation_root / manifest["artifacts"]["panel_file"]
    with panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    all_records = load_incremental_records(samples_path, manifest)
    store = GenReturnWindowStore(panel, manifest["model_contract"]["sector_labels"])
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer_dir"]).to(device).eval()
    model = (
        Kronos.from_pretrained(plan["checkpoint_dir"], local_files_only=True)
        .to(device)
        .eval()
    )
    arms_by_name = {arm["name"]: arm for arm in ARMS}
    shards = Path(plan["shards"])
    deadline = float(plan["deadline"])
    tasks = [task for index, task in enumerate(plan["tasks"]) if index % world_size == rank]
    print(json.dumps({"phase": "worker_started", "rank": rank, "tasks": len(tasks),
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    for task in tasks:
        arm = arms_by_name[task["arm"]]
        shard = shards / f"{arm['name']}_{task['date']}.csv.gz"
        if shard.is_file():
            continue
        if time.time() > deadline:
            print(json.dumps({"phase": "worker_deadline", "rank": rank,
                              "skipped_from": task}), flush=True)
            break
        started = time.time()
        date_records = [row for row in all_records if row["asof_date"] == task["date"]]
        result = decode_records(
            arm, model, tokenizer, date_records, store, device,
            effective_batch=int(plan["effective_batch"]), use_amp=True,
            label=plan["checkpoint_label"],
        )
        staging = shard.with_name(shard.name + ".tmp")
        result.to_csv(staging, index=False, compression="gzip")
        staging.replace(shard)
        day = {}
        try:
            day_metrics = score_predictions(result)["by_signal_date"]
            day = day_metrics[0] if day_metrics else {}
        except Exception as exc:  # metrics are best-effort; shard is already saved
            day = {"metric_error": repr(exc)}
        print(json.dumps({"phase": "shard_done", "rank": rank, "arm": arm["name"],
                          "date": task["date"], "rows": int(len(result)),
                          "seconds": round(time.time() - started, 1),
                          "return10d_rank_ic": day.get("return10d_rank_ic"),
                          "utility_rank_ic": day.get("utility_rank_ic"),
                          "pairwise_accuracy": day.get("pairwise_accuracy"),
                          "pairwise_pairs": day.get("pairwise_pairs")}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"phase": "worker_finished", "rank": rank}), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Baseline 2 generative return OOS for Beta Seg155"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PATH"))
    parser.add_argument("--arm", action="append", default=None,
                        help="Arm name filter (default: all locked arms)")
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dates", nargs="*", default=None,
                        help="Optional asof_date filter for shard workers")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pickle

    evaluation_root, manifest = find_evaluation_root(args.input_root)
    samples_path = evaluation_root / manifest["artifacts"]["samples_file"]
    panel_path = evaluation_root / manifest["artifacts"]["panel_file"]
    with panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    records = load_incremental_records(samples_path, manifest)
    if args.dates:
        wanted = set(args.dates)
        records = [row for row in records if row["asof_date"] in wanted]
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]
    store = GenReturnWindowStore(panel, manifest["model_contract"]["sector_labels"])
    tokenizer = KronosTokenizer.from_pretrained(str(args.tokenizer)).to(device).eval()

    arms = list(ARMS)
    if args.arm:
        wanted_arms = set(args.arm)
        arms = [arm for arm in arms if arm["name"] in wanted_arms]
        if not arms:
            raise RuntimeError(f"No arms matched {sorted(wanted_arms)}")

    summaries = []
    for label, path_str in args.checkpoint:
        model_path = Path(path_str)
        model, use_aux = load_model(model_path, device)
        weights = model_path / "model.safetensors"
        model_sha = sha256_file(weights) if weights.is_file() else None
        print(
            json.dumps(
                {
                    "phase": "checkpoint_loaded",
                    "label": label,
                    "path": str(model_path),
                    "sha256": model_sha,
                    "use_beta_v21_auxiliary": use_aux,
                    "note": (
                        "aux heads unused for generative AR decode"
                        if use_aux
                        else "forecast-only checkpoint (preferred for baseline 2)"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for arm in arms:
            started = time.time()
            print(
                json.dumps(
                    {
                        "phase": "decode_start",
                        "label": label,
                        "arm": arm["name"],
                        "records": len(records),
                        "effective_batch": args.effective_batch,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            frame = decode_records(
                arm,
                model,
                tokenizer,
                records,
                store,
                device,
                effective_batch=args.effective_batch,
                use_amp=use_amp,
                label=label,
            )
            metrics = score_predictions(frame)
            seconds = time.time() - started
            summary = build_summary(
                label, model_path, arm, metrics, seconds, use_aux, model_sha
            )
            summaries.append(summary)
            stem = f"{label}_{arm['name']}"
            frame.to_csv(
                args.output_dir / f"{stem}_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            (args.output_dir / f"{stem}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "phase": "decode_done",
                        "label": label,
                        "arm": arm["name"],
                        "return10d_rank_ic_daily": summary["return10d_rank_ic_daily"],
                        "utility_rank_ic_daily": summary["utility_rank_ic_daily"],
                        "pairwise_accuracy": summary["pairwise_accuracy"],
                        "vs_c2": summary["vs_c2_rank_return10d_ic"],
                        "vs_seg19_return_head": summary[
                            "vs_seg19_return_head_return10d_ic"
                        ],
                        "seconds": round(seconds, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = {
        "evaluation_name": EXPECTED_EVALUATION_NAME,
        "baseline": "baseline_2_generative_return",
        "training": False,
        "score_definition": (
            "predicted_return_d10 = mean_N AR samples of "
            "(denorm_close_d10 / last_close - 1); NOT return_head"
        ),
        "arms": [arm["name"] for arm in arms],
        "checkpoints": summaries,
        "reference": REFERENCE,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"phase": "comparison", **{
        k: comparison[k] for k in ("baseline", "arms")
    }, "n_summaries": len(summaries)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    if len(_sys.argv) >= 2 and _sys.argv[1] == "--worker-plan":
        raise SystemExit(
            worker_from_plan(
                Path(_sys.argv[2]),
                int(_os.environ["KRONOS_SWEEP_RANK"]),
                int(_os.environ.get("KRONOS_SWEEP_WORLD", "2")),
            )
        )
    raise SystemExit(main())
