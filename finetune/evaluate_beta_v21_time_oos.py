"""True time-OOS evaluation for Beta v2.1 C1 ranking checkpoints.

Reads the sealed package kronos_beta_v2_time_oos_through_20260903
(evaluation_manifest.json / evaluation_panel.pkl / evaluation_samples.jsonl)
and scores one or more checkpoints with the training contract:

- teacher-forcing CE (forecast + weighted_forecast with C1 horizon weights)
- expected_utility_score from return+barrier heads
- same-day pairwise accuracy with |utility gap| >= 0.005
- mean within-date Spearman rank IC (+ ICIR, % positive days)

Never uses val_data.pkl. Never trains.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import Kronos, KronosTokenizer

from beta_v21 import (
    expected_utility_score,
    mean_within_date_spearman,
    same_date_pairwise_accuracy,
)
from dataset import build_beta_v21_labels


FEATURES = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT + 1
EXPECTED_EVALUATION_NAME = "kronos_beta_v2_time_oos_through_20260903"
FORECAST_HORIZON_WEIGHTS = (
    1.364,
    1.364,
    1.364,
    1.136,
    1.136,
    0.909,
    0.909,
    0.682,
    0.682,
    0.455,
)
MINIMUM_UTILITY_GAP = 0.005


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_features(index) -> np.ndarray:
    index = pd.DatetimeIndex(index)
    return np.column_stack(
        [index.minute, index.hour, index.weekday, index.day, index.month]
    ).astype(np.float32)


def find_evaluation_root(input_root: Path) -> tuple[Path, dict]:
    matches = []
    for manifest_path in Path(input_root).glob("**/evaluation_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("name") != EXPECTED_EVALUATION_NAME:
            continue
        if manifest.get("purpose") != "evaluation_only_never_train_or_tune":
            continue
        isolation = manifest.get("temporal_isolation", {})
        if not isolation.get("targets_strictly_after_training_target_end"):
            continue
        if not isolation.get("incremental_signal_start"):
            continue
        matches.append((manifest_path.parent, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {EXPECTED_EVALUATION_NAME} package under {input_root}, "
            f"found {len(matches)}"
        )
    root, manifest = matches[0]
    artifacts = manifest["artifacts"]
    for key, sha_key in (("panel_file", "panel_sha256"), ("samples_file", "samples_sha256")):
        path = root / artifacts[key]
        actual = sha256_file(path)
        if actual != artifacts[sha_key]:
            raise RuntimeError(f"Evaluation artifact SHA mismatch: {path}")
    return root, manifest


def load_incremental_records(samples_path: Path, manifest: dict) -> list[dict]:
    isolation = manifest["temporal_isolation"]
    start = isolation["incremental_signal_start"]
    end = isolation["incremental_signal_end"]
    records = []
    with Path(samples_path).open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("set") not in {"incremental_future_all", "future_all"}:
                continue
            asof = record["asof_date"]
            if start <= asof <= end:
                records.append(record)
    if not records:
        raise RuntimeError("No incremental_future_all records in sealed signal range")
    return records


class OOSWindowStore:
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
            "labels": labels,
        }


def iter_batches(records, store, batch_size):
    for offset in range(0, len(records), batch_size):
        yield [store.prepare(record) for record in records[offset : offset + batch_size]]


def stack_batch(items, device):
    x = torch.as_tensor(np.stack([item["x"] for item in items]), device=device)
    stamp = torch.as_tensor(np.stack([item["stamp"] for item in items]), device=device)
    sector = torch.as_tensor(
        [item["sector_id"] for item in items], device=device, dtype=torch.long
    )
    percentile = torch.as_tensor(
        [item["size_percentile"] for item in items],
        device=device,
        dtype=torch.float32,
    )
    labels = {
        key: torch.stack([item["labels"][key] for item in items]).to(device)
        for key in (
            "return_targets",
            "return_scales",
            "barrier_target",
            "barrier_valid",
            "utility",
            "date_id",
        )
    }
    return {
        "x": x,
        "stamp": stamp,
        "sector": sector,
        "percentile": percentile,
        "labels": labels,
        "meta": items,
    }


def objective_token_slices(token_len, lookback, predict):
    history = slice(0, lookback - 1)
    forecast = slice(lookback - 1, lookback - 1 + predict)
    if forecast.stop > token_len:
        raise ValueError("token length shorter than lookback+predict contract")
    return history, forecast


def batch_token_losses(logits, targets):
    """Match train_predictor compute_predictor_losses for forecast mode."""
    history_slice, forecast_slice = objective_token_slices(
        targets[0].shape[1], LOOKBACK, PREDICT
    )

    def dual_mean(selected):
        s1 = F.cross_entropy(
            logits[0][:, selected].transpose(1, 2),
            targets[0][:, selected],
            reduction="mean",
        )
        s2 = F.cross_entropy(
            logits[1][:, selected].transpose(1, 2),
            targets[1][:, selected],
            reduction="mean",
        )
        return (s1 + s2) / 2

    forecast_loss = dual_mean(forecast_slice)
    history_loss = dual_mean(history_slice)
    weights = torch.as_tensor(
        FORECAST_HORIZON_WEIGHTS, device=logits[0].device, dtype=logits[0].dtype
    )
    weights = weights / weights.sum()
    weighted_s1 = F.cross_entropy(
        logits[0][:, forecast_slice].transpose(1, 2),
        targets[0][:, forecast_slice],
        reduction="none",
    ).mean(0)
    weighted_s2 = F.cross_entropy(
        logits[1][:, forecast_slice].transpose(1, 2),
        targets[1][:, forecast_slice],
        reduction="none",
    ).mean(0)
    weighted_forecast = (torch.sum(weighted_s1 * weights) + torch.sum(weighted_s2 * weights)) / 2
    return {
        "forecast_loss": float(forecast_loss.item()),
        "history_loss": float(history_loss.item()),
        "weighted_forecast_loss": float(weighted_forecast.item()),
        "batch_samples": int(targets[0].shape[0]),
    }


def top_bottom_spread(scores: np.ndarray, utilities: np.ndarray, frac: float = 0.2) -> float:
    order = np.argsort(-scores)
    count = max(1, int(len(order) * frac))
    return float(utilities[order[:count]].mean() - utilities[order[-count:]].mean())


def summarize_checkpoint(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    score_mode = str(frame["score_mode"].iloc[0]) if "score_mode" in frame else "expected_utility"
    has_scores = frame["score"].notna().all() if len(frame) else False

    daily = []
    pairwise = None
    pair_count = 0
    rank_ic = None
    rank_ic_dates = 0
    mean_ic = None
    ic_std = None
    positive_rate = None
    mean_tb_util = None
    mean_tb_ret = None

    if has_scores:
        scores = torch.as_tensor(frame["score"].to_numpy(np.float32))
        utilities = torch.as_tensor(frame["utility"].to_numpy(np.float32))
        date_ids = torch.as_tensor(frame["date_id"].to_numpy(np.int64))
        pairwise, pair_count = same_date_pairwise_accuracy(
            scores, utilities, date_ids, minimum_gap=MINIMUM_UTILITY_GAP
        )
        rank_ic, rank_ic_dates = mean_within_date_spearman(scores, utilities, date_ids)
        for asof, group in frame.groupby("asof_date"):
            if len(group) < 2:
                continue
            g_scores = torch.as_tensor(group["score"].to_numpy(np.float32))
            g_utils = torch.as_tensor(group["utility"].to_numpy(np.float32))
            g_dates = torch.as_tensor(group["date_id"].to_numpy(np.int64))
            day_ic, _ = mean_within_date_spearman(g_scores, g_utils, g_dates)
            day_pw, day_pairs = same_date_pairwise_accuracy(
                g_scores, g_utils, g_dates, minimum_gap=MINIMUM_UTILITY_GAP
            )
            daily.append(
                {
                    "asof_date": asof,
                    "samples": int(len(group)),
                    "rank_ic": day_ic,
                    "pairwise_accuracy": day_pw,
                    "pairwise_pairs": day_pairs,
                    "top_bottom_utility_spread": top_bottom_spread(
                        group["score"].to_numpy(np.float64),
                        group["utility"].to_numpy(np.float64),
                    ),
                    "top_bottom_return10d_spread": top_bottom_spread(
                        group["score"].to_numpy(np.float64),
                        group["return_10d"].to_numpy(np.float64),
                    ),
                }
            )
        daily_frame = pd.DataFrame(daily)
        ic_series = daily_frame["rank_ic"].dropna() if len(daily_frame) else pd.Series(dtype=float)
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else None
        mean_ic = float(ic_series.mean()) if len(ic_series) else None
        positive_rate = float((ic_series > 0).mean()) if len(ic_series) else None
        if len(daily_frame):
            mean_tb_util = float(daily_frame["top_bottom_utility_spread"].mean())
            mean_tb_ret = float(daily_frame["top_bottom_return10d_spread"].mean())
    else:
        for asof, group in frame.groupby("asof_date"):
            daily.append({"asof_date": asof, "samples": int(len(group))})

    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "score_mode": score_mode,
        "has_ranking_scores": bool(has_scores),
        "pairwise_accuracy": pairwise,
        "pairwise_pairs": int(pair_count),
        "rank_ic": rank_ic,
        "rank_ic_dates": int(rank_ic_dates),
        "rank_icir": (mean_ic / ic_std) if mean_ic is not None and ic_std else None,
        "rank_ic_positive_rate": positive_rate,
        "mean_daily_rank_ic": mean_ic,
        "mean_top_bottom_utility_spread": mean_tb_util,
        "mean_top_bottom_return10d_spread": mean_tb_ret,
        "forecast_loss": float(
            np.average(frame["forecast_loss"], weights=frame["batch_samples"])
        ),
        "history_loss": float(
            np.average(frame["history_loss"], weights=frame["batch_samples"])
        ),
        "weighted_forecast_loss": float(
            np.average(frame["weighted_forecast_loss"], weights=frame["batch_samples"])
        ),
        "by_signal_date": daily,
    }


def evaluate_checkpoint(
    label: str,
    model_path: Path,
    tokenizer,
    records,
    store,
    device,
    batch_size: int,
    use_amp: bool,
) -> dict:
    print(f"=== Evaluating {label} from {model_path} ===", flush=True)
    model = (
        Kronos.from_pretrained(str(model_path), local_files_only=True)
        .to(device)
        .eval()
    )
    use_aux = bool(getattr(model, "use_beta_v21_auxiliary", False))
    score_mode = "expected_utility" if use_aux else "forecast_only_no_ranking_score"
    if not use_aux:
        print(
            f"{label}: use_beta_v21_auxiliary=False (forecast-floor checkpoint). "
            "Will report WFL/forecast only; pairwise/rank_ic left null.",
            flush=True,
        )
    started = time.time()
    rows = []
    loss_accum = defaultdict(float)
    sample_accum = 0
    total = len(records)
    for batch_index, items in enumerate(iter_batches(records, store, batch_size), 1):
        batch = stack_batch(items, device)
        with torch.no_grad():
            encoded = tokenizer.encode(batch["x"], half=True)
            token_in = [encoded[0][:, :-1], encoded[1][:, :-1]]
            token_out = [encoded[0][:, 1:], encoded[1][:, 1:]]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                model_output = model(
                    token_in[0],
                    token_in[1],
                    batch["stamp"][:, :-1, :],
                    sector_id=batch["sector"],
                    size_percentile=batch["percentile"],
                    use_teacher_forcing=True,
                    s1_targets=token_out[0],
                    return_auxiliary=use_aux,
                    asof_index=LOOKBACK - 1,
                )
            if use_aux:
                logits, auxiliary = model_output
                auxiliary = {key: value.float() for key, value in auxiliary.items()}
                scores = expected_utility_score(
                    auxiliary["return"],
                    auxiliary["barrier"],
                    batch["labels"]["return_scales"],
                )
            else:
                logits = model_output
                scores = None
            logits = [part.float() for part in logits]
            losses = batch_token_losses(logits, token_out)
        for index, item in enumerate(items):
            rows.append(
                {
                    "label": label,
                    "symbol": item["symbol"],
                    "asof_date": item["asof_date"],
                    "target_date": item["target_date"],
                    "direction": item["direction"],
                    "return_10d": item["return_10d"],
                    "score": None if scores is None else float(scores[index].item()),
                    "utility": float(batch["labels"]["utility"][index].item()),
                    "date_id": int(batch["labels"]["date_id"][index].item()),
                    "forecast_loss": losses["forecast_loss"],
                    "history_loss": losses["history_loss"],
                    "weighted_forecast_loss": losses["weighted_forecast_loss"],
                    "batch_samples": losses["batch_samples"],
                    "score_mode": score_mode,
                }
            )
        sample_accum += losses["batch_samples"]
        for key in ("forecast_loss", "history_loss", "weighted_forecast_loss"):
            loss_accum[key] += losses[key] * losses["batch_samples"]
        done = min(batch_index * batch_size, total)
        if batch_index % 50 == 0 or done >= total:
            print(
                f"{label}: {done:,}/{total:,}  "
                f"WFL_running={loss_accum['weighted_forecast_loss']/max(sample_accum,1):.6f}",
                flush=True,
            )
    summary = summarize_checkpoint(rows)
    summary.update(
        {
            "label": label,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path / "model.safetensors")
            if (model_path / "model.safetensors").is_file()
            else None,
            "seconds": time.time() - started,
            "device": str(device),
            "batch_size": batch_size,
            "minimum_utility_gap": MINIMUM_UTILITY_GAP,
            "forecast_horizon_weights": list(FORECAST_HORIZON_WEIGHTS),
            "evaluation_name": EXPECTED_EVALUATION_NAME,
            "score_mode": score_mode,
            "has_ranking_scores": bool(use_aux),
        }
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"{label} DONE pairwise={summary['pairwise_accuracy']} "
        f"rank_ic={summary['rank_ic']} WFL={summary['weighted_forecast_loss']:.8f} "
        f"in {summary['seconds']:.1f}s",
        flush=True,
    )
    return {"summary": summary, "rows": rows}


def pick_best(summaries: list[dict]) -> dict:
    """Primary: pairwise among ranking-capable models; else lowest WFL."""

    ranking_capable = [item for item in summaries if item.get("has_ranking_scores")]
    pool = ranking_capable or summaries

    def key(item):
        has_rank = 1.0 if item.get("has_ranking_scores") else 0.0
        return (
            has_rank,
            float(item["pairwise_accuracy"] if item.get("pairwise_accuracy") is not None else -1.0),
            float(item["rank_ic"] if item.get("rank_ic") is not None else -1.0),
            -float(item["weighted_forecast_loss"] or 1e9),
        )

    ordered = sorted(pool, key=key, reverse=True)
    all_ordered = sorted(summaries, key=key, reverse=True)
    return {
        "ranking_rule": (
            "ranking-capable first; then pairwise_accuracy desc, rank_ic desc, "
            "weighted_forecast_loss asc. Forecast-only (no aux heads) cannot win "
            "on pairwise."
        ),
        "best_label": ordered[0]["label"],
        "ordered_labels": [item["label"] for item in all_ordered],
        "ranking_capable_labels": [item["label"] for item in ranking_capable],
    }


def resolve_checkpoint(path_or_glob: str, input_root: Path) -> Path:
    candidate = Path(path_or_glob)
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate
    matches = sorted(input_root.glob(path_or_glob))
    matches = [path for path in matches if (path / "config.json").is_file()]
    if len(matches) != 1:
        # Also accept checkpoints/best_model nesting
        nested = []
        for path in input_root.glob(path_or_glob):
            best = path / "checkpoints" / "best_model"
            if (best / "config.json").is_file():
                nested.append(best)
            elif (path / "config.json").is_file():
                nested.append(path)
        matches = nested
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one checkpoint for {path_or_glob!r}, found {matches}"
        )
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH_OR_GLOB"),
        required=True,
        help="Repeatable LABEL PATH_OR_GLOB pairs",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args(argv)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_root, manifest = find_evaluation_root(input_root)
    samples_path = evaluation_root / manifest["artifacts"]["samples_file"]
    panel_path = evaluation_root / manifest["artifacts"]["panel_file"]
    records = load_incremental_records(samples_path, manifest)
    if args.max_samples > 0:
        rng = np.random.default_rng(args.seed)
        dates = sorted({record["asof_date"] for record in records})
        sampled = []
        per_date = max(1, args.max_samples // max(len(dates), 1))
        for date in dates:
            pool = [record for record in records if record["asof_date"] == date]
            take = min(per_date, len(pool))
            sampled.extend(pool[int(i)] for i in rng.choice(len(pool), take, replace=False))
        records = sampled
        print(f"Smoke subsample: {len(records)} records", flush=True)

    with panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    store = OOSWindowStore(panel, manifest["model_contract"]["sector_labels"])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        use_amp = True
    else:
        device = torch.device("cpu")
        use_amp = False
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()

    summaries = []
    comparison_rows = []
    for label, path_or_glob in args.checkpoint:
        model_path = resolve_checkpoint(path_or_glob, input_root)
        result = evaluate_checkpoint(
            label,
            model_path,
            tokenizer,
            records,
            store,
            device,
            args.batch_size,
            use_amp,
        )
        summaries.append(result["summary"])
        (output_dir / f"{label}_summary.json").write_text(
            json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n"
        )
        pd.DataFrame(result["rows"]).to_csv(
            output_dir / f"{label}_predictions.csv", index=False
        )
        comparison_rows.append(
            {
                "label": label,
                "score_mode": result["summary"].get("score_mode"),
                "has_ranking_scores": result["summary"].get("has_ranking_scores"),
                "pairwise_accuracy": result["summary"]["pairwise_accuracy"],
                "rank_ic": result["summary"]["rank_ic"],
                "rank_icir": result["summary"]["rank_icir"],
                "rank_ic_positive_rate": result["summary"]["rank_ic_positive_rate"],
                "weighted_forecast_loss": result["summary"]["weighted_forecast_loss"],
                "forecast_loss": result["summary"]["forecast_loss"],
                "mean_top_bottom_utility_spread": result["summary"][
                    "mean_top_bottom_utility_spread"
                ],
                "seconds": result["summary"]["seconds"],
                "model_sha256": result["summary"]["model_sha256"],
            }
        )

    decision = pick_best(summaries)
    report = {
        "evaluation_name": EXPECTED_EVALUATION_NAME,
        "evaluation_root": str(evaluation_root),
        "temporal_isolation": manifest["temporal_isolation"],
        "sample_count": len(records),
        "checkpoints": comparison_rows,
        "decision": decision,
        "metrics_contract": {
            "pairwise_accuracy": "same-day pairs with |utility gap| >= 0.005; score ties count wrong",
            "rank_ic": "mean within-date Spearman(score, utility)",
            "weighted_forecast_loss": "C1 horizon weights 1.364..0.455 teacher-forcing CE",
            "score": "expected_utility_score(return_head, barrier_head, return_scales)",
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"BEST={decision['best_label']} order={decision['ordered_labels']}", flush=True)


if __name__ == "__main__":
    main()
