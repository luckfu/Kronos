"""13-date diagnostic OOS for the 1e-5 campaign best checkpoint.

Zero training. Evaluates C2-kernel best_model (local seg 189, global 289)
on the sealed 13-date window with the same evaluator / seed / fp16
settings used by every Stage3 OOS kernel. Compare D10 pooled Rank IC to
C2 best same-footing 0.1844.
"""
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SOURCE_COMMIT = "0" * 40
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage2_c2_best_wc_1e5_best_oos")
INPUT = Path("/kaggle/input")
HARD_LIMIT_SECONDS = 39600
PURPOSE = ("exploratory_only; 13-date OOS design-contaminated; "
           "production candidate requires sealed fresh-window OOS")
EXPECTED_C2_BEST_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
EXPECTED_WC_BEST_SHA = "8fb3d3d7a76fc2bb0f0277d8682b19c912a3c5b657c91416a3d5e3b5e7c3e268"
EXPECTED_WC_BEST_SEGMENT = 189
EXPECTED_WC_BEST_FORECAST = 2.279949903488159
EXPECTED_PARENT_SEGMENTS = 200
C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.18440434213929857
LABEL = "wc_1e5_best_seg189"


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-wc-1e5-best-oos-")) / "repo"
        try:
            run(["git", "init", str(repo)])
            run(["git", "remote", "add", "origin", "https://github.com/luckfu/Kronos.git"], cwd=repo)
            run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError(f"Source commit mismatch: {actual}")
            return repo
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def metric_or_none(value):
    value = float(value)
    return None if not math.isfinite(value) else value


def summarize(frame, top_fraction=0.10):
    import numpy as np
    import pandas as pd

    by_horizon = []
    by_date_h10 = []
    for horizon in range(1, 11):
        pred_col = f"predicted_return_d{horizon}"
        actual_col = f"actual_return_d{horizon}"
        pred, actual = frame[pred_col], frame[actual_col]
        daily = []
        for date, rows in frame.groupby("asof_date", sort=True):
            rank_ic = rows[pred_col].corr(rows[actual_col], method="spearman")
            count = max(1, int(math.ceil(len(rows) * top_fraction)))
            ranked = rows.sort_values(pred_col)
            spread = ranked.tail(count)[actual_col].mean() - ranked.head(count)[actual_col].mean()
            daily.append({
                "asof_date": str(date),
                "samples": int(len(rows)),
                "rank_ic": metric_or_none(rank_ic),
                "direction_accuracy": float((np.sign(rows[pred_col]) == np.sign(rows[actual_col])).mean()),
                "top_bottom_10pct": float(spread),
            })
        daily_ic = pd.Series([row["rank_ic"] for row in daily], dtype=float).dropna()
        ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        metrics = {
            "horizon_day": horizon,
            "samples": int(len(frame)),
            "direction_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "mae": float((pred - actual).abs().mean()),
            "pooled_rank_ic": metric_or_none(pred.corr(actual, method="spearman")),
            "daily_rank_ic_mean": metric_or_none(daily_ic.mean()),
            "daily_rank_ic_std": metric_or_none(ic_std),
            "daily_icir": metric_or_none(daily_ic.mean() / ic_std),
            "daily_icir_annualized_sqrt252": metric_or_none(daily_ic.mean() / ic_std * math.sqrt(252)),
            "positive_ic_ratio": float((daily_ic > 0).mean()),
            "mean_top_bottom_10pct": float(np.mean([row["top_bottom_10pct"] for row in daily])),
        }
        by_horizon.append(metrics)
        if horizon == 10:
            by_date_h10 = daily
    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "by_horizon": by_horizon,
        "by_signal_date_d10": by_date_h10,
    }


def agreement(left, right, name):
    pred_cols = [f"predicted_return_d{h}" for h in range(1, 11)]
    merged = left[["identity"] + pred_cols].merge(
        right[["identity"] + pred_cols], on="identity",
        suffixes=("_new", "_ref"), validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"{name} identity coverage mismatch: {len(merged)} vs {len(left)}/{len(right)}")
    rows = []
    for horizon in range(1, 11):
        delta = (merged[f"predicted_return_d{horizon}_new"] - merged[f"predicted_return_d{horizon}_ref"]).abs()
        spearman = merged[f"predicted_return_d{horizon}_new"].corr(
            merged[f"predicted_return_d{horizon}_ref"], method="spearman"
        )
        rows.append({
            "horizon_day": horizon,
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "spearman": metric_or_none(spearman),
        })
    return {"name": name, "identities": int(len(merged)), "by_horizon": rows}


def write_summary(payload):
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = OUTPUT / "shards"
    shards.mkdir(exist_ok=True)

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    if not any(name in gpu for name in ("P100", "T4")):
        raise RuntimeError(f"Expected P100 or T4, got {gpu}")

    repo = clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import pandas as pd
    import torch
    from finetune.evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda:0")

    in_c2 = lambda path: "cosine-refinement-c2" in str(path)
    best_file = find_one(
        "**/small_0.1_stage2_c2_best_wc_1e5_c2/checkpoints/best_model/model.safetensors"
    )
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors", in_c2)
    cosine_best_file = find_one(
        "**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors",
        in_c2,
    )
    c2_predictions = find_one(
        "**/kronos_small_0_1_stage2_oos/predictions.csv.gz",
        lambda path: "c2-alpha-oos-evaluation" in str(path),
    )

    source_root = best_file.parent.parent.parent
    progress = json.loads((source_root / "progress.json").read_text())
    completed = int(progress.get("completed_segments", progress.get("current_segment", 0)))
    if completed != EXPECTED_PARENT_SEGMENTS or progress.get("status") != "completed":
        raise RuntimeError(f"C2 training is not a completed 200-segment parent: {progress}")
    best_metric = json.loads((best_file.parent / "best_metric.json").read_text())
    if int(best_metric.get("segment", -1)) != EXPECTED_WC_BEST_SEGMENT:
        raise RuntimeError(f"Unexpected 1e-5 best segment: {best_metric}")
    if not math.isclose(
        float(best_metric["forecast_loss"]), EXPECTED_WC_BEST_FORECAST, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"Unexpected 1e-5 best forecast CE: {best_metric}")

    best_sha = sha256_file(best_file)
    tokenizer_sha = sha256_file(tokenizer_file)
    cosine_best_sha = sha256_file(cosine_best_file)
    if best_sha != EXPECTED_WC_BEST_SHA:
        raise RuntimeError(f"1e-5 best sha mismatch: {best_sha}")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tokenizer_sha}")
    if cosine_best_sha != EXPECTED_C2_BEST_SHA:
        raise RuntimeError(f"C2 cosine best sha mismatch: {cosine_best_sha}")
    if best_sha == cosine_best_sha:
        raise RuntimeError("1e-5 best equals C2 cosine best; nothing new to evaluate")

    manifests = []
    for path in INPUT.glob("**/evaluation_manifest.json"):
        if not ((path.parent / "evaluation_panel.pkl").is_file()
                and (path.parent / "evaluation_samples.jsonl").is_file()):
            continue
        candidate = json.loads(path.read_text())
        if candidate.get("temporal_isolation", {}).get("incremental_signal_start"):
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one sealed evaluation manifest, found {manifests}")
    evaluation_root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    isolation = manifest["temporal_isolation"]
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")

    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    sample_groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    sample_key = "incremental_future_all" if "incremental_future_all" in sample_groups else "future_all"
    start = isolation.get("incremental_signal_start", isolation.get("future_signal_start"))
    end = isolation.get("incremental_signal_end", isolation.get("future_signal_end"))
    records = [row for row in sample_groups[sample_key] if start <= row["asof_date"] <= end]
    if not records:
        raise RuntimeError("No sealed OOS records")
    dates = sorted({row["asof_date"] for row in records})

    reference = pd.read_csv(c2_predictions)
    c2_ref = reference[reference["model"] == "c2_best_segment_179"].copy()
    if c2_ref["identity"].duplicated().any() or len(c2_ref) != 66986:
        raise RuntimeError(f"C2 reference invalid: rows={len(c2_ref)}")
    if len(c2_ref) != len(records):
        raise RuntimeError(f"C2 prediction count mismatch: {len(c2_ref)} != {len(records)}")

    provenance = {
        "wc_1e5_best_weights": {
            "path": str(best_file),
            "sha256": best_sha,
            "local_segment": EXPECTED_WC_BEST_SEGMENT,
            "global_segment": EXPECTED_WC_BEST_SEGMENT + 100,
            "val_forecast_ce": EXPECTED_WC_BEST_FORECAST,
        },
        "c2_cosine_best_weights": {"path": str(cosine_best_file), "sha256": cosine_best_sha},
        "tokenizer": {"path": str(tokenizer_file), "sha256": tokenizer_sha},
        "c2_reference_predictions": {"path": str(c2_predictions), "sha256": sha256_file(c2_predictions)},
        "evaluation_root": str(evaluation_root),
        "gpu": gpu,
    }
    print(json.dumps({"phase": "inputs_resolved", **provenance}, ensure_ascii=False), flush=True)

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    remaining = []
    timed_out = False
    pending = [date for date in dates if not (shards / f"{LABEL}_{date}.csv.gz").is_file()]
    if pending:
        model = Kronos.from_pretrained(
            best_file.parent, num_sectors=86, num_size_buckets=0, context_layer=6,
            use_size_percentile=True, size_mlp_hidden_dim=64,
        ).to(device).eval()
        for date in pending:
            if time.time() - started > HARD_LIMIT_SECONDS:
                remaining = [{"label": LABEL, "dates": [date] + pending[pending.index(date) + 1:]}]
                timed_out = True
                break
            date_records = [row for row in records if row["asof_date"] == date]
            result = evaluate_predictions(
                LABEL, model, tokenizer, date_records, store, device,
                64, 1, 20260906, True,
            )
            shard = shards / f"{LABEL}_{date}.csv.gz"
            result.to_csv(shard, index=False, compression="gzip")
            print({"saved": str(shard), "rows": len(result), "elapsed_sec": time.time() - started}, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    metrics = {}
    sanity = {}
    parts = [pd.read_csv(path) for path in sorted(shards.glob(f"{LABEL}_*.csv.gz"))]
    if not timed_out and len(parts) == len(dates):
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != len(records):
            raise RuntimeError(f"{LABEL} prediction count mismatch: {len(frame)} != {len(records)}")
        metrics[LABEL] = summarize(frame)
        (OUTPUT / f"metrics_{LABEL}.json").write_text(
            json.dumps(metrics[LABEL], ensure_ascii=False, indent=2) + "\n"
        )
        sanity[LABEL] = agreement(frame, c2_ref, "wc_1e5_best_vs_c2_best_segment_179")
        print(json.dumps({
            "phase": "model_complete",
            "label": LABEL,
            "d10": metrics[LABEL]["by_horizon"][9],
        }, ensure_ascii=False), flush=True)
    elif not remaining:
        remaining = [{"label": LABEL, "have_shards": len(parts), "need": len(dates)}]

    d10 = metrics.get(LABEL, {}).get("by_horizon", [{}] * 10)[9].get("pooled_rank_ic")
    payload = {
        "status": "partial" if timed_out or remaining else "complete",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "elapsed_sec": time.time() - started,
        "evaluated": LABEL,
        "c2_best_same_footing_d10_pooled_rank_ic": C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC,
        "c2_best_same_footing_note": (
            "C2 best not re-run; 0.18440434213929857 reproduced by P0b alpha=0, P0a init, "
            "P1 init, freeze-probe B init with this evaluator/seed"
        ),
        "provenance": provenance,
        "sanity_policy": "identity-merge max|delta|/mean|delta|/spearman vs C2 best reference; soft report only",
        "sanity": sanity,
        "metrics": metrics,
        "remaining": remaining,
        "decision_tree_note": (
            f"Compare {LABEL} D10 pooled_rank_ic {d10} to C2 best same-footing 0.1844; exploratory only."
        ),
    }
    write_summary(payload)
    if timed_out:
        sys.exit(0)


if __name__ == "__main__":
    main()
