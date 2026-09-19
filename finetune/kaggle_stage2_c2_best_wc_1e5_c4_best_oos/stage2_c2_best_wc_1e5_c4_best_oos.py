"""13-date OOS for this 1e-5 round's campaign best: C4 best g528.

Does not re-run C2 init (0.1844) or C2-kernel best (0.1836).
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


SOURCE_COMMIT = "5b3ca1fd1ee6ded6e8e30a0dbb1ad7faa5a45e0a"
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage2_c2_best_wc_1e5_c4_best_oos")
INPUT = Path("/kaggle/input")
HARD_LIMIT_SECONDS = 39600
PURPOSE = ("exploratory_only; 13-date OOS design-contaminated; "
           "production candidate requires sealed fresh-window OOS")
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_CACHE = Path("/kaggle/working/kronos_tokenizer_base")
C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.18440434213929857
WC_1E5_C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.18364682977280178
LABEL = "wc_1e5_c4_best_g528"
EXPECTED_SHA = "9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075"
EXPECTED_LOCAL_SEGMENT = 21
EXPECTED_GLOBAL_SEGMENT = 528
EXPECTED_FORECAST = 2.2728769779205322


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
        repo = Path(tempfile.mkdtemp(prefix="kronos-wc-1e5-c4-best-oos-")) / "repo"
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


def resolve_tokenizer():
    matches = [
        path for path in INPUT.glob("**/Kronos-Tokenizer-base/model.safetensors")
        if "cosine-refinement-c2" in str(path)
    ]
    if len(matches) == 1:
        tokenizer_dir = matches[0].parent
    elif not matches:
        run(["python", "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        tokenizer_dir = Path(snapshot_download(
            TOKENIZER_REPO, local_dir=str(TOKENIZER_CACHE), local_dir_use_symlinks=False,
        ))
    else:
        raise RuntimeError(f"Expected one tokenizer, found {matches}")
    tokenizer_sha = sha256_file(tokenizer_dir / "model.safetensors")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tokenizer_sha}")
    return tokenizer_dir, tokenizer_sha


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
    tokenizer_dir, tokenizer_sha = resolve_tokenizer()

    best_file = find_one(
        "**/small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors"
    )
    best_metric = json.loads((best_file.parent / "best_metric.json").read_text())
    if int(best_metric.get("segment", -1)) != EXPECTED_LOCAL_SEGMENT:
        raise RuntimeError(f"Unexpected C4 best segment: {best_metric}")
    if not math.isclose(float(best_metric["forecast_loss"]), EXPECTED_FORECAST, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Unexpected C4 best forecast CE: {best_metric}")
    best_sha = sha256_file(best_file)
    if best_sha != EXPECTED_SHA:
        raise RuntimeError(f"C4 best sha mismatch: {best_sha}")

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

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    remaining = []
    timed_out = False
    print(json.dumps({
        "phase": "inputs_resolved",
        "label": LABEL,
        "global_segment": EXPECTED_GLOBAL_SEGMENT,
        "local_segment": EXPECTED_LOCAL_SEGMENT,
        "val_forecast_ce": EXPECTED_FORECAST,
        "weights_sha256": best_sha,
        "tokenizer_sha256": tokenizer_sha,
        "gpu": gpu,
    }, ensure_ascii=False), flush=True)

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
    parts = [pd.read_csv(path) for path in sorted(shards.glob(f"{LABEL}_*.csv.gz"))]
    if not timed_out and len(parts) == len(dates):
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != len(records):
            raise RuntimeError(f"{LABEL} prediction count mismatch: {len(frame)} != {len(records)}")
        metrics[LABEL] = summarize(frame)
        (OUTPUT / f"metrics_{LABEL}.json").write_text(
            json.dumps(metrics[LABEL], ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({
            "phase": "model_complete", "label": LABEL,
            "d10": metrics[LABEL]["by_horizon"][9],
        }, ensure_ascii=False), flush=True)
    elif not remaining:
        remaining = [{"label": LABEL, "have_shards": len(parts), "need": len(dates)}]

    write_summary({
        "status": "partial" if timed_out or remaining else "complete",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "evaluated": LABEL,
        "global_segment": EXPECTED_GLOBAL_SEGMENT,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "elapsed_sec": time.time() - started,
        "c2_best_same_footing_d10_pooled_rank_ic": C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC,
        "wc_1e5_c2_best_same_footing_d10_pooled_rank_ic": WC_1E5_C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC,
        "already_measured_not_rerun": ["c2_init_0.1844", "wc_1e5_c2_best_g289_0.1836"],
        "metrics": metrics,
        "remaining": remaining,
    })
    if timed_out:
        sys.exit(0)


if __name__ == "__main__":
    main()
