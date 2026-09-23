"""Exploratory fixed-protocol 18-day OOS for Stage3 AR-vol C2.

The 18-day window has already been used for C2 analysis; it is not a new
confirmatory holdout. Reuse the original C2 predictions and locked decoder.
"""
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


SOURCE_COMMIT = "2d22d91de1bf1672b8723e43d01bcb7248dd105e"
BASE_SHA256 = "c5c9e2f6cbc91bb1378c2c9420a325cbfb724c61c4f241bce4213961b19b61f6"
MODEL_SHA256 = "dccd7aefed346a88a0a96166bff31d891da08a9a55452a483621b5a16128154e"
BASELINE_SHA256 = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
OUTPUT = Path("/kaggle/working/stage3_ar_vol_18d_oos")
INPUT = Path("/kaggle/input")
BASE_FILE = OUTPUT / "c2_18d_alpha_oos_pinned.py"
WORLD_SIZE = 2
HARD_LIMIT_SECONDS = 39600
ARM = {
    "name": "ar_vol_c2_prod_t065_p80_n5",
    "checkpoint": "ar_vol_c2_seg8",
    "sample_count": 5,
    "temperature": 0.65,
    "top_p": 0.8,
    "seed": 20260906,
    "protocol": "production",
}


def load_base():
    if not BASE_FILE.is_file():
        url = (
            f"https://raw.githubusercontent.com/luckfu/Kronos/{SOURCE_COMMIT}/"
            "finetune/kaggle_c2_18d_alpha_oos/c2_18d_alpha_oos.py"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != BASE_SHA256:
            raise RuntimeError("Pinned evaluation source SHA mismatch")
        BASE_FILE.write_bytes(body)
    elif hashlib.sha256(BASE_FILE.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError("Cached evaluation source SHA mismatch")
    spec = importlib.util.spec_from_file_location("c2_18d_alpha_oos_pinned", BASE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matched_frames(candidate, baseline, expected_samples):
    import numpy as np

    if len(candidate) != expected_samples or len(baseline) != expected_samples:
        raise RuntimeError("OOS prediction row count mismatch")
    for name, frame in (("candidate", candidate), ("baseline", baseline)):
        if frame["identity"].isna().any() or frame["identity"].duplicated().any():
            raise RuntimeError(f"{name} has missing or duplicate identities")
    candidate = candidate.sort_values("identity").reset_index(drop=True)
    baseline = baseline.sort_values("identity").reset_index(drop=True)
    if not candidate["identity"].equals(baseline["identity"]):
        raise RuntimeError("Candidate and C2 baseline OOS identities differ")
    for column in ("asof_date", "symbol", "sector", "size_decile"):
        if not candidate[column].equals(baseline[column]):
            raise RuntimeError(f"Candidate and C2 baseline {column} differ")
    if not np.allclose(
        candidate["actual_return_d10"], baseline["actual_return_d10"],
        rtol=0, atol=1e-8,
    ):
        raise RuntimeError("Candidate and C2 baseline targets differ")
    return candidate, baseline


def worker(base):
    import torch
    from finetune.evaluate_v1_beta_checkpoints import WindowStore
    from model import Kronos, KronosTokenizer

    rank = int(os.environ["KRONOS_OOS_RANK"])
    plan = json.loads((OUTPUT / "plan.json").read_text())
    device = torch.device("cuda:0")
    manifest = json.loads((Path(plan["evaluation_root"]) / "evaluation_manifest.json").read_text())
    panel, records, _, _, _ = base.read_records(Path(plan["evaluation_root"]), manifest)
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer_dir"]).to(device).eval()
    model = Kronos.from_pretrained(
        plan["model_dir"], num_sectors=86, num_size_buckets=0,
        context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tasks = plan["dates"][rank::WORLD_SIZE]
    for date in tasks:
        if time.time() > plan["deadline"]:
            raise TimeoutError(f"Worker {rank} exceeded deadline at {date}")
        shard = OUTPUT / "shards" / f"{date}.csv.gz"
        rows = [row for row in records if row["asof_date"] == date]
        started = time.time()
        frame = base.decode_date(ARM, model, tokenizer, rows, store, device)
        frame.to_csv(shard, index=False, compression="gzip")
        print(json.dumps({
            "phase": "shard_done", "rank": rank, "date": date,
            "rows": len(frame), "seconds": round(time.time() - started, 1),
        }), flush=True)
    del model, tokenizer, store, panel
    gc.collect()
    torch.cuda.empty_cache()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    if os.environ.get("KRONOS_OOS_RANK") is not None:
        sys.path.insert(0, os.environ["KRONOS_OOS_REPO"])
        sys.path.insert(0, str(Path(os.environ["KRONOS_OOS_REPO"]) / "finetune"))
        worker(base)
        return

    started = time.time()
    (OUTPUT / "shards").mkdir(exist_ok=True)
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(gpus) != WORLD_SIZE or any("T4" not in name for name in gpus):
        raise RuntimeError(f"Expected two Tesla T4 GPUs, found {gpus}")

    repo = base.clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))
    import pandas as pd

    tokenizer_dir, tokenizer_sha = base.resolve_tokenizer()
    model_file = base.find_one(
        "**/stage3_joint_path_smoke/checkpoints/best_model/model.safetensors",
        lambda path: "s3-ar-vol-c2" in str(path),
    )
    if base.sha256_file(model_file) != MODEL_SHA256:
        raise RuntimeError("AR-vol C2 checkpoint SHA mismatch")
    metric = json.loads((model_file.parent / "best_metric.json").read_text())
    if (metric.get("segment"), metric.get("step"), metric.get("definition")) != (
        8, 10000, "causal_token_ce+0.15*ar_log_vol_reinforce"
    ) or not math.isclose(metric["validation_objective"], 2.2400715205110995, rel_tol=1e-9):
        raise RuntimeError(f"AR-vol C2 checkpoint metric mismatch: {metric}")
    parent_manifest = json.loads((model_file.parents[2] / "experiment_manifest.json").read_text())
    if parent_manifest["sha256"]["best"] != BASELINE_SHA256:
        raise RuntimeError("AR-vol C2 was not initialized from the sealed C2 best")

    evaluation_root, manifest = base.load_evaluation()
    panel, records, _, start, end = base.read_records(evaluation_root, manifest)
    dates = sorted({row["asof_date"] for row in records})
    if len(dates) != 18 or len(records) != 92751:
        raise RuntimeError(f"Unexpected sealed OOS set: {len(dates)} dates, {len(records)} rows")
    del panel
    gc.collect()

    baseline_file = base.find_one("**/predictions_prod_t065_p80_n5.csv.gz")
    baseline = pd.read_csv(baseline_file)
    if len(baseline) != len(records) or baseline["asof_date"].nunique() != 18:
        raise RuntimeError("Original C2 OOS predictions are incomplete")
    expected_protocol = {
        "checkpoint": "c2_best",
        "sample_count": 5,
        "temperature": 0.65,
        "top_p": 0.8,
        "seed": 20260906,
    }
    for column, value in expected_protocol.items():
        if not baseline[column].eq(value).all():
            raise RuntimeError(f"Original C2 baseline {column} is not locked: {value}")
    baseline_metrics = base.summarize(baseline)
    if not math.isclose(baseline_metrics["by_horizon"][9]["pooled_rank_ic"], 0.2372795397036252, abs_tol=1e-9):
        raise RuntimeError("Original C2 baseline metric mismatch")
    baseline_vol = base.volatility_summary(baseline)
    if not math.isclose(baseline_vol["calibration_ratio_realized_over_predicted"], 2.102, abs_tol=0.01):
        raise RuntimeError("Original C2 baseline volatility mismatch")

    (OUTPUT / "plan.json").write_text(json.dumps({
        "evaluation_root": str(evaluation_root),
        "tokenizer_dir": str(tokenizer_dir),
        "model_dir": str(model_file.parent),
        "dates": dates,
        "deadline": started + HARD_LIMIT_SECONDS,
    }, indent=2))
    print(json.dumps({
        "phase": "inputs_verified", "dates": len(dates), "rows": len(records),
        "signal_start": start, "signal_end": end, "model_sha256": MODEL_SHA256,
        "tokenizer_sha256": tokenizer_sha, "baseline_file": str(baseline_file),
    }), flush=True)
    script = str(Path(__file__).resolve())
    processes = []
    for rank in range(WORLD_SIZE):
        env = os.environ.copy()
        env.update(
            PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(rank),
            KRONOS_OOS_RANK=str(rank), KRONOS_OOS_REPO=str(repo),
        )
        processes.append(subprocess.Popen([sys.executable, "-u", script], env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"OOS worker exit codes: {codes}")

    parts = sorted((OUTPUT / "shards").glob("*.csv.gz"))
    if len(parts) != len(dates):
        raise RuntimeError(f"Incomplete OOS shards: {len(parts)}/{len(dates)}")
    candidate = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    candidate, baseline = matched_frames(candidate, baseline, len(records))
    candidate_metrics = base.summarize(candidate)
    candidate_vol = base.volatility_summary(candidate)
    candidate_alpha = base.alpha_diagnostics(candidate)
    baseline_alpha = base.alpha_diagnostics(baseline)
    baseline_metrics = base.summarize(baseline)
    daily_candidate = candidate_alpha["by_signal_date"]
    daily_baseline = baseline_alpha["by_signal_date"]
    delta = [
        current["double_residual_rank_ic"] - previous["double_residual_rank_ic"]
        for current, previous in zip(daily_candidate, daily_baseline)
    ]
    if any(c["asof_date"] != b["asof_date"] for c, b in zip(daily_candidate, daily_baseline)):
        raise RuntimeError("Daily alpha dates do not align")
    result = {
        "status": "complete",
        "interpretation": "exploratory_reused_18d_not_fresh_confirmation",
        "training_performed": False,
        "decode_search_performed": False,
        "evaluation_source_commit": SOURCE_COMMIT,
        "evaluation_source_sha256": BASE_SHA256,
        "evaluation_name": manifest["name"],
        "model_sha256": MODEL_SHA256,
        "parent_c2_sha256": BASELINE_SHA256,
        "checkpoint": metric,
        "decode": ARM,
        "signal_dates": len(dates),
        "samples": len(records),
        "baseline": {
            "volatility": baseline_vol, "alpha": baseline_alpha,
            "d10": baseline_metrics["by_horizon"][9],
        },
        "candidate": {
            "volatility": candidate_vol, "alpha": candidate_alpha,
            "d10": candidate_metrics["by_horizon"][9],
        },
        "paired_delta": {
            "vol_calibration_ratio": (
                candidate_vol["calibration_ratio_realized_over_predicted"]
                - baseline_vol["calibration_ratio_realized_over_predicted"]
            ),
            "d10_pooled_rank_ic": (
                candidate_metrics["by_horizon"][9]["pooled_rank_ic"]
                - baseline_metrics["by_horizon"][9]["pooled_rank_ic"]
            ),
            "daily_double_residual_ic_mean": sum(delta) / len(delta),
            "daily_double_residual_ic_nw": base.overlapping_icir(delta),
            "daily_double_residual_ic_block_bootstrap": base.moving_block_bootstrap(delta),
        },
        "elapsed_sec": time.time() - started,
    }
    candidate.to_csv(OUTPUT / "predictions_ar_vol_c2.csv.gz", index=False, compression="gzip")
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "phase": "completed",
        "candidate_vol_ratio": candidate_vol["calibration_ratio_realized_over_predicted"],
        "baseline_vol_ratio": baseline_vol["calibration_ratio_realized_over_predicted"],
        "candidate_d10_rank_ic": candidate_metrics["by_horizon"][9]["pooled_rank_ic"],
        "candidate_double_residual_rank_ic": candidate_alpha["pooled_double_residual_rank_ic"],
    }), flush=True)


if __name__ == "__main__":
    main()
