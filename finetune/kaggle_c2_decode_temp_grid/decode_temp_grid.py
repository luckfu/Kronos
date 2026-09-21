"""C2-best temperature / top_p grid on the sealed 13-date window.

Zero training. Only Cosine C2 best (seg 179). Last sweep showed T=0.6 /
top_p=0.9 / N=16 is best for D10 Rank IC (0.225) but under-predicts path
volatility by ~2x, while T=1.0 / top_p=1.0 calibrates closer to 1.0 and
destroys ranking. This grid searches the in-between so ranking IC and
volatility accuracy can be chosen together.

Joint selection is pre-registered and applied only to the N=8 arms:
  1. Keep arms whose D10 pooled Rank IC is within 0.010 of the best N=8 IC.
  2. Among those, pick the smallest |log(calibration_ratio)|.
  3. Ties break on higher vol Rank IC, then higher return IC.

Production T=0.65 / top_p=0.8 / N=5 is scored as a baseline, not in the
joint set. 2x T4, (arm, date) round-robin, each task reseeds.
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


SOURCE_COMMIT = "e4b92bb32aa47d676ebcba70b5c8427bbc404c03"
OUTPUT = Path("/kaggle/working/kronos_c2_decode_temp_grid")
INPUT = Path("/kaggle/input")
REPO_DIR = Path("/kaggle/working/kronos_repo")
HARD_LIMIT_SECONDS = 39600
PURPOSE = ("exploratory_only; 13-date OOS design-contaminated; "
           "decode-side diagnostic, not a production promotion result")
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_CACHE = Path("/kaggle/working/kronos_tokenizer_base")
EFFECTIVE_BATCH = 64
WORLD_SIZE = 2
JOINT_IC_SLACK = 0.010
DECODE_SEED = 20260906

CHECKPOINTS = {
    "c2_best": {
        "glob": "**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors",
        "sha256": "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a",
        "segment": 179,
        "val_forecast_ce": 2.2944018841,
        "baseline_d10_pooled_rank_ic": 0.18440434213929857,
        "n16_t060_p90_d10_pooled_rank_ic": 0.22465231870461855,
    },
}


def _arm(name, temperature, top_p, sample_count):
    return {
        "name": name,
        "checkpoint": "c2_best",
        "sample_count": sample_count,
        "temperature": temperature,
        "top_p": top_p,
        "seed": DECODE_SEED,
    }


# Production baseline first, then the N=8 T x top_p grid. Cost ~ sample_count.
ARMS = [_arm("prod_t065_p80_n5", 0.65, 0.8, 5)]
for temperature in (0.50, 0.60, 0.65, 0.70, 0.80, 1.00):
    for top_p in (0.8, 0.9):
        tag = f"t{int(round(temperature * 100)):03d}_p{int(round(top_p * 100)):02d}_n8"
        ARMS.append(_arm(tag, temperature, top_p, 8))
for temperature in (0.80, 1.00):
    ARMS.append(_arm(f"t{int(round(temperature * 100)):03d}_p100_n8", temperature, 1.0, 8))
ARMS_BY_NAME = {arm["name"]: arm for arm in ARMS}


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def clone_source():
    for attempt in range(1, 4):
        repo = REPO_DIR if attempt == 1 else Path(tempfile.mkdtemp(prefix="kronos-c2-temp-grid-")) / "repo"
        try:
            if repo.exists():
                subprocess.run(["rm", "-rf", str(repo)], check=True)
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


def load_evaluation():
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
    manifest = json.loads(manifests[0].read_text())
    isolation = manifest["temporal_isolation"]
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")
    return manifests[0].parent, manifest


def read_records(evaluation_root, manifest):
    from finetune.evaluate_v1_beta_checkpoints import load_samples

    isolation = manifest["temporal_isolation"]
    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    sample_key = "incremental_future_all" if "incremental_future_all" in groups else "future_all"
    start = isolation.get("incremental_signal_start", isolation.get("future_signal_start"))
    end = isolation.get("incremental_signal_end", isolation.get("future_signal_end"))
    records = [row for row in groups[sample_key] if start <= row["asof_date"] <= end]
    if not records:
        raise RuntimeError("No sealed OOS records")
    return panel, records, sample_key, start, end


def to_daily(cumulative):
    """Convert cumulative returns from a common base into per-step returns."""
    import numpy as np

    previous = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], axis=-1
    )
    return (1.0 + cumulative) / (1.0 + previous) - 1.0


def decode_date(arm, model, tokenizer, records, store, device):
    import numpy as np
    import pandas as pd
    import torch
    from finetune.evaluate_v1_beta_checkpoints import (
        FEATURES, LOOKBACK, PREDICT, batches, stack_batch,
    )
    from model.kronos import auto_regressive_inference

    sample_count = int(arm["sample_count"])
    batch_size = max(1, EFFECTIVE_BATCH // sample_count)
    close_index = FEATURES.index("close")
    torch.manual_seed(arm["seed"])
    np.random.seed(arm["seed"])

    rows = []
    for items in batches(records, store, batch_size):
        batch = stack_batch(items, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            forecast = auto_regressive_inference(
                tokenizer,
                model,
                batch["x"][:, :LOOKBACK],
                batch["stamp"][:, :LOOKBACK],
                batch["stamp"][:, LOOKBACK: LOOKBACK + PREDICT],
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
        future = forecast[:, :, -PREDICT:, close_index]
        for index, item in enumerate(items):
            scale = float(item["std"][close_index]) + 1e-5
            shift = float(item["mean"][close_index])
            last_close = float(item["x"][LOOKBACK - 1, close_index]) * scale + shift
            paths = future[index].astype(np.float64) * scale + shift
            cumulative = paths / last_close - 1.0
            mean_cumulative = cumulative.mean(axis=0)
            actual = (
                item["x"][LOOKBACK: LOOKBACK + PREDICT, close_index].astype(np.float64)
                * scale + shift
            ) / last_close - 1.0

            predicted_daily = to_daily(cumulative)
            actual_daily = to_daily(actual)
            row = {
                "model": arm["name"],
                "checkpoint": arm["checkpoint"],
                "sample_count": sample_count,
                "temperature": float(arm["temperature"]),
                "top_p": float(arm["top_p"]),
                "seed": int(arm["seed"]),
                **{
                    key: item[key]
                    for key in (
                        "identity", "symbol", "asof_date", "target_date",
                        "direction", "sector", "size_decile", "return_10d",
                    )
                },
                "predicted_return_10d": float(mean_cumulative[-1]),
                "predicted_mean_horizon_return": float(mean_cumulative.mean()),
                "predicted_path_vol": float(predicted_daily.std(axis=1).mean()),
                "predicted_terminal_dispersion": (
                    float(cumulative[:, -1].std(ddof=1)) if sample_count > 1 else None
                ),
                "realized_path_vol": float(actual_daily.std()),
            }
            for horizon in range(PREDICT):
                row[f"predicted_return_d{horizon + 1}"] = float(mean_cumulative[horizon])
                row[f"actual_return_d{horizon + 1}"] = float(actual[horizon])
            rows.append(row)
    return pd.DataFrame(rows)


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


def volatility_summary(frame):
    """Score the predictive spread as a 10-day volatility forecast."""
    import numpy as np
    import pandas as pd

    valid = frame[(frame["predicted_path_vol"] > 0) & (frame["realized_path_vol"] > 0)]
    if valid.empty:
        return None
    predicted = valid["predicted_path_vol"].to_numpy(dtype=float)
    realized = valid["realized_path_vol"].to_numpy(dtype=float)
    daily_ic = []
    for _, rows in valid.groupby("asof_date", sort=True):
        daily_ic.append(rows["predicted_path_vol"].corr(rows["realized_path_vol"], method="spearman"))
    daily_ic = pd.Series(daily_ic, dtype=float).dropna()
    qlike = float(np.mean(np.log(predicted ** 2) + (realized ** 2) / (predicted ** 2)))
    payload = {
        "samples": int(len(valid)),
        "pooled_vol_rank_ic": metric_or_none(
            valid["predicted_path_vol"].corr(valid["realized_path_vol"], method="spearman")
        ),
        "daily_vol_rank_ic_mean": metric_or_none(daily_ic.mean()),
        "qlike": metric_or_none(qlike),
        "mean_predicted_path_vol": float(predicted.mean()),
        "mean_realized_path_vol": float(realized.mean()),
        "calibration_ratio_realized_over_predicted": float(np.mean(realized / predicted)),
    }
    dispersion = valid["predicted_terminal_dispersion"]
    if dispersion.notna().any():
        payload["pooled_dispersion_vs_abs_return_rank_ic"] = metric_or_none(
            dispersion.corr(valid["return_10d"].abs(), method="spearman")
        )
    return payload


def attenuation_fit(points):
    """Fit IC(N) = IC_inf / sqrt(1 + r/N) via least squares on 1/IC^2 vs 1/N."""
    usable = [(n, ic) for n, ic in points if n > 0 and ic and ic > 0]
    if len(usable) < 2:
        return None
    xs = [1.0 / n for n, _ in usable]
    ys = [1.0 / (ic ** 2) for _, ic in usable]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    if intercept <= 0:
        return None
    return {
        "points": [{"sample_count": n, "pooled_rank_ic": ic} for n, ic in usable],
        "ic_infinity": math.sqrt(1.0 / intercept),
        "noise_to_signal_variance_ratio": slope / intercept,
    }


def write_summary(payload):
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def worker_main():
    rank = int(os.environ["KRONOS_SWEEP_RANK"])
    plan = json.loads(Path(os.environ["KRONOS_SWEEP_PLAN"]).read_text())
    repo = Path(plan["repo"])
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import torch
    from finetune.evaluate_v1_beta_checkpoints import WindowStore
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(f"worker {rank} has no GPU")
    device = torch.device("cuda:0")
    evaluation_root = Path(plan["evaluation_root"])
    manifest = json.loads((evaluation_root / "evaluation_manifest.json").read_text())
    panel, records, _, _, _ = read_records(evaluation_root, manifest)
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])

    tokenizer = KronosTokenizer.from_pretrained(Path(plan["tokenizer_dir"])).to(device).eval()
    models = {}
    for key, directory in plan["checkpoint_dirs"].items():
        models[key] = Kronos.from_pretrained(
            Path(directory), num_sectors=86, num_size_buckets=0, context_layer=6,
            use_size_percentile=True, size_mlp_hidden_dim=64,
        ).to(device).eval()

    shards = Path(plan["shards"])
    deadline = float(plan["deadline"])
    tasks = [task for index, task in enumerate(plan["tasks"]) if index % WORLD_SIZE == rank]
    print(json.dumps({
        "phase": "worker_started", "rank": rank, "tasks": len(tasks),
        "gpu": torch.cuda.get_device_name(0),
    }), flush=True)

    for task in tasks:
        arm = ARMS_BY_NAME[task["arm"]]
        shard = shards / f"{arm['name']}_{task['date']}.csv.gz"
        if shard.is_file():
            continue
        if time.time() > deadline:
            print(json.dumps({"phase": "worker_deadline", "rank": rank, "skipped_from": task}), flush=True)
            break
        started = time.time()
        date_records = [row for row in records if row["asof_date"] == task["date"]]
        result = decode_date(arm, models[arm["checkpoint"]], tokenizer, date_records, store, device)
        staging = shard.with_suffix(".tmp")
        result.to_csv(staging, index=False, compression="gzip")
        os.replace(staging, shard)
        print(json.dumps({
            "phase": "shard_done", "rank": rank, "arm": arm["name"], "date": task["date"],
            "rows": int(len(result)), "seconds": round(time.time() - started, 1),
        }), flush=True)

    for model in models.values():
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"phase": "worker_finished", "rank": rank}), flush=True)


def parent_main():
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = OUTPUT / "shards"
    shards.mkdir(exist_ok=True)

    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if len(gpu_names) != WORLD_SIZE or any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"Expected exactly two Tesla T4 GPUs, found {gpu_names}")

    repo = clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import pandas as pd

    tokenizer_dir, tokenizer_sha = resolve_tokenizer()
    checkpoint_dirs = {}
    provenance = {}
    for key, spec in CHECKPOINTS.items():
        weights = find_one(spec["glob"])
        actual = sha256_file(weights)
        if actual != spec["sha256"]:
            raise RuntimeError(f"{key} sha mismatch: {actual}")
        metric = json.loads((weights.parent / "best_metric.json").read_text())
        if int(metric.get("segment", -1)) != spec["segment"]:
            raise RuntimeError(f"{key} unexpected segment: {metric}")
        if not math.isclose(float(metric["forecast_loss"]), spec["val_forecast_ce"], rel_tol=1e-9):
            raise RuntimeError(f"{key} unexpected val forecast CE: {metric}")
        checkpoint_dirs[key] = str(weights.parent)
        provenance[key] = {
            "path": str(weights), "sha256": actual, "segment": spec["segment"],
            "val_forecast_ce": spec["val_forecast_ce"],
            "baseline_d10_pooled_rank_ic": spec["baseline_d10_pooled_rank_ic"],
        }

    evaluation_root, manifest = load_evaluation()
    panel, records, sample_key, start, end = read_records(evaluation_root, manifest)
    dates = sorted({row["asof_date"] for row in records})
    del panel  # only the workers need it resident
    gc.collect()

    tasks = [{"arm": arm["name"], "date": date} for arm in ARMS for date in dates]
    plan_path = OUTPUT / "plan.json"
    plan_path.write_text(json.dumps({
        "repo": str(repo),
        "tokenizer_dir": str(tokenizer_dir),
        "checkpoint_dirs": checkpoint_dirs,
        "evaluation_root": str(evaluation_root),
        "shards": str(shards),
        "deadline": started + HARD_LIMIT_SECONDS,
        "tasks": tasks,
    }, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({
        "phase": "inputs_resolved",
        "gpus": gpu_names,
        "tokenizer_sha256": tokenizer_sha,
        "checkpoints": provenance,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "signal_dates": len(dates),
        "records": len(records),
        "arms": [
            {k: arm[k] for k in ("name", "checkpoint", "sample_count", "temperature", "top_p", "seed")}
            for arm in ARMS
        ],
        "tasks": len(tasks),
        "effective_batch": EFFECTIVE_BATCH,
    }, ensure_ascii=False), flush=True)

    script = str(Path(globals().get("__file__") or sys.argv[0]).resolve())
    processes = []
    for rank in range(WORLD_SIZE):
        environment = os.environ.copy()
        environment.update({
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(rank),
            "KRONOS_SWEEP_RANK": str(rank),
            "KRONOS_SWEEP_PLAN": str(plan_path),
        })
        processes.append(subprocess.Popen([sys.executable, "-u", script], env=environment))
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise RuntimeError(f"Worker exit codes: {failures}")

    metrics = {}
    volatility = {}
    incomplete = []
    for arm in ARMS:
        parts = sorted(shards.glob(f"{arm['name']}_*.csv.gz"))
        if len(parts) != len(dates):
            incomplete.append({"arm": arm["name"], "have": len(parts), "need": len(dates)})
            continue
        frame = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
        if len(frame) != len(records):
            raise RuntimeError(f"{arm['name']} row mismatch: {len(frame)} != {len(records)}")
        metrics[arm["name"]] = summarize(frame)
        volatility[arm["name"]] = volatility_summary(frame)
        (OUTPUT / f"metrics_{arm['name']}.json").write_text(
            json.dumps(metrics[arm["name"]], ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({
            "phase": "arm_complete", "arm": arm["name"],
            "d10_pooled_rank_ic": metrics[arm["name"]]["by_horizon"][9]["pooled_rank_ic"],
            "vol": volatility[arm["name"]],
        }, ensure_ascii=False), flush=True)

    def pooled(name):
        if name not in metrics:
            return None
        return metrics[name]["by_horizon"][9]["pooled_rank_ic"]

    grid_rows = []
    for arm in ARMS:
        ic = pooled(arm["name"])
        vol = volatility.get(arm["name"]) or {}
        cal = vol.get("calibration_ratio_realized_over_predicted")
        log_cal = abs(math.log(cal)) if cal and cal > 0 else None
        grid_rows.append({
            "name": arm["name"],
            "temperature": arm["temperature"],
            "top_p": arm["top_p"],
            "sample_count": arm["sample_count"],
            "d10_pooled_rank_ic": ic,
            "d10_mae": None if arm["name"] not in metrics else metrics[arm["name"]]["by_horizon"][9]["mae"],
            "pooled_vol_rank_ic": vol.get("pooled_vol_rank_ic"),
            "calibration_ratio": cal,
            "abs_log_calibration": log_cal,
            "in_joint_set": arm["sample_count"] == 8,
        })

    joint_pool = [row for row in grid_rows if row["in_joint_set"] and row["d10_pooled_rank_ic"] is not None]
    best_ic_row = max(joint_pool, key=lambda row: row["d10_pooled_rank_ic"]) if joint_pool else None
    best_cal_row = None
    if any(row["abs_log_calibration"] is not None for row in joint_pool):
        best_cal_row = min(
            [row for row in joint_pool if row["abs_log_calibration"] is not None],
            key=lambda row: (row["abs_log_calibration"], -(row["pooled_vol_rank_ic"] or -1)),
        )
    feasible = []
    recommended = None
    if best_ic_row is not None:
        ceiling = best_ic_row["d10_pooled_rank_ic"]
        feasible = [
            row for row in joint_pool
            if row["abs_log_calibration"] is not None
            and row["d10_pooled_rank_ic"] >= ceiling - JOINT_IC_SLACK
        ]
        if feasible:
            recommended = min(
                feasible,
                key=lambda row: (
                    row["abs_log_calibration"],
                    -(row["pooled_vol_rank_ic"] or -1),
                    -(row["d10_pooled_rank_ic"] or -1),
                ),
            )

    write_summary({
        "status": "complete" if not incomplete else "partial",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "question": (
            "On C2 best, which (T, top_p) at N=8 jointly keeps D10 Rank IC "
            "and path-volatility calibration?"
        ),
        "joint_rule": {
            "applies_to": "sample_count=8",
            "ic_slack": JOINT_IC_SLACK,
            "steps": [
                "keep N=8 arms within 0.010 D10 pooled Rank IC of the best N=8 arm",
                "minimize |log(realized/predicted path vol)|",
                "break ties on higher vol Rank IC, then higher return IC",
            ],
        },
        "checkpoints": provenance,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "elapsed_sec": time.time() - started,
        "grid": grid_rows,
        "best_return_ic": best_ic_row,
        "best_vol_calibration": best_cal_row,
        "joint_feasible": feasible,
        "joint_recommendation": recommended,
        "production_baseline": next((row for row in grid_rows if row["name"] == "prod_t065_p80_n5"), None),
        "volatility": volatility,
        "metrics": metrics,
        "incomplete": incomplete,
    })


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    if os.environ.get("KRONOS_SWEEP_RANK") is not None:
        worker_main()
    else:
        parent_main()


if __name__ == "__main__":
    main()
