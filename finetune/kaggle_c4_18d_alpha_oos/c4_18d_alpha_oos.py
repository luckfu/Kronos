"""C4-best confirmatory OOS on the 18-day sealed window through 2026-09-03.

Zero training. Decode is locked before seeing this window:

  production  T=0.65 / top_p=0.8 / N=5
  ranking     T=0.60 / top_p=0.9 / N=16

No temperature or sample_count search. After D10 Rank IC, the kernel
cross-sectionally residualizes predicted and actual 10-day returns on
industry dummies + size decile, then scores overlapping daily ICs with
Newey-West HAC (lag 10), non-overlapping strands, and a moving-block
bootstrap. Residual Rank IC above 0.15 is the pre-registered bar for
style-stripped stock-selection alpha.

2x T4, (arm, date) round-robin, each task reseeds.
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


SOURCE_COMMIT = "2d22d91de1bf1672b8723e43d01bcb7248dd105e"
OUTPUT = Path("/kaggle/working/kronos_c4_18d_alpha_oos")
INPUT = Path("/kaggle/input")
REPO_DIR = Path("/kaggle/working/kronos_repo")
HARD_LIMIT_SECONDS = 39600
PURPOSE = (
    "confirmatory_18d_sealed; decode locked a priori; "
    "industry+size residual Rank IC and overlapping-ICIR diagnostics; "
    "not a decode search"
)
EXPECTED_EVAL_NAME = "kronos_beta_v2_time_oos_through_20260903"
EXPECTED_SIGNAL_START = "2026-08-11"
EXPECTED_SIGNAL_END = "2026-09-03"
EXPECTED_SIGNAL_DATES = 18
EXPECTED_SAMPLES = 92751
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_CACHE = Path("/kaggle/working/kronos_tokenizer_base")
EFFECTIVE_BATCH = 64
WORLD_SIZE = 2
DECODE_SEED = 20260906
NEWEY_WEST_LAG = 10
NONOVERLAP_STRIDE = 10
BOOTSTRAP_BLOCK = 10
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260921
RESIDUAL_IC_BAR = 0.15

CHECKPOINTS = {
    "wc_1e5_c4_best": {
        "glob": "**/small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors",
        "sha256": "9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075",
        "segment": 21,
        "val_forecast_ce": 2.2728769779205322,
        "baseline_13d_n1_d10_pooled_rank_ic": 0.17918741695116097,
    },
}

ARMS = [
    {
        "name": "prod_t065_p80_n5",
        "checkpoint": "wc_1e5_c4_best",
        "sample_count": 5,
        "temperature": 0.65,
        "top_p": 0.8,
        "seed": DECODE_SEED,
        "protocol": "production",
    },
    {
        "name": "rank_t060_p90_n16",
        "checkpoint": "wc_1e5_c4_best",
        "sample_count": 16,
        "temperature": 0.6,
        "top_p": 0.9,
        "seed": DECODE_SEED,
        "protocol": "ranking",
    },
]
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
        repo = REPO_DIR if attempt == 1 else Path(tempfile.mkdtemp(prefix="kronos-c4-18d-")) / "repo"
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
    matches = list(INPUT.glob("**/Kronos-Tokenizer-base/model.safetensors"))
    if len(matches) > 1:
        matches = [path for path in matches if "wc-1e5-c4" in str(path)]
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
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
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
    if manifest.get("name") != EXPECTED_EVAL_NAME:
        raise RuntimeError(
            f"Expected {EXPECTED_EVAL_NAME}, found {manifest.get('name')}"
        )
    isolation = manifest["temporal_isolation"]
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")
    if isolation.get("incremental_signal_start") != EXPECTED_SIGNAL_START:
        raise RuntimeError(f"Unexpected signal start: {isolation}")
    if isolation.get("incremental_signal_end") != EXPECTED_SIGNAL_END:
        raise RuntimeError(f"Unexpected signal end: {isolation}")
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


def residualize_column(frame, column):
    """Cross-section OLS residual on intercept + size_decile + industry dummies."""
    import numpy as np
    import pandas as pd

    result = np.full(len(frame), np.nan, dtype=np.float64)
    for _, rows in frame.groupby("asof_date", sort=True):
        loc = rows.index.to_numpy()
        y = rows[column].to_numpy(dtype=np.float64)
        size = rows["size_decile"].to_numpy(dtype=np.float64)
        dummies = pd.get_dummies(rows["sector"], drop_first=True).to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(rows)), size, dummies])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        result[loc] = y - design @ coef
    return result


def newey_west_mean_se(values, lag):
    import numpy as np

    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return None
    centered = series - series.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    max_lag = min(int(lag), n - 1)
    for k in range(1, max_lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (max_lag + 1.0)
        var += 2.0 * weight * gamma
    if var <= 0:
        return None
    return math.sqrt(var / n)


def overlapping_icir(daily_ics, lag=NEWEY_WEST_LAG):
    import numpy as np

    series = np.asarray(daily_ics, dtype=np.float64)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return None
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    se_nw = newey_west_mean_se(series, lag)
    nw_std = None if se_nw is None else se_nw * math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "naive_std": metric_or_none(std),
        "naive_icir": None if not std else metric_or_none(mean / std),
        "naive_icir_ann_sqrt252": (
            None if not std else metric_or_none(mean / std * math.sqrt(252))
        ),
        "newey_west_lag": lag,
        "newey_west_se_of_mean": se_nw,
        "newey_west_tstat": None if se_nw in (None, 0) else metric_or_none(mean / se_nw),
        "newey_west_std": nw_std,
        "newey_west_icir": None if not nw_std else metric_or_none(mean / nw_std),
        "newey_west_icir_ann_sqrt252": (
            None if not nw_std else metric_or_none(mean / nw_std * math.sqrt(252))
        ),
    }


def nonoverlap_strands(daily_ics, stride=NONOVERLAP_STRIDE):
    import numpy as np

    series = np.asarray(daily_ics, dtype=np.float64)
    strands = []
    for offset in range(stride):
        values = series[offset::stride]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        strands.append({
            "offset": offset,
            "n": int(len(values)),
            "mean": mean,
            "icir": None if not std else metric_or_none(mean / std),
            "icir_ann_sqrt_252_over_stride": (
                None if not std else metric_or_none(
                    mean / std * math.sqrt(252 / stride)
                )
            ),
        })
    means = [row["mean"] for row in strands if row["n"] > 0]
    return {
        "stride": stride,
        "strands": strands,
        "mean_of_strand_means": metric_or_none(float(np.mean(means))) if means else None,
    }


def moving_block_bootstrap(daily_ics, block=BOOTSTRAP_BLOCK, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    import numpy as np

    series = np.asarray(daily_ics, dtype=np.float64)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    means = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([
            np.take(series, (np.arange(block) + start) % n) for start in starts
        ])[:n]
        means[index] = sample.mean()
    mean = float(series.mean())
    return {
        "block": block,
        "reps": reps,
        "seed": seed,
        "mean": mean,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "frac_positive": float((means > 0).mean()),
    }


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
                "direction_accuracy": float(
                    (np.sign(rows[pred_col]) == np.sign(rows[actual_col])).mean()
                ),
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
            "daily_icir_annualized_sqrt252": metric_or_none(
                daily_ic.mean() / ic_std * math.sqrt(252)
            ),
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
    import numpy as np
    import pandas as pd

    valid = frame[(frame["predicted_path_vol"] > 0) & (frame["realized_path_vol"] > 0)]
    if valid.empty:
        return None
    predicted = valid["predicted_path_vol"].to_numpy(dtype=float)
    realized = valid["realized_path_vol"].to_numpy(dtype=float)
    daily_ic = []
    for _, rows in valid.groupby("asof_date", sort=True):
        daily_ic.append(
            rows["predicted_path_vol"].corr(rows["realized_path_vol"], method="spearman")
        )
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


def alpha_diagnostics(frame):
    import pandas as pd

    work = frame.reset_index(drop=True).copy()
    work["residual_predicted_d10"] = residualize_column(work, "predicted_return_d10")
    work["residual_actual_d10"] = residualize_column(work, "actual_return_d10")
    daily = []
    for date, rows in work.groupby("asof_date", sort=True):
        daily.append({
            "asof_date": str(date),
            "samples": int(len(rows)),
            "raw_rank_ic": metric_or_none(
                rows["predicted_return_d10"].corr(rows["actual_return_d10"], method="spearman")
            ),
            "signal_vs_residual_return_rank_ic": metric_or_none(
                rows["predicted_return_d10"].corr(rows["residual_actual_d10"], method="spearman")
            ),
            "double_residual_rank_ic": metric_or_none(
                rows["residual_predicted_d10"].corr(rows["residual_actual_d10"], method="spearman")
            ),
            "signal_vs_size_rank_ic": metric_or_none(
                rows["predicted_return_d10"].corr(rows["size_decile"], method="spearman")
            ),
        })
    daily_raw = [row["raw_rank_ic"] for row in daily]
    daily_single = [row["signal_vs_residual_return_rank_ic"] for row in daily]
    daily_double = [row["double_residual_rank_ic"] for row in daily]
    double_series = pd.Series(daily_double, dtype=float).dropna()
    double_mean = metric_or_none(double_series.mean())
    return {
        "controls": ["sector_dummies_drop_first", "size_decile"],
        "residual_ic_bar": RESIDUAL_IC_BAR,
        "pooled_raw_rank_ic": metric_or_none(
            work["predicted_return_d10"].corr(work["actual_return_d10"], method="spearman")
        ),
        "pooled_signal_vs_residual_return_rank_ic": metric_or_none(
            work["predicted_return_d10"].corr(work["residual_actual_d10"], method="spearman")
        ),
        "pooled_double_residual_rank_ic": metric_or_none(
            work["residual_predicted_d10"].corr(work["residual_actual_d10"], method="spearman")
        ),
        "daily_double_residual_rank_ic_mean": double_mean,
        "clears_residual_bar": (
            None if double_mean is None else bool(double_mean > RESIDUAL_IC_BAR)
        ),
        "overlapping_raw": overlapping_icir(daily_raw),
        "overlapping_double_residual": overlapping_icir(daily_double),
        "nonoverlap_raw": nonoverlap_strands(daily_raw),
        "nonoverlap_double_residual": nonoverlap_strands(daily_double),
        "block_bootstrap_raw_mean_ic": moving_block_bootstrap(daily_raw),
        "block_bootstrap_double_residual_mean_ic": moving_block_bootstrap(daily_double),
        "by_signal_date": daily,
        "single_residual_daily_mean": metric_or_none(
            pd.Series(daily_single, dtype=float).mean()
        ),
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
            "baseline_13d_n1_d10_pooled_rank_ic": spec["baseline_13d_n1_d10_pooled_rank_ic"],
        }

    evaluation_root, manifest = load_evaluation()
    panel, records, sample_key, start, end = read_records(evaluation_root, manifest)
    dates = sorted({row["asof_date"] for row in records})
    if len(dates) != EXPECTED_SIGNAL_DATES or len(records) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SIGNAL_DATES} dates / {EXPECTED_SAMPLES} samples, "
            f"found {len(dates)} / {len(records)}"
        )
    del panel
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
        "evaluation_name": manifest["name"],
        "checkpoints": provenance,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "signal_dates": len(dates),
        "records": len(records),
        "arms": [
            {k: arm[k] for k in (
                "name", "checkpoint", "sample_count", "temperature", "top_p", "seed", "protocol"
            )}
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
    alpha = {}
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
        alpha[arm["name"]] = alpha_diagnostics(frame)
        frame.to_csv(OUTPUT / f"predictions_{arm['name']}.csv.gz", index=False, compression="gzip")
        (OUTPUT / f"metrics_{arm['name']}.json").write_text(
            json.dumps(metrics[arm["name"]], ensure_ascii=False, indent=2) + "\n"
        )
        (OUTPUT / f"alpha_{arm['name']}.json").write_text(
            json.dumps(alpha[arm["name"]], ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({
            "phase": "arm_complete",
            "arm": arm["name"],
            "protocol": arm["protocol"],
            "d10_pooled_rank_ic": metrics[arm["name"]]["by_horizon"][9]["pooled_rank_ic"],
            "double_residual_rank_ic": alpha[arm["name"]]["pooled_double_residual_rank_ic"],
            "clears_residual_bar": alpha[arm["name"]]["clears_residual_bar"],
            "nw_tstat_raw": (alpha[arm["name"]]["overlapping_raw"] or {}).get("newey_west_tstat"),
        }, ensure_ascii=False), flush=True)

    write_summary({
        "status": "complete" if not incomplete else "partial",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "decode_search_performed": False,
        "source_commit": SOURCE_COMMIT,
        "evaluation_name": manifest["name"],
        "locked_protocols": [
            {k: arm[k] for k in (
                "name", "protocol", "temperature", "top_p", "sample_count", "seed"
            )}
            for arm in ARMS
        ],
        "residual_ic_bar": RESIDUAL_IC_BAR,
        "question": (
            "On the sealed 18-day window through 2026-09-03, do locked production "
            "and ranking decode protocols on WC 1e-5 C4 best keep D10 Rank IC "
            "after industry+size residualization, and does overlapping ICIR "
            "survive Newey-West / non-overlap / block bootstrap?"
        ),
        "checkpoints": provenance,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "elapsed_sec": time.time() - started,
        "alpha": alpha,
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
