"""Zero-training C2 -> P0a/P1 parameter and prediction drift diagnostic.

This kernel is diagnostic-only. It consumes already-produced checkpoints and
13-date OOS prediction shards; it performs neither training nor inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage3_c2_drift_diagnostic")
PURPOSE = (
    "diagnostic_only; zero_training; zero_inference; existing 13-date OOS is "
    "design-contaminated and must not be used for production model selection"
)
C2_SHA256 = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
EXPECTED_ROWS = 66986
STEPS = {0: 0, 5: 1565, 10: 3130, 15: 4695}
BRANCHES = {
    "p0a": {
        "checkpoint_source": "kronos-small-0-1-stage3-ce-only-control",
        "oos_source": "kronos-small-0-1-stage3-ce-only-oos",
        "prediction_prefix": "ce_only",
    },
    "p1": {
        "checkpoint_source": "kronos-small-0-1-stage3-p1-ce-rank",
        "oos_source": "kronos-small-0-1-stage3-p1-ce-rank-oos",
        "prediction_prefix": "p1",
    },
}
CHECKPOINT_GLOBS = {
    5: "stage3_joint_path_smoke/checkpoints/milestone_seg05/model.safetensors",
    10: "stage3_joint_path_smoke/checkpoints/milestone_seg10/model.safetensors",
    15: "stage3_joint_path_smoke/checkpoints/last_model/model.safetensors",
}
PREDICTION_SUFFIXES = {
    0: "seg00_init",
    5: "seg05",
    10: "seg10",
    15: "seg15_last",
}
HORIZONS = tuple(range(1, 11))
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260916


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_root(slug: str) -> Path:
    matches = [path for path in INPUT.glob(f"notebooks/*/{slug}") if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one mounted source {slug}, found {matches}")
    return matches[0]


def find_one(root: Path, pattern: str) -> Path:
    matches = [path for path in root.glob(f"**/{pattern}") if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} below {root}, found {matches}")
    return matches[0]


def resolve_inputs():
    c2_root = source_root("kronos-small-0-1-stage2-cosine-refinement-c2")
    c2 = find_one(
        c2_root,
        "small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors",
    )
    if sha256_file(c2) != C2_SHA256:
        raise RuntimeError(f"C2 SHA mismatch: {sha256_file(c2)}")

    checkpoints = {}
    shard_roots = {}
    summaries = {}
    for branch, spec in BRANCHES.items():
        checkpoint_root = source_root(spec["checkpoint_source"])
        oos_root = source_root(spec["oos_source"])
        checkpoints[(branch, 0)] = c2
        for segment, pattern in CHECKPOINT_GLOBS.items():
            checkpoints[(branch, segment)] = find_one(checkpoint_root, pattern)
        shard_roots[branch] = find_one(
            oos_root, f"kronos_small_0_1_stage3_{'ce_only' if branch == 'p0a' else 'p1_ce_rank'}_oos/summary.json"
        ).parent / "shards"
        summaries[branch] = json.loads((shard_roots[branch].parent / "summary.json").read_text())
    return c2, checkpoints, shard_roots, summaries


def tensor_groups(name: str):
    groups = ["all"]
    if name.startswith("sector_emb."):
        groups += ["sector_embedding", "condition_branch"]
    elif name.startswith("size_mlp."):
        groups += ["size_mlp", "condition_branch"]
    elif name.startswith("size_emb."):
        groups += ["size_embedding", "condition_branch"]
    elif name.startswith("transformer."):
        block = int(name.split(".", 2)[1])
        groups += [f"transformer_block_{block + 1:02d}", "transformer_backbone"]
        if block < 6:
            groups.append("blocks_1_6")
        else:
            groups.append("blocks_7_8")
    elif name.startswith("embedding."):
        groups += ["token_embedding", "backbone_non_transformer"]
    elif name.startswith("time_emb."):
        groups += ["temporal_embedding", "backbone_non_transformer"]
    elif name.startswith("dep_layer."):
        groups += ["dependency_layer", "backbone_non_transformer"]
    elif name.startswith("head."):
        groups.append("forecast_head")
    elif name.startswith("norm."):
        groups += ["final_norm", "backbone_non_transformer"]
    elif name.startswith(("return_head.", "barrier_head.")):
        groups.append("auxiliary_heads")
    else:
        groups.append("unclassified")

    if not name.startswith(("sector_emb.", "size_emb.", "size_mlp.", "head.",
                            "return_head.", "barrier_head.")):
        groups.append("backbone")
    return groups


def new_accumulator():
    return {
        "tensor_count": 0,
        "parameter_count": 0,
        "baseline_sq": 0.0,
        "checkpoint_sq": 0.0,
        "delta_sq": 0.0,
        "dot": 0.0,
        "max_abs_delta": 0.0,
        "changed_parameters": 0,
    }


def accumulate(acc, baseline, current):
    baseline = baseline.double()
    current = current.double()
    delta = current - baseline
    acc["tensor_count"] += 1
    acc["parameter_count"] += baseline.numel()
    acc["baseline_sq"] += float((baseline * baseline).sum())
    acc["checkpoint_sq"] += float((current * current).sum())
    acc["delta_sq"] += float((delta * delta).sum())
    acc["dot"] += float((baseline * current).sum())
    acc["max_abs_delta"] = max(acc["max_abs_delta"], float(delta.abs().max()))
    acc["changed_parameters"] += int((delta != 0).sum())


def finalize_weight_row(branch, segment, group, acc):
    baseline_norm = math.sqrt(acc["baseline_sq"])
    checkpoint_norm = math.sqrt(acc["checkpoint_sq"])
    delta_norm = math.sqrt(acc["delta_sq"])
    count = acc["parameter_count"]
    cosine = acc["dot"] / max(baseline_norm * checkpoint_norm, 1e-30)
    return {
        "branch": branch,
        "segment": segment,
        "step": STEPS[segment],
        "module": group,
        "tensor_count": acc["tensor_count"],
        "parameter_count": count,
        "changed_parameters": acc["changed_parameters"],
        "changed_fraction": acc["changed_parameters"] / max(count, 1),
        "baseline_l2": baseline_norm,
        "checkpoint_l2": checkpoint_norm,
        "delta_l2": delta_norm,
        "relative_l2_drift": delta_norm / max(baseline_norm, 1e-30),
        "baseline_rms": math.sqrt(acc["baseline_sq"] / max(count, 1)),
        "delta_rms": math.sqrt(acc["delta_sq"] / max(count, 1)),
        "cosine_similarity": cosine,
        "max_abs_delta": acc["max_abs_delta"],
    }


def compute_weight_drift(c2: Path, checkpoints):
    module_rows = []
    tensor_rows = []
    with safe_open(c2, framework="pt", device="cpu") as baseline_file:
        baseline_keys = set(baseline_file.keys())
        for branch in BRANCHES:
            for segment in (5, 10, 15):
                path = checkpoints[(branch, segment)]
                by_group = defaultdict(new_accumulator)
                with safe_open(path, framework="pt", device="cpu") as current_file:
                    current_keys = set(current_file.keys())
                    if current_keys != baseline_keys:
                        missing = sorted(baseline_keys - current_keys)
                        extra = sorted(current_keys - baseline_keys)
                        raise RuntimeError(
                            f"State keys differ for {branch} seg{segment}: "
                            f"missing={missing}, extra={extra}"
                        )
                    for name in sorted(baseline_keys):
                        baseline = baseline_file.get_tensor(name)
                        current = current_file.get_tensor(name)
                        if baseline.shape != current.shape:
                            raise RuntimeError(f"Shape mismatch for {name}")
                        tensor_acc = new_accumulator()
                        accumulate(tensor_acc, baseline, current)
                        tensor_rows.append(
                            finalize_weight_row(branch, segment, name, tensor_acc)
                            | {"tensor_name": name}
                        )
                        for group in tensor_groups(name):
                            accumulate(by_group[group], baseline, current)
                module_rows.extend(
                    finalize_weight_row(branch, segment, group, acc)
                    for group, acc in sorted(by_group.items())
                )
                print(
                    json.dumps({
                        "phase": "weight_drift",
                        "branch": branch,
                        "segment": segment,
                        "all": next(row for row in module_rows
                                    if row["branch"] == branch
                                    and row["segment"] == segment
                                    and row["module"] == "all"),
                    }),
                    flush=True,
                )
    return pd.DataFrame(module_rows), pd.DataFrame(tensor_rows)


def read_prediction_set(shard_root: Path, prefix: str, segment: int):
    label = f"{prefix}_{PREDICTION_SUFFIXES[segment]}"
    paths = sorted(shard_root.glob(f"{label}_*.csv.gz"))
    if not paths:
        raise RuntimeError(f"No shards for {label} below {shard_root}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if len(frame) != EXPECTED_ROWS or frame["identity"].duplicated().any():
        raise RuntimeError(
            f"Invalid {label} predictions: rows={len(frame)}, "
            f"duplicates={int(frame['identity'].duplicated().sum())}"
        )
    return frame.sort_values("identity").reset_index(drop=True)


def spearman(left, right):
    return finite_float(pd.Series(left).corr(pd.Series(right), method="spearman"))


def pearson(left, right):
    return finite_float(pd.Series(left).corr(pd.Series(right), method="pearson"))


def rank_ic(prediction, actual):
    return spearman(prediction, actual)


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None, None
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return finite_float(low), finite_float(high)


def cross_section_metrics(base, current, horizon, rng):
    pred_col = f"predicted_return_d{horizon}"
    actual_col = f"actual_return_d{horizon}"
    daily_rows = []
    for date, base_day in base.groupby("asof_date", sort=True):
        current_day = current[current["asof_date"] == date]
        merged = base_day[["identity", pred_col, actual_col]].merge(
            current_day[["identity", pred_col, actual_col]],
            on="identity",
            suffixes=("_base", "_current"),
            validate="one_to_one",
        )
        if len(merged) != len(base_day):
            raise RuntimeError(f"Identity mismatch on {date}")
        base_actual = merged[f"{actual_col}_base"].to_numpy()
        current_actual = merged[f"{actual_col}_current"].to_numpy()
        if not np.allclose(base_actual, current_actual, rtol=0, atol=1e-7, equal_nan=True):
            raise RuntimeError(f"Actual-return mismatch on {date} H{horizon}")
        baseline = merged[f"{pred_col}_base"].to_numpy(dtype=np.float64)
        prediction = merged[f"{pred_col}_current"].to_numpy(dtype=np.float64)
        actual = current_actual.astype(np.float64)
        n_tail = max(1, int(math.ceil(len(merged) * 0.10)))
        base_order = np.argsort(baseline)
        current_order = np.argsort(prediction)
        base_bottom, base_top = set(base_order[:n_tail]), set(base_order[-n_tail:])
        curr_bottom, curr_top = set(current_order[:n_tail]), set(current_order[-n_tail:])
        tail_intersection = len(base_bottom & curr_bottom) + len(base_top & curr_top)
        tail_union = len(base_bottom | curr_bottom) + len(base_top | curr_top)
        daily_rows.append({
            "asof_date": date,
            "samples": len(merged),
            "cross_section_spearman_vs_c2": spearman(baseline, prediction),
            "current_rank_ic": rank_ic(prediction, actual),
            "baseline_rank_ic": rank_ic(baseline, actual),
            "top_bottom_retention": tail_intersection / (2 * n_tail),
            "top_bottom_jaccard": tail_intersection / max(tail_union, 1),
        })
    daily = pd.DataFrame(daily_rows)
    ci_low, ci_high = bootstrap_mean_ci(daily["cross_section_spearman_vs_c2"], rng)
    ic_ci_low, ic_ci_high = bootstrap_mean_ci(daily["current_rank_ic"], rng)
    return daily, {
        "daily_cross_section_spearman_mean": finite_float(
            daily["cross_section_spearman_vs_c2"].mean()
        ),
        "daily_cross_section_spearman_std": finite_float(
            daily["cross_section_spearman_vs_c2"].std(ddof=0)
        ),
        "daily_cross_section_spearman_ci95_low": ci_low,
        "daily_cross_section_spearman_ci95_high": ci_high,
        "daily_rank_ic_mean": finite_float(daily["current_rank_ic"].mean()),
        "daily_rank_ic_std": finite_float(daily["current_rank_ic"].std(ddof=0)),
        "daily_rank_ic_ci95_low": ic_ci_low,
        "daily_rank_ic_ci95_high": ic_ci_high,
        "top_bottom_retention_mean": finite_float(daily["top_bottom_retention"].mean()),
        "top_bottom_jaccard_mean": finite_float(daily["top_bottom_jaccard"].mean()),
    }


def residual_alpha(base, current, horizon):
    pred_col = f"predicted_return_d{horizon}"
    actual_col = f"actual_return_d{horizon}"
    residuals = []
    actuals = []
    betas = []
    daily_ics = []
    for date in sorted(base["asof_date"].unique()):
        baseline = base.loc[base["asof_date"] == date, pred_col].to_numpy(dtype=np.float64)
        prediction = current.loc[current["asof_date"] == date, pred_col].to_numpy(dtype=np.float64)
        actual = current.loc[current["asof_date"] == date, actual_col].to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(baseline)), baseline])
        intercept, beta = np.linalg.lstsq(design, prediction, rcond=None)[0]
        residual = prediction - (intercept + beta * baseline)
        residuals.append(residual)
        actuals.append(actual)
        betas.append(beta)
        daily_ics.append(rank_ic(residual, actual))
    return {
        "projection_beta_mean": finite_float(np.mean(betas)),
        "residual_rank_ic_pooled": rank_ic(np.concatenate(residuals), np.concatenate(actuals)),
        "residual_daily_rank_ic_mean": finite_float(np.mean(daily_ics)),
    }


def prediction_drift_row(branch, segment, horizon, base, current, rng):
    pred_col = f"predicted_return_d{horizon}"
    actual_col = f"actual_return_d{horizon}"
    baseline = base[pred_col].to_numpy(dtype=np.float64)
    prediction = current[pred_col].to_numpy(dtype=np.float64)
    actual = current[actual_col].to_numpy(dtype=np.float64)
    delta = prediction - baseline
    daily, aggregate = cross_section_metrics(base, current, horizon, rng)
    row = {
        "branch": branch,
        "segment": segment,
        "step": STEPS[segment],
        "horizon": horizon,
        "samples": len(current),
        "pooled_prediction_spearman_vs_c2": spearman(baseline, prediction),
        "pooled_prediction_pearson_vs_c2": pearson(baseline, prediction),
        "current_pooled_rank_ic": rank_ic(prediction, actual),
        "baseline_pooled_rank_ic": rank_ic(baseline, actual),
        "mean_abs_drift": finite_float(np.mean(np.abs(delta))),
        "median_abs_drift": finite_float(np.quantile(np.abs(delta), 0.50)),
        "p95_abs_drift": finite_float(np.quantile(np.abs(delta), 0.95)),
        "p99_abs_drift": finite_float(np.quantile(np.abs(delta), 0.99)),
        "max_abs_drift": finite_float(np.max(np.abs(delta))),
        "prediction_std_ratio": finite_float(
            np.std(prediction) / max(np.std(baseline), 1e-30)
        ),
        "sign_flip_rate": finite_float(np.mean(np.signbit(prediction) != np.signbit(baseline))),
        **aggregate,
        **residual_alpha(base, current, horizon),
    }
    daily.insert(0, "horizon", horizon)
    daily.insert(0, "segment", segment)
    daily.insert(0, "branch", branch)
    return row, daily


def compute_prediction_drift(shard_roots):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    horizon_rows = []
    daily_frames = []
    frames = {}
    for branch, spec in BRANCHES.items():
        for segment in (0, 5, 10, 15):
            frames[(branch, segment)] = read_prediction_set(
                shard_roots[branch], spec["prediction_prefix"], segment
            )
        base = frames[(branch, 0)]
        for segment in (5, 10, 15):
            current = frames[(branch, segment)]
            if not base["identity"].equals(current["identity"]):
                raise RuntimeError(f"Prediction identities differ for {branch} seg{segment}")
            for horizon in HORIZONS:
                row, daily = prediction_drift_row(
                    branch, segment, horizon, base, current, rng
                )
                horizon_rows.append(row)
                daily_frames.append(daily)
            d10 = horizon_rows[-1]
            print(
                json.dumps({
                    "phase": "prediction_drift",
                    "branch": branch,
                    "segment": segment,
                    "d10": d10,
                }),
                flush=True,
            )
    return pd.DataFrame(horizon_rows), pd.concat(daily_frames, ignore_index=True), frames


def metric_at_horizon(summary, model, horizon):
    rows = summary["metrics"][model]["by_horizon"]
    for row in rows:
        if int(row.get("horizon_day", row.get("horizon", -1))) == horizon:
            return row
    if len(rows) == 10:
        return rows[horizon - 1]
    raise RuntimeError(f"No H{horizon} metric for {model}")


def validation_objective(checkpoint_root: Path, segment: int):
    if segment == 0:
        payload = json.loads(find_one(checkpoint_root, "stage3_joint_path_smoke/c2_causal_baseline.json").read_text())
        return payload["validation"]["token_loss"]
    if segment in (5, 10):
        payload = json.loads(find_one(
            checkpoint_root,
            f"stage3_joint_path_smoke/checkpoints/milestone_seg{segment:02d}/milestone.json",
        ).read_text())
        return payload["validation_objective"]
    best_metric = json.loads(find_one(
        checkpoint_root,
        "stage3_joint_path_smoke/checkpoints/best_model/best_metric.json",
    ).read_text())
    value = best_metric.get(
        "validation_objective",
        best_metric.get("total_loss", best_metric.get("metric", best_metric.get("value"))),
    )
    if value is None:
        raise RuntimeError(f"Unknown best_metric schema: {best_metric}")
    return value


def build_trajectory(module_drift, prediction_drift, summaries):
    rows = []
    for branch, spec in BRANCHES.items():
        checkpoint_root = source_root(spec["checkpoint_source"])
        prefix = spec["prediction_prefix"]
        for segment in (0, 5, 10, 15):
            model = f"{prefix}_{PREDICTION_SUFFIXES[segment]}"
            h10 = metric_at_horizon(summaries[branch], model, 10)
            row = {
                "branch": branch,
                "segment": segment,
                "step": STEPS[segment],
                "validation_objective": validation_objective(checkpoint_root, segment),
                "oos_d10_rank_ic": h10["pooled_rank_ic"],
                "oos_d10_mae": h10["mae"],
                "oos_d10_daily_ic_mean": h10["daily_rank_ic_mean"],
                "oos_d10_daily_ic_std": h10["daily_rank_ic_std"],
                "oos_d10_top_bottom_spread": h10["mean_top_bottom_10pct"],
            }
            if segment == 0:
                for module in ("all", "backbone", "condition_branch", "blocks_7_8",
                               "forecast_head", "sector_embedding", "size_mlp"):
                    row[f"{module}_relative_l2_drift"] = 0.0
                row["d10_cross_section_spearman_vs_c2"] = 1.0
                row["d10_top_bottom_retention"] = 1.0
                row["d10_residual_rank_ic"] = None
            else:
                weights = module_drift[
                    (module_drift["branch"] == branch)
                    & (module_drift["segment"] == segment)
                ].set_index("module")
                for module in ("all", "backbone", "condition_branch", "blocks_7_8",
                               "forecast_head", "sector_embedding", "size_mlp"):
                    row[f"{module}_relative_l2_drift"] = (
                        finite_float(weights.loc[module, "relative_l2_drift"])
                        if module in weights.index else None
                    )
                pred = prediction_drift[
                    (prediction_drift["branch"] == branch)
                    & (prediction_drift["segment"] == segment)
                    & (prediction_drift["horizon"] == 10)
                ].iloc[0]
                row["d10_cross_section_spearman_vs_c2"] = finite_float(
                    pred["daily_cross_section_spearman_mean"]
                )
                row["d10_top_bottom_retention"] = finite_float(
                    pred["top_bottom_retention_mean"]
                )
                row["d10_residual_rank_ic"] = finite_float(
                    pred["residual_rank_ic_pooled"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_summary(trajectory):
    rows = []
    for branch, frame in trajectory.groupby("branch"):
        trained = frame[frame["segment"] > 0]
        for drift_column in (
            "all_relative_l2_drift",
            "backbone_relative_l2_drift",
            "condition_branch_relative_l2_drift",
            "blocks_7_8_relative_l2_drift",
            "forecast_head_relative_l2_drift",
        ):
            rows.append({
                "branch": branch,
                "drift_measure": drift_column,
                "pearson_drift_vs_d10_rank_ic": pearson(
                    trained[drift_column], trained["oos_d10_rank_ic"]
                ),
                "spearman_drift_vs_d10_rank_ic": spearman(
                    trained[drift_column], trained["oos_d10_rank_ic"]
                ),
                "points": len(trained),
                "warning": "descriptive_only; n=3 checkpoints per branch",
            })
    return pd.DataFrame(rows)


def records(frame):
    return [
        {key: finite_float(value) if isinstance(value, (float, np.floating)) else value
         for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def main():
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    c2, checkpoints, shard_roots, summaries = resolve_inputs()
    provenance = {
        "c2": {"path": str(c2), "sha256": sha256_file(c2)},
        "checkpoints": {
            f"{branch}_seg{segment:02d}": {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for (branch, segment), path in checkpoints.items()
        },
        "oos_shard_roots": {branch: str(path) for branch, path in shard_roots.items()},
    }
    print(json.dumps({"phase": "inputs_resolved", **provenance}), flush=True)

    module_drift, tensor_drift = compute_weight_drift(c2, checkpoints)
    prediction_drift, daily_drift, _ = compute_prediction_drift(shard_roots)
    trajectory = build_trajectory(module_drift, prediction_drift, summaries)
    drift_ic_correlations = correlation_summary(trajectory)

    module_drift.to_csv(OUTPUT / "weight_drift_by_module.csv", index=False)
    tensor_drift.to_csv(OUTPUT / "weight_drift_by_tensor.csv", index=False)
    prediction_drift.to_csv(OUTPUT / "prediction_drift_by_horizon.csv", index=False)
    daily_drift.to_csv(OUTPUT / "prediction_drift_by_date.csv", index=False)
    trajectory.to_csv(OUTPUT / "checkpoint_trajectory.csv", index=False)
    drift_ic_correlations.to_csv(OUTPUT / "drift_ic_correlations.csv", index=False)

    summary = {
        "status": "complete",
        "purpose": PURPOSE,
        "training_performed": False,
        "inference_performed": False,
        "baseline": "C2 best segment 179",
        "c2_frozen_sha256": C2_SHA256,
        "evaluation_warning": (
            "13-date OOS is design-contaminated; results diagnose mechanism only "
            "and cannot select a production model"
        ),
        "definitions": {
            "relative_l2_drift": "||W_t - W_C2||_2 / ||W_C2||_2",
            "condition_branch": "sector_emb + size_emb(if present) + size_mlp",
            "blocks_7_8": "1-based transformer blocks 7 and 8 (state keys transformer.6/.7)",
            "backbone": "all tensors except condition embeddings/MLP and output/auxiliary heads",
            "top_bottom_retention": (
                "mean same-tail member retention for top and bottom deciles; "
                "random expectation is 0.10"
            ),
            "residual_alpha": (
                "within-date OLS residual after projecting checkpoint prediction "
                "on C2 prediction; diagnostic only"
            ),
            "validation_objective_warning": (
                "P0a uniform CE and P1 weighted CE + history use different objective "
                "definitions; compare each trajectory to its own C2 baseline only"
            ),
        },
        "bootstrap": {
            "unit": "signal_date",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "dates": 13,
        },
        "provenance": provenance,
        "trajectory": records(trajectory),
        "d10_prediction_drift": records(
            prediction_drift[prediction_drift["horizon"] == 10]
        ),
        "module_weight_drift": records(
            module_drift[module_drift["module"].isin([
                "all", "backbone", "condition_branch", "blocks_7_8",
                "forecast_head", "sector_embedding", "size_mlp",
            ])]
        ),
        "drift_ic_correlations": records(drift_ic_correlations),
        "next_decision": (
            "Use module localization and drift magnitude to choose short 1/3/5-segment "
            "trainable-mask probes; do not continue full-model CE training"
        ),
        "elapsed_sec": time.time() - started,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "phase": "complete",
        "output": str(OUTPUT),
        "elapsed_sec": summary["elapsed_sec"],
        "trajectory": summary["trajectory"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
