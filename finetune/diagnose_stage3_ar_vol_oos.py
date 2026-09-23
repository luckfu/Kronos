"""Attribute the AR-vol C2 OOS change versus the frozen C2 baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = Path("artifacts/c2_18d_alpha_oos_frozen/kronos_c2_18d_alpha_oos/predictions_prod_t065_p80_n5.csv.gz")
CANDIDATE = Path("/tmp/arvol18d_output_20260923/stage3_ar_vol_18d_oos/predictions_ar_vol_c2.csv.gz")
OUTPUT = Path("finetune/reports/stage3_ar_vol_oos_attribution_20260923.json")
BASE_MODEL = Path("artifacts/kronos_small_0_1_stage2_cosine_refinement_c2_output/kronos_small_v21/outputs/models/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors")
CANDIDATE_MODEL = Path("/tmp/arvolc2_output_20260923/stage3_joint_path_smoke/checkpoints/best_model/model.safetensors")
HORIZONS = range(1, 11)


def rank_ic(frame, pred, actual):
    value = frame[pred].corr(frame[actual], method="spearman")
    return None if pd.isna(value) else float(value)


def mean_daily(frame, pred, actual, group=None):
    rows = []
    keys = ["asof_date"] if group is None else ["asof_date", group]
    for key, part in frame.groupby(keys, sort=True, observed=True):
        if len(part) < 8:
            continue
        value = rank_ic(part, pred, actual)
        if value is not None:
            rows.append(value)
    return None if not rows else float(np.mean(rows))


def horizon_table(frame):
    rows = []
    for horizon in HORIZONS:
        pred, actual = f"predicted_return_d{horizon}", f"actual_return_d{horizon}"
        rows.append({
            "horizon": horizon,
            "pooled_rank_ic": rank_ic(frame, pred, actual),
            "daily_rank_ic_mean": mean_daily(frame, pred, actual),
            "direction_accuracy": float((np.sign(frame[pred]) == np.sign(frame[actual])).mean()),
            "mae": float((frame[pred] - frame[actual]).abs().mean()),
        })
    return rows


def date_table(base, cand):
    rows = []
    for date in sorted(base["asof_date"].unique()):
        b = base[base.asof_date == date]
        c = cand[cand.asof_date == date]
        rows.append({
            "asof_date": date,
            "samples": len(b),
            "baseline_d10_rank_ic": rank_ic(b, "predicted_return_d10", "actual_return_d10"),
            "candidate_d10_rank_ic": rank_ic(c, "predicted_return_d10", "actual_return_d10"),
            "delta_d10_rank_ic": rank_ic(c, "predicted_return_d10", "actual_return_d10")
            - rank_ic(b, "predicted_return_d10", "actual_return_d10"),
            "baseline_pred_std": float(b.predicted_return_d10.std()),
            "candidate_pred_std": float(c.predicted_return_d10.std()),
            "baseline_pred_mean": float(b.predicted_return_d10.mean()),
            "candidate_pred_mean": float(c.predicted_return_d10.mean()),
            "baseline_vol_ratio": float((b.realized_path_vol / b.predicted_path_vol).mean()),
            "candidate_vol_ratio": float((c.realized_path_vol / c.predicted_path_vol).mean()),
        })
    return rows


def group_table(base, cand, column):
    rows = []
    for value in sorted(base[column].dropna().unique()):
        b = base[base[column] == value]
        c = cand[cand[column] == value]
        b_ic = mean_daily(b, "predicted_return_d10", "actual_return_d10")
        c_ic = mean_daily(c, "predicted_return_d10", "actual_return_d10")
        rows.append({
            "group": str(value),
            "samples": len(b),
            "baseline_daily_d10_rank_ic": b_ic,
            "candidate_daily_d10_rank_ic": c_ic,
            "delta_daily_d10_rank_ic": None if b_ic is None or c_ic is None else c_ic - b_ic,
            "baseline_pred_std": float(b.predicted_return_d10.std()),
            "candidate_pred_std": float(c.predicted_return_d10.std()),
            "baseline_mae": float((b.predicted_return_d10 - b.actual_return_d10).abs().mean()),
            "candidate_mae": float((c.predicted_return_d10 - c.actual_return_d10).abs().mean()),
        })
    return rows


def distribution(base, cand):
    merged = base[["identity", "predicted_return_d10"]].merge(
        cand[["identity", "predicted_return_d10"]], on="identity", suffixes=("_baseline", "_candidate"),
    )
    b = merged.predicted_return_d10_baseline
    c = merged.predicted_return_d10_candidate
    top = {}
    for fraction in (0.01, 0.05, 0.10):
        n = max(1, int(len(merged) * fraction))
        bi = set(merged.nlargest(n, "predicted_return_d10_baseline").identity)
        ci = set(merged.nlargest(n, "predicted_return_d10_candidate").identity)
        top[f"top_{int(fraction * 100)}pct"] = {
            "n": n,
            "overlap": len(bi & ci) / n,
        }
    return {
        "prediction_rank_spearman": float(b.corr(c, method="spearman")),
        "prediction_pearson": float(b.corr(c)),
        "baseline": {
            "mean": float(b.mean()), "std": float(b.std()),
            "q01": float(b.quantile(.01)), "q50": float(b.quantile(.50)), "q99": float(b.quantile(.99)),
            "positive_rate": float((b > 0).mean()),
        },
        "candidate": {
            "mean": float(c.mean()), "std": float(c.std()),
            "q01": float(c.quantile(.01)), "q50": float(c.quantile(.50)), "q99": float(c.quantile(.99)),
            "positive_rate": float((c > 0).mean()),
        },
        "top_overlap": top,
    }


def weight_drift():
    from safetensors.torch import load_file

    base = load_file(str(BASE_MODEL))
    cand = load_file(str(CANDIDATE_MODEL))
    groups = {}
    for key in base:
        prefix = key.split(".")[0]
        delta = (cand[key].float() - base[key].float())
        original = base[key].float()
        item = groups.setdefault(prefix, [0.0, 0.0, 0])
        item[0] += float(delta.square().sum())
        item[1] += float(original.square().sum())
        item[2] += 1
    return {
        key: {
            "tensors": value[2],
            "relative_l2": float((value[0] / max(value[1], 1e-30)) ** 0.5),
        }
        for key, value in sorted(
            groups.items(),
            key=lambda item: item[1][0] / max(item[1][1], 1e-30),
            reverse=True,
        )
    }


def main():
    base = pd.read_csv(BASELINE).sort_values("identity").reset_index(drop=True)
    cand = pd.read_csv(CANDIDATE).sort_values("identity").reset_index(drop=True)
    if not base.identity.equals(cand.identity):
        raise RuntimeError("OOS identities do not align")
    for column in ("asof_date", "symbol", "sector", "size_decile"):
        if not base[column].equals(cand[column]):
            raise RuntimeError(f"OOS metadata mismatch: {column}")
    result = {
        "status": "complete",
        "window": {
            "signal_dates": int(base.asof_date.nunique()),
            "samples": len(base),
            "start": str(base.asof_date.min()),
            "end": str(base.asof_date.max()),
        },
        "horizons": {
            "baseline": horizon_table(base),
            "candidate": horizon_table(cand),
        },
        "dates": date_table(base, cand),
        "sectors": group_table(base, cand, "sector"),
        "size_deciles": group_table(base, cand, "size_decile"),
        "distribution": distribution(base, cand),
        "weight_drift": weight_drift(),
        "volatility": {
            "baseline_mean_predicted": float(base.predicted_path_vol.mean()),
            "candidate_mean_predicted": float(cand.predicted_path_vol.mean()),
            "realized_mean": float(base.realized_path_vol.mean()),
            "baseline_ratio": float((base.realized_path_vol / base.predicted_path_vol).mean()),
            "candidate_ratio": float((cand.realized_path_vol / cand.predicted_path_vol).mean()),
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["distribution"], ensure_ascii=False, indent=2))
    print(json.dumps(result["volatility"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
