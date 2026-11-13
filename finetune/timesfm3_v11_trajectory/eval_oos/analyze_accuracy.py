"""Accuracy of the TimesFM-3 forecasts (not ranking): how close is the predicted return to the realised one?

Reported per pre-registered segment for the base model (arm B1) and the fine-tuned
SweetSpot, with zero-forecast baselines, calibration by predicted magnitude, the share of
forecasts within a tolerance, and the accuracy a best-possible linear rescale would reach.

    python3 finetune/timesfm3_v11_trajectory/eval_oos/analyze_accuracy.py \\
        --finetuned scratch/tfm3/eval/eval_predictions_sweetspot_window.npz \\
        --base scratch/tfm3/eval_base/eval_predictions_base_channel.npz \\
        --out finetune/reports/20261113_timesfm3_accuracy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = {"A_selection_window": ("2026-08-03", "2026-09-17"), "B_new_window": ("2026-09-18", "2026-10-23"),
            "C_early": ("2026-07-17", "2026-07-31"), "all": ("2026-07-17", "2026-10-23")}
TOLERANCES = (0.02, 0.05, 0.10)
BUCKETS = ((-1.0, -0.15), (-0.15, -0.05), (-0.05, 0.0), (0.0, 0.05), (0.05, 0.15), (0.15, 1.0))


def load(path: str, column: str) -> pd.DataFrame:
    data = np.load(path)
    frame = pd.DataFrame({"symbol": data["symbols"].astype(str), "date": data["dates"].astype(str),
                          "actual": data["true_ret"][:, -1].astype(float)})
    for h in range(10):
        frame[f"{column}_d{h + 1}"] = data["pred_ret"][:, h].astype(float)
        frame[f"actual_d{h + 1}"] = data["true_ret"][:, h].astype(float)
    frame[column] = frame[f"{column}_d10"]          # the 10-day forecast the accuracy metrics use
    return frame


def accuracy(pred: np.ndarray, actual: np.ndarray) -> dict:
    error = pred - actual
    zero = np.abs(actual)
    return {
        "samples": int(len(actual)),
        "mae": float(np.abs(error).mean()), "mae_zero": float(zero.mean()),
        "rmse": float(np.sqrt((error ** 2).mean())), "rmse_zero": float(np.sqrt((actual ** 2).mean())),
        "mae_skill_vs_zero": float(1.0 - np.abs(error).mean() / zero.mean()),
        "mse_skill_vs_zero": float(1.0 - (error ** 2).mean() / (actual ** 2).mean()),
        "mean_error": float(error.mean()), "median_error": float(np.median(error)),
        "r2_about_zero": float(1.0 - (error ** 2).sum() / (actual ** 2).sum()),
        "pearson": float(np.corrcoef(pred, actual)[0, 1]),
        "share_within": {f"{int(t * 100)}pct": float((np.abs(error) <= t).mean()) for t in TOLERANCES},
        "sign_hit": float((np.sign(pred) == np.sign(actual)).mean()),
        "sign_hit_majority": float(max((actual > 0).mean(), (actual < 0).mean())),
        "mean_pred": float(pred.mean()), "mean_actual": float(actual.mean()),
        "pred_std": float(pred.std()), "actual_std": float(actual.std()),
    }


def daily_breakdown(frame: pd.DataFrame, column: str) -> dict:
    """How much of the error is a per-day level miss vs the cross-section?"""
    pred, actual = frame[column], frame["actual"]
    pred_day = frame.groupby("date")[column].transform("mean")
    actual_day = frame.groupby("date")["actual"].transform("mean")
    level = pred_day - actual_day
    residual = (pred - pred_day) - (actual - actual_day)
    return {
        "mse_total": float(((pred - actual) ** 2).mean()),
        "mse_from_day_level": float((level ** 2).mean()),
        "mse_from_cross_section": float((residual ** 2).mean()),
        "day_level_share": float((level ** 2).mean() / ((pred - actual) ** 2).mean()),
        "daily_level_corr": float(np.corrcoef(
            frame.groupby("date")[column].mean(), frame.groupby("date")["actual"].mean())[0, 1]),
        "mean_abs_day_level_error": float(level.abs().mean()),
        "mean_abs_cross_section_error": float(residual.abs().mean()),
    }


def calibration(frame: pd.DataFrame, column: str) -> list[dict]:
    """What actually happens to names the model rates in each predicted-return bucket."""
    rows = []
    for low, high in BUCKETS:
        part = frame[(frame[column] > low) & (frame[column] <= high)]
        if not len(part):
            continue
        rows.append({"bucket": f"({low:+.0%}, {high:+.0%}]", "rows": int(len(part)),
                     "mean_pred": float(part[column].mean()), "mean_actual": float(part["actual"].mean()),
                     "median_actual": float(part["actual"].median()),
                     "share_up": float((part["actual"] > 0).mean()),
                     "mae": float((part[column] - part["actual"]).abs().mean())})
    return rows


def best_rescale(frame: pd.DataFrame, column: str) -> dict:
    """Diagnostic ceiling: the single scale factor with the lowest MAE (fitted on the same data)."""
    pred, actual = frame[column].to_numpy(), frame["actual"].to_numpy()
    grid = np.linspace(0.0, 1.5, 151)
    maes = [np.abs(scale * pred - actual).mean() for scale in grid]
    best = int(np.argmin(maes))
    per_day = frame.groupby("date")[column].transform("mean").to_numpy()
    centred = pred - per_day
    centred_actual = actual - frame.groupby("date")["actual"].transform("mean").to_numpy()
    return {"best_scale": float(grid[best]), "mae_at_best_scale": float(maes[best]),
            "mae_at_scale_1": float(np.abs(pred - actual).mean()),
            "cross_section_slope": float((centred * centred_actual).sum() / (centred ** 2).sum())}


def horizon_curve(frame: pd.DataFrame, column: str) -> dict:
    out = {}
    for h in range(1, 11):
        pred, actual = frame[f"{column}_d{h}"].to_numpy(), frame[f"actual_d{h}"].to_numpy()
        out[str(h)] = {"mae": float(np.abs(pred - actual).mean()), "mae_zero": float(np.abs(actual).mean()),
                       "sign_hit": float((np.sign(pred) == np.sign(actual)).mean()),
                       "sign_hit_majority": float(max((actual > 0).mean(), (actual < 0).mean())),
                       "mean_pred": float(pred.mean()), "mean_actual": float(actual.mean()),
                       "pearson": float(np.corrcoef(pred, actual)[0, 1])}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tuned, base = load(args.finetuned, "tuned"), load(args.base, "base")
    if not tuned[["symbol", "date"]].equals(base[["symbol", "date"]]):
        raise SystemExit("the two runs are not aligned")
    if not np.array_equal(tuned["actual"].to_numpy(), base["actual"].to_numpy()):
        raise SystemExit("labels differ between the two runs")
    # the base frame carries the same labels; dropping them keeps the merge from suffixing
    shared = ["actual"] + [f"actual_d{h}" for h in range(1, 11)]
    frame = tuned.merge(base.drop(columns=shared), on=["symbol", "date"], validate="one_to_one")

    report = {"rows": int(len(frame)), "signal_days": int(frame["date"].nunique()), "segments": {}}
    for name, (start, end) in SEGMENTS.items():
        part = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        report["segments"][name] = {
            "signal_start": start, "signal_end": end, "signal_days": int(part["date"].nunique()),
            "market_mean_10d": float(part["actual"].mean()),
            "finetuned": {**accuracy(part["tuned"].to_numpy(), part["actual"].to_numpy()),
                          "daily": daily_breakdown(part, "tuned"), "calibration": calibration(part, "tuned"),
                          "best_rescale": best_rescale(part, "tuned"),
                          "horizons": horizon_curve(part, "tuned")},
            "base": {**accuracy(part["base"].to_numpy(), part["actual"].to_numpy()),
                     "daily": daily_breakdown(part, "base"), "calibration": calibration(part, "base"),
                     "best_rescale": best_rescale(part, "base"),
                     "horizons": horizon_curve(part, "base")},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for name, seg in report["segments"].items():
        print(f"\n== {name} {seg['signal_start']}..{seg['signal_end']}: {seg['signal_days']}d, "
              f"{seg['finetuned']['samples']} rows, market mean {seg['market_mean_10d']:+.2%}")
        print(f"   {'model':10s} {'MAE':>7s} {'零预测':>7s} {'MAE 改善':>9s} {'RMSE':>7s} {'符号命中':>8s} {'多数类':>7s} "
              f"{'误差<=2%':>9s} {'<=5%':>7s} {'<=10%':>7s} {'平均误差':>9s}")
        for label in ("finetuned", "base"):
            m = seg[label]
            s = m["share_within"]
            print(f"   {label:10s} {m['mae']:7.2%} {m['mae_zero']:7.2%} {m['mae_skill_vs_zero']:+9.1%} {m['rmse']:7.2%} "
                  f"{m['sign_hit']:8.1%} {m['sign_hit_majority']:7.1%} {s['2pct']:9.1%} {s['5pct']:7.1%} {s['10pct']:7.1%} "
                  f"{m['mean_error']:+9.2%}")
        for label in ("finetuned", "base"):
            d = seg[label]["daily"]
            print(f"      {label:10s} 误差来源: 当日整体水平 {d['day_level_share']:.0%} / 截面 {1 - d['day_level_share']:.0%}; "
                  f"当日均值相关 {d['daily_level_corr']:+.2f}; 平均绝对误差 水平 {d['mean_abs_day_level_error']:.2%} + 截面 {d['mean_abs_cross_section_error']:.2%}")
            r = seg[label]["best_rescale"]
            print(f"      {label:10s} 事后最优缩放 ×{r['best_scale']:.2f} 时 MAE {r['mae_at_best_scale']:.2%} "
                  f"(现在是 {r['mae_at_scale_1']:.2%}); 去当日均值后的截面斜率 {r['cross_section_slope']:+.2f}")
        for label in ("finetuned", "base"):
            print(f"      {label:10s} 按预测幅度分组的实际结果: " + "; ".join(
                f"{c['bucket']} 预测{c['mean_pred']:+.1%}→实际{c['mean_actual']:+.1%}（上涨{c['share_up']:.0%}, {c['rows']}条）"
                for c in seg[label]["calibration"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
