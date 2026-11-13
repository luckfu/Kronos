"""SweetSpot on the 65-day package: pre-registered segments, baselines, head-of-ranking view, C2 pairing (CPU only).

Reads the kernel's ``eval_predictions_sweetspot_window.npz``.  Segments, the pipeline check and the
pass criterion are frozen in ``finetune/reports/20261112_timesfm3_sweetspot_new_window_prereg_cn.md``.

    python3 finetune/timesfm3_v11_trajectory/eval_oos/analyze_new_window.py \\
        --npz scratch/tfm3/eval/eval_predictions_sweetspot_window.npz \\
        --c2 P16_SPARSE=scratch/oos65/c2_p16_s3 --c2 P16_DENSE=scratch/c2_roll/eval/c2 \\
        --out finetune/reports/20261112_timesfm3_sweetspot_new_window.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = {"A_selection_window": ("2026-08-03", "2026-09-17"), "B_new_window": ("2026-09-18", "2026-10-23"),
            "C_early": ("2026-07-17", "2026-07-31")}
PUBLISHED_A = {"direction": 0.5859, "daily_ic": 0.0956, "samples": 175428}
CHECK_TOLERANCE = {"direction": 0.01, "daily_ic": 0.015}
CRITERION = {"direction_minus_majority": 0.03, "daily_ic": 0.05}
RANK_GROUPS = (("top2", 1, 2), ("top10", 1, 10), ("rank11_100", 11, 100), ("rank101_1000", 101, 1000),
               ("rest", 1001, 10**9))
BOOTSTRAP = {"block": 5, "replicates": 2000, "seed": 20261112}


def block_ci(values: np.ndarray) -> list[float] | None:
    n, block = len(values), BOOTSTRAP["block"]
    if n < 2 * block:
        return None
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = np.empty(BOOTSTRAP["replicates"])
    for r in range(BOOTSTRAP["replicates"]):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        draws[r] = values[idx[:n]].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def daily(frame: pd.DataFrame, column: str, method: str) -> pd.Series:
    return frame.groupby("date").apply(lambda g: g[column].corr(g["actual"], method=method), include_groups=False)


def segment_metrics(frame: pd.DataFrame) -> dict:
    pred, actual = frame["pred"].to_numpy(), frame["actual"].to_numpy()
    hit = np.sign(pred) == np.sign(actual)
    majority = max((actual > 0).mean(), (actual < 0).mean())
    hit_by_day = pd.Series(hit).groupby(frame["date"].to_numpy()).mean()
    majority_by_day = frame.groupby("date")["actual"].apply(lambda a: max((a > 0).mean(), (a < 0).mean()))
    pearson_ic, rank_ic = daily(frame, "pred", "pearson"), daily(frame, "pred", "spearman")
    centred_pred = pred - frame.groupby("date")["pred"].transform("mean").to_numpy()
    centred_actual = actual - frame.groupby("date")["actual"].transform("mean").to_numpy()
    return {
        "samples": int(len(frame)), "signal_days": int(frame["date"].nunique()),
        "direction": float(hit.mean()), "majority_direction": float(majority),
        "direction_minus_majority": float(hit.mean() - majority),
        # The pooled majority can flip between days; this is the same edge with each day's own majority.
        "direction_minus_daily_majority": float((hit_by_day - majority_by_day).mean()),
        "predicted_up_share": float((pred > 0).mean()), "actual_up_share": float((actual > 0).mean()),
        "excess_direction": float((np.sign(centred_pred) == np.sign(centred_actual)).mean()),
        "daily_ic_pearson": float(pearson_ic.mean()), "daily_ic_pearson_ci95": block_ci(pearson_ic.to_numpy()),
        "daily_rank_ic": float(rank_ic.mean()), "daily_rank_ic_ci95": block_ci(rank_ic.to_numpy()),
        "positive_ic_days": float((pearson_ic > 0).mean()),
        "pooled_ic": float(np.corrcoef(pred, actual)[0, 1]),
        "mae": float(np.abs(pred - actual).mean()), "mae_zero": float(np.abs(actual).mean()),
        "rmse": float(np.sqrt(((pred - actual) ** 2).mean())), "rmse_zero": float(np.sqrt((actual ** 2).mean())),
        "mean_pred": float(pred.mean()), "mean_actual": float(actual.mean()),
        "pred_cross_section_std": float(frame.groupby("date")["pred"].std().mean()),
        "actual_cross_section_std": float(frame.groupby("date")["actual"].std().mean()),
        "slope_actual_on_pred_cross_section": float((centred_pred * centred_actual).sum() / (centred_pred ** 2).sum()),
    }


def head_view(frame: pd.DataFrame) -> dict:
    """Forecast vs outcome by rank position, and the top-10 edge split by what the market did."""
    work = frame.copy()
    work["rank"] = work.groupby("date")["pred"].rank(ascending=False, method="first")
    work["market"] = work.groupby("date")["actual"].transform("mean")
    out = {"by_rank": {}}
    for name, low, high in RANK_GROUPS:
        group = work[(work["rank"] >= low) & (work["rank"] <= high)]
        out["by_rank"][name] = {"rows": int(len(group)), "mean_pred": float(group["pred"].mean()),
                                "mean_actual": float(group["actual"].mean()),
                                "median_error": float((group["actual"] - group["pred"]).median()),
                                "excess_vs_market": float((group["actual"] - group["market"]).mean()),
                                "share_up": float((group["actual"] > 0).mean()),
                                "share_reaching_forecast": float((group["actual"] >= group["pred"]).mean())}
    per_day = work[work["rank"] <= 10].groupby("date").agg(excess=("actual", "mean"), market=("market", "first"))
    per_day["excess"] -= per_day["market"]
    up, down = per_day[per_day["market"] > 0], per_day[per_day["market"] <= 0]
    bottom = work.groupby("date").apply(
        lambda g: g.nsmallest(max(1, len(g) // 10), "pred")["actual"].mean() - g["actual"].mean(), include_groups=False)
    top = work.groupby("date").apply(
        lambda g: g.nlargest(max(1, len(g) // 10), "pred")["actual"].mean() - g["actual"].mean(), include_groups=False)
    out["top10"] = {"excess_all": float(per_day["excess"].mean()), "excess_ci95": block_ci(per_day["excess"].to_numpy()),
                    "up_market_days": int(len(up)), "excess_up_market": float(up["excess"].mean()) if len(up) else None,
                    "down_market_days": int(len(down)),
                    "excess_down_market": float(down["excess"].mean()) if len(down) else None,
                    "corr_with_market": float(np.corrcoef(per_day["market"], per_day["excess"])[0, 1])}
    out["deciles"] = {"top_decile_excess": float(top.mean()), "bottom_decile_excess": float(bottom.mean())}
    return out


def pair_with_c2(frame: pd.DataFrame, shard_dir: str) -> dict:
    columns = ["symbol", "asof_date", "predicted_return_d10", "actual_return_d10"]
    c2 = pd.concat([pd.read_csv(p, usecols=columns, dtype={"symbol": str})
                    for p in sorted(glob.glob(str(Path(shard_dir) / "*.csv.gz")))], ignore_index=True)
    c2 = c2.rename(columns={"asof_date": "date", "predicted_return_d10": "c2"})
    merged = frame.merge(c2, on=["symbol", "date"], how="inner")
    if not np.allclose(merged["actual"], merged["actual_return_d10"], atol=1e-5):
        raise SystemExit("labels differ between the TimesFM run and the C2 shards")
    out = {}
    for name, (start, end) in {**SEGMENTS, "all": ("2026-07-17", "2026-10-23")}.items():
        part = merged[(merged["date"] >= start) & (merged["date"] <= end)].copy()
        if part["date"].nunique() < 3:
            continue
        for column in ("pred", "c2"):
            part["r_" + column] = part.groupby("date")[column].rank(pct=True)
        part["fused"] = (part["r_pred"] + part["r_c2"]) / 2.0
        ics = {column: daily(part, column, "spearman") for column in ("pred", "c2", "fused")}
        agreement = part.groupby("date").apply(lambda g: g["pred"].corr(g["c2"], method="spearman"), include_groups=False)
        entry = {"rows": int(len(part)), "signal_days": int(part["date"].nunique()),
                 "rank_ic": {k: float(v.mean()) for k, v in ics.items()},
                 "prediction_rank_corr": float(agreement.mean())}
        for left, right in (("pred", "c2"), ("fused", "c2"), ("fused", "pred")):
            diff = (ics[left] - ics[right]).to_numpy()
            entry[f"{left}_minus_{right}"] = {"mean": float(diff.mean()), "ci95": block_ci(diff)}
        out[name] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--c2", action="append", default=[], help="name=<16-path C2 shard dir>")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    frame = pd.DataFrame({"symbol": data["symbols"].astype(str), "date": data["dates"].astype(str),
                          "pred": data["pred_ret"][:, -1].astype(float), "actual": data["true_ret"][:, -1].astype(float)})
    if frame["date"].max() > "2026-10-23" or frame.duplicated(["symbol", "date"]).any():
        raise SystemExit("unexpected dates or duplicate rows in the prediction file")
    report = {"samples": int(len(frame)), "signal_days": int(frame["date"].nunique()),
              "published_A": PUBLISHED_A, "criterion": CRITERION, "bootstrap": BOOTSTRAP, "segments": {}}
    for name, (start, end) in SEGMENTS.items():
        part = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        report["segments"][name] = {"signal_start": start, "signal_end": end, **segment_metrics(part),
                                    "head": head_view(part)}
    a, b = report["segments"]["A_selection_window"], report["segments"]["B_new_window"]
    report["pipeline_check"] = {
        "direction": a["direction"], "daily_ic": a["daily_ic_pearson"], "samples": a["samples"],
        "passed": bool(abs(a["direction"] - PUBLISHED_A["direction"]) <= CHECK_TOLERANCE["direction"]
                       and abs(a["daily_ic_pearson"] - PUBLISHED_A["daily_ic"]) <= CHECK_TOLERANCE["daily_ic"])}
    report["verdict_B"] = {
        "direction_minus_majority": b["direction_minus_majority"], "daily_ic": b["daily_ic_pearson"],
        "passed": bool(b["direction_minus_majority"] >= CRITERION["direction_minus_majority"]
                       and b["daily_ic_pearson"] >= CRITERION["daily_ic"])}
    report["with_c2"] = {item.split("=", 1)[0]: pair_with_c2(frame, item.split("=", 1)[1]) for item in args.c2}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    check = report["pipeline_check"]
    print(f"pipeline check (segment A vs published): direction {check['direction']:.4f} vs {PUBLISHED_A['direction']}, "
          f"daily IC {check['daily_ic']:.4f} vs {PUBLISHED_A['daily_ic']}, samples {check['samples']} vs "
          f"{PUBLISHED_A['samples']} -> {'PASSED' if check['passed'] else 'FAILED'}")
    for name, seg in report["segments"].items():
        print(f"\n== {name} {seg['signal_start']}..{seg['signal_end']}: {seg['signal_days']} days, {seg['samples']} rows")
        print(f"   direction {seg['direction']:.2%} vs majority {seg['majority_direction']:.2%} "
              f"(edge {seg['direction_minus_majority']:+.2%}, per-day edge {seg['direction_minus_daily_majority']:+.2%}); "
              f"predicted up {seg['predicted_up_share']:.0%}, actual up {seg['actual_up_share']:.0%}")
        print(f"   daily IC {seg['daily_ic_pearson']:+.4f} {seg['daily_ic_pearson_ci95']}, rank IC {seg['daily_rank_ic']:+.4f}, "
              f"positive days {seg['positive_ic_days']:.0%}")
        print(f"   MAE {seg['mae']:.2%} vs zero forecast {seg['mae_zero']:.2%}; mean forecast {seg['mean_pred']:+.2%}, "
              f"actual {seg['mean_actual']:+.2%}; cross-section std forecast {seg['pred_cross_section_std']:.2%} / "
              f"actual {seg['actual_cross_section_std']:.2%}; slope {seg['slope_actual_on_pred_cross_section']:+.2f}")
        for group, row in seg["head"]["by_rank"].items():
            print(f"      {group:13s} n={row['rows']:6d} forecast {row['mean_pred']:+.1%} actual {row['mean_actual']:+.1%} "
                  f"vs market {row['excess_vs_market']:+.1%} | up {row['share_up']:.0%}, reached forecast "
                  f"{row['share_reaching_forecast']:.0%}")
        top = seg["head"]["top10"]
        print(f"      top-10 excess {top['excess_all']:+.2%} {top['excess_ci95']}; market up ({top['up_market_days']}d) "
              f"{top['excess_up_market']}, market down ({top['down_market_days']}d) {top['excess_down_market']}; "
              f"deciles top {seg['head']['deciles']['top_decile_excess']:+.2%} / bottom "
              f"{seg['head']['deciles']['bottom_decile_excess']:+.2%}")
    verdict = report["verdict_B"]
    print(f"\nverdict on B: direction edge {verdict['direction_minus_majority']:+.2%} (needs >= +3.00%), daily IC "
          f"{verdict['daily_ic']:+.4f} (needs >= 0.05) -> {'PASS' if verdict['passed'] else 'FAIL'}")
    for name, segments in report["with_c2"].items():
        for seg, e in segments.items():
            print(f"with C2 [{name}] {seg}: {e['signal_days']}d, rank IC TimesFM {e['rank_ic']['pred']:+.4f} / C2 "
                  f"{e['rank_ic']['c2']:+.4f} / fused {e['rank_ic']['fused']:+.4f}; agreement {e['prediction_rank_corr']:+.3f}; "
                  f"fused-C2 {e['fused_minus_c2']['mean']:+.4f} {e['fused_minus_c2']['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
