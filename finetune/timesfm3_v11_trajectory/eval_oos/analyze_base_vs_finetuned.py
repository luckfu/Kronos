"""Base TimesFM-3 vs the fine-tuned SweetSpot on identical rows (CPU only).

Arm B1 is the same 8 channels with channel 7 (sector) held at zero, no LoRA.  Both runs cover
the same 335,312 window/date pairs in the same order and share bit-identical labels, so every
number here is paired.

Pre-registered in finetune/reports/20261113_timesfm3_base_vs_finetuned_prereg_cn.md: the verdict
is the paired daily Rank IC difference on segment B, 5-day block bootstrap, 2000 draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = {"A_selection_window": ("2026-08-03", "2026-09-17"), "B_new_window": ("2026-09-18", "2026-10-23"),
            "C_early": ("2026-07-17", "2026-07-31")}
RANK_GROUPS = (("top10", 1, 10), ("rank11_100", 11, 100), ("rest", 101, 10**9))
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


def daily_ic(frame: pd.DataFrame, column: str, method: str) -> pd.Series:
    return frame.groupby("date").apply(lambda g: g[column].corr(g["actual"], method=method), include_groups=False)


def describe(frame: pd.DataFrame, column: str) -> dict:
    pred, actual = frame[column].to_numpy(), frame["actual"].to_numpy()
    hit = np.sign(pred) == np.sign(actual)
    majority = max((actual > 0).mean(), (actual < 0).mean())
    pearson, rank = daily_ic(frame, column, "pearson"), daily_ic(frame, column, "spearman")
    centred_p = pred - frame.groupby("date")[column].transform("mean").to_numpy()
    centred_a = actual - frame.groupby("date")["actual"].transform("mean").to_numpy()
    out = {"samples": int(len(frame)), "signal_days": int(frame["date"].nunique()),
           "direction": float(hit.mean()), "majority_direction": float(majority),
           "direction_minus_majority": float(hit.mean() - majority),
           "excess_direction": float((np.sign(centred_p) == np.sign(centred_a)).mean()),
           "daily_ic": float(pearson.mean()), "daily_ic_ci95": block_ci(pearson.to_numpy()),
           "daily_rank_ic": float(rank.mean()), "positive_ic_days": float((pearson > 0).mean()),
           "pooled_ic": float(np.corrcoef(pred, actual)[0, 1]),
           "mae": float(np.abs(pred - actual).mean()), "mae_zero": float(np.abs(actual).mean()),
           "rmse": float(np.sqrt(((pred - actual) ** 2).mean())),
           "mean_pred": float(pred.mean()), "mean_actual": float(actual.mean()),
           "predicted_up_share": float((pred > 0).mean()),
           "pred_cross_section_std": float(frame.groupby("date")[column].std().mean()),
           "actual_cross_section_std": float(frame.groupby("date")["actual"].std().mean())}
    work = frame.copy()
    work["rank"] = work.groupby("date")[column].rank(ascending=False, method="first")
    work["market"] = work.groupby("date")["actual"].transform("mean")
    out["by_rank"] = {}
    for name, low, high in RANK_GROUPS:
        group = work[(work["rank"] >= low) & (work["rank"] <= high)]
        out["by_rank"][name] = {"rows": int(len(group)), "mean_pred": float(group[column].mean()),
                                "mean_actual": float(group["actual"].mean()),
                                "excess_vs_market": float((group["actual"] - group["market"]).mean()),
                                "share_up": float((group["actual"] > 0).mean()),
                                "share_reaching_forecast": float((group["actual"] >= group[column]).mean())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    def load(path: str, column: str) -> pd.DataFrame:
        data = np.load(path)
        return pd.DataFrame({"symbol": data["symbols"].astype(str), "date": data["dates"].astype(str),
                             column: data["pred_ret"][:, -1].astype(float),
                             "actual": data["true_ret"][:, -1].astype(float)})

    base, tuned = load(args.base, "base"), load(args.finetuned, "tuned")
    if not (base[["symbol", "date"]].equals(tuned[["symbol", "date"]])):
        raise SystemExit("the two runs are not aligned on the same rows")
    if not np.array_equal(base["actual"].to_numpy(), tuned["actual"].to_numpy()):
        raise SystemExit("labels differ between the two runs")
    frame = tuned.merge(base[["symbol", "date", "base"]], on=["symbol", "date"], validate="one_to_one")
    frame["market"] = frame.groupby("date")["actual"].transform("mean")

    report = {"rows": int(len(frame)), "signal_days": int(frame["date"].nunique()),
              "bootstrap": BOOTSTRAP, "segments": {},
              "prediction_rank_corr": float(frame.groupby("date").apply(
                  lambda g: g["tuned"].corr(g["base"], method="spearman"), include_groups=False).mean())}
    for name, (start, end) in {**SEGMENTS, "all": ("2026-07-17", "2026-10-23")}.items():
        part = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        tuned_ic, base_ic = daily_ic(part, "tuned", "pearson"), daily_ic(part, "base", "pearson")
        diff = (tuned_ic - base_ic).to_numpy()
        ci = block_ci(diff)
        report["segments"][name] = {
            "signal_start": start, "signal_end": end, "samples": int(len(part)),
            "signal_days": int(part["date"].nunique()),
            "finetuned": describe(part, "tuned"), "base": describe(part, "base"),
            "d_ic_finetuned_minus_base": {"mean": float(diff.mean()), "ci95": ci,
                                          "verdict": "better" if ci and ci[0] > 0 else "worse" if ci and ci[1] < 0 else "inconclusive",
                                          "wins": int((diff > 0).sum()), "days": int(len(diff))}}

    verdict = report["segments"]["B_new_window"]["d_ic_finetuned_minus_base"]
    report["verdict_B"] = {"mean": verdict["mean"], "ci95": verdict["ci95"],
                           "passed": bool(verdict["ci95"] and verdict["ci95"][0] > 0)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"aligned rows {len(frame)} / {report['signal_days']} days; "
          f"same-day prediction rank corr {report['prediction_rank_corr']:+.3f}")
    for name, seg in report["segments"].items():
        f, b = seg["finetuned"], seg["base"]
        print(f"\n== {name} {seg['signal_start']}..{seg['signal_end']}: {seg['signal_days']}d, {seg['samples']} rows")
        print(f"   {'model':10s} {'direction':>9s} {'vs majority':>11s} {'daily IC':>9s} {'day-IC CI':>18s} "
              f"{'MAE':>7s} {'mean pred':>10s} {'pred up':>8s}")
        for label, m in (("fine-tuned", f), ("base B1", b)):
            ci = m["daily_ic_ci95"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            print(f"   {label:10s} {m['direction']:9.2%} {m['direction_minus_majority']:+11.2%} {m['daily_ic']:+9.4f} "
                  f"{ci_text:>18s} {m['mae']:7.2%} {m['mean_pred']:+10.2%} {m['predicted_up_share']:8.0%}")
        d = seg["d_ic_finetuned_minus_base"]
        ci = d["ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        print(f"   daily IC difference (fine-tuned - base): {d['mean']:+.4f} {ci_text} -> {d['verdict']} "
              f"({d['wins']}/{d['days']} days)")
        for label, m in (("fine-tuned", f), ("base B1", b)):
            head = m["by_rank"]["top10"]
            rest = m["by_rank"]["rest"]
            print(f"      {label:10s} top-10: forecast {head['mean_pred']:+.1%} actual {head['mean_actual']:+.1%} "
                  f"vs market {head['excess_vs_market']:+.1%} reached {head['share_reaching_forecast']:.0%} | "
                  f"rest: forecast {rest['mean_pred']:+.1%} actual {rest['mean_actual']:+.1%} vs market {rest['excess_vs_market']:+.1%}")
    print(f"\nverdict on B (pre-registered): {report['verdict_B']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
