#!/usr/bin/env python3
"""T1: how much Rank IC is left in sampling more paths? (CPU only, nothing fitted on labels)

Each stored path is the model's forecast plus independent sampling noise, so an
average of k paths has IC(k) = IC_inf * sqrt(k / (k + c)), where c is the noise
variance of one path relative to the signal variance.  IC is measured at
k = 1, 2, 4, 8, 16 (disjoint groups of the 16 stored paths, averaged over
groups) for both the path mean and the path median, and the two parameters are
fitted by least squares.

Pre-registered verdict (finetune/reports/20261112_c2_ceiling_tests_prereg_cn.md):
IC_inf - IC(16) >= 0.01 -> "worth adding paths", otherwise "not worth it".

    python3 finetune/c2_ceiling/path_count_ceiling.py --shards scratch/c2_roll/eval/c2 \\
        --date-from 2026-09-15 --date-to 2026-10-23 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD, SEAL_AFTER, KS = 0.01, "2026-10-23", (1, 2, 4, 8, 16)
KS_ARRAY = np.asarray(KS, dtype=float)


def daily_ic(dates: np.ndarray, score: np.ndarray, actual: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"d": dates, "s": score, "a": actual})
    return frame.groupby("d").apply(lambda g: g["s"].corr(g["a"], method="spearman"), include_groups=False)


def fit_ceiling(ks: np.ndarray, ics: np.ndarray) -> tuple[float, float]:
    """Least squares for IC_inf and c on a grid of c (IC_inf is linear given c)."""
    best = (np.inf, np.nan, np.nan)
    for c in np.exp(np.linspace(np.log(0.01), np.log(200.0), 4000)):
        basis = np.sqrt(ks / (ks + c))
        ic_inf = float(basis @ ics / (basis @ basis))
        error = float(((ic_inf * basis - ics) ** 2).sum())
        if error < best[0]:
            best = (error, ic_inf, float(c))
    return best[1], best[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.date_to > SEAL_AFTER:
        raise SystemExit(f"Refusing: signal dates after {SEAL_AFTER} are sealed")

    columns = ["asof_date", "actual_return_d10"] + [f"path{j:02d}_d10" for j in range(16)]
    frame = pd.concat([pd.read_csv(p, usecols=columns) for p in sorted(glob.glob(f"{args.shards}/*.csv.gz"))])
    frame = frame[(frame.asof_date >= args.date_from) & (frame.asof_date <= args.date_to)]
    dates, actual = frame.asof_date.to_numpy(), frame.actual_return_d10.to_numpy()
    paths = frame[columns[2:]].to_numpy()

    # The sqrt law is exact for the path MEAN; the median of k paths is noisier by a
    # k-dependent factor, so its curve fits worse and under-states the ceiling
    # (checked on synthetic data).  Both are fitted; the verdict uses the larger
    # ceiling against the production number, IC of the 16-path median.
    rng, n, curves = np.random.default_rng(20261112), frame.asof_date.nunique(), {}
    blocks = []
    for _ in range(1000):
        index = []
        while len(index) < n:
            start = int(rng.integers(0, n - 5 + 1))
            index.extend(range(start, start + 5))
        blocks.append(index[:n])
    for name, aggregate in (("median", np.median), ("mean", np.mean)):
        table = pd.concat([
            pd.concat([daily_ic(dates, aggregate(paths[:, start:start + k], axis=1), actual)
                       for start in range(0, 16, k)], axis=1).mean(axis=1)      # disjoint groups of k paths
            for k in KS], axis=1).to_numpy()
        ics = table.mean(axis=0)
        ic_inf, noise_ratio = fit_ceiling(KS_ARRAY, ics)
        fitted = ic_inf * np.sqrt(KS_ARRAY / (KS_ARRAY + noise_ratio))
        curves[name] = {
            "ic_by_paths": {str(k): float(v) for k, v in zip(KS, ics)}, "ic_inf": ic_inf,
            "noise_to_signal_c": noise_ratio, "fit_max_abs_error": float(np.abs(fitted - ics).max()),
            "ic_at_32_predicted": float(ic_inf * np.sqrt(32 / (32 + noise_ratio))),
            "ic_at_64_predicted": float(ic_inf * np.sqrt(64 / (64 + noise_ratio))),
            "_resampled_ic_inf": [fit_ceiling(KS_ARRAY, table[index].mean(axis=0))[0] for index in blocks],
            "_resampled_ic16": [float(table[index, -1].mean()) for index in blocks],
        }
    production = curves["median"]["ic_by_paths"]["16"]
    gap = max(curves["median"]["ic_inf"], curves["mean"]["ic_inf"]) - production
    resampled = (np.maximum(curves["median"]["_resampled_ic_inf"], curves["mean"]["_resampled_ic_inf"])
                 - np.asarray(curves["median"]["_resampled_ic16"]))
    for curve in curves.values():
        del curve["_resampled_ic_inf"], curve["_resampled_ic16"]
    report = {
        "shards": args.shards, "date_from": args.date_from, "date_to": args.date_to,
        "rows": int(len(frame)), "dates": int(n), "model": "IC(k) = IC_inf * sqrt(k / (k + c))",
        "curves": curves, "production_ic_median_16": production,
        "headroom_inf_minus_16": gap,
        "headroom_ci95": [float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))],
        "threshold": THRESHOLD, "verdict": "worth_adding_paths" if gap >= THRESHOLD else "not_worth_adding_paths",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
