#!/usr/bin/env python3
"""T2 / T3 / T4: can a small model on top of frozen C2 rank better than C2's 16-path median?

Protocol is frozen in finetune/reports/20261112_c2_ceiling_tests_prereg_cn.md.
This script does everything that happens BEFORE the single look at the test
window: build features, pick the ridge penalty on a time split inside the
fitting window, refit on the whole fitting window, and write test-window scores
as shards.  The comparison itself is rolling_compare.py.

  fitting window   signal dates 2026-07-17 .. 2026-08-28 (31 days)
      sub-fit      first 15 dates;  10 dates skipped (label overlap);  validation = last 6
  test window      signal dates 2026-09-15 .. 2026-10-23, scored once per test

Target and features are within-date percentile ranks minus 0.5 (hidden states:
within-date z-scores), so every model is a cross-sectional ranker.  Ridge
regression in closed form; no dependency beyond numpy/pandas.

    python3 finetune/c2_ceiling/fit_candidates.py --test T2 \\
        --fit-shards scratch/c2_ceiling/c2_early --test-shards scratch/c2_roll/eval/c2 \\
        --out-dir scratch/c2_ceiling/cand
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIT_FROM, FIT_TO, TEST_FROM, TEST_TO = "2026-07-17", "2026-08-28", "2026-09-15", "2026-10-23"
SUBFIT_DAYS, EMBARGO_DAYS, VALIDATION_DAYS = 15, 10, 6
ALPHAS = {"T2": (1, 10, 100, 1000, 10000), "T4": (1, 10, 100, 1000, 10000),
          "T3": (10, 100, 1000, 10000, 100000)}
STACK = ([f"path_median_d{h}" for h in range(1, 11)]
         + ["path_mean_d10", "path_share_up_d5", "path_share_up_d10", "path_std_d10"])
FACTORS = ["ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "vol_20", "amt_20", "amt_ratio",
           "size_pct", "range_1", "pos_60"]
KEYS = ["identity", "symbol", "asof_date", "target_date", "sector", "size_decile", "last_close"]


def load_shards(directory: str, date_from: str, date_to: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{directory}/*.csv.gz"))
    if not files:
        raise SystemExit(f"No shards in {directory}")
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame["asof_date"] = frame["asof_date"].astype(str)
    frame = frame[(frame.asof_date >= date_from) & (frame.asof_date <= date_to)].reset_index(drop=True)
    if frame.identity.duplicated().any() or frame.asof_date.max() > TEST_TO:
        raise SystemExit(f"Bad shard set in {directory}")
    paths = [f"path{j:02d}_d10" for j in range(16)]
    if set(paths) <= set(frame.columns):          # single-path shards carry labels only (enough for T3)
        frame["path_std_d10"] = frame[paths].std(axis=1, ddof=1)
    return frame


def rank_centered(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Within-date percentile rank minus 0.5; missing values sit at the middle."""
    ranked = frame.groupby("asof_date")[columns].rank(pct=True) - 0.5
    return ranked.fillna(0.0).to_numpy(dtype=np.float64)


def zscore_by_date(dates: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    out = np.empty_like(matrix, dtype=np.float32)
    for date in np.unique(dates):
        rows = dates == date
        block = matrix[rows].astype(np.float32)
        out[rows] = (block - block.mean(0)) / (block.std(0) + 1e-6)
    return out


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T.astype(np.float64) @ x.astype(np.float64)
    return np.linalg.solve(gram + alpha * np.eye(x.shape[1]), x.T.astype(np.float64) @ y)


def mean_daily_ic(dates: np.ndarray, score: np.ndarray, actual: np.ndarray) -> float:
    frame = pd.DataFrame({"d": dates, "s": score, "a": actual})
    return float(frame.groupby("d").apply(lambda g: g["s"].corr(g["a"], method="spearman"),
                                          include_groups=False).mean())


def split_fit_window(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    unique = np.array(sorted(set(dates)))
    if len(unique) < SUBFIT_DAYS + EMBARGO_DAYS + VALIDATION_DAYS:
        raise SystemExit(f"Fitting window has only {len(unique)} dates")
    sub, val = unique[:SUBFIT_DAYS], unique[-VALIDATION_DAYS:]
    info = {"fit_dates": len(unique), "sub_fit": [sub[0], sub[-1]], "validation": [val[0], val[-1]]}
    return np.isin(dates, sub), np.isin(dates, val), info


def build_stack(frame: pd.DataFrame, factors: pd.DataFrame | None) -> tuple[np.ndarray, list[str]]:
    columns = list(STACK)
    if factors is not None:
        frame = frame.merge(factors, on=["symbol", "asof_date"], how="left", validate="one_to_one")
        columns += FACTORS
    return rank_centered(frame, columns), columns


def load_hidden(pattern: str, frame: pd.DataFrame | None):
    """OOS files align to shard rows by identity; validation files bring their own labels."""
    parts = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(pattern))]
    if not parts:
        raise SystemExit(f"No hidden-state files match {pattern}")
    hidden = np.concatenate([p["hidden"] for p in parts])
    if frame is not None:
        position = pd.Series(np.arange(len(hidden)), index=np.concatenate([p["identity"] for p in parts]))
        if not position.index.is_unique or not frame.identity.isin(position.index).all():
            raise SystemExit("Hidden states do not cover the shard rows one to one")
        return hidden[position.loc[frame.identity].to_numpy()]
    dates = np.concatenate([p["asof_date"] for p in parts]).astype(str)
    returns = np.concatenate([p["ret10"] for p in parts])
    target = pd.Series(returns).groupby(dates).rank(pct=True).to_numpy() - 0.5
    return hidden, dates, target


def write_scores(frame: pd.DataFrame, score: np.ndarray, label: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = frame[KEYS + [f"actual_return_d{h}" for h in range(1, 11)]].copy()
    out["predicted_return_d10"] = score
    for date, rows in out.groupby("asof_date"):
        rows.to_csv(out_dir / f"{label}_{date}.csv.gz", index=False, compression="gzip")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=["T2", "T3", "T4"])
    parser.add_argument("--fit-shards", required=True)
    parser.add_argument("--test-shards", required=True)
    parser.add_argument("--factors", help="price_factors.csv.gz (T4)")
    parser.add_argument("--hidden-oos", help="glob of oos_hidden_*.npz (T3)")
    parser.add_argument("--hidden-val", help="glob of val_hidden_*.npz (T3)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fit-from", default=FIT_FROM, help="code tests only; the registered value is the default")
    parser.add_argument("--fit-to", default=FIT_TO, help="code tests only; the registered value is the default")
    parser.add_argument("--skip-test-window", action="store_true", help="selection only; do not read test rows")
    args = parser.parse_args()
    if args.fit_to >= TEST_FROM:
        raise SystemExit("The fitting window must end before the test window starts")

    fit = load_shards(args.fit_shards, args.fit_from, args.fit_to)
    # A label is 10 of the stock's OWN traded bars ahead, so a stock that was halted
    # can have a label ending inside the test window.  Those rows are not fitted on.
    late = fit.target_date.astype(str) >= TEST_FROM
    fit = fit[~late].reset_index(drop=True)
    fit_dates = fit.asof_date.to_numpy()
    y = rank_centered(fit, ["actual_return_d10"])[:, 0]
    sub, val, info = split_fit_window(fit_dates)
    report = {"test": args.test, "fit_window": [args.fit_from, args.fit_to], "registered_fit_window": [FIT_FROM, FIT_TO],
              "fit_rows": int(len(fit)), "fit_rows_dropped_label_in_test_window": int(late.sum()), "split": info,
              "baseline_validation_ic": mean_daily_ic(fit_dates[val], fit.predicted_return_d10.to_numpy()[val],
                                                      fit.actual_return_d10.to_numpy()[val])}
    actual_val = fit.actual_return_d10.to_numpy()[val]

    if args.test in {"T2", "T4"}:
        factors = pd.read_csv(args.factors, dtype={"symbol": str}) if args.test == "T4" else None
        if args.test == "T4" and factors.asof_date.max() > TEST_TO:
            raise SystemExit("Factor file reaches past the test window")
        x, columns = build_stack(fit, factors)
        variants = {"window": (x, y, fit_dates, sub)}
        report["features"] = columns
    else:
        x = zscore_by_date(fit_dates, load_hidden(args.hidden_oos, fit))
        vx, vdates, vy = load_hidden(args.hidden_val, None)
        vx = zscore_by_date(vdates, vx)
        variants = {"a_window": (x, y, fit_dates, sub), "b_val_symbols": (vx, vy, vdates, None),
                    "c_both": (np.concatenate([x, vx]), np.concatenate([y, vy]),
                               np.concatenate([fit_dates, vdates]), np.concatenate([sub, np.ones(len(vy), bool)]))}
        report["val_symbol_rows"] = int(len(vy))

    # Selection: fit on the sub-fit dates (validation-symbol rows are always usable:
    # they end 2026-07-02, before the fitting window starts), score the last 6 dates.
    grid = {}
    for name, (vx_all, vy_all, _dates, mask) in variants.items():
        rows = np.ones(len(vy_all), bool) if mask is None else mask
        for alpha in ALPHAS[args.test]:
            weights = ridge(vx_all[rows], vy_all[rows], alpha)
            grid[f"{name}|{alpha}"] = mean_daily_ic(fit_dates[val], x[val] @ weights, actual_val)
    best = max(grid, key=grid.get)
    name, alpha = best.split("|")
    report.update(selection_grid=grid, selected={"variant": name, "alpha": float(alpha), "validation_ic": grid[best]})

    if not args.skip_test_window:
        full_x, full_y, _dates, _mask = variants[name]       # refit on every row of the chosen variant
        weights = ridge(full_x, full_y, float(alpha))
        test = load_shards(args.test_shards, TEST_FROM, TEST_TO)
        if args.test in {"T2", "T4"}:
            test_x, _ = build_stack(test, factors)
        else:
            test_x = zscore_by_date(test.asof_date.to_numpy(), load_hidden(args.hidden_oos, test))
        write_scores(test, test_x @ weights, args.test.lower(), Path(args.out_dir) / args.test.lower())
        report.update(test_rows=int(len(test)), test_dates=int(test.asof_date.nunique()),
                      refit_rows=int(len(full_y)),
                      weights={c: float(w) for c, w in zip(report.get("features", []), weights)})
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f"{args.test.lower()}_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "weights"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
