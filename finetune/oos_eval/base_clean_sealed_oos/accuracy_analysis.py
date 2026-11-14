"""Forecast-accuracy view of sealed-OOS shards: single path vs multi-path (no GPU).

The Rank IC reports ask "is the cross-sectional order right".  This one asks
"is the forecast close to what happened": D10 return error against the
predict-nothing baseline, and direction accuracy against always guessing the
majority direction.  Rank IC is reported alongside as a secondary number.

    python3 finetune/oos_eval/base_clean_sealed_oos/accuracy_analysis.py \
        --single scratch/oos65/c2 --multi scratch/oos65/c2_p16_s3 \
        --out finetune/reports/20261111_c2_p16_accuracy.json

Rows are paired on `identity`; labels must match bit for bit.  Confidence
intervals are a 5-day block bootstrap over signal dates (2000 draws).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED, BLOCK, DRAWS = 20261111, 5, 2000


def load(directory):
    files = sorted(glob.glob(str(Path(directory) / "*.csv.gz")))
    if not files:
        raise SystemExit(f"No shards in {directory}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True), len(files)


def block_ci(daily, seed=SEED):
    """Mean of a per-date series with a moving-block bootstrap 95% CI."""
    values = np.asarray(daily, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    block = min(BLOCK, n)
    reps = np.empty(DRAWS)
    for r in range(DRAWS):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        reps[r] = values[idx[:n]].mean()
    return dict(mean=float(values.mean()), ci95=[float(np.quantile(reps, .025)), float(np.quantile(reps, .975))])


def per_date(frame, column):
    """Per-date sums, so pooled metrics can be rebuilt inside the bootstrap."""
    p = frame[column].to_numpy()
    a = frame["actual"].to_numpy()
    work = pd.DataFrame(dict(
        date=frame["asof_date"].to_numpy(), n=1.0,
        abs_err=np.abs(p - a), sq_err=(p - a) ** 2, abs_zero=np.abs(a), sq_zero=a ** 2,
        hit=(np.sign(p) == np.sign(a)).astype(float), up=(a > 0).astype(float), down=(a < 0).astype(float),
    ))
    daily = work.groupby("date").sum()
    daily["ic"] = frame.groupby("asof_date").apply(
        lambda day: day[column].corr(day["actual"], method="spearman"), include_groups=False)
    centred_p = p - frame.groupby("asof_date")[column].transform("mean").to_numpy()
    centred_a = a - frame.groupby("asof_date")["actual"].transform("mean").to_numpy()
    daily["excess_hit"] = pd.Series((np.sign(centred_p) == np.sign(centred_a)).astype(float)).groupby(
        frame["asof_date"].to_numpy()).sum()
    return daily


def pooled(daily):
    n = daily.n.sum()
    majority = max(daily.up.sum(), daily.down.sum()) / n
    return dict(
        mae=float(daily.abs_err.sum() / n), rmse=float(np.sqrt(daily.sq_err.sum() / n)),
        mae_zero=float(daily.abs_zero.sum() / n), rmse_zero=float(np.sqrt(daily.sq_zero.sum() / n)),
        mae_gain_vs_zero=float(1 - daily.abs_err.sum() / daily.abs_zero.sum()),
        mse_gain_vs_zero=float(1 - daily.sq_err.sum() / daily.sq_zero.sum()),
        direction=float(daily.hit.sum() / n), majority_direction=float(majority),
        direction_gain_vs_majority=float(daily.hit.sum() / n - majority),
        excess_direction=float(daily.excess_hit.sum() / n),
        daily_rank_ic=float(daily.ic.mean()), icir=float(daily.ic.mean() / daily.ic.std(ddof=1)),
    )


def bootstrap(daily_by_name, seed=SEED):
    """Block bootstrap of every pooled metric, and of each pair's difference."""
    names = list(daily_by_name)
    dates = daily_by_name[names[0]].index
    n = len(dates)
    block = min(BLOCK, n)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in names}
    for _ in range(DRAWS):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        idx = idx[:n]
        for name in names:
            draws[name].append(pooled(daily_by_name[name].iloc[idx]))
    keys = ("mae", "rmse", "mae_gain_vs_zero", "mse_gain_vs_zero", "direction",
            "direction_gain_vs_majority", "excess_direction", "daily_rank_ic")
    out = {"ci95": {}, "paired": {}}
    for name in names:
        out["ci95"][name] = {k: [float(np.quantile([d[k] for d in draws[name]], q)) for q in (.025, .975)] for k in keys}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            out["paired"][f"{right}-{left}"] = {
                k: dict(delta=float(pooled(daily_by_name[right])[k] - pooled(daily_by_name[left])[k]),
                        ci95=[float(np.quantile([b[k] - a[k] for b, a in zip(draws[right], draws[left])], q))
                              for q in (.025, .975)])
                for k in keys}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", required=True, help="single-path shard dir")
    parser.add_argument("--multi", required=True, help="multi-path shard dir (BASE_OOS_PATHS>1)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-subset", action="store_true",
                        help="timing probes only: --multi covers part of each date")
    args = parser.parse_args()

    single, n_single = load(args.single)
    multi, n_multi = load(args.multi)
    path_cols = sorted(c for c in multi.columns if c.startswith("path") and c.endswith("_d10") and c[4:6].isdigit())
    if not path_cols:
        raise SystemExit("--multi shards carry no per-path columns")
    single = single[single.asof_date.isin(set(multi.asof_date))]
    if args.allow_subset:
        single = single[single.identity.isin(set(multi.identity))]
    merged = multi.merge(single[["identity", "predicted_return_d10", "actual_return_d10"]],
                         on="identity", suffixes=("", "_single"), validate="one_to_one")
    if len(merged) != len(multi) or len(merged) != len(single):
        raise SystemExit(f"Unpaired rows: multi={len(multi)} single={len(single)} merged={len(merged)}")
    if not np.array_equal(merged.actual_return_d10.to_numpy(), merged.actual_return_d10_single.to_numpy()):
        raise SystemExit("Paired labels differ")
    paths = merged[path_cols].to_numpy()
    if not np.allclose(np.median(paths, 1), merged.path_median_d10) or not np.allclose(paths.mean(1), merged.path_mean_d10):
        raise SystemExit("Per-path columns do not reproduce the stored aggregates")

    frame = pd.DataFrame(dict(
        asof_date=merged.asof_date, actual=merged.actual_return_d10,
        single=merged.predicted_return_d10_single,      # historical single-path run
        median=merged.path_median_d10,                  # production aggregate
        mean=merged.path_mean_d10,
        vote=merged.path_share_up_d10 - 0.5,            # sign = majority of paths
        path00=paths[:, 0],                             # one path of the new run
    ))
    names = ["single", "median", "mean"]
    daily = {name: per_date(frame, name) for name in names + ["vote", "path00"]}
    result = dict(
        inputs=dict(single=args.single, multi=args.multi, single_files=n_single, multi_files=n_multi,
                    partial_dates=bool(args.allow_subset)),
        rows=int(len(frame)), dates=int(frame.asof_date.nunique()),
        first=str(frame.asof_date.min()), last=str(frame.asof_date.max()), n_paths=len(path_cols),
        horizon="d10", bootstrap=dict(block=BLOCK, draws=DRAWS, seed=SEED),
        d10={name: pooled(daily[name]) for name in names + ["path00"]},
        direction_by_path_vote=dict(direction=pooled(daily["vote"])["direction"],
                                    majority_direction=pooled(daily["vote"])["majority_direction"]),
    )
    result.update(bootstrap({name: daily[name] for name in names}))

    # How many paths are enough: aggregate the first k of the 16 stored paths.
    curve = {}
    for k in (1, 2, 4, 8, len(path_cols)):
        for agg, fn in (("median", np.median), ("mean", np.mean)):
            frame["_k"] = fn(paths[:, :k], axis=1)
            curve[f"{agg}_{k}"] = {m: v for m, v in pooled(per_date(frame, "_k")).items()
                                   if m in ("mae", "rmse", "direction", "daily_rank_ic")}
    result["first_k_paths"] = curve

    # Every horizon, for the three forecasts that exist at every horizon.
    horizons = {}
    for h in range(1, 11):
        a = merged[f"actual_return_d{h}"].to_numpy()
        row = dict(mae_zero=float(np.abs(a).mean()), rmse_zero=float(np.sqrt((a ** 2).mean())),
                   majority_direction=float(max((a > 0).mean(), (a < 0).mean())))
        single_h = single.set_index("identity").loc[merged.identity, f"predicted_return_d{h}"].to_numpy()
        for name, p in (("single", single_h), ("median", merged[f"path_median_d{h}"].to_numpy()),
                        ("mean", merged[f"path_mean_d{h}"].to_numpy())):
            row[name] = dict(mae=float(np.abs(p - a).mean()), rmse=float(np.sqrt(((p - a) ** 2).mean())),
                             direction=float((np.sign(p) == np.sign(a)).mean()))
        horizons[str(h)] = row
    result["horizons"] = horizons

    # Calibration of the path spread and of the forecast scale (diagnostic).
    a = frame.actual.to_numpy()
    p = frame["mean"].to_numpy()
    lo, hi = np.quantile(paths, .1, axis=1), np.quantile(paths, .9, axis=1)
    result["diagnostic_post_hoc"] = dict(
        note="computed on this window; not a pre-registered result",
        mean_path_std=float(paths.std(1, ddof=1).mean()), actual_std=float(a.std()),
        coverage_of_10_90_path_band=float(((a >= lo) & (a <= hi)).mean()),
        best_scale_for_path_mean=float((p * a).sum() / (p * p).sum()),
        share_predicted_up=float((p > 0).mean()), share_actual_up=float((a > 0).mean()),
        mean_predicted=float(p.mean()), mean_actual=float(a.mean()),
    )
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    show = {name: {k: round(v, 5) for k, v in result["d10"][name].items()} for name in result["d10"]}
    print(json.dumps(dict(rows=result["rows"], dates=result["dates"], d10=show), indent=1))


if __name__ == "__main__":
    main()
