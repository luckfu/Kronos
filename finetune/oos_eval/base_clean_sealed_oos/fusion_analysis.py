"""CPU-only rank-fusion / incremental-signal analysis over sealed-OOS shards.

Each --model points at a directory of shards written by evaluate_base_sealed.py
(one `<label>_<date>.csv.gz` per signal date; see fetch_kernel_shards.py).
Rows are aligned on `identity` and the labels must match exactly, so every
comparison is paired on the same (symbol, as-of) windows.

Metric is the evaluator's: mean over dates of the D10 Spearman IC.

The equal-weight fusion is the pre-registered result.  The weight grid is
selected on the OOS window itself and is printed as a diagnostic only.

    python3 finetune/oos_eval/base_clean_sealed_oos/fusion_analysis.py \
        --model base=scratch/oos65/pretrained_base \
        --model c2=scratch/oos65/c2 \
        --raw /path/to/combined_raw.csv \
        --out finetune/reports/20261111_oos65_fusion.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = ["identity", "symbol", "asof_date", "sector", "size_decile",
        "predicted_return_d10", "actual_return_d10"]


def load_model(directory: Path) -> pd.DataFrame:
    shards = sorted(directory.glob("*.csv.gz"))
    if not shards:
        raise SystemExit(f"No shards in {directory}")
    frame = pd.concat([pd.read_csv(p, usecols=KEEP) for p in shards], ignore_index=True)
    frame["asof_date"] = frame["asof_date"].astype(str)
    if frame.identity.duplicated().any():
        raise SystemExit(f"Duplicate identities in {directory}")
    return frame


def align(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(models)
    base = models[names[0]][["identity", "symbol", "asof_date", "sector",
                             "size_decile", "actual_return_d10"]]
    merged = base.copy()
    for name in names:
        part = models[name][["identity", "predicted_return_d10", "actual_return_d10"]].rename(
            columns={"predicted_return_d10": f"p_{name}", "actual_return_d10": f"a_{name}"})
        merged = merged.merge(part, on="identity", how="inner")
    for name in names:
        if not np.allclose(merged[f"a_{name}"], merged["actual_return_d10"], atol=1e-9, equal_nan=True):
            raise SystemExit(f"Labels differ between {names[0]} and {name}: not a paired set")
        merged = merged.drop(columns=f"a_{name}")
    return merged


def ranks(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("asof_date")[column].rank(pct=True, method="average")


def daily_ic(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("asof_date").apply(
        lambda day: day[column].corr(day["actual_return_d10"], method="spearman"),
        include_groups=False)


def ic_stats(series: pd.Series) -> dict:
    values = series.dropna().to_numpy()
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return dict(dates=int(len(values)), ic=float(values.mean()), ic_std=std,
                icir=float(values.mean() / std) if std else float("nan"),
                positive_ratio=float((values > 0).mean()))


def top_bottom(frame: pd.DataFrame, column: str, q: float = 0.1) -> float:
    def spread(day):
        lo, hi = day[column].quantile(q), day[column].quantile(1 - q)
        return day.loc[day[column] >= hi, "actual_return_d10"].mean() - \
            day.loc[day[column] <= lo, "actual_return_d10"].mean()
    return float(frame.groupby("asof_date").apply(spread, include_groups=False).mean())


def paired_bootstrap(left: pd.Series, right: pd.Series, seed=20261101, block=5, reps=2000) -> dict:
    dates = sorted(set(left.dropna().index) & set(right.dropna().index))
    d = (right.reindex(dates) - left.reindex(dates)).to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = []
        while len(idx) < len(dates):
            start = int(rng.integers(0, max(1, len(dates) - block + 1)))
            idx.extend(range(start, min(start + block, len(dates))))
        draws.append(float(np.nanmean(d[idx[:len(dates)]])))
    return dict(dates=len(dates), delta_ic=float(np.nanmean(d)),
                ci95=[float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                block=block, replicates=reps, seed=seed)


def reversal_10d(raw_path: Path, frame: pd.DataFrame) -> pd.Series:
    """Past 10-traded-row return at as-of, matching the evaluator's volume>0 rows."""
    raw = pd.read_csv(raw_path, usecols=["symbol", "date", "close", "volume"])
    raw["date"] = raw["date"].astype(str)
    raw = raw[raw["volume"] > 0].sort_values(["symbol", "date"])
    raw["ret10"] = raw.groupby("symbol")["close"].transform(lambda c: c / c.shift(10) - 1)
    keyed = raw.set_index(["symbol", "date"])["ret10"]
    keys = pd.MultiIndex.from_arrays([frame["symbol"].astype(str), frame["asof_date"]])
    keyed.index = pd.MultiIndex.from_arrays([keyed.index.get_level_values(0).astype(str),
                                             keyed.index.get_level_values(1)])
    return pd.Series(keyed.reindex(keys).to_numpy(), index=frame.index)


def residualize(frame: pd.DataFrame, target: str, controls: list[str], sector: bool) -> pd.Series:
    """Per-date OLS residual of a rank signal on rank controls (+ sector dummies)."""
    out = pd.Series(np.nan, index=frame.index)
    for _, day in frame.groupby("asof_date"):
        day = day.dropna(subset=[target, *controls])
        if len(day) < 50:
            continue
        cols = [np.ones(len(day))] + [day[c].to_numpy() for c in controls]
        X = np.column_stack(cols)
        if sector:
            dummies = pd.get_dummies(day["sector"], drop_first=True, dtype=float).to_numpy()
            X = np.column_stack([X, dummies])
        y = day[target].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[day.index] = y - X @ beta
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=shard_dir")
    parser.add_argument("--raw", help="combined_raw.csv of the same OOS package (enables reversal control)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    specs = dict(item.split("=", 1) for item in args.model)
    models = {name: load_model(Path(path)) for name, path in specs.items()}
    frame = align(models)
    names = list(models)
    report: dict = dict(
        models=specs,
        rows={name: int(len(m)) for name, m in models.items()},
        aligned_rows=int(len(frame)), aligned_dates=int(frame.asof_date.nunique()),
        first_date=frame.asof_date.min(), last_date=frame.asof_date.max(),
    )

    for name in names:
        frame[f"r_{name}"] = ranks(frame, f"p_{name}")
    ic = {name: daily_ic(frame, f"r_{name}") for name in names}
    report["single"] = {name: dict(**ic_stats(ic[name]), top_bottom_10pct=top_bottom(frame, f"r_{name}"))
                        for name in names}

    report["paired_single"] = {f"{b}-{a}": paired_bootstrap(ic[a], ic[b])
                               for a, b in combinations(names, 2)}

    report["prediction_corr"] = {}
    for a, b in combinations(names, 2):
        corr = frame.groupby("asof_date").apply(
            lambda day: day[f"r_{a}"].corr(day[f"r_{b}"]), include_groups=False)
        ic_corr = float(np.corrcoef(*(s.reindex(corr.index).to_numpy() for s in (ic[a], ic[b])))[0, 1])
        report["prediction_corr"][f"{a}~{b}"] = dict(
            mean_daily_rank_corr=float(corr.mean()), min=float(corr.min()), max=float(corr.max()),
            daily_ic_series_corr=ic_corr)

    frame["r_fused"] = frame[[f"r_{n}" for n in names]].mean(axis=1)
    fused_ic = daily_ic(frame, "r_fused")
    report["fusion_equal_weight"] = dict(
        **ic_stats(fused_ic), top_bottom_10pct=top_bottom(frame, "r_fused"),
        paired_vs={name: paired_bootstrap(ic[name], fused_ic) for name in names})

    if len(names) == 2:
        a, b = names
        grid = {}
        for w in np.round(np.arange(0, 1.01, 0.1), 1):
            frame["_w"] = w * frame[f"r_{a}"] + (1 - w) * frame[f"r_{b}"]
            grid[f"{a}={w:.1f}"] = float(daily_ic(frame, "_w").mean())
        frame = frame.drop(columns="_w")
        report["fusion_weight_grid_DIAGNOSTIC_ONLY"] = dict(
            note="weights selected on the OOS window; do not report as a result", ic=grid)

    controls = ["r_size"]
    frame["r_size"] = frame.groupby("asof_date")["size_decile"].rank(pct=True, method="average")
    if args.raw:
        frame["ret10"] = reversal_10d(Path(args.raw), frame)
        frame["r_rev"] = ranks(frame, "ret10")
        controls.append("r_rev")
        report["reversal_factor"] = dict(
            ic_of_minus_ret10=ic_stats(daily_ic(frame.assign(_neg=-frame["ret10"]), "_neg")),
            missing=int(frame["ret10"].isna().sum()))
    report["neutralized"] = {"controls": controls + ["sector_dummies"]}
    for name in [*names, "fused"]:
        resid = residualize(frame, f"r_{name}", controls, sector=True)
        report["neutralized"][name] = ic_stats(daily_ic(frame.assign(_res=resid), "_res"))

    report["incremental"] = {}
    for a, b in ((x, y) for x in names for y in names if x != y):
        resid = residualize(frame, f"r_{a}", [f"r_{b}"], sector=False)
        report["incremental"][f"{a}|{b}"] = ic_stats(daily_ic(frame.assign(_res=resid), "_res"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
