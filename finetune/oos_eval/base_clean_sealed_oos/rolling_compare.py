"""Paired comparison of candidate models against one baseline on sealed-OOS shards.

Each `name=dir` points at a directory of `<label>_<date>.csv.gz` shards written
by evaluate_base_sealed.py.  Every candidate is inner-joined with the baseline
on `identity` (labels must match), so both models are scored on exactly the
same (symbol, as-of) windows.  Only the 10-day horizon is used
(`predicted_return_d10` vs `actual_return_d10`).

Per model: daily Rank IC, size/reversal/sector-neutral IC, MAE/RMSE against the
always-zero forecast, direction hit rate, top/bottom-decile excess return.
Paired deltas (candidate - baseline) carry a 95% CI from a moving-block
bootstrap over signal dates, both models sharing the resampled dates.

The verdict follows the pre-registered RULE below and nothing else.

    python3 finetune/oos_eval/base_clean_sealed_oos/rolling_compare.py \
        --base c2=scratch/oos65/c2 \
        --cand seg03=<shard_dir> --cand seg06=<shard_dir> \
        --raw scratch/oos65/combined_raw.csv \
        [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD] \
        --out finetune/reports/rolling_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_analysis as fa  # noqa: E402

RULE = ("A candidate passes only if the 95% CI lower bound of "
        "(candidate − baseline) daily Rank IC is > 0.")
SEED, BLOCK, REPS = 20261111, 5, 2000
ACTUAL = "actual_return_d10"
# metric -> (per-date numerator, per-date denominator); the metric is sum/sum.
RATIOS = {
    "daily_rank_ic": ("ic", "ic_n"), "neutral_ic": ("nic", "nic_n"),
    "mae": ("abs", "n"), "mae_zero": ("abs0", "n"),
    "direction": ("hit", "n"), "top10_excess": ("top", "top_n"),
    "bottom10_excess": ("bot", "bot_n"), "mean_pred": ("pred", "n"),
    "pred_xs_std": ("std", "one"),
}
PAIRED = ["daily_rank_ic", "neutral_ic", "mae", "top10_excess", "direction"]


def iso_date(text: str) -> str:
    return date.fromisoformat(text).isoformat()


def load(directory: Path, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    """All shards of one model (fa.load_model aborts on duplicate identities), date-filtered."""
    frame = fa.load_model(directory)
    keep = pd.Series(True, index=frame.index)
    if date_from:
        keep &= frame["asof_date"] >= date_from
    if date_to:
        keep &= frame["asof_date"] <= date_to
    return frame[keep].reset_index(drop=True)


def pair(base: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inner-join on identity (fa.align aborts on label mismatch) and describe coverage."""
    joined = fa.align({"base": base, "cand": cand})
    joined["ret10"] = joined["identity"].map(base.set_index("identity")["ret10"])
    complete = joined.dropna(subset=["p_base", "p_cand", ACTUAL]).reset_index(drop=True)
    coverage = dict(
        base_rows=int(len(base)), base_dates=int(base.asof_date.nunique()),
        cand_rows=int(len(cand)), cand_dates=int(cand.asof_date.nunique()),
        joined_rows=int(len(complete)), joined_dates=int(complete.asof_date.nunique()),
        dropped_nan_rows=int(len(joined) - len(complete)),
        first_date=complete.asof_date.min() if len(complete) else None,
        last_date=complete.asof_date.max() if len(complete) else None)
    return complete, coverage


def neutral_daily_ic(frame: pd.DataFrame, column: str) -> pd.Series:
    """Daily IC of the rank signal residualised on size, reversal and sector."""
    work = frame.assign(_r=fa.ranks(frame, column))
    resid = fa.residualize(work, "_r", ["r_size", "r_rev"], sector=True)
    return fa.daily_ic(work.assign(_res=resid), "_res")


def per_date(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-date sums from which every pooled metric (and bootstrap draw) is rebuilt."""
    pred, actual = frame[column], frame[ACTUAL]
    rank = fa.ranks(frame, column)
    excess = actual - frame.groupby("asof_date")[ACTUAL].transform("mean")
    rows = pd.DataFrame({
        "asof_date": frame["asof_date"], "n": 1.0, "pred": pred,
        "abs": (pred - actual).abs(), "sq": (pred - actual) ** 2,
        "abs0": actual.abs(), "sq0": actual ** 2,
        "hit": (np.sign(pred) == np.sign(actual)).astype(float),
        "up": (actual > 0).astype(float), "down": (actual < 0).astype(float),
        "top": excess * (rank > 0.9), "top_n": (rank > 0.9).astype(float),
        "bot": excess * (rank <= 0.1), "bot_n": (rank <= 0.1).astype(float)})
    table = rows.groupby("asof_date").sum()
    table["std"] = frame.groupby("asof_date")[column].std()
    table["one"] = 1.0
    for name, series in (("ic", fa.daily_ic(frame, column)),
                         ("nic", neutral_daily_ic(frame, column))):
        table[name] = series.reindex(table.index)
        table[f"{name}_n"] = table[name].notna().astype(float)
    return table


def summarize(table: pd.DataFrame) -> dict:
    out = {name: float(np.nansum(table[num]) / table[den].sum())
           for name, (num, den) in RATIOS.items()}
    n = table["n"].sum()
    ic, nic = fa.ic_stats(table["ic"]), fa.ic_stats(table["nic"])
    out.update(
        icir=ic["icir"], positive_ratio=ic["positive_ratio"],
        neutral_icir=nic["icir"], neutral_positive_ratio=nic["positive_ratio"],
        rmse=float(np.sqrt(table["sq"].sum() / n)),
        rmse_zero=float(np.sqrt(table["sq0"].sum() / n)),
        majority_direction=float(max(table["up"].sum(), table["down"].sum()) / n),
        rows=int(n), dates=int(len(table)))
    return out


def block_indices(n: int) -> np.ndarray:
    """REPS x n matrix of moving-block resampled date positions."""
    block = min(BLOCK, n)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n - block + 1, size=(REPS, -(-n // block)))
    return (starts[:, :, None] + np.arange(block)).reshape(REPS, -1)[:, :n]


def paired_deltas(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    idx = block_indices(len(base))

    def draws(table: pd.DataFrame, metric: str) -> tuple[float, np.ndarray]:
        num, den = (table[c].to_numpy() for c in RATIOS[metric])
        return np.nansum(num) / den.sum(), np.nansum(num[idx], axis=1) / den[idx].sum(axis=1)

    out = {}
    for metric in PAIRED:
        (b_point, b_draws), (c_point, c_draws) = draws(base, metric), draws(cand, metric)
        low, high = np.nanquantile(c_draws - b_draws, [0.025, 0.975])
        out[metric] = dict(delta=float(c_point - b_point), ci95=[float(low), float(high)])
    return out


def verdict(ci95: list[float]) -> str:
    if ci95[0] > 0:
        return "pass"
    return "fail_worse" if ci95[1] < 0 else "fail_inconclusive"


def compare(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    frame, coverage = pair(base, cand)
    if frame.empty:
        raise SystemExit("No joined rows between baseline and candidate")
    frame["r_size"] = frame.groupby("asof_date")["size_decile"].rank(pct=True, method="average")
    frame["r_rev"] = fa.ranks(frame, "ret10")
    tables = {side: per_date(frame, f"p_{side}") for side in ("base", "cand")}
    paired = paired_deltas(tables["base"], tables["cand"])
    rank_corr = frame.groupby("asof_date").apply(
        lambda day: day["p_cand"].corr(day["p_base"], method="spearman"), include_groups=False)
    return dict(
        coverage=coverage, coverage_complete=coverage["joined_rows"] == coverage["base_rows"],
        baseline=summarize(tables["base"]), candidate=summarize(tables["cand"]),
        paired=paired, prediction_rank_corr=float(rank_corr.mean()),
        verdict=verdict(paired["daily_rank_ic"]["ci95"]))


def print_table(report: dict) -> None:
    cols = ["daily_rank_ic", "neutral_ic", "mae", "mae_zero", "direction", "top10_excess"]
    base = report["base"]["name"]
    print(f"{'model':<28}{'rows':>8}{'dates':>6}" + "".join(f"{c:>15}" for c in cols))
    for name, result in report["candidates"].items():
        for label, key in ((f"{base} (rows of {name})", "baseline"), (name, "candidate")):
            m = result[key]
            print(f"{label:<28}{m['rows']:>8}{m['dates']:>6}" + "".join(f"{m[c]:>+15.4f}" for c in cols))
    print(f"\n{'candidate - ' + base:<28}{'dIC':>9}{'ci95':>22}{'rank_corr':>11}  verdict")
    for name, result in report["candidates"].items():
        d = result["paired"]["daily_rank_ic"]
        ci = f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
        print(f"{name:<28}{d['delta']:>+9.4f}{ci:>22}{result['prediction_rank_corr']:>11.4f}  {result['verdict']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="name=shard_dir of the baseline")
    parser.add_argument("--cand", action="append", required=True, help="name=shard_dir (repeatable)")
    parser.add_argument("--raw", required=True, help="combined_raw.csv of the same OOS package")
    parser.add_argument("--date-from", type=iso_date, help="inclusive lower bound on asof_date")
    parser.add_argument("--date-to", type=iso_date, help="inclusive upper bound on asof_date")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_name, base_dir = args.base.split("=", 1)
    cands = dict(item.split("=", 1) for item in args.cand)
    if len(cands) != len(args.cand) or base_name in cands:
        raise SystemExit("Model names must be unique across --base and --cand")
    base = load(Path(base_dir), args.date_from, args.date_to)
    if base.empty:
        raise SystemExit(f"Baseline {base_name} has no rows inside the date filter")
    base["ret10"] = fa.reversal_10d(Path(args.raw), base)

    report: dict = dict(
        rule=RULE, horizon="d10", base=dict(name=base_name, dir=base_dir),
        date_from=args.date_from, date_to=args.date_to, raw=args.raw,
        bootstrap=dict(kind="moving_block_over_dates", block=BLOCK, replicates=REPS, seed=SEED),
        candidates={})
    for name, directory in cands.items():
        result = compare(base, load(Path(directory), args.date_from, args.date_to))
        if not result["coverage_complete"]:
            c = result["coverage"]
            print(f"WARNING: {name} covers {c['joined_rows']}/{c['base_rows']} baseline rows "
                  f"({c['joined_dates']}/{c['base_dates']} dates); comparison uses the joined rows only",
                  file=sys.stderr)
        report["candidates"][name] = dict(dir=directory, **result)
    report["any_pass"] = any(r["verdict"] == "pass" for r in report["candidates"].values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
