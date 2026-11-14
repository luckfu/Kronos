"""C2 decode grid: is there a detectable difference between (T, top_p) arms? (CPU only)

Selection segment first (all arms), then the confirmation segment (only the arm the
pre-registered ordering picks, plus the control).  Rules and ordering are frozen in
finetune/reports/20261114_c2_decode_grid_prereg_cn.md.

    python3 finetune/oos_eval/base_clean_sealed_oos/decode_grid_analysis.py \\
        --arm t065_p080=scratch/c2_decode/select/t065_p080 \\
        --arm t050_p080=scratch/c2_decode/select/t050_p080 \\
        --raw scratch/oos65/combined_raw.csv --segment select \\
        --out scratch/c2_decode/select_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = {"select": ("2026-07-17", "2026-09-17"), "confirm": ("2026-09-18", "2026-10-23")}
SEAL_AFTER = "2026-10-23"
BOOTSTRAP = {"block": 5, "replicates": 2000, "seed": 20261114}
FLAT_BAND = 0.02            # all arms' mean IC inside best ± FLAT_BAND ...
FLAT_PAIRWISE = 0.0         # ... and every pairwise 95% CI spans zero
PICK_IC_MARGIN = 0.010      # arms within this of the best IC are candidates


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


def load_shards(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / "*.csv.gz")))
    if not files:
        raise SystemExit(f"no shards in {directory}")
    frame = pd.concat([pd.read_csv(f, usecols=["identity", "symbol", "asof_date", "predicted_return_d10",
                                               "actual_return_d10"], dtype={"symbol": str}) for f in files],
                      ignore_index=True)
    frame["asof_date"] = frame["asof_date"].astype(str)
    if frame["asof_date"].max() > SEAL_AFTER:
        raise SystemExit(f"shards contain a sealed-window date: {frame['asof_date'].max()}")
    if frame.duplicated("identity").any():
        raise SystemExit(f"duplicate identities in {directory}")
    return frame.rename(columns={"asof_date": "date", "predicted_return_d10": "pred",
                                 "actual_return_d10": "actual"})


def arm_metrics(frame: pd.DataFrame) -> dict:
    pred, actual = frame["pred"].to_numpy(), frame["actual"].to_numpy()
    rank_ic = frame.groupby("date").apply(lambda g: g["pred"].corr(g["actual"], method="spearman"),
                                          include_groups=False)
    pearson = frame.groupby("date").apply(lambda g: g["pred"].corr(g["actual"]), include_groups=False)
    mae = float(np.abs(pred - actual).mean())
    zero = float(np.abs(actual).mean())
    pred_xs = float(frame.groupby("date")["pred"].std().mean())
    actual_xs = float(frame.groupby("date")["actual"].std().mean())
    return {"rows": int(len(frame)), "days": int(frame["date"].nunique()),
            "rank_ic": float(rank_ic.mean()), "rank_ic_ci95": block_ci(rank_ic.to_numpy()),
            "pearson_ic": float(pearson.mean()),
            "mae": mae, "mae_zero": zero, "mae_skill_vs_zero": float(1.0 - mae / zero),
            "calibration_ratio": float(pred_xs / actual_xs),
            "mean_pred": float(pred.mean()), "mean_actual": float(actual.mean()),
            "_daily_rank_ic": rank_ic, "_daily_pearson": pearson}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, help="name=<shard dir>")
    parser.add_argument("--segment", choices=sorted(SEGMENTS), default="select")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start, end = SEGMENTS[args.segment]
    frames, report = {}, {"segment": args.segment, "window": [start, end], "bootstrap": BOOTSTRAP,
                          "rules": {"flat_band": FLAT_BAND, "pick_ic_margin": PICK_IC_MARGIN}, "arms": {}}
    for item in args.arm:
        name, directory = item.split("=", 1)
        frame = load_shards(directory)
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        frames[name] = frame
        report["arms"][name] = arm_metrics(frame)

    common = None
    for frame in frames.values():
        keys = set(frame["identity"])
        common = keys if common is None else common & keys
    if not common:
        raise SystemExit("arms share no rows")
    names = list(frames)
    for name in names:
        frames[name] = frames[name][frames[name]["identity"].isin(common)].set_index("identity").sort_index()
    if len({len(frames[name]) for name in names}) != 1:
        raise SystemExit("arms do not cover the same rows after the intersection")
    labels = frames[names[0]]["actual"].to_numpy()
    for name in names:
        if not np.array_equal(frames[name]["actual"].to_numpy(), labels):
            raise SystemExit(f"labels differ for {name}")
    report["paired_rows"] = int(len(labels))

    pairwise = {}
    for left, right in combinations(names, 2):
        diff = (report["arms"][left]["_daily_rank_ic"] - report["arms"][right]["_daily_rank_ic"]).dropna().to_numpy()
        ci = block_ci(diff)
        pairwise[f"{left}-{right}"] = {"mean": float(diff.mean()), "ci95": ci,
                                       "verdict": "different" if ci and (ci[0] > 0 or ci[1] < 0) else "indistinguishable"}
    report["pairwise_rank_ic"] = pairwise

    ics = {name: report["arms"][name]["rank_ic"] for name in names}
    best = max(ics.values())
    spread = max(ics.values()) - min(ics.values())
    any_different = any(v["verdict"] == "different" for v in pairwise.values())
    report["verdict"] = {
        "best_arm": max(ics, key=ics.get), "best_rank_ic": best, "ic_spread": spread,
        "all_within_band": bool(spread <= 2 * FLAT_BAND),
        "any_pairwise_different": any_different,
        "flat": bool((not any_different) and spread <= 2 * FLAT_BAND),
    }
    if any_different:
        candidates = [n for n in names if ics[n] >= best - PICK_IC_MARGIN]
        ranked = sorted(candidates, key=lambda n: (-report["arms"][n]["mae_skill_vs_zero"],
                                                   abs(report["arms"][n]["calibration_ratio"] - 1.0),
                                                   -ics[n]))
        report["verdict"]["candidates_within_margin"] = candidates
        report["verdict"]["arm_to_confirm"] = ranked[0]
    for name in names:
        for key in ("_daily_rank_ic", "_daily_pearson"):
            report["arms"][name].pop(key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"segment {args.segment} {start}..{end}: {report['paired_rows']} paired rows, "
          f"{report['arms'][names[0]]['days']} days")
    print(f"{'arm':14s} {'Rank IC':>8s} {'IC CI':>18s} {'MAE':>7s} {'零预测':>7s} {'MAE skill':>9s} {'校准比':>7s} {'平均预测':>9s}")
    for name, m in sorted(report["arms"].items(), key=lambda kv: -kv[1]["rank_ic"]):
        ci = "n/a" if m["rank_ic_ci95"] is None else f"[{m['rank_ic_ci95'][0]:+.3f}, {m['rank_ic_ci95'][1]:+.3f}]"
        print(f"{name:14s} {m['rank_ic']:+8.4f} {ci:>18s} {m['mae']:7.2%} {m['mae_zero']:7.2%} "
              f"{m['mae_skill_vs_zero']:+9.1%} {m['calibration_ratio']:7.2f} {m['mean_pred']:+9.2%}")
    print(f"\nIC spread across arms: {spread:.4f}; any pairwise CI excluding zero: {any_different}")
    v = report["verdict"]
    print(f"verdict: {'FLAT (no detectable difference)' if v['flat'] else 'candidate to confirm: ' + v.get('arm_to_confirm', '?')}")
    for pair, p in pairwise.items():
        if p["verdict"] == "different":
            print(f"   {pair}: {p['mean']:+.4f} {p['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
