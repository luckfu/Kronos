"""Local OOS IC + executable k=10/20 backtest evaluator for Experiment G1 scores.

Strictly evaluates on 2026-07-17 .. 2026-10-23 (65d, 335,312 rows) aligned with
C2 (16-path T=0.65, p=0.80) and FUSE_RIDGE_C2, keeping 2026-10-24+ untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OOS_MOD = ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OOS_MOD))

import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

OOS_START = "2026-07-17"
OOS_END = "2026-10-23"


def evaluate_g1(g1_csv: Path, c2_dir: Path, raw_path: Path, out_json: Path) -> dict:
    c2 = fa.load_model(c2_dir)
    c2["asof_date"] = c2["asof_date"].astype(str)
    c2 = c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)
    assert c2["asof_date"].max() <= OOS_END, "Sealed window violation!"

    g1 = pd.read_csv(g1_csv)
    g1["asof_date"] = g1["asof_date"].astype(str)
    assert g1["asof_date"].max() <= OOS_END
    score_cols = [c for c in g1.columns if c.startswith("p_")]
    merged = c2.merge(g1[["identity"] + score_cols], on="identity", how="inner", validate="one_to_one")
    if len(merged) != len(c2):
        raise RuntimeError(f"Alignment mismatch: merged={len(merged)} vs c2={len(c2)}")

    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")
    cols = {"C2": "predicted_return_d10"}
    for sc in score_cols:
        short = sc[2:].upper()
        merged[short] = merged[sc]
        merged[f"r_{short}"] = fa.ranks(merged, sc)
        fuse_name = f"FUSE_{short}"
        merged[fuse_name] = 0.5 * merged["r_c2"] + 0.5 * merged[f"r_{short}"]
        cols[short] = sc
        cols[fuse_name] = fuse_name

    merged["rev10"] = fa.reversal_10d(raw_path, merged)
    raw_df = pd.read_csv(raw_path, dtype={"symbol": str})
    raw_df["date"] = raw_df["date"].astype(str)
    traded = raw_df[raw_df["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("symbol")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    liq_map = traded[["symbol", "date", "med_amt_20d"]].rename(columns={"date": "asof_date"})
    merged = merged.merge(liq_map, on=["symbol", "asof_date"], how="left")
    liq = merged[merged["med_amt_20d"] >= 5e7].copy()

    base_ic = fa.daily_ic(merged, "predicted_return_d10")
    merged["resid_actual"] = fa.residualize(merged, "actual_return_d10", ["size_decile", "rev10"], sector=True)

    cross_section = {}
    c2_corrs = {}
    for name, col in cols.items():
        ic_s = fa.daily_ic(merged, col)
        merged[f"res_{name}"] = fa.residualize(merged, col, ["size_decile", "rev10"], sector=True)
        nic_s = merged.groupby("asof_date", sort=False).apply(
            lambda g: g[f"res_{name}"].corr(g["resid_actual"], method="spearman"), include_groups=False
        )
        univ_daily = merged.groupby("asof_date", sort=False)["actual_return_d10"].mean()
        t10 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(10)
        t20 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(20)
        ex10 = float((t10.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean())
        ex20 = float((t20.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean())
        entry = {
            **fa.ic_stats(ic_s),
            "ic_seg_a": float(ic_s.loc[ic_s.index <= "2026-09-10"].mean()),
            "ic_seg_b": float(ic_s.loc[ic_s.index >= "2026-09-15"].mean()),
            "neutral_ic": float(nic_s.mean()),
            "neutral_icir": float(nic_s.mean() / nic_s.std(ddof=1)),
            "top10pct_excess": fa.top_bottom(merged, col, 0.1),
            "liq_top10_excess": ex10,
            "liq_top20_excess": ex20,
        }
        if name != "C2":
            entry["paired_vs_C2"] = fa.paired_bootstrap(base_ic, ic_s, seed=20261115)
            if not name.startswith("FUSE_"):
                c2_corrs[name] = float(
                    merged.groupby("asof_date").apply(lambda g: g["r_c2"].corr(g[f"r_{name}"]), include_groups=False).mean()
                )
        cross_section[name] = entry

    # Executable k=10, 20 backtests
    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_dates = sorted(merged["asof_date"].unique())
    signal_days = [market.date_index[d] for d in signal_dates]
    phases = [signal_days[p::10] for p in range(10)]
    score_mats = {
        name: market.score_matrix(merged[["symbol", "asof_date", col]].rename(columns={col: "predicted_return_d10"}))
        for name, col in cols.items()
    }
    bench = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases]
    dummy_gate = np.full(len(market.dates), 2, dtype=int)
    books = {}
    c2_daily_by_k = {}
    for k in (10, 20):
        for name in cols:
            runs = [
                nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats[name], k, dummy_gate, False), days, k, last_day)
                for days in phases
            ]
            nets = [r["daily"]["ret"] for r in runs]
            if name == "C2":
                c2_daily_by_k[k] = nets
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
            drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
            mean_net = float(np.mean([fnb.total(r) for r in nets]))
            mean_dd = float(np.mean(drawdowns))
            diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
            diff_c2 = pd.concat([r - c for r, c in zip(nets, c2_daily_by_k[k])], axis=1).mean(axis=1).to_numpy()
            books[f"{name}_k{k}"] = {
                "k": k, "model": name,
                "mean_net": mean_net,
                "mean_universe_net": float(np.mean([fnb.total(b) for b in bench])),
                "mean_net_excess": float(np.mean(excess)),
                "beat_pool_phases": int(sum(x > 0 for x in excess)),
                "phases": len(phases),
                "mean_max_drawdown": mean_dd,
                "calmar_like": float(mean_net / abs(mean_dd)) if abs(mean_dd) > 1e-9 else 0.0,
                "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                "vs_universe_ci95": sub.block_ci(diff_univ),
                "vs_c2_ci95": sub.block_ci(diff_c2),
            }

    result = {"c2_rank_corrs": c2_corrs, "cross_section": cross_section, "books": books}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-csv", required=True)
    parser.add_argument("--c2-dir", default="scratch/non_pv/c2_p16_65d")
    parser.add_argument("--raw", default="scratch/oos65/combined_raw.csv")
    parser.add_argument("--out", default="finetune/reports/20261115_gpu_g1_results.json")
    args = parser.parse_args()
    res = evaluate_g1(ROOT / args.g1_csv, ROOT / args.c2_dir, ROOT / args.raw, ROOT / args.out)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
