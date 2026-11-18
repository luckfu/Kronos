"""Pre-registered study: Broad-market & equal-weight regime gates for C2 (16-path p16).

Evaluates five pre-specified causal regime gates on C2 (T=0.65, top_p=0.80, 16-path median)
over the 65-day clean OOS (2026-07-17..2026-10-23) with thresholds calibrated strictly on
2026-02-02..2026-07-02 and zero exposure to 2026-10-24+.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

OOS_START = "2026-07-17"
OOS_END = "2026-10-23"


def build_broad_regime_table(raw_path: Path, non_pv_dir: Path) -> pd.DataFrame:
    """Compute strictly causal daily regime indicators and 5 pre-registered gate scores."""
    raw = pd.read_csv(raw_path, dtype={"symbol": str})
    raw["date"] = raw["date"].astype(str)
    traded = raw[raw["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["ret1"] = traded.groupby("symbol")["close"].pct_change()
    traded["ma20"] = traded.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    traded["above_ma20"] = (traded["close"] > traded["ma20"]).astype(float)

    breadth = traded.dropna(subset=["ma20"]).groupby("date")["above_ma20"].mean().rename("breadth_ma20")
    ew_ret = traded.groupby("date")["ret1"].mean().fillna(0.0)
    ew_idx = (1.0 + ew_ret).cumprod()
    ew_ma20 = ew_idx.rolling(20, min_periods=20).mean()
    ew_above = (ew_idx > ew_ma20).astype(int).rename("ew_above_ma20")

    table = pd.DataFrame({"breadth_ma20": breadth, "ew_above_ma20": ew_above}).dropna()
    for name, fname in [("csi300", "idx_sh000300.parquet"), ("csi1000", "idx_sh000852.parquet")]:
        idx = pd.read_parquet(non_pv_dir / fname).copy()
        idx["date"] = idx["date"].astype(str)
        idx = idx.sort_values("date")
        idx["ma20"] = idx["close"].rolling(20, min_periods=20).mean()
        idx[f"{name}_above_ma20"] = (idx["close"] > idx["ma20"]).astype(int)
        table = table.join(idx.set_index("date")[[f"{name}_above_ma20"]], how="inner")

    table["RAW"] = 2
    table["G_CSI300"] = table["csi300_above_ma20"] + (table["breadth_ma20"] >= 0.35).astype(int)
    table["G_CSI1000"] = table["csi1000_above_ma20"] + (table["breadth_ma20"] >= 0.35).astype(int)
    table["G_EW"] = table["ew_above_ma20"] + (table["breadth_ma20"] >= 0.35).astype(int)
    table["G_BREADTH"] = np.where(
        table["breadth_ma20"] < 0.30, 0, np.where(table["breadth_ma20"] < 0.45, 1, 2)
    )
    return table


def evaluate_gates(raw_path: Path, c2_dir: Path, regime: pd.DataFrame) -> dict:
    """Run few-name backtests (k=10, 20; step=10; 10 phases) for each pre-registered gate."""
    c2_df = fa.load_model(c2_dir)
    c2_df["asof_date"] = c2_df["asof_date"].astype(str)
    c2_df = c2_df[(c2_df["asof_date"] >= OOS_START) & (c2_df["asof_date"] <= OOS_END)].copy()
    assert c2_df["asof_date"].max() <= OOS_END, "Sealed window violation!"

    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_dates = sorted(c2_df["asof_date"].unique())
    signal_days = [market.date_index[d] for d in signal_dates]
    score_mat = market.score_matrix(c2_df[["symbol", "asof_date", "predicted_return_d10"]])

    step = 10
    phases = [signal_days[p::step] for p in range(step)]
    bench = [fnb.universe(market, score_mat, days[0], last_day) for days in phases]

    gates = ["RAW", "G_CSI300", "G_CSI1000", "G_EW", "G_BREADTH"]
    gate_arrays = {}
    gate_counts = {}
    for gname in gates:
        arr = np.full(len(market.dates), 2, dtype=int)
        for d, idx in market.date_index.items():
            if d in regime.index:
                arr[idx] = int(regime.loc[d, gname])
        gate_arrays[gname] = arr
        counts = regime.loc[signal_dates, gname].value_counts().sort_index().to_dict()
        gate_counts[gname] = {int(k): int(v) for k, v in counts.items()}

    books = {}
    raw_daily_by_k = {}
    for k in (10, 20):
        for gname in gates:
            use_gate = gname != "RAW"
            runs = [
                nrs.run_gated_slots(
                    market, nrs.gated_plan(market, score_mat, k, gate_arrays[gname], use_gate), days, k, last_day
                )
                for days in phases
            ]
            nets = [r["daily"]["ret"] for r in runs]
            if gname == "RAW":
                raw_daily_by_k[k] = nets
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
            drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
            mean_net = float(np.mean([fnb.total(r) for r in nets]))
            mean_dd = float(np.mean(drawdowns))
            diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
            diff_raw = pd.concat([r - c for r, c in zip(nets, raw_daily_by_k[k])], axis=1).mean(axis=1).to_numpy()
            books[f"{gname}_k{k}"] = {
                "k": k,
                "gate": gname,
                "mean_net": mean_net,
                "mean_universe_net": float(np.mean([fnb.total(b) for b in bench])),
                "mean_net_excess": float(np.mean(excess)),
                "net_excess_range": [float(min(excess)), float(max(excess))],
                "beat_pool_phases": int(sum(x > 0 for x in excess)),
                "phases": len(phases),
                "mean_max_drawdown": mean_dd,
                "calmar_like": float(mean_net / abs(mean_dd)) if abs(mean_dd) > 1e-9 else 0.0,
                "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                "vs_universe_ci95": sub.block_ci(diff_univ),
                "vs_raw_ci95": sub.block_ci(diff_raw),
                "vs_raw_mean_total_diff": float(np.mean([fnb.total(r) - fnb.total(c) for r, c in zip(nets, raw_daily_by_k[k])])),
            }
    return {"signal_dates": len(signal_dates), "gate_oos_day_counts": gate_counts, "books": books}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="scratch/oos65/combined_raw.csv")
    parser.add_argument("--nonpv-dir", default="scratch/non_pv")
    parser.add_argument("--c2-dir", default="scratch/non_pv/c2_p16_65d")
    parser.add_argument("--out", default="finetune/reports/20261115_broad_regime_gate_results.json")
    args = parser.parse_args()

    regime = build_broad_regime_table(ROOT / args.raw, ROOT / args.nonpv_dir)
    out = evaluate_gates(ROOT / args.raw, ROOT / args.c2_dir, regime)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
