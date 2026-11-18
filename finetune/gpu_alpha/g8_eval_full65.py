"""Full 65-day OOS evaluation of G8 (G2_FT_VARLEN 16-path AR) standalone and fused with G1/G4/G5."""
from __future__ import annotations

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


def evaluate_g8(
    g8_shards_dir: Path,
    c2_dir: Path,
    g1_csv: Path,
    g4_csv: Path,
    g5_csv: Path,
    raw_path: Path,
    out_json: Path,
) -> dict:
    c2 = fa.load_model(c2_dir)
    c2["asof_date"] = c2["asof_date"].astype(str)
    c2 = c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)

    g8 = fa.load_model(g8_shards_dir)
    g8["asof_date"] = g8["asof_date"].astype(str)
    g8 = g8[(g8["asof_date"] >= OOS_START) & (g8["asof_date"] <= OOS_END)].reset_index(drop=True)
    if len(g8) != len(c2):
        raise RuntimeError(f"G8 row count mismatch: g8={len(g8)} vs c2={len(c2)}")

    g1 = pd.read_csv(g1_csv)[["identity", "p_g1_univ_last"]]
    g4 = pd.read_csv(g4_csv)[["identity", "p_g4_base_ridge_m5"]]
    g5 = pd.read_csv(g5_csv)[["identity", "p_g5_lowalpha"]]

    merged = (
        c2.merge(g8[["identity", "predicted_return_d10"]].rename(columns={"predicted_return_d10": "p_g8"}), on="identity", validate="one_to_one")
        .merge(g1, on="identity", validate="one_to_one")
        .merge(g4, on="identity", validate="one_to_one")
        .merge(g5, on="identity", validate="one_to_one")
    )
    if len(merged) != 335312:
        raise RuntimeError(f"Expected 335312 rows, got {len(merged)}")

    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")
    merged["r_g8"] = fa.ranks(merged, "p_g8")
    merged["r_g1"] = fa.ranks(merged, "p_g1_univ_last")
    merged["r_g4"] = fa.ranks(merged, "p_g4_base_ridge_m5")
    merged["r_g5"] = fa.ranks(merged, "p_g5_lowalpha")

    models = {
        "C2": "r_c2",
        "G8_FT_AR16": "r_g8",
        "FUSE_C2_G8": None,
        "FUSE_G8_G1": None,
        "FUSE_G8_G5": None,
        "FUSE_C2_G1": None,
        "ENS_EQ3_C2_G1_G4": None,
        "ENS_EQ3_G8_G1_G4": None,
        "ENS_EQ3_G8_G5_G4": None,
        "ENS_EQ4_C2_G8_G1_G4": None,
        "ENS_EQ4_C2_G8_G5_G4": None,
    }
    merged["FUSE_C2_G8"] = 0.5 * merged["r_c2"] + 0.5 * merged["r_g8"]
    merged["FUSE_G8_G1"] = 0.5 * merged["r_g8"] + 0.5 * merged["r_g1"]
    merged["FUSE_G8_G5"] = 0.5 * merged["r_g8"] + 0.5 * merged["r_g5"]
    merged["FUSE_C2_G1"] = 0.5 * merged["r_c2"] + 0.5 * merged["r_g1"]
    merged["ENS_EQ3_C2_G1_G4"] = (merged["r_c2"] + merged["r_g1"] + merged["r_g4"]) / 3.0
    merged["ENS_EQ3_G8_G1_G4"] = (merged["r_g8"] + merged["r_g1"] + merged["r_g4"]) / 3.0
    merged["ENS_EQ3_G8_G5_G4"] = (merged["r_g8"] + merged["r_g5"] + merged["r_g4"]) / 3.0
    merged["ENS_EQ4_C2_G8_G1_G4"] = 0.25 * (merged["r_c2"] + merged["r_g8"] + merged["r_g1"] + merged["r_g4"])
    merged["ENS_EQ4_C2_G8_G5_G4"] = 0.25 * (merged["r_c2"] + merged["r_g8"] + merged["r_g5"] + merged["r_g4"])
    for name in list(models):
        if models[name] is None:
            models[name] = name

    merged["rev10"] = fa.reversal_10d(raw_path, merged)
    merged["resid_actual"] = fa.residualize(merged, "actual_return_d10", ["size_decile", "rev10"], sector=True)

    raw_df = pd.read_csv(raw_path, dtype={"symbol": str})
    raw_df["date"] = raw_df["date"].astype(str)
    traded = raw_df[raw_df["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("symbol")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    merged = merged.merge(traded[["symbol", "date", "med_amt_20d"]].rename(columns={"date": "asof_date"}), on=["symbol", "asof_date"], how="left")
    liq = merged[merged["med_amt_20d"] >= 5e7].copy()

    base_ic = fa.daily_ic(merged, "r_c2")
    univ_daily = merged.groupby("asof_date", sort=False)["actual_return_d10"].mean()
    c2_g8_corr = float(merged.groupby("asof_date").apply(lambda g: g["r_c2"].corr(g["r_g8"]), include_groups=False).mean())

    cross_section = {}
    for name, col in models.items():
        ic_s = fa.daily_ic(merged, col)
        merged[f"res_{name}"] = fa.residualize(merged, col, ["size_decile", "rev10"], sector=True)
        nic_s = merged.groupby("asof_date", sort=False).apply(
            lambda g: g[f"res_{name}"].corr(g["resid_actual"], method="spearman"), include_groups=False
        )
        t10 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(10)
        t20 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(20)
        entry = {
            **fa.ic_stats(ic_s),
            "ic_seg_a": float(ic_s.loc[ic_s.index <= "2026-09-10"].mean()),
            "ic_seg_b": float(ic_s.loc[ic_s.index >= "2026-09-15"].mean()),
            "neutral_ic": float(nic_s.mean()),
            "neutral_icir": float(nic_s.mean() / nic_s.std(ddof=1)),
            "top10pct_excess": fa.top_bottom(merged, col, 0.1),
            "liq_top10_excess": float((t10.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean()),
            "liq_top20_excess": float((t20.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean()),
        }
        if name != "C2":
            entry["paired_vs_C2"] = fa.paired_bootstrap(base_ic, ic_s, seed=20261117)
        cross_section[name] = entry

    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_days = [market.date_index[d] for d in sorted(merged["asof_date"].unique())]
    phases = [signal_days[p::10] for p in range(10)]
    score_mats = {
        name: market.score_matrix(merged[["symbol", "asof_date", col]].rename(columns={col: "predicted_return_d10"}))
        for name, col in models.items()
    }
    bench = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases]
    dummy_gate = np.full(len(market.dates), 2, dtype=int)
    books, c2_daily = {}, {}
    for k in (10, 20):
        for name in models:
            runs = [nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats[name], k, dummy_gate, False), days, k, last_day) for days in phases]
            nets = [r["daily"]["ret"] for r in runs]
            if name == "C2":
                c2_daily[k] = nets
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
            dds = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
            mn = float(np.mean([fnb.total(r) for r in nets]))
            mdd = float(np.mean(dds))
            books[f"{name}_k{k}"] = {
                "k": k, "model": name,
                "mean_net": mn, "mean_net_excess": float(np.mean(excess)),
                "beat_pool_phases": int(sum(x > 0 for x in excess)), "phases": len(phases),
                "mean_max_drawdown": mdd, "calmar_like": float(mn / abs(mdd)) if abs(mdd) > 1e-9 else 0.0,
                "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                "vs_universe_ci95": sub.block_ci(pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()),
                "vs_c2_ci95": sub.block_ci(pd.concat([r - c for r, c in zip(nets, c2_daily[k])], axis=1).mean(axis=1).to_numpy()),
            }

    out = {"c2_g8_rank_corr": c2_g8_corr, "cross_section": cross_section, "books": books}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> int:
    res = evaluate_g8(
        g8_shards_dir=ROOT / "scratch/gpu_alpha_g8_out/base_clean_sealed_oos/shards",
        c2_dir=ROOT / "scratch/non_pv/c2_p16_65d",
        g1_csv=ROOT / "scratch/gpu_alpha_g1_out/g1_out/oos_g1_scores.csv.gz",
        g4_csv=ROOT / "scratch/gpu_alpha_g4_out/g4_out/oos_g4_scores.csv.gz",
        g5_csv=ROOT / "scratch/gpu_alpha_g5_out/g5_out/oos_g5_scores.csv.gz",
        raw_path=ROOT / "scratch/oos65/combined_raw.csv",
        out_json=ROOT / "finetune/reports/20261117_gpu_g8_results.json",
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
