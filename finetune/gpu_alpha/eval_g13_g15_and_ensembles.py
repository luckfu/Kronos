"""Evaluate G13/G15 (paired 33d AR16) and G1-G18 cross-experiment 65d ensembles."""
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


def _load_oos_c2() -> pd.DataFrame:
    c2 = fa.load_model(ROOT / "scratch/non_pv/c2_p16_65d")
    c2["asof_date"] = c2["asof_date"].astype(str)
    return c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)


def _attach_liquidity(merged: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path, dtype={"symbol": str})
    raw_df["date"] = raw_df["date"].astype(str)
    traded = raw_df[raw_df["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("symbol")["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).median()
    )
    return merged.merge(
        traded[["symbol", "date", "med_amt_20d"]].rename(columns={"date": "asof_date"}),
        on=["symbol", "asof_date"],
        how="left",
    )


def _eval_cross_section(
    merged: pd.DataFrame,
    liq: pd.DataFrame,
    univ_daily: pd.Series,
    base_ic: pd.Series,
    models: dict[str, str],
    extra_ref_ic: pd.Series | None = None,
) -> dict:
    cross_section = {}
    for name, col in models.items():
        ic_s = fa.daily_ic(merged, col)
        merged[f"_res_{name}"] = fa.residualize(merged, col, ["size_decile", "rev10"], sector=True)
        nic_s = merged.groupby("asof_date", sort=False).apply(
            lambda g: g[f"_res_{name}"].corr(g["resid_actual"], method="spearman"),
            include_groups=False,
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
        if extra_ref_ic is not None and name != "G8_FT_AR16":
            entry["paired_vs_G8"] = fa.paired_bootstrap(extra_ref_ic, ic_s, seed=20261118)
        cross_section[name] = entry
    return cross_section


def evaluate_g13_g15(out_json: Path) -> dict:
    raw_path = ROOT / "scratch/oos65/combined_raw.csv"
    c2 = _load_oos_c2()
    g8 = fa.load_model(ROOT / "scratch/gpu_alpha_g8_out/base_clean_sealed_oos/shards")
    g13 = fa.load_model(ROOT / "scratch/gpu_alpha_g13_out/base_clean_sealed_oos/shards")
    g15 = fa.load_model(ROOT / "scratch/gpu_alpha_g15_out/base_clean_sealed_oos/shards")
    g1 = pd.read_csv(ROOT / "scratch/gpu_alpha_g1_out/g1_out/oos_g1_scores.csv.gz")[["identity", "p_g1_univ_last"]]
    g9 = pd.read_csv(ROOT / "scratch/gpu_alpha_g9_out/g9_out/oos_g9_scores.csv.gz")[["identity", "p_g9_base_l60_832"]]
    g14 = pd.read_csv(ROOT / "scratch/gpu_alpha_g14_out/g14_out/oos_g14_scores.csv.gz")[["identity", "p_g14_base_l45_832"]]
    g16 = pd.read_csv(ROOT / "scratch/gpu_alpha_g16_out/g16_out/oos_g16_scores.csv.gz")[["identity", "p_g16_pt_small_l45"]]

    merged = (
        g13[["identity", "predicted_return_d10"]].rename(columns={"predicted_return_d10": "p_g13"})
        .merge(g15[["identity", "predicted_return_d10"]].rename(columns={"predicted_return_d10": "p_g15"}), on="identity", validate="one_to_one")
        .merge(c2, on="identity", validate="one_to_one")
        .merge(g8[["identity", "predicted_return_d10"]].rename(columns={"predicted_return_d10": "p_g8"}), on="identity", validate="one_to_one")
        .merge(g1, on="identity", validate="one_to_one")
        .merge(g9, on="identity", validate="one_to_one")
        .merge(g14, on="identity", validate="one_to_one")
        .merge(g16, on="identity", validate="one_to_one")
    )
    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")
    merged["r_g8"] = fa.ranks(merged, "p_g8")
    merged["r_g13"] = fa.ranks(merged, "p_g13")
    merged["r_g15"] = fa.ranks(merged, "p_g15")
    merged["r_g1"] = fa.ranks(merged, "p_g1_univ_last")
    merged["r_g9"] = fa.ranks(merged, "p_g9_base_l60_832")
    merged["r_g14"] = fa.ranks(merged, "p_g14_base_l45_832")
    merged["r_g16"] = fa.ranks(merged, "p_g16_pt_small_l45")

    merged["FUSE_C2_G1"] = 0.5 * (merged["r_c2"] + merged["r_g1"])
    merged["FUSE_G8_G1"] = 0.5 * (merged["r_g8"] + merged["r_g1"])
    merged["FUSE_G13_G1"] = 0.5 * (merged["r_g13"] + merged["r_g1"])
    merged["FUSE_C2_G13"] = 0.5 * (merged["r_c2"] + merged["r_g13"])
    merged["FUSE_C2_G15"] = 0.5 * (merged["r_c2"] + merged["r_g15"])
    merged["FUSE_G15_G1"] = 0.5 * (merged["r_g15"] + merged["r_g1"])
    merged["FUSE_G15_G9L60"] = 0.5 * (merged["r_g15"] + merged["r_g9"])
    merged["FUSE_G15_G16L45"] = 0.5 * (merged["r_g15"] + merged["r_g16"])
    merged["ENS_C2_G13_G14L45_G16L45"] = 0.25 * (merged["r_c2"] + merged["r_g13"] + merged["r_g14"] + merged["r_g16"])
    merged["ENS_C2_G15_G14L45_G16L45"] = 0.25 * (merged["r_c2"] + merged["r_g15"] + merged["r_g14"] + merged["r_g16"])

    models = {
        "C2": "r_c2",
        "G8_FT_AR16": "r_g8",
        "G13_FT_L120_AR16": "r_g13",
        "G15_BASE_L60_AR16": "r_g15",
        "G1_UNIV_LAST": "r_g1",
        "G9_BASE_L60": "r_g9",
        "G14_BASE_L45": "r_g14",
        "G16_PT_SMALL_L45": "r_g16",
        "FUSE_C2_G1": "FUSE_C2_G1",
        "FUSE_G8_G1": "FUSE_G8_G1",
        "FUSE_G13_G1": "FUSE_G13_G1",
        "FUSE_C2_G13": "FUSE_C2_G13",
        "FUSE_C2_G15": "FUSE_C2_G15",
        "FUSE_G15_G1": "FUSE_G15_G1",
        "FUSE_G15_G9L60": "FUSE_G15_G9L60",
        "FUSE_G15_G16L45": "FUSE_G15_G16L45",
        "ENS_C2_G13_G14L45_G16L45": "ENS_C2_G13_G14L45_G16L45",
        "ENS_C2_G15_G14L45_G16L45": "ENS_C2_G15_G14L45_G16L45",
    }

    merged["rev10"] = fa.reversal_10d(raw_path, merged)
    merged["resid_actual"] = fa.residualize(merged, "actual_return_d10", ["size_decile", "rev10"], sector=True)
    merged = _attach_liquidity(merged, raw_path)
    liq = merged[merged["med_amt_20d"] >= 5e7].copy()
    univ_daily = merged.groupby("asof_date", sort=False)["actual_return_d10"].mean()
    base_ic = fa.daily_ic(merged, "r_c2")
    g8_ic = fa.daily_ic(merged, "r_g8")

    rank_corrs = {}
    for left in ("r_g13", "r_g15"):
        rank_corrs[left] = {
            right: float(merged.groupby("asof_date").apply(lambda g: g[left].corr(g[right]), include_groups=False).mean())
            for right in ("r_c2", "r_g8", "r_g1", "r_g9", "r_g14", "r_g16")
        }

    out = {
        "n_rows": int(len(merged)),
        "n_dates": int(merged["asof_date"].nunique()),
        "dates": sorted(merged["asof_date"].unique().tolist()),
        "rank_corrs": rank_corrs,
        "cross_section": _eval_cross_section(merged, liq, univ_daily, base_ic, models, extra_ref_ic=g8_ic),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def evaluate_g1_g18_ensembles(out_json: Path) -> dict:
    raw_path = ROOT / "scratch/oos65/combined_raw.csv"
    c2 = _load_oos_c2()
    g8 = fa.load_model(ROOT / "scratch/gpu_alpha_g8_out/base_clean_sealed_oos/shards")
    g1 = pd.read_csv(ROOT / "scratch/gpu_alpha_g1_out/g1_out/oos_g1_scores.csv.gz")[["identity", "p_g1_univ_last"]]
    g12 = pd.read_csv(ROOT / "scratch/gpu_alpha_g12_out/g12_out/oos_g12_scores.csv.gz")[["identity", "p_g12_base_l30_832"]]
    g14 = pd.read_csv(ROOT / "scratch/gpu_alpha_g14_out/g14_out/oos_g14_scores.csv.gz")[["identity", "p_g14_base_l45_832"]]
    g16 = pd.read_csv(ROOT / "scratch/gpu_alpha_g16_out/g16_out/oos_g16_scores.csv.gz")[["identity", "p_g16_pt_small_l45"]]
    g17 = pd.read_csv(ROOT / "scratch/gpu_alpha_g17_out/g17_out/oos_g17_scores.csv.gz")[["identity", "p_g17_base_l60_full", "p_g17_aniso_l60_orth"]]

    merged = (
        c2
        .merge(g8[["identity", "predicted_return_d10"]].rename(columns={"predicted_return_d10": "p_g8"}), on="identity", validate="one_to_one")
        .merge(g1, on="identity", validate="one_to_one")
        .merge(g12, on="identity", validate="one_to_one")
        .merge(g14, on="identity", validate="one_to_one")
        .merge(g16, on="identity", validate="one_to_one")
        .merge(g17, on="identity", validate="one_to_one")
    )
    if len(merged) != 335312:
        raise RuntimeError(f"Expected 335312 rows, got {len(merged)}")

    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")
    merged["r_g8"] = fa.ranks(merged, "p_g8")
    merged["r_g1"] = fa.ranks(merged, "p_g1_univ_last")
    merged["r_g12_l30"] = fa.ranks(merged, "p_g12_base_l30_832")
    merged["r_g14_l45"] = fa.ranks(merged, "p_g14_base_l45_832")
    merged["r_g16_l45"] = fa.ranks(merged, "p_g16_pt_small_l45")
    merged["r_g17_l60"] = fa.ranks(merged, "p_g17_base_l60_full")
    merged["r_g17_aniso"] = fa.ranks(merged, "p_g17_aniso_l60_orth")

    merged["ENS_ZS2_G14L45_G16L45"] = 0.5 * (merged["r_g14_l45"] + merged["r_g16_l45"])
    merged["ENS_ZS3_L45_L30"] = (merged["r_g14_l45"] + merged["r_g16_l45"] + merged["r_g12_l30"]) / 3.0
    merged["ENS_C2_G14L45_G16L45"] = (merged["r_c2"] + merged["r_g14_l45"] + merged["r_g16_l45"]) / 3.0
    merged["ENS_C2_G1_G14L45"] = (merged["r_c2"] + merged["r_g1"] + merged["r_g14_l45"]) / 3.0
    merged["ENS_C2_G1_G16L45"] = (merged["r_c2"] + merged["r_g1"] + merged["r_g16_l45"]) / 3.0
    merged["ENS_C2_G1_G14L45_G16L45"] = 0.25 * (merged["r_c2"] + merged["r_g1"] + merged["r_g14_l45"] + merged["r_g16_l45"])
    merged["ENS_C2_G8_G1_G14L45_G16L45"] = 0.2 * (merged["r_c2"] + merged["r_g8"] + merged["r_g1"] + merged["r_g14_l45"] + merged["r_g16_l45"])
    merged["ENS_C2_G8_G1_G14L45_G16L45_G12L30"] = (merged["r_c2"] + merged["r_g8"] + merged["r_g1"] + merged["r_g14_l45"] + merged["r_g16_l45"] + merged["r_g12_l30"]) / 6.0
    merged["ENS_C2_G1_G14L45_G16L45_G17ANISO"] = 0.2 * (merged["r_c2"] + merged["r_g1"] + merged["r_g14_l45"] + merged["r_g16_l45"] + merged["r_g17_aniso"])
    merged["ENS_G8_G1_G14L45_G16L45_G17L60"] = 0.2 * (merged["r_g8"] + merged["r_g1"] + merged["r_g14_l45"] + merged["r_g16_l45"] + merged["r_g17_l60"])

    models = {
        "C2": "r_c2",
        "ENS_ZS2_G14L45_G16L45": "ENS_ZS2_G14L45_G16L45",
        "ENS_ZS3_L45_L30": "ENS_ZS3_L45_L30",
        "ENS_C2_G14L45_G16L45": "ENS_C2_G14L45_G16L45",
        "ENS_C2_G1_G14L45": "ENS_C2_G1_G14L45",
        "ENS_C2_G1_G16L45": "ENS_C2_G1_G16L45",
        "ENS_C2_G1_G14L45_G16L45": "ENS_C2_G1_G14L45_G16L45",
        "ENS_C2_G8_G1_G14L45_G16L45": "ENS_C2_G8_G1_G14L45_G16L45",
        "ENS_C2_G8_G1_G14L45_G16L45_G12L30": "ENS_C2_G8_G1_G14L45_G16L45_G12L30",
        "ENS_C2_G1_G14L45_G16L45_G17ANISO": "ENS_C2_G1_G14L45_G16L45_G17ANISO",
        "ENS_G8_G1_G14L45_G16L45_G17L60": "ENS_G8_G1_G14L45_G16L45_G17L60",
    }

    merged["rev10"] = fa.reversal_10d(raw_path, merged)
    merged["resid_actual"] = fa.residualize(merged, "actual_return_d10", ["size_decile", "rev10"], sector=True)
    merged = _attach_liquidity(merged, raw_path)
    liq = merged[merged["med_amt_20d"] >= 5e7].copy()
    univ_daily = merged.groupby("asof_date", sort=False)["actual_return_d10"].mean()
    base_ic = fa.daily_ic(merged, "r_c2")

    cross_section = _eval_cross_section(merged, liq, univ_daily, base_ic, models)

    pair_corrs = {}
    cols = ["r_c2", "r_g8", "r_g1", "r_g12_l30", "r_g14_l45", "r_g16_l45", "r_g17_l60"]
    for a in cols:
        pair_corrs[a] = {
            b: float(merged.groupby("asof_date").apply(lambda g: g[a].corr(g[b]), include_groups=False).mean())
            for b in cols
        }

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
            runs = [
                nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats[name], k, dummy_gate, False), days, k, last_day)
                for days in phases
            ]
            nets = [r["daily"]["ret"] for r in runs]
            if name == "C2":
                c2_daily[k] = nets
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
            dds = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
            mn = float(np.mean([fnb.total(r) for r in nets]))
            mdd = float(np.mean(dds))
            books[f"{name}_k{k}"] = {
                "k": k,
                "model": name,
                "mean_net": mn,
                "mean_net_excess": float(np.mean(excess)),
                "beat_pool_phases": int(sum(x > 0 for x in excess)),
                "phases": len(phases),
                "mean_max_drawdown": mdd,
                "calmar_like": float(mn / abs(mdd)) if abs(mdd) > 1e-9 else 0.0,
                "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                "vs_universe_ci95": sub.block_ci(pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()),
                "vs_c2_ci95": sub.block_ci(pd.concat([r - c for r, c in zip(nets, c2_daily[k])], axis=1).mean(axis=1).to_numpy()),
            }

    out = {"pair_corrs": pair_corrs, "cross_section": cross_section, "books": books}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["g13_g15", "ensembles"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = evaluate_g13_g15(args.out) if args.mode == "g13_g15" else evaluate_g1_g18_ensembles(args.out)
    print(f"wrote {args.out} keys={list(res.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
