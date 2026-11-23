"""Wave 3 (G1-G34) Final Cross-Experiment Master Ensemble Evaluator.

Evaluates strictly on 65-day OOS (2026-07-17 .. 2026-10-23, 335,312 rows) using
equal-weight cross-sectional rank averaging (zero OOS weight tuning), keeping
2026-10-24+ sealed acceptance window 100% untouched.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

for _mod in ("sklearn", "sklearn.linear_model", "xgboost"):
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        if _mod == "sklearn.linear_model":
            _m.Ridge = object
        if _mod == "xgboost":
            _m.XGBRegressor = object
        sys.modules[_mod] = _m

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

SOURCES: dict[str, tuple[str, list[str]]] = {
    "g24": (
        "scratch/gpu_alpha_g24_out/oos_g24_scores.csv.gz",
        ["p_g24_pts_l45_full2560_multi"],
    ),
    "g25": (
        "scratch/gpu_alpha_g25_out/oos_g25_scores.csv.gz",
        ["p_g25_base_l52_pool"],
    ),
    "g27": (
        "scratch/gpu_alpha_g27_out/oos_g27_scores.csv.gz",
        ["p_g27_film_regime_mlp", "p_g27_resid_mlp_cosic", "p_g27_lgbm_nonlin"],
    ),
    "g29": (
        "scratch/gpu_alpha_g29_out/oos_g29_scores.csv.gz",
        ["p_g29_mini_l45_pool_regime", "p_g29_mini_l30_45_raw", "p_g29_trifam_l45_decay"],
    ),
    "g30": (
        "scratch/gpu_alpha_g30_out/oos_g30_scores.csv.gz",
        [
            "p_g30_quad_ridge_regime",
            "p_g30_quad_resid_cosic",
            "p_g30_quad_film_cosic",
            "p_g30_quad_film_rankw",
            "p_g30_quad_xstock_attn",
            "p_g30_quad_lgbm_multi",
            "p_g30_quad_lgbm_ret5",
        ],
    ),
    "g31": (
        "scratch/gpu_alpha_g31_out/oos_g31_scores.csv.gz",
        [
            "p_g31_mini_s1_stats",
            "p_g31_mini_l45_dep_s2",
            "p_g31_mini_l30_45_rich_volnorm",
            "p_g31_c2_l45_rich_regime",
            "p_g31_minic2_joint_ret5",
            "p_g31_minic2_joint_regime",
            "p_g31_minic2_joint_decay365",
        ],
    ),
    "g32": (
        "scratch/gpu_alpha_g32_out/oos_g32_scores.csv.gz",
        [
            "p_g32_c2_l45_ar16_d10_mean",
            "p_g32_c2_l45_ar16_d10_med",
            "p_g32_c2_l45_ar16_d10_probup",
        ],
    ),
    "g33": (
        "scratch/gpu_alpha_g33_out/oos_g33_scores.csv.gz",
        [
            "p_g33_base_l45_all_dep_regime",
            "p_g33_base_l52_rich_ret20",
            "p_g33_base_l52_ft_ridge_pool",
        ],
    ),
    "g34": (
        "scratch/gpu_alpha_g34_out/oos_g34_scores.csv.gz",
        [
            "p_g34_dep_c2_raw",
            "p_g34_dep_quad_raw",
            "p_g34_dep_quad_decay365",
            "p_g34_dep_quad_regime",
            "p_g34_dep_plus_stats_regime",
            "p_g34_s1s2_stats_quad40",
        ],
    ),
}

ENSEMBLES: dict[str, list[str]] = {
    "W2_CHAMP_FILM27": [
        "r_c2",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_PURE_AR_DUALWIN": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
    ],
    "W3_PURE_AR_DUALWIN_PROBUP": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g32_c2_l45_ar16_d10_probup",
    ],
    "W3_C2ONLY_AR_RICH": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_c2_l45_rich_regime",
    ],
    "W3_C2ONLY_FULL4": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
    ],
    "W3_LIGHT_C2AR_MINIC2": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
    ],
    "W3_LIGHT_C2AR_MINIC2_5WAY": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_minic2_joint_ret5",
        "r_p_g31_mini_l45_dep_s2",
    ],
    "W3_LIGHT_C2AR_MINI29_G31": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_ret5",
        "r_p_g31_mini_l30_45_rich_volnorm",
    ],
    "W3_LIGHT_C2AR_MINI_PLUS_S1": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_mini_l45_dep_s2",
        "r_p_g31_mini_s1_stats",
    ],
    "W3_FLAGSHIP_AR_PLUS_W2CHAMP": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_FLAGSHIP_G30_NONLIN_AR": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g30_quad_film_cosic",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g30_quad_lgbm_multi",
    ],
    "W3_FLAGSHIP_G30_AR_CORE": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
    ],
    "W3_FLAGSHIP_5PILLAR_APEX": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_dep_quad_regime",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_FLAGSHIP_5PILLAR_BALANCED": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g32_c2_l45_ar16_d10_probup",
        "r_p_g31_minic2_joint_ret5",
        "r_p_g33_base_l45_all_dep_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_dep_quad_decay365",
        "r_p_g30_quad_film_cosic",
        "r_p_g30_quad_xstock_attn",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_FLAGSHIP_5PILLAR_WITH_S1S2": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_dep_plus_stats_regime",
        "r_p_g34_s1s2_stats_quad40",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_FLAGSHIP_HIGH_CALMAR": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_probup",
        "r_p_g29_mini_l30_45_raw",
        "r_p_g31_minic2_joint_decay365",
        "r_p_g33_base_l45_all_dep_regime",
        "r_p_g34_dep_quad_decay365",
        "r_p_g30_quad_film_rankw",
        "r_p_g27_resid_mlp_cosic",
    ],
    "W3_FLAGSHIP_6PILLAR_GRAND": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g33_base_l52_ft_ridge_pool",
        "r_p_g34_dep_quad_raw",
        "r_p_g30_quad_film_cosic",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g30_quad_lgbm_multi",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_NO_AR_QUAD_NONLIN_DEP": [
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_dep_quad_regime",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_SUPREME_AR_W2_LGBMRET5": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
        "r_p_g30_quad_lgbm_ret5",
    ],
    "W3_SUPREME_5PILLAR_LGBMRET5": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_dep_quad_regime",
        "r_p_g30_quad_lgbm_ret5",
        "r_p_g30_quad_xstock_attn",
    ],
    "W3_NO_AR_LGBMRET5_CORE": [
        "r_p_g30_quad_lgbm_ret5",
        "r_p_g30_quad_xstock_attn",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
    ],
}


def main() -> int:
    c2_dir = ROOT / "scratch" / "non_pv" / "c2_p16_65d"
    raw_path = ROOT / "scratch" / "oos65" / "combined_raw.csv"
    out_json = ROOT / "finetune" / "reports" / "20261123_gpu_wave3_final_ensembles_results.json"

    c2 = fa.load_model(c2_dir)
    c2["asof_date"] = c2["asof_date"].astype(str)
    c2 = c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)
    assert c2["asof_date"].max() <= OOS_END, "Sealed window violation!"

    merged = c2.copy()
    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")

    for tag, (rel_path, req_cols) in SOURCES.items():
        df = pd.read_csv(ROOT / rel_path, usecols=["identity", "asof_date"] + req_cols)
        df["asof_date"] = df["asof_date"].astype(str)
        assert df["asof_date"].max() <= OOS_END, f"Sealed violation in {tag}"
        merged = merged.merge(df[["identity"] + req_cols], on="identity", how="inner", validate="one_to_one")
        if len(merged) != len(c2):
            raise RuntimeError(f"Alignment mismatch after {tag}: {len(merged)} vs {len(c2)}")
        for sc in req_cols:
            merged[f"r_{sc}"] = fa.ranks(merged, sc)

    cols: dict[str, str] = {"C2": "predicted_return_d10"}
    ens_def: dict[str, list[str]] = {"C2": ["predicted_return_d10"]}
    for name, members in ENSEMBLES.items():
        merged[name] = merged[members].mean(axis=1)
        cols[name] = name
        ens_def[name] = list(members)

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
    for name, col in cols.items():
        ic_s = fa.daily_ic(merged, col)
        res_col = fa.residualize(merged, col, ["size_decile", "rev10"], sector=True)
        tmp = pd.DataFrame({"asof_date": merged["asof_date"], "res": res_col, "act": merged["resid_actual"]})
        nic_s = tmp.groupby("asof_date", sort=False).apply(
            lambda g: g["res"].corr(g["act"], method="spearman"), include_groups=False
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
        cross_section[name] = entry

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

    result = {"ensembles": ens_def, "cross_section": cross_section, "books": books}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"{'NAME':<32} {'IC':>7} {'ICIR':>5} {'POS':>6} {'SEGA/B':>11} {'NIC':>7} {'NICIR':>5} {'dIC':>7} {'CI_LO':>7} {'LIQ10':>7} {'K10_NET':>8} {'K10_DD':>8} {'K10_CAL':>7} {'K10_B':>5} {'K20_NET':>8} {'K20_DD':>8} {'K20_CAL':>7}")
    for name in cols:
        v = cross_section[name]
        p = v.get("paired_vs_C2", {})
        dic = p.get("delta_ic", 0.0)
        ci_lo = p.get("ci95", [0.0, 0.0])[0]
        b10 = books[f"{name}_k10"]
        b20 = books[f"{name}_k20"]
        print(
            f"{name:<32} {v['ic']:7.4f} {v['icir']:5.2f} {v['positive_ratio']*100:5.1f}% "
            f"{v['ic_seg_a']:.3f}/{v['ic_seg_b']:.3f} {v['neutral_ic']:7.4f} {v['neutral_icir']:5.2f} "
            f"{dic:+7.4f} {ci_lo:+7.4f} {v['liq_top10_excess']*100:+6.2f}% "
            f"{b10['mean_net']*100:+7.2f}% {b10['mean_max_drawdown']*100:+7.2f}% {b10['calmar_like']:7.3f} {b10['beat_pool_phases']:2d}/10 "
            f"{b20['mean_net']*100:+7.2f}% {b20['mean_max_drawdown']*100:+7.2f}% {b20['calmar_like']:7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
