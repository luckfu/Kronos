"""Wave 4 (G1-G38 + E1 + G38 Hazard Gate) Final Cross-Experiment Master Ensemble Evaluator.

Evaluates strictly on 65-day OOS (2026-07-17 .. 2026-10-23, 335,312 rows) using equal-weight
cross-sectional rank averaging (zero OOS weight tuning), keeping 2026-10-24+ sealed untouched.
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

import eval_adaptive_execution_shield as e1  # noqa: E402
import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

OOS_START = "2026-07-17"
OOS_END = "2026-10-23"

SOURCES: dict[str, tuple[str, list[str]]] = {
    "g25": ("scratch/gpu_alpha_g25_out/oos_g25_scores.csv.gz", ["p_g25_base_l52_pool"]),
    "g27": ("scratch/gpu_alpha_g27_out/oos_g27_scores.csv.gz", ["p_g27_film_regime_mlp", "p_g27_resid_mlp_cosic"]),
    "g29": ("scratch/gpu_alpha_g29_out/oos_g29_scores.csv.gz", ["p_g29_mini_l45_pool_regime"]),
    "g30": (
        "scratch/gpu_alpha_g30_out/oos_g30_scores.csv.gz",
        ["p_g30_quad_film_rankw", "p_g30_quad_xstock_attn", "p_g30_quad_lgbm_ret5"],
    ),
    "g31": (
        "scratch/gpu_alpha_g31_out/oos_g31_scores.csv.gz",
        [
            "p_g31_mini_l45_dep_s2",
            "p_g31_c2_l45_rich_regime",
            "p_g31_minic2_joint_ret5",
            "p_g31_minic2_joint_regime",
        ],
    ),
    "g32": ("scratch/gpu_alpha_g32_out/oos_g32_scores.csv.gz", ["p_g32_c2_l45_ar16_d10_mean"]),
    "g33": ("scratch/gpu_alpha_g33_out/oos_g33_scores.csv.gz", ["p_g33_base_l52_rich_ret20"]),
    "g34": (
        "scratch/gpu_alpha_g34_out/oos_g34_scores.csv.gz",
        ["p_g34_dep_c2_raw", "p_g34_dep_plus_stats_regime", "p_g34_s1s2_stats_quad40"],
    ),
    "g35": (
        "scratch/gpu_alpha_g35_out/oos_g35_scores.csv.gz",
        [
            "p_g35_ridge_cvar10",
            "p_g35_ridge_floor10",
            "p_g35_ridge_nowhipsaw",
            "p_g35_ridge_calmar10",
            "p_g35_ridge_crashentry",
            "p_g35_s1s2_cvar10",
            "p_g35_tailshield_cvar_shield35",
            "p_g35_tailshield_cvar_shield70",
            "p_g35_tailshield_nowhipsaw_shield35",
            "p_g35_lgbm_cvar10",
            "p_g35_lgbm_nowhipsaw",
        ],
    ),
    "g36": (
        "scratch/gpu_alpha_g36_out/oos_g36_scores.csv.gz",
        [
            "p_g36_quad_liq_ridge",
            "p_g36_quad_lambdarank_r10",
            "p_g36_quad_lambdarank_multi",
            "p_g36_quad_focal_listnet",
            "p_g36_quad_xstock_topk",
            "p_g36_hybrid_lambdarank_lin",
            "p_g36_hybrid_topk_all",
        ],
    ),
    "g37": (
        "scratch/gpu_alpha_g37_out/oos_g37_scores.csv.gz",
        [
            "p_g37_c2only_ridge_regime",
            "p_g37_c2only_distill_resid",
            "p_g37_c2only_distill_xstock",
            "p_g37_c2only_distill_lgbm",
            "p_g37_minic2_ridge_regime",
            "p_g37_minic2_distill_resid",
            "p_g37_minic2_distill_xstock",
            "p_g37_minic2_distill_lgbm",
        ],
    ),
    "g38": (
        "scratch/gpu_alpha_g38_out/oos_g38_scores.csv.gz",
        [
            "p_g38_ridge_r5",
            "p_g38_ridge_fwd5_10",
            "p_g38_mlp_r5",
            "p_g38_mlp_fwd5_10",
            "p_g38_mlp_r10",
            "p_g38_consistent_d5_d10",
            "p_g38_lgbm_fwd5_10",
            "p_g38_lgbm_r10",
        ],
    ),
}

ENSEMBLES: dict[str, list[str]] = {
    # Baselines (W1-W3)
    "ENS_W2F_C2_MINI29": ["r_c2", "r_p_g29_mini_l45_pool_regime"],
    "W3_C2ONLY_FULL4": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
    ],
    "W3_LIGHT_C2AR_MINIC2_5WAY": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_minic2_joint_ret5",
        "r_p_g31_mini_l45_dep_s2",
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
    "W3_SUPREME_AR_W2_LGBMRET5": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
        "r_p_g30_quad_lgbm_ret5",
    ],
    # Track A: Zero-new-weight C2-Only Distilled
    "W4_C2ONLY_DISTILL_CORE": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g37_c2only_distill_resid",
        "r_p_g37_c2only_distill_xstock",
    ],
    "W4_C2ONLY_DISTILL_FULL": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
        "r_p_g37_c2only_ridge_regime",
        "r_p_g37_c2only_distill_resid",
        "r_p_g37_c2only_distill_xstock",
        "r_p_g37_c2only_distill_lgbm",
    ],
    # Track B: Ultra-Light C2 + 4.1M Mini Distilled
    "W4_LIGHT_MINIC2_DISTILL": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g37_minic2_distill_resid",
        "r_p_g37_minic2_distill_xstock",
    ],
    "W4_LIGHT_MINIC2_FULL": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_minic2_joint_ret5",
        "r_p_g37_minic2_ridge_regime",
        "r_p_g37_minic2_distill_resid",
        "r_p_g37_minic2_distill_xstock",
        "r_p_g37_minic2_distill_lgbm",
    ],
    # Track C: Liquid Pool Top-10/20 LambdaRank & Focal-ListNet Sharpening
    "W4_TOPK_SHARP_CORE": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g36_quad_lambdarank_r10",
        "r_p_g36_quad_focal_listnet",
        "r_p_g36_quad_xstock_topk",
    ],
    "W4_TOPK_SHARP_FLAGSHIP": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g30_quad_xstock_attn",
        "r_p_g36_quad_lambdarank_r10",
        "r_p_g36_quad_lambdarank_multi",
        "r_p_g36_quad_focal_listnet",
        "r_p_g36_quad_xstock_topk",
    ],
    # Track D: Downside Crash-Shield + Term-Structure Continuation + High-Calmar Flagships
    "W4_CRASH_SHIELD_5PILLAR": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g35_tailshield_nowhipsaw_shield35",
        "r_p_g35_ridge_crashentry",
        "r_p_g38_consistent_d5_d10",
        "r_p_g30_quad_xstock_attn",
    ],
    "W4_TERMSTRUCT_CONT_CORE": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g38_consistent_d5_d10",
        "r_p_g38_mlp_fwd5_10",
        "r_p_g38_lgbm_fwd5_10",
        "r_p_g35_lgbm_nowhipsaw",
    ],
    "W4_APEX_GRAND": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_s1s2_stats_quad40",
        "r_p_g30_quad_lgbm_ret5",
        "r_p_g30_quad_xstock_attn",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g36_quad_lambdarank_r10",
        "r_p_g36_quad_focal_listnet",
        "r_p_g38_consistent_d5_d10",
    ],
}

SHIELD_MODELS = [
    "C2",
    "W3_LIGHT_C2AR_MINIC2_5WAY",
    "W3_FLAGSHIP_5PILLAR_WITH_S1S2",
    "W4_LIGHT_MINIC2_DISTILL",
    "W4_TOPK_SHARP_FLAGSHIP",
    "W4_CRASH_SHIELD_5PILLAR",
    "W4_APEX_GRAND",
]


def compute_hazard_gate_tiers(
    breadth_ma20: pd.Series,
    p_crash_hazard: pd.Series,
    warn_thresh: float = 0.55,
    critical_thresh: float = 0.70,
) -> pd.Series:
    """Map causal breadth + G38 P_crash_hazard probabilities into discrete {0, 1, 2} allocation tiers."""
    base = np.where(breadth_ma20 < 0.30, 0, np.where(breadth_ma20 < 0.45, 1, 2))
    out = np.where(
        breadth_ma20 < 0.30,
        0,
        np.where(
            p_crash_hazard >= critical_thresh,
            0,
            np.where(p_crash_hazard >= warn_thresh, np.maximum(base - 1, 0), base),
        ),
    )
    return pd.Series(out.astype(int), index=breadth_ma20.index, name="G_HAZARD_G38")


def main() -> int:
    c2_dir = ROOT / "scratch" / "non_pv" / "c2_p16_65d"
    raw_path = ROOT / "scratch" / "oos65" / "combined_raw.csv"
    haz_path = ROOT / "scratch" / "gpu_alpha_g38_out" / "oos_g38_regime_hazard.csv"
    out_json = ROOT / "finetune" / "reports" / "20261127_gpu_wave4_final_ensembles_results.json"

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
    phases10 = [signal_days[p::10] for p in range(10)]
    score_mats = {
        name: market.score_matrix(merged[["symbol", "asof_date", col]].rename(columns={col: "predicted_return_d10"}))
        for name, col in cols.items()
    }
    bench10 = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases10]
    dummy_gate = np.full(len(market.dates), 2, dtype=int)
    books = {}
    c2_daily_by_k = {}
    for k in (10, 20):
        for name in cols:
            runs = [
                nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats[name], k, dummy_gate, False), days, k, last_day)
                for days in phases10
            ]
            nets = [r["daily"]["ret"] for r in runs]
            if name == "C2":
                c2_daily_by_k[k] = nets
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench10)]
            drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
            mean_net = float(np.mean([fnb.total(r) for r in nets]))
            mean_dd = float(np.mean(drawdowns))
            diff_univ = pd.concat([r - b for r, b in zip(nets, bench10)], axis=1).mean(axis=1).to_numpy()
            diff_c2 = pd.concat([r - c for r, c in zip(nets, c2_daily_by_k[k])], axis=1).mean(axis=1).to_numpy()
            books[f"{name}_k{k}"] = {
                "k": k, "model": name,
                "mean_net": mean_net,
                "mean_universe_net": float(np.mean([fnb.total(b) for b in bench10])),
                "mean_net_excess": float(np.mean(excess)),
                "beat_pool_phases": int(sum(x > 0 for x in excess)),
                "phases": len(phases10),
                "mean_max_drawdown": mean_dd,
                "calmar_like": float(mean_net / abs(mean_dd)) if abs(mean_dd) > 1e-9 else 0.0,
                "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                "vs_universe_ci95": sub.block_ci(diff_univ),
                "vs_c2_ci95": sub.block_ci(diff_c2),
            }

    # Causal Execution Shield + G38 Hazard Gate matrix
    reg, _ = e1.compute_causal_breadth_and_vol_mask(raw_path)
    haz_df = pd.read_csv(haz_path)
    haz_df["asof_date"] = haz_df["asof_date"].astype(str)
    haz_idx = haz_df.set_index("asof_date")
    aligned_b20 = reg["breadth_ma20"].reindex(haz_idx.index).ffill().fillna(0.5)
    reg.loc[haz_idx.index, "G_HAZARD_G38"] = compute_hazard_gate_tiers(aligned_b20, haz_idx["p_crash_hazard"])
    reg["G_HAZARD_G38"] = reg["G_HAZARD_G38"].fillna(reg["G_BREADTH"]).astype(int)

    gate_arrays = {}
    for gname in ("RAW", "G_BREADTH", "G_BREADTH_VEL", "G_HAZARD_G38"):
        arr = np.full(len(market.dates), 2, dtype=int)
        if gname != "RAW":
            for d, idx in market.date_index.items():
                if d in reg.index:
                    arr[idx] = int(reg.loc[d, gname])
        gate_arrays[gname] = arr

    shield_books = {}
    for step in (10, 5):
        phases = [signal_days[p::step] for p in range(step)]
        bench = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases]
        for k in (10, 20):
            for shield in ("RAW", "G_BREADTH", "G_BREADTH_VEL", "G_HAZARD_G38"):
                use_gate = shield != "RAW"
                for mname in SHIELD_MODELS:
                    runs = [
                        nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats[mname], k, gate_arrays[shield], use_gate), days, k, last_day)
                        for days in phases
                    ]
                    nets = [r["daily"]["ret"] for r in runs]
                    excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
                    drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
                    mean_net = float(np.mean([fnb.total(r) for r in nets]))
                    mean_dd = float(np.mean(drawdowns))
                    shield_books[f"step{step}_k{k}_{shield}_{mname}"] = {
                        "step": step, "k": k, "shield": shield, "model": mname,
                        "mean_net": mean_net,
                        "mean_net_excess": float(np.mean(excess)),
                        "beat_pool_phases": int(sum(x > 0 for x in excess)),
                        "phases": len(phases),
                        "mean_max_drawdown": mean_dd,
                        "calmar_like": float(mean_net / abs(mean_dd)) if abs(mean_dd) > 1e-9 else 0.0,
                        "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                    }

    gate_counts = {
        g: {int(k): int(v) for k, v in reg.loc[signal_dates, g].value_counts().sort_index().items()}
        for g in ("G_BREADTH", "G_BREADTH_VEL", "G_HAZARD_G38")
    }
    result = {
        "ensembles": ens_def,
        "cross_section": cross_section,
        "books": books,
        "shield_gate_counts": gate_counts,
        "shield_books": shield_books,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"{'NAME':<32} {'IC':>7} {'ICIR':>5} {'POS':>6} {'SEGA/B':>11} {'NIC':>7} {'dIC':>7} {'LIQ10':>7} {'K10_NET':>8} {'K10_DD':>8} {'K10_CAL':>7} {'K20_NET':>8} {'K20_DD':>8} {'K20_CAL':>7}")
    for name in cols:
        v = cross_section[name]
        dic = v.get("paired_vs_C2", {}).get("delta_ic", 0.0)
        b10, b20 = books[f"{name}_k10"], books[f"{name}_k20"]
        print(
            f"{name:<32} {v['ic']:7.4f} {v['icir']:5.2f} {v['positive_ratio']*100:5.1f}% "
            f"{v['ic_seg_a']:.3f}/{v['ic_seg_b']:.3f} {v['neutral_ic']:7.4f} {dic:+7.4f} "
            f"{v['liq_top10_excess']*100:+6.2f}% {b10['mean_net']*100:+7.2f}% {b10['mean_max_drawdown']*100:+7.2f}% "
            f"{b10['calmar_like']:7.3f} {b20['mean_net']*100:+7.2f}% {b20['mean_max_drawdown']*100:+7.2f}% {b20['calmar_like']:7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
