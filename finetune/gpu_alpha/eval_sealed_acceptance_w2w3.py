"""Formal Sealed Acceptance Evaluator for Wave 2 & Wave 3 (2026-10-26 .. 2026-11-20).

Strictly follows 20261123_sealed_acceptance_prereg_cn.md:
  1. Verifies live 2026-11-20 production parity (head_val_alpha1e4.npy vs ranking.csv).
  2. Evaluates ALL 15 pre-registered models/ensembles + single-head diagnostics on:
     - Primary D10 (10 mature dates: 2026-10-26..2026-11-06 -> target ..2026-11-20)
       with sub-regime split: breadth >= 30% (active, 3d) vs breadth < 30% (defensive, 7d)
     - Extended D5 (15 mature dates: 2026-10-26..2026-11-13 -> target ..2026-11-20)
       with sub-regime split: breadth >= 30% (active, 7d) vs breadth < 30% (defensive, 8d)
  3. Zero OOS weight tuning (pure equal-weight cross-sectional rank averaging).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEALED_DIR = ROOT / "scratch" / "sealed_20261026_20261120"
sys.path.insert(0, str(ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
import fusion_analysis as fa  # noqa: E402

ENSEMBLES: dict[str, list[str]] = {
    "FUSE_VAL_1E4": ["r_c2", "r_p_val_1e4"],
    "ENS_W2F_C2_MINI29": ["r_c2", "r_p_g29_mini_l45_pool_regime"],
    "W2_CHAMP_FILM27": [
        "r_c2",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_PURE_AR_DUALWIN": ["r_c2", "r_p_g32_c2_l45_ar16_d10_mean"],
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
    "W3_FLAGSHIP_AR_PLUS_W2CHAMP": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
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
    "W3_NO_AR_LGBMRET5_CORE": [
        "r_p_g30_quad_lgbm_ret5",
        "r_p_g30_quad_xstock_attn",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
    ],
}

SINGLE_DIAG: list[str] = [
    "p_val_1e4",
    "p_g32_c2_l45_ar16_d10_mean",
    "p_g32_c2_l45_ar16_d10_med",
    "p_g32_c2_l45_ar16_d10_probup",
    "p_g29_mini_l45_pool_regime",
    "p_g25_base_l52_pool",
    "p_g27_film_regime_mlp",
    "p_g27_resid_mlp_cosic",
    "p_g27_lgbm_nonlin",
    "p_g30_quad_lgbm_ret5",
    "p_g30_quad_lgbm_multi",
    "p_g30_quad_xstock_attn",
    "p_g30_quad_resid_cosic",
    "p_g30_quad_film_cosic",
    "p_g30_quad_film_rankw",
    "p_g30_quad_ridge_decay",
    "p_g31_minic2_joint_regime",
    "p_g31_minic2_joint_ret5",
    "p_g31_c2_l45_rich_regime",
    "p_g31_mini_l45_dep_s2",
    "p_g31_mini_s1_stats",
    "p_g33_base_l52_rich_ret20",
    "p_g33_base_l45_all_dep_regime",
    "p_g34_dep_c2_raw",
    "p_g34_dep_quad_regime",
    "p_g34_s1s2_stats_quad40",
]


def _eval_subset(
    sub: pd.DataFrame,
    models: dict[str, str],
    b_map: dict[str, float],
    date_filter: list[str] | None = None,
) -> dict:
    df = sub[sub["asof"].isin(date_filter)].copy() if date_filter is not None else sub.copy()
    dates = sorted(df["asof"].unique())
    if not dates:
        return {"n_dates": 0, "models": {}}

    liq = df[df["med_amt_20d"] >= 5e4].copy()
    df["resid_actual"] = fa.residualize(df, "actual_return", ["size_decile", "rev10"], sector=True)
    df_fa = df.rename(columns={"actual_return": "actual_return_d10"})
    base_c2_ic = fa.daily_ic(df_fa, models["C2"])
    base_val_ic = fa.daily_ic(df_fa, models["FUSE_VAL_1E4"])
    univ_daily = df.groupby("asof", sort=False)["actual_return"].mean()

    res = {}
    for name, col in models.items():
        ic_s = fa.daily_ic(df_fa, col)
        res_sig = fa.residualize(df, col, ["size_decile", "rev10"], sector=True)
        tmp = pd.DataFrame({"asof": df["asof"], "sig": res_sig, "act": df["resid_actual"]})
        nic_s = tmp.groupby("asof", sort=False).apply(
            lambda g: g["sig"].corr(g["act"], method="spearman"), include_groups=False
        )

        def _t10_univ(day: pd.DataFrame) -> float:
            hi = day[col].quantile(0.9)
            return float(day.loc[day[col] >= hi, "actual_return"].mean() - day["actual_return"].mean())

        t10u = df.groupby("asof", sort=False).apply(_t10_univ, include_groups=False)
        t10 = liq.sort_values(["asof", col], ascending=[True, False]).groupby("asof", sort=False).head(10)
        t20 = liq.sort_values(["asof", col], ascending=[True, False]).groupby("asof", sort=False).head(20)
        t10_ex = t10.groupby("asof")["actual_return"].mean() - univ_daily
        t20_ex = t20.groupby("asof")["actual_return"].mean() - univ_daily

        entry = {
            **fa.ic_stats(ic_s),
            "neutral_ic": float(nic_s.mean()),
            "neutral_icir": float(nic_s.mean() / nic_s.std(ddof=1)) if len(nic_s) > 1 and nic_s.std(ddof=1) else float("nan"),
            "top10pct_vs_univ": float(t10u.mean()),
            "top10pct_pos_days": int((t10u > 0).sum()),
            "liq_top10_excess": float(t10_ex.mean()),
            "liq_top10_pos_days": int((t10_ex > 0).sum()),
            "liq_top20_excess": float(t20_ex.mean()),
            "liq_top20_pos_days": int((t20_ex > 0).sum()),
            "daily_ic": {str(k): round(float(v), 6) for k, v in ic_s.items()},
        }
        if name != "C2" and len(dates) >= 3:
            entry["paired_vs_C2"] = fa.paired_bootstrap(base_c2_ic, ic_s, block=min(3, len(dates)), seed=20261123)
        if name != "FUSE_VAL_1E4" and len(dates) >= 3:
            entry["paired_vs_VAL1E4"] = fa.paired_bootstrap(base_val_ic, ic_s, block=min(3, len(dates)), seed=20261123)
        res[name] = entry

    return {
        "n_dates": len(dates),
        "dates": dates,
        "breadth_by_date": {d: round(b_map[d], 4) for d in dates},
        "n_rows": int(len(df)),
        "univ_mean_ret": float(univ_daily.mean()),
        "models": res,
    }


def evaluate_horizon(
    merged: pd.DataFrame,
    kline: pd.DataFrame,
    cal: list[str],
    cal_idx: dict[str, int],
    b_map: dict[str, float],
    horizon: int,
    models: dict[str, str],
) -> dict:
    valid_dates = [d for d in sorted(merged["asof"].unique()) if cal_idx[d] + horizon < len(cal)]
    sub = merged[merged["asof"].isin(valid_dates)].copy()

    ac_map = kline.set_index(["code", "date"])["adj_close"]
    target_dates = [cal[cal_idx[d] + horizon] for d in sub["asof"]]
    k0 = pd.MultiIndex.from_arrays([sub["code"], sub["asof"]])
    k1 = pd.MultiIndex.from_arrays([sub["code"], target_dates])
    c0 = ac_map.reindex(k0).to_numpy(dtype=np.float64)
    c1 = ac_map.reindex(k1).to_numpy(dtype=np.float64)
    sub["actual_return"] = c1 / c0 - 1.0
    sub["asof_date"] = sub["asof"]
    sub["sector"] = sub["sector_id"]
    sub["size_decile"] = (np.floor(sub["size_percentile"].clip(0, 0.999999) * 10) + 1).astype(int)

    active_dates = [d for d in valid_dates if b_map[d] >= 0.30]
    def_dates = [d for d in valid_dates if b_map[d] < 0.30]
    return {
        "horizon_days": horizon,
        "last_target_date": cal[cal_idx[valid_dates[-1]] + horizon],
        "ALL": _eval_subset(sub, models, b_map, None),
        "ACTIVE_B30": _eval_subset(sub, models, b_map, active_dates),
        "DEFENSIVE_LT30": _eval_subset(sub, models, b_map, def_dates),
    }


def main() -> int:
    ranks = pd.read_parquet(SEALED_DIR / "sealed_shadow_rankings.parquet")
    kline = pd.read_parquet(SEALED_DIR / "sealed_kline_150d.parquet")
    hid = pd.read_parquet(SEALED_DIR / "sealed_w2w3_hidden_scores.parquet")
    ar = pd.read_parquet(SEALED_DIR / "sealed_g32_ar_scores.parquet")

    ranks["asof"] = ranks["asof"].astype(str)
    ranks["code"] = ranks["code"].astype(str)
    ranks["identity"] = ranks["asof"] + "|" + ranks["code"]
    kline["date"] = kline["date"].astype(str)
    kline["code"] = kline["code"].astype(str)

    hid_cols = [c for c in hid.columns if c.startswith("p_")]
    ar_cols = [c for c in ar.columns if c.startswith("p_")]
    merged = ranks.merge(hid[["identity"] + hid_cols], on="identity", how="inner", validate="one_to_one")
    merged = merged.merge(ar[["identity"] + ar_cols], on="identity", how="inner", validate="one_to_one")
    if len(merged) != len(ranks):
        raise RuntimeError(f"Alignment mismatch: {len(merged)} vs {len(ranks)}")

    # Kline derived features
    kline = kline.sort_values(["code", "date"]).copy()
    kline["adj_close"] = kline["close"] * kline["adj_factor"]
    kline["ma20"] = kline.groupby("code")["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    asof_dates = sorted(merged["asof"].unique())
    b_map = (
        kline[kline["date"].isin(asof_dates)]
        .groupby("date")
        .apply(lambda g: float((g["adj_close"] > g["ma20"]).mean()), include_groups=False)
        .to_dict()
    )

    traded = kline[kline["volume"] > 0].sort_values(["code", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("code")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    traded["rev10_raw"] = traded.groupby("code")["adj_close"].transform(lambda c: c / c.shift(10) - 1.0)
    feat_map = traded[["code", "date", "med_amt_20d", "rev10_raw"]].rename(columns={"date": "asof"})
    merged = merged.merge(feat_map, on=["code", "asof"], how="left")
    merged["rev10"] = merged.groupby("asof")["rev10_raw"].rank(pct=True, method="average")

    # Parity check against live 2026-11-20 ranking.csv
    d20 = merged[merged["asof"] == "2026-11-20"].copy()
    d20["r_c2_chk"] = d20["predicted_return_d10"].rank(pct=True, method="average")
    d20["r_val_chk"] = d20["p_val_1e4"].rank(pct=True, method="average")
    d20["f_val_chk"] = 0.5 * d20["r_c2_chk"] + 0.5 * d20["r_val_chk"]
    parity = {
        "asof": "2026-11-20",
        "n": int(len(d20)),
        "score_hidden_corr": float(d20["score_hidden"].corr(d20["p_val_1e4"])),
        "score_fused_corr": float(d20["score_fused"].corr(d20["f_val_chk"])),
        "score_fused_max_abs_diff": float((d20["score_fused"] - d20["f_val_chk"]).abs().max()),
    }

    # Pre-registered cross-sectional rank columns
    rank_cols = {"r_c2": merged.groupby("asof")["predicted_return_d10"].rank(pct=True, method="average")}
    for c in hid_cols + ar_cols:
        rank_cols[f"r_{c}"] = merged.groupby("asof")[c].rank(pct=True, method="average")
    merged = pd.concat([merged, pd.DataFrame(rank_cols, index=merged.index)], axis=1)

    ens_frame = {}
    for name, members in ENSEMBLES.items():
        ens_frame[name] = merged[members].mean(axis=1)
    merged = pd.concat([merged, pd.DataFrame(ens_frame, index=merged.index)], axis=1)

    models: dict[str, str] = {"C2": "predicted_return_d10"}
    for name in ENSEMBLES:
        models[name] = name
    for sc in SINGLE_DIAG:
        models[f"SINGLE_{sc[2:].upper()}"] = sc

    cal = sorted(kline["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    out = {
        "parity_2026_11_20": parity,
        "ensembles": ENSEMBLES,
        "D10": evaluate_horizon(merged, kline, cal, cal_idx, b_map, 10, models),
        "D5": evaluate_horizon(merged, kline, cal, cal_idx, b_map, 5, models),
    }

    out_path = ROOT / "finetune" / "reports" / "20261123_sealed_acceptance_w2w3_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote:", out_path)
    print("PARITY 2026-11-20:", json.dumps(parity))

    prereg_keys = ["C2"] + list(ENSEMBLES.keys()) + ["SINGLE_G32_C2_L45_AR16_D10_MEAN", "SINGLE_G30_QUAD_LGBM_RET5", "SINGLE_G30_QUAD_XSTOCK_ATTN", "SINGLE_G31_MINIC2_JOINT_REGIME", "SINGLE_G29_MINI_L45_POOL_REGIME", "SINGLE_G33_BASE_L52_RICH_RET20", "SINGLE_G34_DEP_QUAD_REGIME"]
    for hz in ("D10", "D5"):
        for reg in ("ALL", "ACTIVE_B30", "DEFENSIVE_LT30"):
            blk = out[hz][reg]
            print(f"\n=== {hz} [{reg}] ({blk['n_dates']}d, univ={blk['univ_mean_ret']*100:+.2f}%) ===")
            print(f"{'NAME':<34} {'IC':>7} {'ICIR':>6} {'POS':>6} {'NIC':>7} {'dC2':>7} {'dVAL':>7} {'T10%':>7} {'LIQ10':>7} {'LIQ20':>7}")
            for m in prereg_keys:
                st = blk["models"][m]
                dc2 = st.get("paired_vs_C2", {}).get("delta_ic", 0.0)
                dv = st.get("paired_vs_VAL1E4", {}).get("delta_ic", 0.0)
                print(
                    f"{m:<34} {st['ic']:+7.4f} {st['icir']:+6.2f} {st['positive_ratio']*100:5.1f}% "
                    f"{st['neutral_ic']:+7.4f} {dc2:+7.4f} {dv:+7.4f} "
                    f"{st['top10pct_vs_univ']*100:+6.2f}% {st['liq_top10_excess']*100:+6.2f}% {st['liq_top20_excess']*100:+6.2f}%"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
