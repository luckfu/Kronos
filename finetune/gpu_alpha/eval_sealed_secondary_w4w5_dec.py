"""December Secondary Sealed Observation Track Evaluator (Wave 4 & Wave 5 Frozen Candidates).

Strictly keeps Primary Track (eval_sealed_review_dec.py & 20261125_sealed_review_dec_prereg_cn.md) untouched.
Reuses ProtocolConfig, compute_breadth_map, classify_dates_by_regime, check_prerequisites, _eval_subset,
and score_candidate directly from eval_sealed_review_dec.py so both tracks share bit-identical adjudication rules
while testing strictly frozen <= 2026-07-02 Wave 4 + Wave 5 No-G32 candidates and causal execution shields.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OOS_MOD))

import eval_no_g32_and_midcycle_shield as e2  # noqa: E402
import eval_sealed_review_dec as esr  # noqa: E402
import eval_wave4_master_ensembles as ew4  # noqa: E402
import few_names_backtest as fnb  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402
from finetune.small_clean_oos import execution_constrained as ec  # noqa: E402
from rank_book_backtest import infer_flags  # noqa: E402

DEFAULT_SECONDARY_DIR = ROOT / "scratch" / "sealed_dec_secondary"
DEFAULT_SECONDARY_OUTPUT = ROOT / "finetune" / "reports" / "202612_sealed_secondary_w4w5_results.json"

SECONDARY_ENSEMBLES: dict[str, list[str]] = {
    "FUSE_VAL_1E4": ["r_c2", "r_p_val_1e4"],
    "NOG32_C2ONLY_DISTILL5": [
        "r_c2",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
        "r_p_g37_c2only_distill_resid",
        "r_p_g37_c2only_distill_xstock",
    ],
    "W5_C2ONLY_TAILFWD_NOG32": [
        "r_c2",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
        "r_p_g37_c2only_distill_resid",
        "r_p_g37_c2only_distill_xstock",
        "r_p_g39_c2only_ridge_fwd5_10",
        "r_p_g39_c2only_mlp_cont_shield",
        "r_p_g39_c2only_xstock_tail",
        "r_p_g39_c2only_lgbm_fwd5_10",
    ],
    "NOG32_LIGHT_MINIC2_DISTILL6": [
        "r_c2",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_mini_l45_dep_s2",
        "r_p_g37_minic2_distill_resid",
        "r_p_g37_minic2_distill_xstock",
    ],
    "W5_LIGHT_NOG32_G39_TAILFWD": [
        "r_c2",
        "r_p_g31_minic2_joint_regime",
        "r_p_g39_minic2_ridge_fwd5_10",
        "r_p_g39_minic2_mlp_cont_shield",
        "r_p_g39_minic2_xstock_tail",
        "r_p_g39_minic2_lgbm_cvar",
    ],
    "W5_LIGHT_NOG32_G40_FTMINI": [
        "r_c2",
        "r_p_g40_ftmini_l45_e2e",
        "r_p_g40_ftmini_ridge_nowhipsaw",
        "r_p_g40_joint_ridge_nowhipsaw",
        "r_p_g40_joint_ridge_fwd5_10",
        "r_p_g40_joint_xstock_tail",
        "r_p_g40_joint_lgbm_cvar10",
    ],
    "W5_LIGHT_NOG32_SUPER_MINIC2": [
        "r_c2",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_mini_l45_dep_s2",
        "r_p_g37_minic2_distill_xstock",
        "r_p_g39_minic2_ridge_fwd5_10",
        "r_p_g39_minic2_mlp_cont_shield",
        "r_p_g40_ftmini_l45_e2e",
        "r_p_g40_joint_ridge_nowhipsaw",
        "r_p_g40_joint_lgbm_cvar10",
    ],
    "NOG32_CRASH_SHIELD_CORE": [
        "r_c2",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g30_quad_resid_cosic",
        "r_p_g35_lgbm_cvar10",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g35_tailshield_nowhipsaw_shield35",
        "r_p_g35_ridge_crashentry",
        "r_p_g38_ridge_fwd5_10",
    ],
    "W5_NOG32_CRASH_SHIELD_APEX": [
        "r_c2",
        "r_p_g27_resid_mlp_cosic",
        "r_p_g30_quad_resid_cosic",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g35_lgbm_cvar10",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g35_ridge_crashentry",
        "r_p_g38_ridge_fwd5_10",
        "r_p_g39_minic2_mlp_cont_shield",
        "r_p_g39_minic2_xstock_tail",
        "r_p_g40_ftmini_l45_e2e",
        "r_p_g40_joint_ridge_nowhipsaw",
        "r_p_g40_joint_lgbm_cvar10",
    ],
}

SECONDARY_SINGLE_REFS: list[str] = ["p_g35_lgbm_cvar10", "p_g40_ftmini_l45_e2e"]

SECONDARY_COMPLEXITY_RANK: dict[str, int] = {
    "NOG32_C2ONLY_DISTILL5": 1,
    "W5_C2ONLY_TAILFWD_NOG32": 1,
    "NOG32_LIGHT_MINIC2_DISTILL6": 2,
    "W5_LIGHT_NOG32_G39_TAILFWD": 2,
    "W5_LIGHT_NOG32_G40_FTMINI": 3,
    "W5_LIGHT_NOG32_SUPER_MINIC2": 3,
    "NOG32_CRASH_SHIELD_CORE": 4,
    "W5_NOG32_CRASH_SHIELD_APEX": 5,
    "SINGLE_G35_LGBM_CVAR10": 4,
    "SINGLE_G40_FTMINI_L45_E2E": 2,
}


def adjudicate_secondary(
    prereq: dict[str, Any],
    scope_a: dict[str, Any],
    scope_b: dict[str, Any],
    candidate_keys: list[str],
    cfg: esr.ProtocolConfig,
) -> dict[str, Any]:
    scorecards = [esr.score_candidate(name, scope_a, scope_b, cfg) for name in candidate_keys]
    for sc in scorecards:
        sc["deployment_complexity_rank"] = SECONDARY_COMPLEXITY_RANK.get(sc["candidate"], 99)
    scorecards.sort(key=lambda x: (-x["total_score"], x["deployment_complexity_rank"]))

    if not prereq["eligible_for_decision"]:
        return {
            "final_action": "DEFER_AUTO_EXTEND",
            "selected_candidate": None,
            "reason": "; ".join(prereq["failures"]),
            "scorecards": scorecards,
        }

    for tier_name in ("A_PROMOTE", "B_SHADOW_CANARY"):
        pool = [s for s in scorecards if s["tier"] == tier_name]
        if pool:
            best_score = pool[0]["total_score"]
            within_tie = [s for s in pool if (best_score - s["total_score"]) <= cfg.tie_margin]
            within_tie.sort(key=lambda x: (x["deployment_complexity_rank"], -x["total_score"]))
            winner = within_tie[0]
            return {
                "final_action": tier_name,
                "selected_candidate": winner["candidate"],
                "reason": f"score={winner['total_score']}, complexity_rank={winner['deployment_complexity_rank']}",
                "scorecards": scorecards,
            }

    return {
        "final_action": "C_HOLD",
        "selected_candidate": "FUSE_VAL_1E4",
        "reason": "No secondary candidate cleared thresholds without red-line vetoes.",
        "scorecards": scorecards,
    }


def build_sealed_market(kline: pd.DataFrame) -> sub.Market:
    """Build a causal sub.Market directly from adjusted sealed_kline_150d parquet."""
    df = kline.copy()
    df["symbol"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)
    adj = df["adj_factor"].astype(float) if "adj_factor" in df.columns else 1.0
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float) * adj
    bars = infer_flags(ec.normalize_bars(df))
    bars["median_amount"] = bars.groupby("symbol")["amount"].transform(
        lambda s: s.rolling(ec.LIQUIDITY_LOOKBACK, min_periods=ec.LIQUIDITY_LOOKBACK).median()
    )
    mkt = sub.Market.__new__(sub.Market)
    mkt.dates = sorted(bars["date"].unique())
    mkt.symbols = sorted(bars["symbol"].unique())
    mkt.date_index = {d: i for i, d in enumerate(mkt.dates)}
    mkt.symbol_index = {s: i for i, s in enumerate(mkt.symbols)}

    def wide(column: str, fill=np.nan, dtype=np.float64):
        out = np.full((len(mkt.dates), len(mkt.symbols)), fill, dtype=dtype)
        rows = bars["date"].map(mkt.date_index).to_numpy()
        cols = bars["symbol"].map(mkt.symbol_index).to_numpy()
        out[rows, cols] = bars[column].to_numpy()
        return out

    close, open_ = wide("close"), wide("open")
    mkt.traded = np.isfinite(close)
    mkt.one_up = wide("one_word_limit_up", False, bool)
    mkt.one_down = wide("one_word_limit_down", False, bool)
    mkt.close = pd.DataFrame(close).ffill().to_numpy()
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), mkt.close[:-1]])
    mkt.open = np.where(mkt.traded, open_, prev_close)
    mkt.high = np.where(mkt.traded, wide("high"), prev_close)
    median = wide("median_amount")
    mkt.liquid = mkt.traded & (median >= ec.MIN_TRAILING_MEDIAN_AMOUNT)
    slip = np.full(median.shape, np.nan)
    for floor, rate in reversed(ec.SLIPPAGE_TIERS_CNY):
        slip[median >= floor] = rate
    mkt.side_cost = pd.DataFrame(slip).ffill().fillna(sub.FALLBACK_SLIPPAGE).to_numpy() + sub.FEE_PER_SIDE
    return mkt


def eval_sealed_execution_overlays(
    kline: pd.DataFrame,
    merged: pd.DataFrame,
    models: dict[str, str],
    haz_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Evaluate RAW, G_BREADTH, G_BREADTH_VEL, G_HAZARD_G38, RAW_MCB, G_HAZARD_G38_MCB on sealed kline."""
    market = build_sealed_market(kline)
    last_day = len(market.dates) - 1
    sig_dates = sorted(merged["asof"].unique())
    sig_days = [market.date_index[d] for d in sig_dates if d in market.date_index]
    if len(sig_days) < 10:
        return {}
    max_dec_day = max(sig_days)

    b_map = esr.compute_breadth_map(kline)
    b_s = pd.Series({d: b_map[d] for d in sorted(b_map)})
    b5 = b_s.rolling(5, min_periods=1).mean()
    db3 = b_s - b_s.shift(3)
    db5 = b_s - b_s.shift(5)
    g_breadth = pd.Series(np.where(b_s < 0.30, 0, np.where(b_s < 0.45, 1, 2)), index=b_s.index)
    vel_warn = (db5 <= -0.12) | (b5 < (b_s - 0.10))
    g_vel = pd.Series(np.where(b_s < 0.30, 0, np.where(vel_warn, np.maximum(g_breadth - 1, 0), g_breadth)), index=b_s.index)
    panic = (b_s < 0.28) | (db3 <= -0.15) | (db5 <= -0.18)
    mcb_s = pd.Series(np.where(panic, 0, np.minimum(2, g_vel)), index=b_s.index)

    if haz_df is not None and not haz_df.empty:
        hz = haz_df.set_index("asof_date")
        al_b = b_s.reindex(hz.index).ffill().fillna(0.5)
        g_haz = g_breadth.copy()
        g_haz.loc[hz.index] = ew4.compute_hazard_gate_tiers(al_b, hz["p_crash_hazard"])
    else:
        g_haz = g_breadth.copy()

    gate_map = {}
    for gname, ser in [("RAW", None), ("G_BREADTH", g_breadth), ("G_BREADTH_VEL", g_vel), ("G_HAZARD_G38", g_haz), ("MCB", mcb_s)]:
        arr = np.full(len(market.dates), 2, dtype=int)
        if ser is not None:
            for d, idx in market.date_index.items():
                if d in ser.index:
                    arr[idx] = int(ser.loc[d])
        gate_map[gname] = arr

    score_mats = {}
    for mname, col in models.items():
        if col in merged.columns:
            sc = merged[["code", "asof", col]].rename(columns={"code": "symbol", "asof": "asof_date", col: "predicted_return_d10"})
            score_mats[mname] = market.score_matrix(sc)

    overlays = [
        ("RAW", "RAW", False),
        ("G_BREADTH", "G_BREADTH", False),
        ("G_BREADTH_VEL", "G_BREADTH_VEL", False),
        ("G_HAZARD_G38", "G_HAZARD_G38", False),
        ("RAW_MCB", "RAW", True),
        ("G_HAZARD_G38_MCB", "G_HAZARD_G38", True),
    ]
    out = {}
    phases10 = [sig_days[p::10] for p in range(10) if sig_days[p::10]]
    bench10 = [fnb.universe(market, score_mats["FUSE_VAL_1E4"], days[0], last_day) for days in phases10]
    for sname, g_key, enable_mcb in overlays:
        use_gate = g_key != "RAW"
        for mname, smat in score_mats.items():
            runs = [
                e2.run_mcb_gated_slots(
                    market, nrs.gated_plan(market, smat, 10, gate_map[g_key], use_gate), days, 10, last_day,
                    mcb_tier=gate_map["MCB"], enable_mcb=enable_mcb, max_decision_day=max_dec_day,
                )
                for days in phases10
            ]
            out[f"k10_{sname}_{mname}"] = e2._summarize_runs(runs, bench10)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dec Secondary Sealed Track Evaluator (W4+W5)")
    parser.add_argument("--sealed-dir", type=Path, default=DEFAULT_SECONDARY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_SECONDARY_OUTPUT)
    parser.add_argument("--parity-date", type=str, default=None)
    parser.add_argument("--strict-d10-start", type=str, default="2026-11-09")
    parser.add_argument("--strict-d5-start", type=str, default="2026-11-16")
    parser.add_argument("--cumulative-start", type=str, default="2026-10-26")
    args = parser.parse_args(argv)

    cfg = esr.ProtocolConfig(
        strict_d10_start=args.strict_d10_start,
        strict_d5_start=args.strict_d5_start,
        cumulative_start=args.cumulative_start,
    )
    fa = esr._import_fusion_analysis()

    ranks = pd.read_parquet(args.sealed_dir / "sealed_shadow_rankings.parquet")
    kline = pd.read_parquet(args.sealed_dir / "sealed_kline_150d.parquet")
    sec_pq = args.sealed_dir / "sealed_w4w5_secondary_scores.parquet"
    sec_df = pd.read_parquet(sec_pq)
    haz_p = args.sealed_dir / "sealed_g38_regime_hazard.csv"
    haz_df = pd.read_csv(haz_p) if haz_p.is_file() else None

    ranks["asof"] = ranks["asof"].astype(str)
    ranks["code"] = ranks["code"].astype(str)
    ranks["identity"] = ranks["asof"] + "|" + ranks["code"]
    kline["date"] = kline["date"].astype(str)
    kline["code"] = kline["code"].astype(str)

    p_cols = [c for c in sec_df.columns if c.startswith("p_")]
    merged = ranks.merge(sec_df[["identity"] + p_cols], on="identity", how="inner", validate="one_to_one")
    if len(merged) != len(ranks):
        raise RuntimeError(f"Alignment mismatch: {len(merged)} vs {len(ranks)}")

    kline = kline.sort_values(["code", "date"]).copy()
    kline["adj_close"] = kline["close"] * kline["adj_factor"]
    b_map = esr.compute_breadth_map(kline)

    traded = kline[kline["volume"] > 0].sort_values(["code", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("code")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    traded["rev10_raw"] = traded.groupby("code")["adj_close"].transform(lambda c: c / c.shift(10) - 1.0)
    merged = merged.merge(traded[["code", "date", "med_amt_20d", "rev10_raw"]].rename(columns={"date": "asof"}), on=["code", "asof"], how="left")
    merged["rev10"] = merged.groupby("asof")["rev10_raw"].rank(pct=True, method="average")

    parity_date = args.parity_date or str(sorted(merged["asof"].unique())[-1])
    dp = merged[merged["asof"] == parity_date].copy()
    dp["r_c2_chk"] = dp["predicted_return_d10"].rank(pct=True, method="average")
    dp["r_val_chk"] = dp["p_val_1e4"].rank(pct=True, method="average")
    dp["f_val_chk"] = 0.5 * dp["r_c2_chk"] + 0.5 * dp["r_val_chk"]
    parity = {
        "asof": parity_date,
        "n": int(len(dp)),
        "score_hidden_corr": float(dp["score_hidden"].corr(dp["p_val_1e4"])),
        "score_fused_corr": float(dp["score_fused"].corr(dp["f_val_chk"])),
    }

    rank_cols = {"r_c2": merged.groupby("asof")["predicted_return_d10"].rank(pct=True, method="average")}
    for c in p_cols:
        rank_cols[f"r_{c}"] = merged.groupby("asof")[c].rank(pct=True, method="average")
    merged = pd.concat([merged, pd.DataFrame(rank_cols, index=merged.index)], axis=1)

    active_ensembles = {
        name: members
        for name, members in SECONDARY_ENSEMBLES.items()
        if all(m in merged.columns for m in members)
    }
    ens_frame = {name: merged[members].mean(axis=1) for name, members in active_ensembles.items()}
    merged = pd.concat([merged, pd.DataFrame(ens_frame, index=merged.index)], axis=1)

    models: dict[str, str] = {"C2": "predicted_return_d10"}
    for name in active_ensembles:
        models[name] = name
    for sc in SECONDARY_SINGLE_REFS:
        if sc in merged.columns:
            models[f"SINGLE_{sc[2:].upper()}"] = sc

    cal = sorted(kline["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    scope_a = {
        "D10": esr.evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 10, cfg.strict_d10_start, models, cfg, fa),
        "D5": esr.evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 5, cfg.strict_d5_start, models, cfg, fa),
    }
    scope_b = {
        "D10": esr.evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 10, cfg.cumulative_start, models, cfg, fa),
        "D5": esr.evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 5, cfg.cumulative_start, models, cfg, fa),
    }

    prereq = esr.check_prerequisites(parity, scope_a["D10"], scope_b["D10"], cfg)
    cand_keys = [k for k in active_ensembles if k != "FUSE_VAL_1E4"]
    decision = adjudicate_secondary(prereq, scope_a, scope_b, cand_keys, cfg)

    overlays = eval_sealed_execution_overlays(kline, merged, models, haz_df) if prereq["eligible_for_decision"] else {}
    out = {
        "protocol": "20261128_sealed_secondary_w4w5_prereg_cn.md",
        "parity_gate": parity,
        "prerequisites": prereq,
        "decision": decision,
        "active_ensembles": active_ensembles,
        "SCOPE_A_STRICT_SEALED": scope_a,
        "SCOPE_B_CUMULATIVE": scope_b,
        "execution_overlays": overlays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote:", args.output)
    print(f"SECONDARY DECISION: {decision['final_action']} -> {decision['selected_candidate']} ({decision['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
