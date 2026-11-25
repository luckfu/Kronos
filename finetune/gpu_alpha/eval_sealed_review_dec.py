"""Formal Sealed Review Evaluator for Dec Multi-Regime Acceptance Protocol.

Strictly implements finetune/reports/20261125_sealed_review_dec_prereg_cn.md:
  1. Verifies live production parity gate on the cutoff date
     (score_hidden_corr > 0.9999 and score_fused_corr > 0.9999).
  2. Evaluates all 10 pre-registered baselines/single-heads/candidates (with
     G32_C2_L45_AR16 permanently removed) using pure equal-weight cross-sectional
     rank averaging (zero OOS weight tuning).
  3. Reports dual time scopes:
     - Scope A (STRICT_SEALED): pure unseen incremental dates
       (D10 asof >= 2026-11-09; D5 asof >= 2026-11-16).
     - Scope B (CUMULATIVE): cumulative multi-regime window (asof >= 2026-10-26).
  4. Stratifies dates into mutually exclusive ex-post market regimes while
     keeping ex-ante causal gating (breadth_ma20 < 0.30) strictly separated:
     - NORMAL_ACTIVE: breadth(asof) >= 0.30, univ_ret in (-0.05, +0.08],
       min_delta_breadth_5d > -0.20 within [asof, target].
     - EXTREME_ACTIVE: breadth(asof) >= 0.30 and shock condition triggered.
     - DEFENSIVE_LT30: breadth(asof) < 0.30.
  5. Enforces prerequisite sample gates (with auto-defer status), computes the
     100-point regime-weighted rubric (65% Normal / 30% Extreme / 5% Defensive),
     checks all 3 red-line vetoes, and outputs deterministic A/B/C decisions
     with lightweight tie-breaking (within 5 points).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEALED_DIR = ROOT / "scratch" / "sealed_dec_review"
DEFAULT_OUTPUT = ROOT / "finetune" / "reports" / "202612_sealed_review_dec_results.json"

# Pre-registered 10-model evaluation universe (G32_C2_L45_AR16 excluded).
ENSEMBLES: dict[str, list[str]] = {
    "FUSE_VAL_1E4": ["r_c2", "r_p_val_1e4"],
    "CAND_C2_G27_RESID": ["r_c2", "r_p_g27_resid_mlp_cosic"],
    "CAND_C2_G30_RESID": ["r_c2", "r_p_g30_quad_resid_cosic"],
    "CAND_C2_G30_LGBMRET5": ["r_c2", "r_p_g30_quad_lgbm_ret5"],
    "CAND_C2_RESID_MINIDEP": [
        "r_c2",
        "r_p_g27_resid_mlp_cosic",
        "r_p_g30_quad_resid_cosic",
        "r_p_g31_mini_l45_dep_s2",
    ],
    "ENS_W2F_C2_MINI29": ["r_c2", "r_p_g29_mini_l45_pool_regime"],
    "W2_CHAMP_FILM27": [
        "r_c2",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
    ],
}

SINGLE_REFS: list[str] = [
    "p_g27_resid_mlp_cosic",
    "p_g30_quad_resid_cosic",
]

# Lower complexity rank = preferred on tie (total score diff <= 5 points).
DEPLOYMENT_COMPLEXITY_RANK: dict[str, int] = {
    "ENS_W2F_C2_MINI29": 1,
    "CAND_C2_G27_RESID": 2,
    "SINGLE_G27_RESID_MLP_COSIC": 2,
    "CAND_C2_G30_RESID": 3,
    "SINGLE_G30_QUAD_RESID_COSIC": 3,
    "CAND_C2_G30_LGBMRET5": 3,
    "W2_CHAMP_FILM27": 4,
    "CAND_C2_RESID_MINIDEP": 5,
}

CANDIDATE_KEYS: list[str] = [
    "CAND_C2_G27_RESID",
    "CAND_C2_G30_RESID",
    "CAND_C2_G30_LGBMRET5",
    "CAND_C2_RESID_MINIDEP",
    "ENS_W2F_C2_MINI29",
    "W2_CHAMP_FILM27",
]


@dataclass(frozen=True)
class ProtocolConfig:
    parity_corr_min: float = 0.9999
    min_cum_d10_dates: int = 20
    min_strict_d10_dates: int = 10
    min_normal_active_d10_dates: int = 8
    breadth_active_threshold: float = 0.30
    normal_ret_low: float = -0.05
    normal_ret_high: float = 0.08
    extreme_breadth_5d_drop: float = -0.20
    strict_d10_start: str = "2026-11-09"
    strict_d5_start: str = "2026-11-16"
    cumulative_start: str = "2026-10-26"
    promote_score_min: float = 80.0
    shadow_score_min: float = 65.0
    tie_margin: float = 5.0


def _import_fusion_analysis():
    sys.path.insert(0, str(ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import fusion_analysis as fa  # noqa: E402

    return fa


def compute_breadth_map(kline: pd.DataFrame) -> dict[str, float]:
    """Compute daily market breadth (close * adj_factor > MA20) across all kline dates."""
    df = kline.sort_values(["code", "date"]).copy()
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"] * df["adj_factor"]
    df["ma20"] = df.groupby("code")["adj_close"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    valid = df.dropna(subset=["ma20"])
    if valid.empty:
        return {}
    return (
        valid.groupby("date")
        .apply(lambda g: float((g["adj_close"] > g["ma20"]).mean()), include_groups=False)
        .to_dict()
    )


def compute_min_delta_breadth_5d(
    asof: str,
    horizon: int,
    cal: list[str],
    cal_idx: dict[str, int],
    b_map: dict[str, float],
) -> float:
    """Minimum 5-trading-day breadth change within [asof, target] (in percentage points)."""
    start_i = cal_idx[asof]
    end_i = start_i + horizon
    window_dates = cal[start_i : end_i + 1]
    if len(window_dates) <= 5:
        return float("nan")
    deltas: list[float] = []
    for k in range(5, len(window_dates)):
        d_cur = window_dates[k]
        d_prev = window_dates[k - 5]
        if d_cur in b_map and d_prev in b_map:
            deltas.append(float(b_map[d_cur] - b_map[d_prev]))
    return min(deltas) if deltas else float("nan")


def classify_dates_by_regime(
    valid_dates: list[str],
    univ_ret_by_date: dict[str, float],
    cal: list[str],
    cal_idx: dict[str, int],
    b_map: dict[str, float],
    horizon: int,
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    """Partition signal dates into NORMAL_ACTIVE, EXTREME_ACTIVE, and DEFENSIVE_LT30."""
    buckets: dict[str, list[str]] = {
        "NORMAL_ACTIVE": [],
        "EXTREME_ACTIVE": [],
        "DEFENSIVE_LT30": [],
    }
    diagnostics: dict[str, dict[str, Any]] = {}

    for d in valid_dates:
        b0 = float(b_map[d])
        u_ret = float(univ_ret_by_date[d])
        min_db5 = compute_min_delta_breadth_5d(d, horizon, cal, cal_idx, b_map)

        if b0 < cfg.breadth_active_threshold:
            regime = "DEFENSIVE_LT30"
            reason = "breadth_lt_0.30"
        else:
            shock_reasons: list[str] = []
            if u_ret <= cfg.normal_ret_low:
                shock_reasons.append(f"univ_ret<={cfg.normal_ret_low:.2f}")
            if u_ret > cfg.normal_ret_high:
                shock_reasons.append(f"univ_ret>{cfg.normal_ret_high:.2f}")
            if not math.isnan(min_db5) and min_db5 <= cfg.extreme_breadth_5d_drop:
                shock_reasons.append(f"min_dB5<={cfg.extreme_breadth_5d_drop:.2f}")

            if shock_reasons:
                regime = "EXTREME_ACTIVE"
                reason = ",".join(shock_reasons)
            else:
                regime = "NORMAL_ACTIVE"
                reason = "normal_active"

        buckets[regime].append(d)
        diagnostics[d] = {
            "regime": regime,
            "reason": reason,
            "breadth_asof": round(b0, 4),
            "univ_ret": round(u_ret, 6),
            "min_delta_breadth_5d": None if math.isnan(min_db5) else round(min_db5, 4),
        }

    return {"buckets": buckets, "by_date": diagnostics}


def _eval_subset(
    sub: pd.DataFrame,
    models: dict[str, str],
    b_map: dict[str, float],
    date_filter: list[str] | None,
    fa: Any,
) -> dict[str, Any]:
    df = sub[sub["asof"].isin(date_filter)].copy() if date_filter is not None else sub.copy()
    dates = sorted(df["asof"].unique())
    if not dates:
        return {"n_dates": 0, "dates": [], "n_rows": 0, "univ_mean_ret": None, "models": {}}

    liq = df[df["med_amt_20d"] >= 5e4].copy()
    df["resid_actual"] = fa.residualize(df, "actual_return", ["size_decile", "rev10"], sector=True)
    df_fa = df.rename(columns={"actual_return": "actual_return_d10"})
    base_c2_ic = fa.daily_ic(df_fa, models["C2"])
    base_val_ic = fa.daily_ic(df_fa, models["FUSE_VAL_1E4"])
    univ_daily = df.groupby("asof", sort=False)["actual_return"].mean()

    res: dict[str, Any] = {}
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

        nic_std = float(nic_s.std(ddof=1)) if len(nic_s) > 1 else float("nan")
        nic_ir = float(nic_s.mean() / nic_std) if len(nic_s) > 1 and nic_std > 0 else float("nan")

        entry: dict[str, Any] = {
            **fa.ic_stats(ic_s),
            "neutral_ic": float(nic_s.mean()),
            "neutral_icir": nic_ir,
            "top10pct_vs_univ": float(t10u.mean()),
            "top10pct_pos_days": int((t10u > 0).sum()),
            "liq_top10_excess": float(t10_ex.mean()),
            "liq_top10_pos_days": int((t10_ex > 0).sum()),
            "liq_top20_excess": float(t20_ex.mean()),
            "liq_top20_pos_days": int((t20_ex > 0).sum()),
            "daily_ic": {str(k): round(float(v), 6) for k, v in ic_s.items()},
        }
        if name != "C2" and len(dates) >= 3:
            entry["paired_vs_C2"] = fa.paired_bootstrap(
                base_c2_ic, ic_s, block=min(3, len(dates)), seed=20261125
            )
        if name != "FUSE_VAL_1E4" and len(dates) >= 3:
            entry["paired_vs_VAL1E4"] = fa.paired_bootstrap(
                base_val_ic, ic_s, block=min(3, len(dates)), seed=20261125
            )
        res[name] = entry

    return {
        "n_dates": len(dates),
        "dates": dates,
        "breadth_by_date": {d: round(b_map[d], 4) for d in dates},
        "n_rows": int(len(df)),
        "univ_mean_ret": float(univ_daily.mean()),
        "models": res,
    }


def evaluate_horizon_scope(
    merged: pd.DataFrame,
    kline: pd.DataFrame,
    cal: list[str],
    cal_idx: dict[str, int],
    b_map: dict[str, float],
    horizon: int,
    start_date: str,
    models: dict[str, str],
    cfg: ProtocolConfig,
    fa: Any,
) -> dict[str, Any]:
    valid_dates = [
        d
        for d in sorted(merged["asof"].unique())
        if d >= start_date and d in cal_idx and cal_idx[d] + horizon < len(cal)
    ]
    if not valid_dates:
        return {
            "horizon_days": horizon,
            "start_date": start_date,
            "last_target_date": None,
            "regime_map": {},
            "ALL": {"n_dates": 0, "models": {}},
            "NORMAL_ACTIVE": {"n_dates": 0, "models": {}},
            "EXTREME_ACTIVE": {"n_dates": 0, "models": {}},
            "DEFENSIVE_LT30": {"n_dates": 0, "models": {}},
        }

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

    univ_ret_by_date = sub.groupby("asof")["actual_return"].mean().to_dict()
    reg_info = classify_dates_by_regime(
        valid_dates=valid_dates,
        univ_ret_by_date=univ_ret_by_date,
        cal=cal,
        cal_idx=cal_idx,
        b_map=b_map,
        horizon=horizon,
        cfg=cfg,
    )
    buckets = reg_info["buckets"]

    return {
        "horizon_days": horizon,
        "start_date": start_date,
        "last_target_date": cal[cal_idx[valid_dates[-1]] + horizon],
        "regime_map": reg_info["by_date"],
        "ALL": _eval_subset(sub, models, b_map, None, fa),
        "NORMAL_ACTIVE": _eval_subset(sub, models, b_map, buckets["NORMAL_ACTIVE"], fa),
        "EXTREME_ACTIVE": _eval_subset(sub, models, b_map, buckets["EXTREME_ACTIVE"], fa),
        "DEFENSIVE_LT30": _eval_subset(sub, models, b_map, buckets["DEFENSIVE_LT30"], fa),
    }


def check_prerequisites(
    parity: dict[str, Any],
    scope_a_d10: dict[str, Any],
    scope_b_d10: dict[str, Any],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    """Verify Parity Gate and minimum sample sizes; trigger auto-defer if unmet."""
    failures: list[str] = []

    h_corr = parity.get("score_hidden_corr", float("nan"))
    f_corr = parity.get("score_fused_corr", float("nan"))
    parity_ok = (
        not math.isnan(h_corr)
        and not math.isnan(f_corr)
        and h_corr > cfg.parity_corr_min
        and f_corr > cfg.parity_corr_min
    )
    if not parity_ok:
        failures.append(
            f"parity_gate_failed(hidden={h_corr:.6f}, fused={f_corr:.6f}, req>{cfg.parity_corr_min})"
        )

    cum_d10 = int(scope_b_d10["ALL"]["n_dates"])
    strict_d10 = int(scope_a_d10["ALL"]["n_dates"])
    norm_d10 = int(scope_b_d10["NORMAL_ACTIVE"]["n_dates"])

    if cum_d10 < cfg.min_cum_d10_dates:
        failures.append(f"cumulative_d10_dates={cum_d10} < {cfg.min_cum_d10_dates}")
    if strict_d10 < cfg.min_strict_d10_dates:
        failures.append(f"strict_d10_dates={strict_d10} < {cfg.min_strict_d10_dates}")
    if norm_d10 < cfg.min_normal_active_d10_dates:
        failures.append(f"normal_active_d10_dates={norm_d10} < {cfg.min_normal_active_d10_dates}")

    eligible = len(failures) == 0
    return {
        "eligible_for_decision": eligible,
        "status": "READY_FOR_ADJUDICATION" if eligible else "DEFER_AUTO_EXTEND",
        "parity_ok": parity_ok,
        "counts": {
            "cumulative_d10_dates": cum_d10,
            "strict_d10_dates": strict_d10,
            "cumulative_normal_active_d10_dates": norm_d10,
            "cumulative_extreme_active_d10_dates": int(scope_b_d10["EXTREME_ACTIVE"]["n_dates"]),
            "cumulative_defensive_d10_dates": int(scope_b_d10["DEFENSIVE_LT30"]["n_dates"]),
        },
        "failures": failures,
    }


def _get_m(block: dict[str, Any], name: str) -> dict[str, Any] | None:
    if block.get("n_dates", 0) == 0:
        return None
    return block.get("models", {}).get(name)


def score_candidate(
    name: str,
    scope_a: dict[str, Any],
    scope_b: dict[str, Any],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    """Compute the 100-point rubric and red-line veto checks for one candidate."""
    b_d10_norm = scope_b["D10"]["NORMAL_ACTIVE"]
    b_d5_norm = scope_b["D5"]["NORMAL_ACTIVE"]
    b_d10_ext = scope_b["D10"]["EXTREME_ACTIVE"]
    b_d5_ext = scope_b["D5"]["EXTREME_ACTIVE"]
    b_d10_def = scope_b["D10"]["DEFENSIVE_LT30"]
    a_d10_norm = scope_a["D10"]["NORMAL_ACTIVE"]

    items: dict[str, dict[str, Any]] = {}

    def _record(key: str, weight: float, mult: float, note: str) -> None:
        items[key] = {
            "weight": weight,
            "tier_mult": mult,
            "score": round(weight * mult, 2),
            "note": note,
        }

    # --- Normal Active (65 pts) ---
    m_n = _get_m(b_d10_norm, name)
    c2_n = _get_m(b_d10_norm, "C2")
    val_n = _get_m(b_d10_norm, "FUSE_VAL_1E4")
    if m_n and c2_n and val_n:
        # 1. D10 IC & ICIR (25)
        if m_n["ic"] > c2_n["ic"] and m_n["ic"] >= val_n["ic"] and m_n["icir"] >= val_n["icir"]:
            _record("normal_d10_ic_icir", 25, 1.0, "beats C2 and >= VAL in IC & ICIR")
        elif m_n["ic"] > c2_n["ic"] and (val_n["ic"] - m_n["ic"]) <= 0.003 and m_n["icir"] > val_n["icir"]:
            _record("normal_d10_ic_icir", 25, 0.5, "within 0.003 of VAL IC with higher ICIR")
        else:
            _record("normal_d10_ic_icir", 25, 0.0, "below C2 or trails VAL IC/ICIR")

        # 2. Normal NIC (15)
        if m_n["neutral_ic"] > c2_n["neutral_ic"] and (m_n["neutral_ic"] - val_n["neutral_ic"]) >= 0.005:
            _record("normal_d10_nic", 15, 1.0, "beats VAL NIC by >= +0.005")
        elif m_n["neutral_ic"] > c2_n["neutral_ic"] and abs(m_n["neutral_ic"] - val_n["neutral_ic"]) <= 0.005:
            _record("normal_d10_nic", 15, 0.5, "beats C2 and within +-0.005 of VAL NIC")
        else:
            _record("normal_d10_nic", 15, 0.0, "trails C2 or lags VAL NIC by > 0.005")

        # 3. Normal liq10/liq20 (15)
        m_liq_avg = 0.5 * (m_n["liq_top10_excess"] + m_n["liq_top20_excess"])
        val_liq_avg = 0.5 * (val_n["liq_top10_excess"] + val_n["liq_top20_excess"])
        if (
            m_n["liq_top10_excess"] > 0
            and m_n["liq_top20_excess"] > 0
            and (m_liq_avg - val_liq_avg) >= 0.005
        ):
            _record("normal_d10_liq", 15, 1.0, "both liq10/20 > 0 and avg >= VAL +0.50%")
        elif (
            m_n["liq_top10_excess"] > 0
            and m_n["liq_top20_excess"] > 0
            and m_n["liq_top10_excess"] >= val_n["liq_top10_excess"]
            and m_n["liq_top20_excess"] >= val_n["liq_top20_excess"]
        ):
            _record("normal_d10_liq", 15, 0.5, "both liq10/20 > 0 and >= VAL")
        else:
            _record("normal_d10_liq", 15, 0.0, "negative liq excess or below VAL")
    else:
        _record("normal_d10_ic_icir", 25, 0.0, "missing normal active D10 data")
        _record("normal_d10_nic", 15, 0.0, "missing normal active D10 data")
        _record("normal_d10_liq", 15, 0.0, "missing normal active D10 data")

    # 4. Normal win-rate & D5 consistency (10)
    m_d5_n = _get_m(b_d5_norm, name)
    if m_n and m_d5_n:
        p10, p5 = m_n["positive_ratio"], m_d5_n["positive_ratio"]
        if p10 >= 0.70 and p5 >= 0.70:
            _record("normal_winrate_d5_consistency", 10, 1.0, f"D10={p10:.1%}, D5={p5:.1%} both >=70%")
        elif (p10 >= 0.70 and p5 >= 0.60) or (p5 >= 0.70 and p10 >= 0.60):
            _record("normal_winrate_d5_consistency", 10, 0.5, f"D10={p10:.1%}, D5={p5:.1%}")
        else:
            _record("normal_winrate_d5_consistency", 10, 0.0, f"D10={p10:.1%}, D5={p5:.1%}")
    else:
        _record("normal_winrate_d5_consistency", 10, 0.0, "missing normal D10/D5 data")

    # --- Extreme Active (30 pts) ---
    m_e = _get_m(b_d10_ext, name)
    val_e = _get_m(b_d10_ext, "FUSE_VAL_1E4")
    if m_e and val_e:
        # 5. Extreme liq10/liq20 floor (15)
        if (
            m_e["liq_top10_excess"] > val_e["liq_top10_excess"]
            and m_e["liq_top20_excess"] > val_e["liq_top20_excess"]
            and m_e["liq_top10_excess"] >= 0.0
        ):
            _record("extreme_d10_liq_floor", 15, 1.0, "beats VAL on liq10/20 and liq10 >= 0%")
        elif (
            m_e["liq_top10_excess"] > val_e["liq_top10_excess"]
            and m_e["liq_top10_excess"] >= -0.01
        ):
            _record("extreme_d10_liq_floor", 15, 0.5, "beats VAL on liq10 and liq10 >= -1.0%")
        else:
            _record("extreme_d10_liq_floor", 15, 0.0, "liq10 < -1.0% or below VAL")

        # 6. Extreme NIC resilience (10)
        if m_e["neutral_ic"] >= 0.05 and m_e["neutral_ic"] > val_e["neutral_ic"]:
            _record("extreme_d10_nic_resilience", 10, 1.0, "NIC >= +0.05 and > VAL")
        elif m_e["neutral_ic"] > 0.0:
            _record("extreme_d10_nic_resilience", 10, 0.5, "NIC > 0")
        else:
            _record("extreme_d10_nic_resilience", 10, 0.0, "NIC <= 0")
    else:
        _record("extreme_d10_liq_floor", 15, 0.0, "missing extreme D10 data")
        _record("extreme_d10_nic_resilience", 10, 0.0, "missing extreme D10 data")

    # 7. Extreme D5 anti-whipsaw (5)
    m_d5_e = _get_m(b_d5_ext, name)
    c2_d5_e = _get_m(b_d5_ext, "C2")
    if m_d5_e and c2_d5_e:
        if m_d5_e["ic"] >= 0.05 and m_d5_e["ic"] >= c2_d5_e["ic"]:
            _record("extreme_d5_anti_whipsaw", 5, 1.0, "D5 IC >= +0.05 and >= C2")
        elif m_d5_e["ic"] > 0.0:
            _record("extreme_d5_anti_whipsaw", 5, 0.5, "D5 IC > 0")
        else:
            _record("extreme_d5_anti_whipsaw", 5, 0.0, "D5 IC <= 0")
    else:
        _record("extreme_d5_anti_whipsaw", 5, 0.0, "missing extreme D5 data")

    # --- Defensive Safety Cushion (5 pts) ---
    m_def = _get_m(b_d10_def, name)
    if m_def:
        if m_def["ic"] >= 0.0:
            _record("defensive_d10_cushion", 5, 1.0, "defensive D10 IC >= 0")
        elif m_def["ic"] > -0.02:
            _record("defensive_d10_cushion", 5, 0.5, "defensive D10 IC in (-0.02, 0)")
        else:
            _record("defensive_d10_cushion", 5, 0.0, "defensive D10 IC <= -0.02")
    else:
        _record("defensive_d10_cushion", 5, 0.5, "no defensive days observed (neutral)")

    total_score = round(sum(v["score"] for v in items.values()), 2)

    # --- Red-line vetoes (Section 5.3) ---
    vetoes: list[str] = []
    m_a_n = _get_m(a_d10_norm, name)
    val_a_n = _get_m(a_d10_norm, "FUSE_VAL_1E4")
    strict_normal_ok_for_promote = False
    if m_a_n and val_a_n:
        if (val_a_n["ic"] - m_a_n["ic"]) > 0.010:
            vetoes.append("redline_1_strict_normal_ic_lag_gt_0.010")
        if m_a_n["liq_top10_excess"] < 0.0 and m_a_n["liq_top10_excess"] < val_a_n["liq_top10_excess"]:
            vetoes.append("redline_1_strict_normal_liq10_neg_and_below_val")

        non_inferior = (
            m_a_n["ic"] >= val_a_n["ic"]
            and m_a_n["neutral_ic"] >= val_a_n["neutral_ic"]
            and m_a_n["liq_top10_excess"] >= val_a_n["liq_top10_excess"]
        )
        strict_wins = sum(
            [
                m_a_n["ic"] > val_a_n["ic"],
                m_a_n["neutral_ic"] > val_a_n["neutral_ic"],
                m_a_n["liq_top10_excess"] > val_a_n["liq_top10_excess"],
            ]
        )
        strict_normal_ok_for_promote = non_inferior and strict_wins >= 2

    if m_e:
        if m_e["liq_top10_excess"] < -0.015:
            vetoes.append("redline_2_extreme_liq10_lt_neg1.5pct")
        if m_e["neutral_ic"] <= 0.0:
            vetoes.append("redline_2_extreme_nic_le_0")

    if m_d5_e and m_d5_e["ic"] <= -0.02:
        vetoes.append("redline_3_extreme_d5_ic_le_neg0.02")

    # --- Tier classification (Section 6) ---
    if vetoes or total_score < cfg.shadow_score_min:
        tier = "C_HOLD"
    elif total_score >= cfg.promote_score_min and strict_normal_ok_for_promote:
        tier = "A_PROMOTE"
    else:
        tier = "B_SHADOW_CANARY"

    return {
        "candidate": name,
        "total_score": total_score,
        "deployment_complexity_rank": DEPLOYMENT_COMPLEXITY_RANK.get(name, 99),
        "strict_normal_ok_for_promote": strict_normal_ok_for_promote,
        "vetoes": vetoes,
        "tier": tier,
        "breakdown": items,
    }


def adjudicate_candidates(
    prereq: dict[str, Any],
    scope_a: dict[str, Any],
    scope_b: dict[str, Any],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    """Score all locked candidates and apply lightweight tie-breaking."""
    scorecards = [score_candidate(name, scope_a, scope_b, cfg) for name in CANDIDATE_KEYS]
    scorecards.sort(key=lambda x: (-x["total_score"], x["deployment_complexity_rank"]))

    if not prereq["eligible_for_decision"]:
        return {
            "final_action": "DEFER_AUTO_EXTEND",
            "selected_candidate": None,
            "reason": "; ".join(prereq["failures"]),
            "scorecards": scorecards,
        }

    promotable = [s for s in scorecards if s["tier"] == "A_PROMOTE"]
    if promotable:
        best_score = promotable[0]["total_score"]
        within_tie = [s for s in promotable if (best_score - s["total_score"]) <= cfg.tie_margin]
        within_tie.sort(key=lambda x: (x["deployment_complexity_rank"], -x["total_score"]))
        winner = within_tie[0]
        return {
            "final_action": "A_PROMOTE",
            "selected_candidate": winner["candidate"],
            "reason": (
                f"score={winner['total_score']} >= {cfg.promote_score_min}, "
                f"complexity_rank={winner['deployment_complexity_rank']} (tie_margin={cfg.tie_margin})"
            ),
            "scorecards": scorecards,
        }

    shadowable = [s for s in scorecards if s["tier"] == "B_SHADOW_CANARY"]
    if shadowable:
        best_score = shadowable[0]["total_score"]
        within_tie = [s for s in shadowable if (best_score - s["total_score"]) <= cfg.tie_margin]
        within_tie.sort(key=lambda x: (x["deployment_complexity_rank"], -x["total_score"]))
        winner = within_tie[0]
        return {
            "final_action": "B_SHADOW_CANARY",
            "selected_candidate": winner["candidate"],
            "reason": f"score={winner['total_score']} in [{cfg.shadow_score_min}, {cfg.promote_score_min}) or strict_normal partial",
            "scorecards": scorecards,
        }

    return {
        "final_action": "C_HOLD",
        "selected_candidate": "FUSE_VAL_1E4",
        "reason": "No candidate cleared shadow/promote thresholds without red-line vetoes.",
        "scorecards": scorecards,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dec Sealed Review Multi-Regime Evaluator")
    p.add_argument("--sealed-dir", type=Path, default=DEFAULT_SEALED_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--parity-date", type=str, default=None, help="Override parity check date (default: latest asof)")
    p.add_argument("--strict-d10-start", type=str, default="2026-11-09")
    p.add_argument("--strict-d5-start", type=str, default="2026-11-16")
    p.add_argument("--cumulative-start", type=str, default="2026-10-26")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = ProtocolConfig(
        strict_d10_start=args.strict_d10_start,
        strict_d5_start=args.strict_d5_start,
        cumulative_start=args.cumulative_start,
    )
    fa = _import_fusion_analysis()

    ranks = pd.read_parquet(args.sealed_dir / "sealed_shadow_rankings.parquet")
    kline = pd.read_parquet(args.sealed_dir / "sealed_kline_150d.parquet")
    hid = pd.read_parquet(args.sealed_dir / "sealed_w2w3_hidden_scores.parquet")

    ranks["asof"] = ranks["asof"].astype(str)
    ranks["code"] = ranks["code"].astype(str)
    ranks["identity"] = ranks["asof"] + "|" + ranks["code"]
    kline["date"] = kline["date"].astype(str)
    kline["code"] = kline["code"].astype(str)

    hid_cols = [c for c in hid.columns if c.startswith("p_")]
    merged = ranks.merge(hid[["identity"] + hid_cols], on="identity", how="inner", validate="one_to_one")
    if len(merged) != len(ranks):
        raise RuntimeError(f"Alignment mismatch: {len(merged)} vs {len(ranks)}")

    kline = kline.sort_values(["code", "date"]).copy()
    kline["adj_close"] = kline["close"] * kline["adj_factor"]
    b_map = compute_breadth_map(kline)

    traded = kline[kline["volume"] > 0].sort_values(["code", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("code")["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).median()
    )
    traded["rev10_raw"] = traded.groupby("code")["adj_close"].transform(lambda c: c / c.shift(10) - 1.0)
    feat_map = traded[["code", "date", "med_amt_20d", "rev10_raw"]].rename(columns={"date": "asof"})
    merged = merged.merge(feat_map, on=["code", "asof"], how="left")
    merged["rev10"] = merged.groupby("asof")["rev10_raw"].rank(pct=True, method="average")

    # Parity Gate on cutoff date (default: latest asof date in panel)
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
        "score_fused_max_abs_diff": float((dp["score_fused"] - dp["f_val_chk"]).abs().max()),
    }

    # Pre-registered equal-weight cross-sectional ranks
    rank_cols = {"r_c2": merged.groupby("asof")["predicted_return_d10"].rank(pct=True, method="average")}
    for c in hid_cols:
        rank_cols[f"r_{c}"] = merged.groupby("asof")[c].rank(pct=True, method="average")
    merged = pd.concat([merged, pd.DataFrame(rank_cols, index=merged.index)], axis=1)

    ens_frame = {name: merged[members].mean(axis=1) for name, members in ENSEMBLES.items()}
    merged = pd.concat([merged, pd.DataFrame(ens_frame, index=merged.index)], axis=1)

    models: dict[str, str] = {"C2": "predicted_return_d10"}
    for name in ENSEMBLES:
        models[name] = name
    for sc in SINGLE_REFS:
        models[f"SINGLE_{sc[2:].upper()}"] = sc

    cal = sorted(kline["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    scope_a = {
        "D10": evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 10, cfg.strict_d10_start, models, cfg, fa),
        "D5": evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 5, cfg.strict_d5_start, models, cfg, fa),
    }
    scope_b = {
        "D10": evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 10, cfg.cumulative_start, models, cfg, fa),
        "D5": evaluate_horizon_scope(merged, kline, cal, cal_idx, b_map, 5, cfg.cumulative_start, models, cfg, fa),
    }

    prereq = check_prerequisites(parity, scope_a["D10"], scope_b["D10"], cfg)
    decision = adjudicate_candidates(prereq, scope_a, scope_b, cfg)

    out = {
        "protocol": "20261125_sealed_review_dec_prereg_cn.md",
        "parity_gate": parity,
        "prerequisites": prereq,
        "decision": decision,
        "ensembles": ENSEMBLES,
        "SCOPE_A_STRICT_SEALED": scope_a,
        "SCOPE_B_CUMULATIVE": scope_b,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote:", args.output)
    print("PARITY:", json.dumps(parity, ensure_ascii=False))
    print("PREREQUISITES:", json.dumps(prereq, ensure_ascii=False))
    print(f"DECISION: {decision['final_action']} -> {decision['selected_candidate']} ({decision['reason']})")
    print(f"\n{'CANDIDATE':<26} {'SCORE':>6} {'TIER':<16} {'COMPLEX':>7} {'VETOES'}")
    for sc in decision["scorecards"]:
        print(
            f"{sc['candidate']:<26} {sc['total_score']:6.2f} {sc['tier']:<16} "
            f"{sc['deployment_complexity_rank']:7d} {','.join(sc['vetoes']) or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
