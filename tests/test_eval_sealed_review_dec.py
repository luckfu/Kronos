"""Unit tests for Dec multi-regime sealed review evaluator (eval_sealed_review_dec.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

import eval_sealed_review_dec as ev  # noqa: E402


def _mk_metrics(
    ic: float = 0.10,
    icir: float = 1.0,
    pos: float = 0.75,
    nic: float = 0.08,
    liq10: float = 0.01,
    liq20: float = 0.01,
) -> dict[str, float]:
    return {
        "ic": ic,
        "icir": icir,
        "positive_ratio": pos,
        "neutral_ic": nic,
        "liq_top10_excess": liq10,
        "liq_top20_excess": liq20,
    }


def _make_scope_fixture(
    overrides_b_norm: dict[str, dict[str, float]] | None = None,
    overrides_b_ext: dict[str, dict[str, float]] | None = None,
    overrides_b_d5_ext: dict[str, dict[str, float]] | None = None,
    overrides_a_norm: dict[str, dict[str, float]] | None = None,
    overrides_b_def: dict[str, dict[str, float]] | None = None,
) -> tuple[dict, dict]:
    base = {
        "C2": _mk_metrics(ic=0.07, icir=0.80, pos=0.65, nic=0.05, liq10=0.002, liq20=0.002),
        "FUSE_VAL_1E4": _mk_metrics(ic=0.09, icir=1.00, pos=0.70, nic=0.07, liq10=0.005, liq20=0.005),
    }
    for name in ev.CANDIDATE_KEYS:
        base[name] = _mk_metrics(ic=0.06, icir=0.70, pos=0.55, nic=0.04, liq10=-0.02, liq20=-0.01)

    def _apply(ov: dict[str, dict[str, float]] | None) -> dict[str, dict[str, float]]:
        d = {k: dict(v) for k, v in base.items()}
        if ov:
            for k, v in ov.items():
                d[k] = dict(v)
        return d

    b_norm = {"n_dates": 10, "models": _apply(overrides_b_norm)}
    b_ext = {"n_dates": 5, "models": _apply(overrides_b_ext)}
    b_d5_ext = {"n_dates": 5, "models": _apply(overrides_b_d5_ext)}
    b_def = {"n_dates": 5, "models": _apply(overrides_b_def)}
    a_norm = {"n_dates": 10, "models": _apply(overrides_a_norm)}

    scope_a = {
        "D10": {"ALL": a_norm, "NORMAL_ACTIVE": a_norm, "EXTREME_ACTIVE": b_ext, "DEFENSIVE_LT30": b_def},
        "D5": {"ALL": a_norm, "NORMAL_ACTIVE": a_norm, "EXTREME_ACTIVE": b_d5_ext, "DEFENSIVE_LT30": b_def},
    }
    scope_b = {
        "D10": {
            "ALL": {"n_dates": 20, "models": b_norm["models"]},
            "NORMAL_ACTIVE": b_norm,
            "EXTREME_ACTIVE": b_ext,
            "DEFENSIVE_LT30": b_def,
        },
        "D5": {
            "ALL": {"n_dates": 20, "models": b_norm["models"]},
            "NORMAL_ACTIVE": b_norm,
            "EXTREME_ACTIVE": b_d5_ext,
            "DEFENSIVE_LT30": b_def,
        },
    }
    return scope_a, scope_b


def test_prereg_candidates_exclude_g32_l45_ar() -> None:
    for ens_name, members in ev.ENSEMBLES.items():
        for m in members:
            assert "g32" not in m.lower(), f"G32 component leaked into {ens_name}: {m}"
            assert "l45_ar" not in m.lower(), f"L45 AR component leaked into {ens_name}: {m}"
    for cand in ev.CANDIDATE_KEYS:
        assert cand in ev.ENSEMBLES
        assert cand in ev.DEPLOYMENT_COMPLEXITY_RANK


def test_classify_dates_by_regime_all_branches() -> None:
    cfg = ev.ProtocolConfig()
    cal = [f"2026-11-{i:02d}" for i in range(1, 28)]
    cal_idx = {d: i for i, d in enumerate(cal)}
    b_map = {d: 0.42 for d in cal}
    b_map["2026-11-02"] = 0.24  # defensive (< 0.30)
    b_map["2026-11-12"] = 0.57  # V-shape peak
    b_map["2026-11-17"] = 0.09  # 5d breadth drop = -0.48 <= -0.20

    univ_ret = {
        "2026-11-01": 0.015,   # NORMAL_ACTIVE
        "2026-11-02": 0.010,   # DEFENSIVE_LT30
        "2026-11-03": -0.055,  # EXTREME_ACTIVE (univ_ret <= -5%)
        "2026-11-04": 0.095,   # EXTREME_ACTIVE (univ_ret > +8%)
        "2026-11-08": 0.005,   # EXTREME_ACTIVE (min_dB5 <= -0.20 in [11-08, 11-18])
    }
    res = ev.classify_dates_by_regime(
        valid_dates=list(univ_ret.keys()),
        univ_ret_by_date=univ_ret,
        cal=cal,
        cal_idx=cal_idx,
        b_map=b_map,
        horizon=10,
        cfg=cfg,
    )
    by_d = res["by_date"]
    assert by_d["2026-11-01"]["regime"] == "NORMAL_ACTIVE"
    assert by_d["2026-11-02"]["regime"] == "DEFENSIVE_LT30"
    assert by_d["2026-11-03"]["regime"] == "EXTREME_ACTIVE"
    assert "univ_ret<=-0.05" in by_d["2026-11-03"]["reason"]
    assert by_d["2026-11-04"]["regime"] == "EXTREME_ACTIVE"
    assert "univ_ret>0.08" in by_d["2026-11-04"]["reason"]
    assert by_d["2026-11-08"]["regime"] == "EXTREME_ACTIVE"
    assert "min_dB5<=-0.20" in by_d["2026-11-08"]["reason"]


def test_check_prerequisites_parity_and_auto_defer() -> None:
    cfg = ev.ProtocolConfig()
    good_parity = {"score_hidden_corr": 0.99999, "score_fused_corr": 0.99999}
    bad_parity = {"score_hidden_corr": 0.9980, "score_fused_corr": 0.99999}

    scope_a_ok = {"ALL": {"n_dates": 10}}
    scope_b_ok = {
        "ALL": {"n_dates": 20},
        "NORMAL_ACTIVE": {"n_dates": 8},
        "EXTREME_ACTIVE": {"n_dates": 5},
        "DEFENSIVE_LT30": {"n_dates": 7},
    }
    res_ok = ev.check_prerequisites(good_parity, scope_a_ok, scope_b_ok, cfg)
    assert res_ok["eligible_for_decision"] is True
    assert res_ok["status"] == "READY_FOR_ADJUDICATION"

    res_bad_p = ev.check_prerequisites(bad_parity, scope_a_ok, scope_b_ok, cfg)
    assert res_bad_p["eligible_for_decision"] is False
    assert res_bad_p["status"] == "DEFER_AUTO_EXTEND"
    assert any("parity_gate_failed" in f for f in res_bad_p["failures"])

    scope_b_short_norm = {**scope_b_ok, "NORMAL_ACTIVE": {"n_dates": 7}}
    res_defer = ev.check_prerequisites(good_parity, {"ALL": {"n_dates": 9}}, scope_b_short_norm, cfg)
    assert res_defer["eligible_for_decision"] is False
    assert res_defer["status"] == "DEFER_AUTO_EXTEND"
    assert len(res_defer["failures"]) == 2


def test_score_candidate_tiers_and_redline_vetoes() -> None:
    cfg = ev.ProtocolConfig()
    strong = _mk_metrics(ic=0.12, icir=1.30, pos=0.80, nic=0.10, liq10=0.015, liq20=0.015)

    # 1. Full score -> A_PROMOTE
    sa, sb = _make_scope_fixture(
        overrides_b_norm={"ENS_W2F_C2_MINI29": strong},
        overrides_b_ext={"ENS_W2F_C2_MINI29": strong},
        overrides_b_d5_ext={"ENS_W2F_C2_MINI29": strong},
        overrides_a_norm={"ENS_W2F_C2_MINI29": strong},
        overrides_b_def={"ENS_W2F_C2_MINI29": strong},
    )
    sc = ev.score_candidate("ENS_W2F_C2_MINI29", sa, sb, cfg)
    assert sc["total_score"] == 100.0
    assert sc["vetoes"] == []
    assert sc["tier"] == "A_PROMOTE"

    # 2. High score on Scope B, but Scope A strict normal IC trails VAL by > 0.010 -> redline 1 veto -> C_HOLD
    weak_strict = _mk_metrics(ic=0.075, icir=1.0, pos=0.70, nic=0.08, liq10=0.01, liq20=0.01)
    sa2, sb2 = _make_scope_fixture(
        overrides_b_norm={"ENS_W2F_C2_MINI29": strong},
        overrides_b_ext={"ENS_W2F_C2_MINI29": strong},
        overrides_b_d5_ext={"ENS_W2F_C2_MINI29": strong},
        overrides_a_norm={"ENS_W2F_C2_MINI29": weak_strict},
        overrides_b_def={"ENS_W2F_C2_MINI29": strong},
    )
    sc2 = ev.score_candidate("ENS_W2F_C2_MINI29", sa2, sb2, cfg)
    assert "redline_1_strict_normal_ic_lag_gt_0.010" in sc2["vetoes"]
    assert sc2["tier"] == "C_HOLD"

    # 3. Extreme liq10 < -1.5% -> redline 2 veto -> C_HOLD
    crash_ext = _mk_metrics(ic=0.05, icir=0.5, pos=0.60, nic=0.03, liq10=-0.020, liq20=-0.010)
    sa3, sb3 = _make_scope_fixture(
        overrides_b_norm={"CAND_C2_G27_RESID": strong},
        overrides_b_ext={"CAND_C2_G27_RESID": crash_ext},
        overrides_b_d5_ext={"CAND_C2_G27_RESID": strong},
        overrides_a_norm={"CAND_C2_G27_RESID": strong},
        overrides_b_def={"CAND_C2_G27_RESID": strong},
    )
    sc3 = ev.score_candidate("CAND_C2_G27_RESID", sa3, sb3, cfg)
    assert "redline_2_extreme_liq10_lt_neg1.5pct" in sc3["vetoes"]
    assert sc3["tier"] == "C_HOLD"

    # 4. Extreme D5 IC <= -0.02 -> redline 3 veto -> C_HOLD
    whipsaw_d5 = _mk_metrics(ic=-0.025, icir=-0.3, pos=0.40, nic=0.01, liq10=0.005, liq20=0.005)
    sa4, sb4 = _make_scope_fixture(
        overrides_b_norm={"CAND_C2_G30_RESID": strong},
        overrides_b_ext={"CAND_C2_G30_RESID": strong},
        overrides_b_d5_ext={"CAND_C2_G30_RESID": whipsaw_d5},
        overrides_a_norm={"CAND_C2_G30_RESID": strong},
        overrides_b_def={"CAND_C2_G30_RESID": strong},
    )
    sc4 = ev.score_candidate("CAND_C2_G30_RESID", sa4, sb4, cfg)
    assert "redline_3_extreme_d5_ic_le_neg0.02" in sc4["vetoes"]
    assert sc4["tier"] == "C_HOLD"

    # 5. Shadow/Canary tier (score >= 65, no vetoes, but < 80)
    mid_norm = _mk_metrics(ic=0.088, icir=1.10, pos=0.72, nic=0.072, liq10=0.006, liq20=0.006)
    sa5, sb5 = _make_scope_fixture(
        overrides_b_norm={"CAND_C2_G27_RESID": mid_norm},
        overrides_b_ext={"CAND_C2_G27_RESID": strong},
        overrides_b_d5_ext={"CAND_C2_G27_RESID": strong},
        overrides_a_norm={"CAND_C2_G27_RESID": mid_norm},
        overrides_b_def={"CAND_C2_G27_RESID": strong},
    )
    sc5 = ev.score_candidate("CAND_C2_G27_RESID", sa5, sb5, cfg)
    assert 65.0 <= sc5["total_score"] < 80.0
    assert sc5["vetoes"] == []
    assert sc5["tier"] == "B_SHADOW_CANARY"


def test_adjudicate_lightweight_tie_breaking() -> None:
    cfg = ev.ProtocolConfig()
    strong = _mk_metrics(ic=0.12, icir=1.30, pos=0.80, nic=0.10, liq10=0.015, liq20=0.015)
    # Slightly lower defensive cushion on ENS_W2F_C2_MINI29 (97.5 vs 100.0 -> diff 2.5 <= 5.0)
    def_mid = _mk_metrics(ic=-0.01, icir=-0.10, pos=0.40, nic=0.01, liq10=0.0, liq20=0.0)

    sa, sb = _make_scope_fixture(
        overrides_b_norm={"ENS_W2F_C2_MINI29": strong, "W2_CHAMP_FILM27": strong},
        overrides_b_ext={"ENS_W2F_C2_MINI29": strong, "W2_CHAMP_FILM27": strong},
        overrides_b_d5_ext={"ENS_W2F_C2_MINI29": strong, "W2_CHAMP_FILM27": strong},
        overrides_a_norm={"ENS_W2F_C2_MINI29": strong, "W2_CHAMP_FILM27": strong},
        overrides_b_def={"ENS_W2F_C2_MINI29": def_mid, "W2_CHAMP_FILM27": strong},
    )
    prereq = ev.check_prerequisites(
        {"score_hidden_corr": 0.99999, "score_fused_corr": 0.99999}, sa["D10"], sb["D10"], cfg
    )
    dec = ev.adjudicate_candidates(prereq, sa, sb, cfg)
    assert dec["final_action"] == "A_PROMOTE"
    assert dec["selected_candidate"] == "ENS_W2F_C2_MINI29"


def test_compute_breadth_map_basic() -> None:
    dates = [f"2026-11-{i:02d}" for i in range(1, 25)]
    rows = []
    # Stock A steadily rises (above MA20), Stock B steadily falls (below MA20)
    for i, d in enumerate(dates):
        rows.append({"code": "sh.600000", "date": d, "close": 10.0 + i * 0.5, "adj_factor": 1.0})
        rows.append({"code": "sz.000001", "date": d, "close": 20.0 - i * 0.5, "adj_factor": 1.0})
    kline = pd.DataFrame(rows)
    b_map = ev.compute_breadth_map(kline)
    assert len(b_map) == 5  # 24 dates with min_periods=20 -> 5 valid dates
    for val in b_map.values():
        assert val == pytest.approx(0.5)


def test_eval_subset_liq_excess_uses_liquid_pool_benchmark() -> None:
    fa = ev._import_fusion_analysis()
    rows = []
    # 25 liquid stocks (med_amt_20d >= 5e4) all have +0.02 actual_return; top-10 by score have +0.05
    for i in range(25):
        rows.append({
            "asof": "2026-11-10",
            "asof_date": "2026-11-10",
            "code": f"sh.600{i:03d}",
            "sector": i % 3,
            "size_decile": (i % 5) + 1,
            "rev10": float(i) / 30.0,
            "med_amt_20d": 1e5,
            "actual_return": 0.05 if i < 10 else 0.02,
            "predicted_return_d10": float(30 - i),
            "FUSE_VAL_1E4": float(30 - i),
        })
    # 25 illiquid micro-caps crash -0.20 (outside liquid pool, must NOT inflate liq10/liq20 excess)
    for i in range(25):
        rows.append({
            "asof": "2026-11-10",
            "asof_date": "2026-11-10",
            "code": f"sz.300{i:03d}",
            "sector": i % 3,
            "size_decile": 1,
            "rev10": 0.0,
            "med_amt_20d": 1e3,
            "actual_return": -0.20,
            "predicted_return_d10": -float(i),
            "FUSE_VAL_1E4": -float(i),
        })
    sub = pd.DataFrame(rows)
    res = ev._eval_subset(
        sub,
        models={"C2": "predicted_return_d10", "FUSE_VAL_1E4": "FUSE_VAL_1E4"},
        b_map={"2026-11-10": 0.40},
        date_filter=None,
        fa=fa,
    )
    liq_mean = (10 * 0.05 + 15 * 0.02) / 25.0  # 0.032
    expected_liq10 = 0.05 - liq_mean           # +0.018 (NOT 0.05 - (-0.084) = +0.134!)
    assert res["models"]["C2"]["liq_top10_excess"] == pytest.approx(expected_liq10, abs=1e-9)
