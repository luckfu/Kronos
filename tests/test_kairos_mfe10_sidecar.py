"""Unit tests for Kairos mfe10≥10% short sidecar helpers."""

from __future__ import annotations

import math

import numpy as np

from modernbert_finance.mfe10_sidecar import (
    PHASE_P_BACKBONE,
    PHASE_P_FREEZE_TOKENIZER_EMBEDS,
    PHASE_P_SHALLOW_LAYERS,
    GATE_DELTA_VS_PRIOR,
    MFE10_DEF,
    MFE10_THRESHOLD,
    PRIOR_STUCK_PATIENCE,
    PRIOR_STUCK_TOL,
    TARGET_NAME,
    binary_from_mfe10,
    binary_log_loss,
    clears_gate,
    constant_prior_log_loss,
    delta_vs_prior,
    eval_summary,
    prior_stuck_stop,
    target_contract,
)


def test_target_is_path_mfe_not_ctc() -> None:
    assert MFE10_THRESHOLD == 0.10
    assert MFE10_DEF == "max(high[T+1:T+10]) / close[T] - 1"
    assert TARGET_NAME == "buy_worth_mfe10pct"
    contract = target_contract()
    assert contract["not_close_to_close"] is True
    assert contract["not_multi_head_r2"] is True


def test_binary_from_mfe10_threshold() -> None:
    mfe = np.array([-0.05, 0.0, 0.099999, 0.10, 0.25])
    y = binary_from_mfe10(mfe)
    np.testing.assert_array_equal(y, np.array([0, 0, 0, 1, 1], dtype=np.float32))


def test_path_mfe_can_differ_from_close_to_close() -> None:
    mfe10 = 0.12
    fwd_ret_10 = 0.02
    assert binary_from_mfe10(mfe10) == 1.0
    assert (fwd_ret_10 >= MFE10_THRESHOLD) is False


def test_constant_prior_log_loss_matches_entropy() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)  # p=0.25
    ll = constant_prior_log_loss(y)
    p = 0.25
    expected = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert abs(ll - expected) < 1e-9


def test_prior_stuck_early_stop() -> None:
    reason, streak = prior_stuck_stop(0.5655, 0.5655, 0)
    assert reason == ""
    assert streak == 1
    reason, streak = prior_stuck_stop(0.5656, 0.5655, streak)
    assert reason == "stuck_at_constant_prior"
    assert streak == PRIOR_STUCK_PATIENCE
    assert PRIOR_STUCK_TOL == 1e-3


def test_prior_stuck_resets_when_escaping() -> None:
    reason, streak = prior_stuck_stop(0.5655, 0.5655, 0)
    assert streak == 1
    reason, streak = prior_stuck_stop(0.50, 0.5655, streak)
    assert reason == ""
    assert streak == 0


def test_gate_delta() -> None:
    assert clears_gate(-0.04) is True
    assert clears_gate(-0.039) is False
    assert GATE_DELTA_VS_PRIOR == -0.04
    assert abs(delta_vs_prior(0.53, 0.5655) - (0.53 - 0.5655)) < 1e-12


def test_eval_summary_prints_prior() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.25).astype(np.float64)
    # Predict constant prior → delta ~ 0
    p = np.full_like(y, 0.25)
    summary = eval_summary(p, y, train_prior=0.25)
    assert "constant_prior_log_loss" in summary
    assert abs(summary["delta_vs_prior"]) < 1e-6
    assert summary["gate_passed"] is False
    assert abs(binary_log_loss(p, y) - summary["model_log_loss"]) < 1e-12


def test_decision_only_phase_p_contract() -> None:
    contract = target_contract()
    assert contract["not_ranking_ic"] is True
    assert PHASE_P_BACKBONE == "shallow"
    assert PHASE_P_SHALLOW_LAYERS == 2
    assert PHASE_P_FREEZE_TOKENIZER_EMBEDS is True
