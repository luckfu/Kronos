"""Unit / tiny smoke tests for Kairos ranking probe helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from modernbert_finance.ranking_probe import (
    MFE10_DEF,
    RANK_IC_BAR,
    TARGET_NAME,
    TOPK_LIFT_BAR,
    clears_rank_gate,
    cs_rank,
    daily_rank_ic,
    ranking_eval_summary,
    safe_spearman,
    target_contract,
    topk_hit_rate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_target_contract_is_ranking_not_binary() -> None:
    c = target_contract()
    assert c["not_binary_mfe10"] is True
    assert c["not_22_layer_binary"] is True
    assert c["not_multi_head_r2"] is True
    assert TARGET_NAME == "mfe10_continuous_rank"
    assert "max(high" in MFE10_DEF
    assert RANK_IC_BAR == 0.05
    assert TOPK_LIFT_BAR == 0.05


def test_perfect_scores_have_rank_ic_one() -> None:
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(5), 40)
    label = rng.normal(size=len(dates))
    # perfect scores = labels
    ic = daily_rank_ic(dates, label, label, min_names=30)
    assert ic["n_days"] == 5
    assert abs(ic["mean"] - 1.0) < 1e-9
    topk = topk_hit_rate(dates, label, label, frac=0.2, min_names=30)
    assert abs(topk["mean_hit"] - 1.0) < 1e-9
    assert topk["lift_vs_chance"] > 0.5


def test_noise_scores_near_chance() -> None:
    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange(40), 80)
    label = rng.normal(size=len(dates))
    score = rng.normal(size=len(dates))
    summary = ranking_eval_summary(score, label, dates)
    assert abs(summary["rank_ic"]["mean"]) < 0.08
    # Pure noise should not clear the ranking gate on a large panel.
    assert summary["gate_passed"] is False


def test_cs_rank_and_gate() -> None:
    dates = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0.1, 0.5, 0.9, 0.2, 0.3, 0.8])
    ranks = cs_rank(dates, y)
    assert abs(ranks[2] - 1.0) < 1e-9  # highest on day0
    assert clears_rank_gate(0.06, 0.0) is True
    assert clears_rank_gate(0.01, 0.06) is True
    assert clears_rank_gate(0.01, 0.01) is False
    assert abs(safe_spearman(y, y) - 1.0) < 1e-9


def test_train_script_syntax_and_knobs() -> None:
    script = ROOT / "finetune/kaggle_kairos_ranking_probe/train_ranking_probe.py"
    src = script.read_text(encoding="utf-8")
    ast.parse(src)
    assert 'BACKBONE_MODE = "identity"' in src
    assert "TARGET_MODE = \"mfe10_continuous\"" in src
    assert "MAX_SEGMENTS_THIS_RUN = 3" in src
    assert "not_22_layer_binary" in src
    assert "F.mse_loss" in src
    assert "validation/rank_ic_mean" in src
    assert "MFE10_THRESHOLD" not in src
    assert "binary_cross_entropy" not in src
