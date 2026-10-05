"""Feeding-mode gate for Beta v2.1 validation denominators."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_predictor import (
    DEFAULT_BETA_V21_SCORE_FEEDING_MODE,
    beta_v21_validation_score,
    resolve_kept_best_loss,
    selection_is_improvement,
    should_reuse_saved_beta_v21_denominators,
    validate_resume_guard,
)


def test_reuse_requires_matching_feeding_mode():
    saved = {
        "csv": "1,1,1,1,1",
        "feeding_mode": DEFAULT_BETA_V21_SCORE_FEEDING_MODE,
    }
    assert should_reuse_saved_beta_v21_denominators(
        saved, DEFAULT_BETA_V21_SCORE_FEEDING_MODE, force_recalibrate=False
    )
    assert not should_reuse_saved_beta_v21_denominators(
        saved, DEFAULT_BETA_V21_SCORE_FEEDING_MODE, force_recalibrate=True
    )
    assert not should_reuse_saved_beta_v21_denominators(
        {"csv": "1,1,1,1,1"}, DEFAULT_BETA_V21_SCORE_FEEDING_MODE, False
    )
    assert not should_reuse_saved_beta_v21_denominators(
        {"csv": "1,1,1,1,1", "feeding_mode": "chronological_segment_date_sort"},
        DEFAULT_BETA_V21_SCORE_FEEDING_MODE,
        False,
    )


def test_resume_guard_can_ignore_denominators():
    saved = {"a": 1, "beta_v21_validation_denominators": "old"}
    current = {"a": 1, "beta_v21_validation_denominators": "new"}
    validate_resume_guard(
        saved, current, ignore_keys=("beta_v21_validation_denominators",)
    )
    try:
        validate_resume_guard(saved, current)
        raise AssertionError("expected mismatch")
    except ValueError as exc:
        assert "beta_v21_validation_denominators" in str(exc)


def test_sparse_ranking_denominator_miscalibrates_beta_v21_score():
    """Dense same-day ranking must not be scored with the sparse denominator.

    Shuffled batches contribute ~0.37 pairs and pull the sample-weighted
    ranking loss toward 0, so a saved ranking denominator can sit near 0.05.
    A random pairwise softplus on a real same-day batch is about ln(2).
    The composite then stops being a ranking metric (and is half forecast).
    """
    metrics = {
        "weighted_forecast_loss": 2.31,
        "return_loss": 0.40,
        "barrier_loss": 0.70,
        "ranking_loss": 0.693147,
    }
    sparse = beta_v21_validation_score(
        metrics,
        {"beta_v21_validation_denominators": "2.31,1,0.40,0.70,0.05"},
    )
    ranking_term = 0.10 * metrics["ranking_loss"] / 0.05
    # Other terms are ~0.9 at their own denominators; the sparse ranking
    # denominator alone pushes the composite above 2 and contributes more
    # than half of it. That is not a ranking loss and not the old forecast best.
    assert ranking_term > 1.0
    assert ranking_term > 0.5 * sparse
    assert sparse > 2.0


def test_forecast_threshold_is_not_reused_for_ranking():
    assert resolve_kept_best_loss("forecast", 2.31236787, "ranking", 0.51) == 0.51
    assert resolve_kept_best_loss("forecast", 2.31236787, "ranking", None) == float("inf")
    assert resolve_kept_best_loss("ranking", 0.42, "ranking", 0.90) == 0.42
    assert resolve_kept_best_loss("forecast", 2.31236787, "forecast", None) == 2.31236787


def test_pairwise_accuracy_threshold_is_not_a_ranking_loss():
    assert resolve_kept_best_loss(
        "ranking", 0.68692991, "pairwise_accuracy", 0.66304848
    ) == 0.66304848
    assert resolve_kept_best_loss(
        "ranking", 0.68692991, "pairwise_accuracy", None
    ) == float("-inf")
    assert resolve_kept_best_loss(
        "pairwise_accuracy", 0.70, "pairwise_accuracy", 0.10
    ) == 0.70
    # Forecast runs stay lower-better and keep the saved forecast number.
    assert resolve_kept_best_loss("forecast", 2.31236787, "forecast", 0.9) == 2.31236787
    assert selection_is_improvement("forecast", 2.30, 2.31) is True
    assert selection_is_improvement("forecast", 2.32, 2.31) is False
    assert selection_is_improvement("pairwise_accuracy", 0.67, 0.66) is True
    assert selection_is_improvement("pairwise_accuracy", 0.65, 0.66) is False
    assert selection_is_improvement("ranking", 0.68, 0.69) is True
