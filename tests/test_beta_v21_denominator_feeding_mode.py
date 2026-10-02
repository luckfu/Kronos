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
