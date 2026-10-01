"""Unit tests for Phase H path-touch MFE label + C2 availability helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modernbert_finance.ablations.teacher_distill_ablations import (
    MFE_TOUCH_THRESHOLD,
    _path_touch_from_daily_close,
    assess_c2_availability,
)


def test_mfe_touch_threshold_default() -> None:
    assert MFE_TOUCH_THRESHOLD == 0.10


def test_path_touch_binary_from_mfe10() -> None:
    mfe = np.array([-0.05, 0.0, 0.099, 0.10, 0.25], dtype=np.float64)
    y = (mfe >= MFE_TOUCH_THRESHOLD).astype(np.int64)
    np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])


def test_path_touch_from_daily_close_uses_max_cum() -> None:
    df = pd.DataFrame(
        {f"actual_return_d{i}": [0.01 * i, 0.05 if i == 3 else 0.01] for i in range(1, 11)}
    )
    # row0: increasing 0.01..0.10 → max 0.10
    # row1: day3=0.05 else 0.01 → max 0.05
    out = _path_touch_from_daily_close(df)
    assert abs(out[0] - 0.10) < 1e-9
    assert abs(out[1] - 0.05) < 1e-9


def test_c2_availability_blocker_when_no_overlap() -> None:
    info = assess_c2_availability("2025-07-03", "2026-07-02")
    assert info["date_overlap_with_val"] is False
    assert info["blocker"] is not None
    assert info["c2_segment"] == 179
