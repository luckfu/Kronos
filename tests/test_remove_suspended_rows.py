import pandas as pd
import pytest

from finetune.remove_suspended_rows import clean_panel, eligible_windows


def test_clean_panel_removes_only_zero_volume_and_preserves_dates():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-07"])
    frame = pd.DataFrame(
        {"volume": [100, 0, 200], "close": [10.0, 10.0, 11.0]},
        index=dates,
    )
    cleaned, stats = clean_panel({"sh.600000": frame})
    assert cleaned["sh.600000"].index.tolist() == [dates[0], dates[2]]
    assert cleaned["sh.600000"].close.tolist() == [10.0, 11.0]
    assert stats["zero_volume_removed"] == 1
    assert stats["symbols_shorter_than_131"] == 1


def test_clean_panel_rejects_unsorted_input():
    frame = pd.DataFrame({"volume": [1, 0]}, index=pd.to_datetime(["2025-01-03", "2025-01-02"]))
    with pytest.raises(ValueError, match="Unsorted"):
        clean_panel({"sh.600000": frame})


def test_eligible_windows_use_actual_asof_dates():
    dates = pd.bdate_range("2025-01-01", periods=132).delete(10)
    frame = pd.DataFrame({"volume": 1}, index=dates)
    assert eligible_windows(frame, dates[119], dates[119]) == 1
    assert eligible_windows(frame, dates[120], dates[120]) == 0
