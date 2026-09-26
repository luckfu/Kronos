import pandas as pd

from modernbert_finance.build_dataset import HORIZON, LOOKBACK
from modernbert_finance.build_targets import WINDOW, target_row


def test_target_row_uses_kronos_window_identity_and_thresholds():
    rows = WINDOW
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = [100.0] * LOOKBACK + [100.0] * HORIZON + [100.0]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000.0] * rows,
            "amount": [100000.0] * rows,
        },
        index=dates,
    )
    frame.iloc[LOOKBACK, frame.columns.get_loc("high")] = 105.0
    frame.iloc[LOOKBACK + 1, frame.columns.get_loc("low")] = 95.0

    row = target_row("sh.600000", frame, 0)

    assert row["start_index"] == 0
    assert row["asof_date"] == dates[LOOKBACK - 1].date().isoformat()
    assert row["up_003"] == 1
    assert row["up_005"] == 1
    assert row["up_008"] == 0
    assert row["down_003"] == 1
    assert row["down_005"] == 1
    assert row["down_008"] == 0
