import json

import numpy as np
import pandas as pd

from modernbert_finance.build_dataset import (
    FIRST_TOUCH_LABELS,
    HORIZON,
    LOOKBACK,
    make_record,
    state_features,
)


def sample_frame(rows=LOOKBACK + HORIZON):
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.arange(100.0, 100.0 + rows)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "amount": np.full(rows, 100000.0),
            "size_percentile": np.full(rows, 0.5),
            "sector": ["test"] * rows,
        },
        index=dates,
    )
    return frame


def test_state_features_use_only_history():
    frame = sample_frame()
    history = frame.iloc[:LOOKBACK].copy()
    before = state_features(history)
    history.iloc[-1, history.columns.get_loc("close")] *= 2
    after = state_features(history)
    assert before["return_20d"] != after["return_20d"]
    assert "mfe10" not in before


def test_record_targets_use_future_and_keep_metadata():
    record = make_record("sh.600000", sample_frame(), 0)
    assert record["symbol"] == "sh.600000"
    serialized = record["state"]["history_120d_normalized"]
    assert len(serialized) == LOOKBACK
    assert serialized[0][0] == -119
    assert serialized[-1][0] == 0
    assert len(serialized[0]) == 7
    assert all(
        len(str(abs(value)).split(".")[-1]) <= 3
        for value in serialized[0][1:]
    )
    assert record["target"]["time_to_mfe"] == HORIZON
    assert record["target"]["mfe10_bucket"] == 3
    assert record["target"]["mfe10_exceedance"] == [1, 1, 0, 0]
    assert record["target"]["mae10_bucket"] == 4
    assert record["target"]["mae10_exceedance"] == [0, 0, 0, 0]
    assert record["target"]["first_touch"] == FIRST_TOUCH_LABELS.index("upside_first")
    assert record["analog"] is None
    json.dumps(record, ensure_ascii=False)


def test_first_touch_is_ordered_and_same_day_both_is_downside():
    frame = sample_frame()
    current_close = float(frame.iloc[LOOKBACK - 1]["close"])
    future = frame.iloc[LOOKBACK:].copy()
    future.iloc[0, future.columns.get_loc("high")] = current_close * 1.06
    future.iloc[0, future.columns.get_loc("low")] = current_close * 0.94
    future.iloc[1:, future.columns.get_loc("high")] = current_close * 1.20
    future.iloc[1:, future.columns.get_loc("low")] = current_close * 0.99
    frame.iloc[LOOKBACK:, :] = future.to_numpy()

    record = make_record("sh.600000", frame, 0)

    assert record["target"]["first_touch"] == FIRST_TOUCH_LABELS.index("downside_first")
