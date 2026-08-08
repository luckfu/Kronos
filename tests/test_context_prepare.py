import pandas as pd
import pytest

from finetune.prepare_a_share_context import load_raw_inputs, make_signal_panel


def synthetic_frame(periods=360):
    dates = pd.date_range("2014-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "symbol": "sh.600000",
            "date": dates,
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1000.0,
            "amount": 10000.0,
            "market_cap": 1e9,
            "size_bucket": 5,
            "size_percentile": 0.55,
        }
    )


def test_context_panel_keeps_120_rows_and_signal_boundary():
    frame = synthetic_frame()
    panel, coverage, skipped = make_signal_panel(
        frame, "2015-01-01", "2015-03-31", lookback=120, predict=10
    )
    assert not skipped
    assert len(panel) == 1
    assert coverage[0]["context_rows"] == 120
    assert coverage[0]["first_signal"] == "2015-01-01"
    assert coverage[0]["last_signal"] == "2015-03-31"
    assert panel["sh.600000"].index[0] == pd.Timestamp("2014-07-18")


def test_duplicate_symbol_date_is_rejected(tmp_path):
    frame = synthetic_frame(periods=3)
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    frame.to_csv(first, index=False)
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match="Duplicate symbol,date"):
        load_raw_inputs([first, second])
