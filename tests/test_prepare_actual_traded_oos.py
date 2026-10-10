import numpy as np
import pandas as pd
import pytest

from finetune.prepare_actual_traded_oos import (
    append_actual_traded, build_candidates, clean_frame,
)


def frame_for(dates):
    return pd.DataFrame({
        "open": 10., "high": 11., "low": 9., "close": 10.,
        "volume": 100., "amount": 1000., "size_percentile": .5,
        "size_bucket": 5., "sector": "test",
    }, index=dates)


def test_zero_volume_is_skipped_without_calendar_reindexing():
    dates = pd.bdate_range("2025-01-01", periods=131)
    frame = frame_for(dates)
    frame.loc[dates[120], "volume"] = 0
    cleaned, removed = clean_frame(frame)
    records = build_candidates({"s": cleaned}, str(dates[119].date()), str(dates[119].date()))
    assert removed == 1
    assert len(cleaned) == 130
    assert records[0]["target_date"] == str(dates[130].date())
    assert dates[120] not in cleaned.index


def test_maturity_requires_130_rows_and_labels_end_at_d10():
    dates = pd.bdate_range("2025-01-01", periods=130)
    frame = frame_for(dates)
    start = str(dates[119].date())
    records = build_candidates({"s": frame}, start, str(dates[-1].date()))
    assert len(records) == 1
    assert records[0]["target_date"] == str(dates[129].date())
    assert len(frame) == 130
    with pytest.raises(ValueError, match="No candidates"):
        build_candidates({"s": frame.iloc[:-1]}, start, start)


def test_refresh_history_replaces_prices_and_drops_absent_symbols():
    dates = pd.bdate_range("2025-01-01", periods=131)
    old = frame_for(dates)
    raw = old.drop(columns=["size_percentile", "size_bucket", "sector"]).reset_index(
        names="date"
    )
    raw["symbol"] = "s"
    raw["market_cap"] = 10000.
    raw["close"] = 20.
    panel, _ = append_actual_traded({"s": old, "absent": old}, raw, True)
    assert set(panel) == {"s"}
    assert np.all(panel["s"]["close"] == 20.)
    assert (panel["s"]["sector"] == "test").all()
