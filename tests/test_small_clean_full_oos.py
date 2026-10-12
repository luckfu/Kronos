import numpy as np
import pandas as pd
import pytest

from finetune.small_clean_oos.evaluate import WindowStore, paired_comparison


def fixture_window():
    dates = pd.bdate_range("2026-01-01", periods=130)
    # The last input and labels exceed the normalization clip: raw closes must survive.
    close = np.ones(130) * 10
    close[119] = 30
    close[120:] = np.arange(40, 50)
    panel = pd.DataFrame({key: close.copy() for key in
                          ("open", "high", "low", "close", "volume", "amount")},
                         index=dates)
    panel["sector"] = "sector_b"
    panel["size_percentile"] = .7
    record = dict(symbol="stock", start_index=0, asof_date=str(dates[119].date()),
                  target_date=str(dates[129].date()), return_10d=49 / 30 - 1,
                  sector="sector_b", size_decile=7)
    return panel, record


def test_final_130_row_window_and_unclipped_prices():
    frame, record = fixture_window()
    prepared = WindowStore({"stock": frame}, ["sector_a", "sector_b"]).prepare(record)
    assert prepared["x"].shape == (120, 6)
    assert prepared["stamp"].shape == (130, 5)
    assert prepared["last_close"] == 30
    assert prepared["actual"][-1] == record["return_10d"]
    assert prepared["sector_id"] == 1
    assert prepared["future_dates"][-1] == record["target_date"]
    assert prepared["x"][-1, 3] == 5


def test_gaps_are_preserved_and_unknown_sector_uses_training_id():
    frame, record = fixture_window()
    dates = frame.index.to_list()
    dates[120:] = pd.bdate_range(frame.index[119] + pd.Timedelta(days=1006), periods=10)
    frame.index = pd.DatetimeIndex(dates)
    frame["sector"] = "unknown"
    record.update(target_date=str(dates[-1].date()), sector="unknown")
    prepared = WindowStore({"stock": frame}, ["known", "unknown"]).prepare(record)
    assert prepared["sector_id"] == 1
    assert prepared["future_dates"][0] == str(dates[120].date())


def test_bad_labels_and_incomplete_windows_rejected():
    frame, record = fixture_window()
    store = WindowStore({"stock": frame}, ["sector_b"])
    with pytest.raises(RuntimeError, match="Raw return"):
        store.prepare(dict(record, return_10d=0))
    with pytest.raises(RuntimeError, match="Incomplete"):
        WindowStore({"stock": frame.iloc[:-1]}, ["sector_b"]).prepare(record)


def test_pairing_guard_rejects_different_sample_or_label():
    def rows():
        frame = pd.DataFrame(dict(identity=["a", "b"]))
        for h in range(1, 11):
            frame[f"actual_return_d{h}"] = [.1, -.1]
        return frame

    c2 = rows()
    with pytest.raises(RuntimeError, match="Unpaired"):
        paired_comparison(c2, rows().assign(identity=["a", "c"]))
    clean = rows()
    clean.loc[0, "actual_return_d10"] = .2
    with pytest.raises(RuntimeError, match="labels differ"):
        paired_comparison(c2, clean)
