"""Checks for the forecast-band calibration on synthetic data (CPU, no data files)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "finetune/oos_eval/base_clean_sealed_oos"))

import band_calibration as bc  # noqa: E402

N_DAYS, N_SYMBOLS = 40, 800


def synthetic(seed=0, noise=0.05, bias=-0.02):
    """Forecasts whose sample band is far too narrow and whose errors are biased."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=N_DAYS + 12).strftime("%Y-%m-%d")
    rows = []
    for d in range(N_DAYS):
        forecast = rng.normal(0.0, 0.02, N_SYMBOLS)
        actual = forecast + bias + rng.normal(0.0, noise, N_SYMBOLS)
        rows.append(pd.DataFrame({"signal": dates[d], "symbol": [f"s{i}" for i in range(N_SYMBOLS)], "h": 10,
                                  "forecast": forecast, "actual": actual, "matures": dates[d + 10],
                                  "vol": rng.uniform(0.01, 0.05, N_SYMBOLS),
                                  "raw_low": forecast - 0.01, "raw_high": forecast + 0.01}))
    data = pd.concat(rows, ignore_index=True)
    data["error"] = data["actual"] - data["forecast"]
    data["rank"] = data.groupby(["signal", "h"])["forecast"].rank(ascending=False, method="first")
    return data, list(dates)


def test_calibrated_band_reaches_nominal_coverage_and_raw_does_not():
    data, _ = synthetic()
    result = bc.evaluate(data)["d10"]["methods"]
    assert result["RAW"]["coverage"] < 0.25
    for method in ("G", "V", "S"):
        assert abs(result[method]["coverage"] - bc.NOMINAL) < 0.02
        assert abs(result[method]["below"] - 0.10) < 0.02 and abs(result[method]["above"] - 0.10) < 0.02
        assert result[method]["interval_score"] < result["RAW"]["interval_score"]


def test_only_matured_labels_enter_the_fit():
    data, dates = synthetic()
    bands = bc.walk_forward(data)
    first = bands["signal"].min()
    # History needs MIN_HISTORY_DATES signal days whose D10 label matured on or before the signal day.
    assert first == dates[10 + bc.MIN_HISTORY_DATES - 1]
    # Corrupting every outcome that had not matured by `probe` must leave that day's band unchanged.
    probe = dates[25]
    before = bands[bands["signal"] == probe][["symbol", "G_low", "G_high", "V_low", "S_high"]].reset_index(drop=True)
    tampered = data.copy()
    late = tampered["matures"] > probe
    tampered.loc[late, "actual"] += 5.0
    tampered["error"] = tampered["actual"] - tampered["forecast"]
    after = bc.walk_forward(tampered)
    after = after[after["signal"] == probe][["symbol", "G_low", "G_high", "V_low", "S_high"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_volatility_buckets_give_wider_bands_to_noisier_names():
    rng = np.random.default_rng(1)
    data, _ = synthetic()
    data["actual"] = data["forecast"] + rng.normal(0.0, 1.0, len(data)) * data["vol"] * 2.0
    data["error"] = data["actual"] - data["forecast"]
    bands = bc.walk_forward(data)
    width = bands["V_high"] - bands["V_low"]
    quiet, noisy = width[bands["vol"] < 0.015].mean(), width[bands["vol"] > 0.045].mean()
    assert noisy > 2.0 * quiet
    scores = bc.evaluate(data)["d10"]["methods"]
    assert scores["V"]["interval_score"] < scores["G"]["interval_score"]


def test_sealed_window_is_refused(tmp_path):
    frame = pd.DataFrame({"signal": ["2026-10-26"], "code": ["sh.600000"], "close_asof": [10.0], "h": [1],
                          "p10": [9.9], "p50": [10.0], "p90": [10.1]})
    path = tmp_path / "x.csv.gz"
    frame.to_csv(path, index=False)
    try:
        bc.load_forecasts(f"production:{path}")
    except SystemExit as exc:
        assert "sealed" in str(exc)
    else:
        raise AssertionError("sealed window was read")
