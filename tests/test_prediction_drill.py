import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "deploy" / "oracle-kronos" / "prediction_drill.py"
SPEC = importlib.util.spec_from_file_location("prediction_drill", SCRIPT)
prediction_drill = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prediction_drill)


def test_command_line_defaults_to_production_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "prediction_drill.py",
        "--asof", "2026-09-18",
        "--sector-map", str(tmp_path / "sectors.json"),
        "--output-dir", str(tmp_path / "output"),
    ])

    args = prediction_drill.parse_args()

    assert args.sample_count == 16


def test_explicit_future_trading_dates_are_strict():
    values = "2026-09-21,2026-09-22,2026-09-23,2026-09-24,2026-09-28,2026-09-29,2026-09-30,2026-10-08,2026-10-09,2026-10-12"
    dates, source = prediction_drill.future_trading_dates(
        pd.Timestamp("2026-09-18").date(), values
    )
    assert source == "command_line"
    assert dates[0] == "2026-09-21"
    assert dates[-1] == "2026-10-12"
    with pytest.raises(RuntimeError, match="exactly 10"):
        prediction_drill.future_trading_dates(
            pd.Timestamp("2026-09-18").date(), "2026-09-21"
        )


def test_payload_uses_latest_normalized_forward_adjustment():
    dates = pd.date_range("2026-01-01", periods=120, freq="D").date
    frame = pd.DataFrame({
        "code": ["sh.600000"] * 120,
        "date": dates,
        "open": np.full(120, 10.0),
        "high": np.full(120, 11.0),
        "low": np.full(120, 9.0),
        "close": np.full(120, 10.0),
        "volume": np.full(120, 100.0),
        "amount": np.full(120, 1000.0),
        "adj_factor": np.linspace(1.0, 2.0, 120),
    })
    eligible = pd.DataFrame([{
        "code": "sh.600000", "sector_id": 63, "size_percentile": 0.55,
    }])
    future = [item.isoformat() for item in pd.date_range("2026-05-01", periods=10).date]
    payload = prediction_drill.make_payload(frame, eligible, future, 50)
    assert len(payload["items"][0]["data"]) == 120
    assert payload["items"][0]["data"][0]["close"] == pytest.approx(5.0)
    assert payload["items"][0]["data"][-1]["close"] == pytest.approx(10.0)
    assert payload["future_timestamps"] == future
    assert payload["temperature"] == 0.60
    assert payload["top_p"] == 0.90
    assert payload["top_k"] == 0
    assert payload["sample_count"] == 50


def test_fingerprint_includes_sampling_parameters(monkeypatch, tmp_path):
    sector_map = tmp_path / "sectors.json"
    sector_map.write_text("{}")
    args = argparse.Namespace(
        asof=pd.Timestamp("2026-09-18").date(),
        sector_map=sector_map,
        inference_url="https://example.test",
        sample_count=16,
        batch_size=12,
    )
    dates = list(pd.date_range("2026-04-01", periods=120).date)
    baseline = prediction_drill.run_fingerprint(
        args, dates, ["2026-09-21"], ["sh.600000"]
    )
    original = prediction_drill.DEFAULT_TOP_P
    monkeypatch.setattr(prediction_drill, "DEFAULT_TOP_P", 0.8)
    changed = prediction_drill.run_fingerprint(
        args, dates, ["2026-09-21"], ["sh.600000"]
    )
    assert changed != baseline
    assert prediction_drill.DEFAULT_TOP_P != original


def test_daily_scheduler_pins_production_sampling():
    scheduler = SCRIPT.with_name("daily_prediction_scheduler.sh").read_text()
    assert "--sample-count 16" in scheduler


def test_response_validation_rejects_wrong_model():
    predictions = [{"close_p50": 10.0} for _ in range(10)]
    response = {
        "results": [{"id": "sh.600000", "predictions": predictions}],
        "meta": {
            "sample_count": 50,
            "model_release": "wrong",
            "model_checkpoint": "Segment@179",
        },
    }
    with pytest.raises(RuntimeError, match="unexpected model release"):
        prediction_drill.validate_response(response, ["sh.600000"], {"sh.600000": 10.0})


def test_chunks_never_exceed_modal_limit():
    batches = list(prediction_drill.chunks([str(index) for index in range(25)], 12))
    assert [len(batch) for batch in batches] == [12, 12, 1]
