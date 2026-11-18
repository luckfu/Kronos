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


def test_compute_market_regime_thresholds():
    def _df(above_count: int, total: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            "trade_status": ["1"] * total,
            "amount": [1e6] * total,
            "invalid_ohlcva_rows": [0] * total,
            "invalid_factor_rows": [0] * total,
            "max_source_switch_log_jump": [0.0] * total,
            "ma20_rows": [20] * total,
            "ma20_adj_close": [10.0] * total,
            "asof_adj_close": [10.5] * above_count + [9.5] * (total - above_count),
        })

    assert prediction_drill.compute_market_regime(_df(2, 10))["level"] == 0
    assert prediction_drill.compute_market_regime(_df(2, 10))["position_pct"] == 0
    assert prediction_drill.compute_market_regime(_df(4, 10))["level"] == 1
    assert prediction_drill.compute_market_regime(_df(4, 10))["position_pct"] == 50
    assert prediction_drill.compute_market_regime(_df(5, 10))["level"] == 2
    assert prediction_drill.compute_market_regime(_df(5, 10))["position_pct"] == 100


def test_build_outputs_fuses_hidden_scores_and_records_regime(tmp_path):
    selected = pd.DataFrame([
        {"code": "sh.600000", "sector_id": 1, "sector_label": "银行", "size_percentile": 0.9, "close": 10.0},
        {"code": "sz.000001", "sector_id": 1, "sector_label": "银行", "size_percentile": 0.8, "close": 10.0},
        {"code": "sz.000063", "sector_id": 2, "sector_label": "通信", "size_percentile": 0.7, "close": 10.0},
    ])
    head_path = tmp_path / "head.npy"
    w = np.zeros(512, dtype=np.float32)
    w[0] = 1.0
    np.save(head_path, w)
    preds = lambda c: [{"close_p50": c} for _ in range(10)]
    batch_payloads = [{
        "asof": "2026-10-23",
        "response": {
            "meta": {"model_release": "small-0.1-cosine-c2", "model_checkpoint": "Segment@179", "sample_count": 16},
            "results": [
                # C2 ranks sh.600000 highest, but hidden feature puts sz.000063 clearly ahead cleanly.
                {"id": "sh.600000", "predictions": preds(10.5), "hidden_state": [-2.0] + [0.0] * 511},
                {"id": "sz.000001", "predictions": preds(10.2), "hidden_state": [0.0] + [0.0] * 511},
                {"id": "sz.000063", "predictions": preds(10.4), "hidden_state": [2.0] + [0.0] * 511},
            ],
        },
    }]
    regime = {"available": True, "breadth_ma20": 0.52, "level": 2, "position_pct": 100, "action": "满仓运行"}
    summary = prediction_drill.build_outputs(
        tmp_path / "out", selected, batch_payloads, "fp1",
        hidden_head_path=head_path, regime=regime,
    )
    assert summary["fusion"]["enabled"] is True
    assert summary["regime"]["position_pct"] == 100
    ranking = pd.read_csv(tmp_path / "out" / "ranking.csv")
    assert {"score_hidden", "score_fused", "rank_fused", "rank_percentile_fused"}.issubset(ranking.columns)
    # sz.000063: r_c2=2/3, r_hidden=1.0 -> avg=5/6; sh.600000: r_c2=1.0, r_hidden=1/3 -> avg=2/3
    assert ranking.iloc[0]["code"] == "sz.000063"
    assert int(ranking.iloc[0]["rank_fused"]) == 1
    assert int(ranking.loc[ranking["code"] == "sh.600000", "rank_d10"].iloc[0]) == 1

    # Fallback gracefully when hidden_state is omitted (e.g., before Modal redeploy)
    for item in batch_payloads[0]["response"]["results"]:
        item.pop("hidden_state")
    fallback = prediction_drill.build_outputs(
        tmp_path / "fallback", selected, batch_payloads, "fp2",
        hidden_head_path=head_path, regime=regime,
    )
    assert fallback["fusion"]["enabled"] is False
    assert fallback["fusion"]["reason"] == "hidden_state_missing"
