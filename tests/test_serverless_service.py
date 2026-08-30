import numpy as np
import pandas as pd
import pytest

from serverless import service


def valid_payload(rows=120, pred_len=10):
    timestamps = pd.bdate_range("2026-01-01", periods=rows)
    return {
        "data": [
            {
                "timestamp": timestamp.isoformat(),
                "open": 10 + index,
                "high": 11 + index,
                "low": 9 + index,
                "close": 10.5 + index,
                "volume": 1_000_000,
                "amount": 10_000_000,
            }
            for index, timestamp in enumerate(timestamps)
        ],
        "future_timestamps": [
            timestamp.isoformat()
            for timestamp in pd.bdate_range(timestamps[-1] + pd.Timedelta(days=1), periods=pred_len)
        ],
        "pred_len": pred_len,
        "sample_count": 3,
        "sector_id": 63,
        "size_percentile": 0.55,
    }


def test_parse_request_accepts_only_caller_supplied_data():
    parsed = service.parse_request(valid_payload())

    assert parsed.frame.columns.tolist() == list(service.FEATURE_COLUMNS)
    assert len(parsed.frame) == 120
    assert parsed.pred_len == 10
    assert parsed.sector_id == 63
    assert parsed.size_percentile == 0.55


def test_parse_request_rejects_symbol_only_requests():
    with pytest.raises(service.RequestError, match="data must be"):
        service.parse_request({"symbol": "600519"})


def test_parse_request_requires_exact_beta_context():
    with pytest.raises(service.RequestError, match="exactly 120"):
        service.parse_request(valid_payload(rows=600))


def test_parse_request_rejects_v6_size_bucket_contract():
    payload = valid_payload()
    payload["size_bucket"] = 5

    with pytest.raises(service.RequestError, match="not supported"):
        service.parse_request(payload)


@pytest.mark.parametrize("field", ["sector_id", "size_percentile"])
def test_parse_request_requires_beta_conditions(field):
    payload = valid_payload()
    payload.pop(field)

    with pytest.raises(service.RequestError, match=f"{field} is required"):
        service.parse_request(payload)


def test_parse_request_reports_invalid_integer_as_request_error():
    payload = valid_payload()
    payload["sample_count"] = "many"

    with pytest.raises(service.RequestError, match="sample_count must be an integer"):
        service.parse_request(payload)


def test_predict_returns_mean_and_close_intervals(monkeypatch):
    class FakePredictor:
        device = "cpu"

        def predict(self, **kwargs):
            samples = np.ones((3, 10, 6), dtype=np.float32)
            samples[1] *= 2
            samples[2] *= 3
            return samples

    monkeypatch.setattr(service, "get_predictor", lambda: FakePredictor())

    result = service.predict(valid_payload())

    assert result["meta"] == {
        "context_rows": 120,
        "pred_len": 10,
        "sample_count": 3,
        "model_device": "cpu",
        "model_release": "beta-v1.2",
        "model_checkpoint": "Best@871",
    }
    assert result["predictions"][0]["close"] == pytest.approx(2.0)
    assert result["predictions"][0]["close_p50"] == pytest.approx(2.0)
    assert np.asarray(result["samples"]["close"]).shape == (3, 10)


def test_predict_batch_runs_one_predictor_batch(monkeypatch):
    calls = []

    class FakePredictor:
        device = "cuda:0"

        def predict_batch(self, **kwargs):
            calls.append(kwargs)
            return [np.ones((3, 10, 6), dtype=np.float32) * value for value in (2, 3)]

    monkeypatch.setattr(service, "get_predictor", lambda: FakePredictor())
    payload = valid_payload()
    item = {
        key: payload[key]
        for key in ("data", "future_timestamps", "sector_id", "size_percentile")
    }
    result = service.predict_batch({
        "items": [{"id": "a", **item}, {"id": "b", **item}],
        "pred_len": 10, "sample_count": 3,
    })

    assert len(calls) == 1
    assert calls[0]["return_samples"] is True
    assert calls[0]["sector_id"] == [63, 63]
    assert calls[0]["size_percentile"] == [0.55, 0.55]
    assert calls[0]["size_bucket"] is None
    assert result["meta"]["batch_size"] == 2
    assert [item["id"] for item in result["results"]] == ["a", "b"]
    assert result["results"][1]["predictions"][0]["close"] == pytest.approx(3.0)
