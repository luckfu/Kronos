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


def test_parse_request_uses_production_sampling_by_default():
    payload = valid_payload()
    payload.pop("sample_count")

    parsed = service.parse_request(payload)

    assert parsed.temperature == 0.60
    assert parsed.top_p == 0.90
    assert parsed.top_k == 0
    assert parsed.sample_count == 16


def test_parse_request_accepts_explicit_top_k():
    payload = valid_payload()
    payload["top_k"] = 20

    parsed = service.parse_request(payload)

    assert parsed.top_k == 20


def test_parse_request_rejects_negative_top_k():
    payload = valid_payload()
    payload["top_k"] = -1

    with pytest.raises(service.RequestError, match="top_k cannot be negative"):
        service.parse_request(payload)


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
            assert kwargs["T"] == 0.60
            assert kwargs["top_p"] == 0.90
            assert kwargs["top_k"] == 0
            assert kwargs["sample_count"] == 16
            samples = np.ones((16, 10, 6), dtype=np.float32)
            samples[8:] *= 3
            return samples

    monkeypatch.setattr(service, "get_predictor", lambda: FakePredictor())

    payload = valid_payload()
    payload.pop("sample_count")
    result = service.predict(payload)

    assert result["meta"] == {
        "context_rows": 120,
        "pred_len": 10,
        "temperature": 0.60,
        "top_p": 0.90,
        "top_k": 0,
        "sample_count": 16,
        "model_device": "cpu",
        "model_release": "small-0.1-cosine-c2",
        "model_checkpoint": "Segment@179",
    }
    assert result["predictions"][0]["close"] == pytest.approx(2.0)
    assert result["predictions"][0]["close_p50"] == pytest.approx(2.0)
    assert np.asarray(result["samples"]["close"]).shape == (16, 10)


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
    assert "hidden_state" not in result["results"][0]


def test_parse_request_validates_include_hidden():
    payload = valid_payload()
    payload["include_hidden"] = True
    parsed = service.parse_request(payload)
    assert parsed.include_hidden is True

    bad = valid_payload()
    bad["include_hidden"] = "yes"
    with pytest.raises(service.RequestError, match="include_hidden must be a boolean"):
        service.parse_request(bad)


def test_predict_and_batch_return_hidden_when_requested(monkeypatch):
    class FakePredictor:
        device = "cpu"

        def predict(self, **kwargs):
            assert kwargs["return_hidden"] is True
            return np.ones((2, 10, 6), dtype=np.float32), np.arange(8, dtype=np.float32)

        def predict_batch(self, **kwargs):
            assert kwargs["return_hidden"] is True
            return (
                [np.ones((2, 10, 6), dtype=np.float32), np.ones((2, 10, 6), dtype=np.float32) * 2],
                np.stack([np.ones(8, dtype=np.float32), np.full(8, 2.0, dtype=np.float32)]),
            )

    monkeypatch.setattr(service, "get_predictor", lambda: FakePredictor())
    payload = valid_payload()
    payload["sample_count"] = 2
    payload["include_hidden"] = True
    single = service.predict(payload)
    assert single["hidden_state"] == pytest.approx(list(range(8)))

    item = {
        key: payload[key]
        for key in ("data", "future_timestamps", "sector_id", "size_percentile")
    }
    batch = service.predict_batch({
        "items": [{"id": "a", **item}, {"id": "b", **item}],
        "pred_len": 10, "sample_count": 2, "include_hidden": True,
    })
    assert batch["results"][0]["hidden_state"] == pytest.approx([1.0] * 8)
    assert batch["results"][1]["hidden_state"] == pytest.approx([2.0] * 8)


def test_kronos_predictor_return_hidden_matches_decode_s1():
    import torch
    from model import kronos as kronos_mod

    torch.manual_seed(7)
    tokenizer = kronos_mod.KronosTokenizer(
        d_in=6, d_model=16, n_heads=2, ff_dim=32,
        n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05,
        group_size=4,
    ).eval()
    model = kronos_mod.Kronos(
        s1_bits=4, s2_bits=4, n_layers=1, d_model=16, n_heads=2, ff_dim=32,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=True, num_sectors=8, context_layer=0, use_size_percentile=True,
    ).eval()
    predictor = kronos_mod.KronosPredictor(model, tokenizer, device="cpu", max_context=64)
    p = valid_payload(rows=24, pred_len=4)
    df = pd.DataFrame(p["data"])[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = pd.Series(pd.to_datetime([row["timestamp"] for row in p["data"]]))
    y_ts = pd.Series(pd.to_datetime(p["future_timestamps"]))

    samples, hidden_batch = predictor.predict_batch(
        [df, df * 1.1], [x_ts, x_ts], [y_ts, y_ts],
        pred_len=4, T=0.6, top_p=0.9, sample_count=3, verbose=False,
        sector_id=[1, 2], size_percentile=[0.2, 0.8],
        return_samples=True, return_hidden=True,
    )
    assert len(samples) == 2
    assert hidden_batch.shape == (2, 16)
    # Verify exact parity with direct decode_s1[:, -1, :] on the normalized window.
    x_raw = df.to_numpy(dtype=np.float32)
    x_norm = np.clip((x_raw - x_raw.mean(0)) / (x_raw.std(0) + 1e-5), -5.0, 5.0)
    stamp = kronos_mod.calc_time_stamps(x_ts).to_numpy(dtype=np.float32)
    with torch.no_grad():
        tokens = tokenizer.encode(torch.from_numpy(x_norm).unsqueeze(0), half=True)
        _, ctx = model.decode_s1(
            tokens[0], tokens[1], torch.from_numpy(stamp).unsqueeze(0),
            sector_id=torch.tensor([1]), size_percentile=torch.tensor([0.2]),
        )
        expected = ctx[0, -1, :].numpy()
    assert np.allclose(hidden_batch[0], expected, atol=1e-5)
