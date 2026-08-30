"""Model-only inference boundary used by serverless HTTP adapters."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

FEATURE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
DEFAULT_MODEL_PATH = (
    "outputs/models/a_share_v6_segment542_latest/checkpoints/last_model"
)
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512
MAX_PRED_LEN = 30
# 10 paths makes P10/P50/P90 very sensitive to a single sampled trajectory.
# The grid-optimized production configuration uses the API maximum of 50 paths.
MAX_SAMPLE_COUNT = 50
DEFAULT_SAMPLE_COUNT = 50
MAX_BATCH_SIZE = 12
INFERENCE_SEED = int(os.getenv("KRONOS_INFERENCE_SEED", "20260817"))


class RequestError(ValueError):
    """An invalid inference request."""


@dataclass(frozen=True)
class InferenceRequest:
    frame: pd.DataFrame
    timestamps: pd.Series
    future_timestamps: pd.Series
    pred_len: int
    temperature: float
    top_p: float
    sample_count: int
    size_bucket: int | None
    size_percentile: float | None


_predictor: KronosPredictor | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _seed_inference(seed: int) -> None:
    """Pin sampling RNGs so the same inputs produce the same paths."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _device() -> str:
    configured = os.getenv("KRONOS_DEVICE")
    if configured:
        return configured
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_predictor() -> KronosPredictor:
    """Lazily load one model per warm runtime."""
    global _predictor
    if _predictor is not None:
        return _predictor

    with _model_lock:
        if _predictor is not None:
            return _predictor
        tokenizer = KronosTokenizer.from_pretrained(
            os.getenv("KRONOS_TOKENIZER_ID", DEFAULT_TOKENIZER_ID)
        ).eval()
        model = Kronos.from_pretrained(
            os.getenv("KRONOS_MODEL_ID", DEFAULT_MODEL_PATH),
            num_sectors=0,
            num_size_buckets=10,
            context_layer=10,
            use_size_percentile=False,
            size_mlp_hidden_dim=64,
        ).eval()
        _predictor = KronosPredictor(
            model, tokenizer, device=_device(), max_context=MAX_CONTEXT
        )
        return _predictor


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise RequestError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RequestError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise RequestError(f"{name} must be an integer")
    return result


def parse_request(payload: Mapping[str, Any] | None) -> InferenceRequest:
    """Validate caller-supplied market data without fetching or persisting it."""
    if not isinstance(payload, Mapping):
        raise RequestError("JSON body must be an object")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise RequestError("data must be a non-empty array of OHLCVA rows")
    if len(rows) > MAX_CONTEXT:
        rows = rows[-MAX_CONTEXT:]
    if not all(isinstance(row, Mapping) for row in rows):
        raise RequestError("every data item must be an object")

    missing = [
        column for column in FEATURE_COLUMNS
        if any(column not in row for row in rows)
    ]
    if missing:
        raise RequestError(f"data is missing required fields: {', '.join(missing)}")
    if any("timestamp" not in row for row in rows):
        raise RequestError("every data item must include timestamp")

    frame = pd.DataFrame(rows)
    timestamps = pd.to_datetime(frame.pop("timestamp"), errors="coerce", utc=True)
    if timestamps.isna().any():
        raise RequestError("data contains an invalid timestamp")
    timestamps = timestamps.dt.tz_convert(None)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise RequestError("data timestamps must be unique and increasing")

    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[:, FEATURE_COLUMNS]
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise RequestError("OHLCVA fields must contain only finite numbers")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise RequestError("price fields must be positive")
    if (frame[["volume", "amount"]] < 0).any().any():
        raise RequestError("volume and amount must be non-negative")

    pred_len = _integer(payload.get("pred_len", 10), "pred_len")
    if not 1 <= pred_len <= MAX_PRED_LEN:
        raise RequestError(f"pred_len must be between 1 and {MAX_PRED_LEN}")
    future_values = payload.get("future_timestamps")
    if not isinstance(future_values, list) or len(future_values) != pred_len:
        raise RequestError("future_timestamps length must equal pred_len")
    future_timestamps = pd.Series(
        pd.to_datetime(future_values, errors="coerce", utc=True)
    ).dt.tz_convert(None)
    if future_timestamps.isna().any():
        raise RequestError("future_timestamps contains an invalid timestamp")
    if (future_timestamps <= timestamps.iloc[-1]).any():
        raise RequestError("future_timestamps must be after the last data timestamp")
    if not future_timestamps.is_monotonic_increasing:
        raise RequestError("future_timestamps must be increasing")

    temperature = _number(payload.get("temperature", 0.65), "temperature")
    top_p = _number(payload.get("top_p", 0.8), "top_p")
    sample_count = _integer(
        payload.get("sample_count", DEFAULT_SAMPLE_COUNT), "sample_count"
    )
    if temperature <= 0:
        raise RequestError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise RequestError("top_p must be in (0, 1]")
    if not 1 <= sample_count <= MAX_SAMPLE_COUNT:
        raise RequestError(
            f"sample_count must be between 1 and {MAX_SAMPLE_COUNT}"
        )

    size_bucket = payload.get("size_bucket")
    if size_bucket is not None:
        size_bucket = _integer(size_bucket, "size_bucket")
        if not 0 <= size_bucket <= 9:
            raise RequestError("size_bucket must be between 0 and 9")
    size_percentile = payload.get("size_percentile")
    if size_percentile is not None:
        size_percentile = _number(size_percentile, "size_percentile")
        if not 0 <= size_percentile <= 1:
            raise RequestError("size_percentile must be between 0 and 1")

    return InferenceRequest(
        frame=frame.reset_index(drop=True),
        timestamps=pd.Series(timestamps.to_numpy(), name="timestamps"),
        future_timestamps=pd.Series(
            future_timestamps.to_numpy(), name="timestamps"
        ),
        pred_len=pred_len,
        temperature=temperature,
        top_p=top_p,
        sample_count=sample_count,
        size_bucket=size_bucket,
        size_percentile=size_percentile,
    )


def predict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    request = parse_request(payload)
    predictor = get_predictor()
    with _inference_lock, torch.inference_mode():
        _seed_inference(INFERENCE_SEED)
        samples = predictor.predict(
            df=request.frame,
            x_timestamp=request.timestamps,
            y_timestamp=request.future_timestamps,
            pred_len=request.pred_len,
            T=request.temperature,
            top_p=request.top_p,
            sample_count=request.sample_count,
            verbose=False,
            size_bucket=request.size_bucket,
            size_percentile=request.size_percentile,
            return_samples=True,
        )

    mean = samples.mean(axis=0)
    close_index = FEATURE_COLUMNS.index("close")
    close_quantiles = np.quantile(
        samples[:, :, close_index], [0.1, 0.5, 0.9], axis=0
    )
    predictions = []
    for index, timestamp in enumerate(request.future_timestamps):
        values = {
            column: float(mean[index, column_index])
            for column_index, column in enumerate(FEATURE_COLUMNS)
        }
        values["high"] = max(
            values["open"], values["high"], values["low"], values["close"]
        )
        values["low"] = min(
            values["open"], values["high"], values["low"], values["close"]
        )
        values["volume"] = max(0.0, values["volume"])
        values["amount"] = max(0.0, values["amount"])
        predictions.append({
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            **values,
            "close_p10": float(close_quantiles[0, index]),
            "close_p50": float(close_quantiles[1, index]),
            "close_p90": float(close_quantiles[2, index]),
        })

    return {
        "predictions": predictions,
        "samples": {
            "close": samples[:, :, close_index].astype(float).tolist(),
        },
        "meta": {
            "context_rows": len(request.frame),
            "pred_len": request.pred_len,
            "sample_count": request.sample_count,
            "model_device": str(predictor.device),
        },
    }


def predict_batch(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Predict several caller-supplied series in one GPU batch."""
    if not isinstance(payload, Mapping):
        raise RequestError("JSON body must be an object")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise RequestError("items must be a non-empty array")
    if len(items) > MAX_BATCH_SIZE:
        raise RequestError(f"items may contain at most {MAX_BATCH_SIZE} series")

    requests = []
    for item in items:
        if not isinstance(item, Mapping):
            raise RequestError("every item must be an object")
        merged = {**payload, **item}
        merged.pop("items", None)
        requests.append(parse_request(merged))
    signatures = {(len(req.frame), req.pred_len, req.sample_count) for req in requests}
    if len(signatures) != 1:
        raise RequestError("all batch items must use the same context, pred_len, and sample_count")

    predictor = get_predictor()
    size_buckets = [req.size_bucket for req in requests]
    if any(value is None for value in size_buckets):
        size_buckets = None
    with _inference_lock, torch.inference_mode():
        _seed_inference(INFERENCE_SEED)
        batch_samples = predictor.predict_batch(
            df_list=[req.frame for req in requests],
            x_timestamp_list=[req.timestamps for req in requests],
            y_timestamp_list=[req.future_timestamps for req in requests],
            pred_len=requests[0].pred_len,
            T=requests[0].temperature,
            top_p=requests[0].top_p,
            sample_count=requests[0].sample_count,
            verbose=False,
            size_bucket=size_buckets,
            return_samples=True,
        )

    results = []
    close_index = FEATURE_COLUMNS.index("close")
    for item, req, samples in zip(items, requests, batch_samples):
        mean = samples.mean(axis=0)
        close_samples = samples[:, :, close_index]
        quantiles = np.quantile(close_samples, [0.1, 0.5, 0.9], axis=0)
        predictions = []
        for index, timestamp in enumerate(req.future_timestamps):
            values = {column: float(mean[index, col]) for col, column in enumerate(FEATURE_COLUMNS)}
            values["high"] = max(values["open"], values["high"], values["low"], values["close"])
            values["low"] = min(values["open"], values["high"], values["low"], values["close"])
            values["volume"] = max(0.0, values["volume"])
            values["amount"] = max(0.0, values["amount"])
            predictions.append({
                "timestamp": pd.Timestamp(timestamp).isoformat(), **values,
                "close_p10": float(quantiles[0, index]),
                "close_p50": float(quantiles[1, index]),
                "close_p90": float(quantiles[2, index]),
            })
        results.append({
            "id": item.get("id"), "predictions": predictions,
            "samples": {"close": close_samples.astype(float).tolist()},
        })
    return {
        "results": results,
        "meta": {
            "batch_size": len(results), "pred_len": requests[0].pred_len,
            "sample_count": requests[0].sample_count, "model_device": str(predictor.device),
        },
    }
