"""Build a strict-causal audit/export dataset from Kronos panel pickles.

The JSONL output is for schema inspection and audit. The ModernBERT training
path should read panel windows directly and keep numeric arrays/token ids,
without text serialization or decimal quantization.

Future rows are used only for targets; every state feature is computed from the
lookback rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOOKBACK = 120
HORIZON = 10
FEATURES = ("open", "high", "low", "close", "volume", "amount")
MFE_BUCKET_EDGES = (-np.inf, 0.0, 0.03, 0.05, 0.08, 0.12, np.inf)
MAE_BUCKET_EDGES = (-np.inf, -0.08, -0.05, -0.03, 0.0, np.inf)
MFE_EXCEEDANCE_THRESHOLDS = (0.03, 0.05, 0.08, 0.12)
MAE_EXCEEDANCE_THRESHOLDS = (0.03, 0.05, 0.08, 0.12)
FIRST_TOUCH_UPSIDE_THRESHOLD = 0.05
FIRST_TOUCH_DOWNSIDE_THRESHOLD = 0.05
FIRST_TOUCH_LABELS = ("upside_first", "downside_first", "neither")
SERIALIZED_SEQUENCE_DECIMALS = 3
SERIALIZED_DECIMALS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite feature value: {value!r}")
    return result


def rounded(value: float) -> float:
    return finite(round(float(value), SERIALIZED_DECIMALS))


def rounded_sequence(value: float) -> float:
    return finite(round(float(value), SERIALIZED_SEQUENCE_DECIMALS))


def bucket(value: float, edges: tuple[float, ...]) -> int:
    return int(np.searchsorted(np.asarray(edges[1:-1]), value, side="right"))


def first_touch_label(
    future: pd.DataFrame,
    current_close: float,
) -> int:
    """Return 0=upside_first, 1=downside_first, 2=neither.

    If both thresholds are touched on the same day, downside wins by the
    conservative labeling rule.
    """
    for high, low in zip(future["high"].to_numpy(), future["low"].to_numpy()):
        upside = high / current_close - 1.0 >= FIRST_TOUCH_UPSIDE_THRESHOLD
        downside = 1.0 - low / current_close >= FIRST_TOUCH_DOWNSIDE_THRESHOLD
        if downside:
            return 1
        if upside:
            return 0
    return 2


def prepare_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: panel index must be a DatetimeIndex")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate dates")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")
    values = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{symbol}: non-finite OHLCVA value")
    if (values[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive price")
    if (values[["volume", "amount"]] < 0).any().any():
        raise ValueError(f"{symbol}: negative volume or amount")
    return frame.copy()


def state_features(history: pd.DataFrame) -> dict[str, Any]:
    """Create a compact state using history only."""
    close = history["close"].to_numpy(dtype=np.float64)
    high = history["high"].to_numpy(dtype=np.float64)
    low = history["low"].to_numpy(dtype=np.float64)
    volume = history["volume"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close))
    last = float(close[-1])
    result: dict[str, Any] = {}

    for days in (3, 5, 10, 20, 60, 120):
        result[f"return_{days}d"] = rounded(
            last / close[max(0, len(close) - days - 1)] - 1.0
        )
    for days in (5, 10, 20, 60):
        result[f"vol_{days}d"] = rounded(
            np.std(log_returns[-days:]) if days <= len(log_returns) else 0.0
        )

    for days in (20, 60, 120):
        result[f"close_vs_{days}d_high"] = rounded(last / np.max(high[-days:]) - 1.0)
        result[f"close_vs_{days}d_low"] = rounded(last / np.min(low[-days:]) - 1.0)

    mean_5 = max(float(np.mean(volume[-5:])), 1e-12)
    mean_20 = max(float(np.mean(volume[-20:])), 1e-12)
    mean_60 = max(float(np.mean(volume[-60:])), 1e-12)
    result["volume_ratio_5d"] = rounded(mean_5 / mean_20)
    result["volume_ratio_20d"] = rounded(mean_20 / mean_60)
    result["volume_trend"] = rounded(mean_5 / mean_60 - 1.0)
    result["range_20d"] = rounded(np.mean((high[-20:] - low[-20:]) / close[-20:]))
    result["up_day_ratio_20d"] = rounded(np.mean(log_returns[-20:] > 0))
    return result


def normalized_history_rows(history: pd.DataFrame) -> list[list[float]]:
    """Return normalized history as [trading_day_offset, *features] rows."""
    values = history.loc[:, FEATURES].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    normalized = (values - mean) / (std + 1e-5)
    rows: list[list[float]] = []
    for offset, row in enumerate(normalized):
        rows.append([
            offset - LOOKBACK + 1,
            *[rounded_sequence(value) for value in row],
        ])
    return rows


def make_record(symbol: str, frame: pd.DataFrame, start: int) -> dict[str, Any]:
    asof_position = start + LOOKBACK - 1
    future_start = asof_position + 1
    future_end = future_start + HORIZON
    history = frame.iloc[start:asof_position + 1]
    future = frame.iloc[future_start:future_end]
    current_close = float(history.iloc[-1]["close"])
    future_high = future["high"].to_numpy(dtype=np.float64)
    future_low = future["low"].to_numpy(dtype=np.float64)
    future_close = future["close"].to_numpy(dtype=np.float64)
    mfe = finite(np.max(future_high) / current_close - 1.0)
    mae = finite(np.min(future_low) / current_close - 1.0)
    time_to_mfe = int(np.argmax(future_high) + 1)
    first_touch = first_touch_label(future, current_close)
    mfe_exceedance = [
        int(mfe >= threshold) for threshold in MFE_EXCEEDANCE_THRESHOLDS
    ]
    mae_exceedance = [
        int(-mae >= threshold) for threshold in MAE_EXCEEDANCE_THRESHOLDS
    ]
    asof = pd.Timestamp(frame.index[asof_position]).date().isoformat()
    metadata = frame.iloc[asof_position]
    sector_label = str(metadata.get("sector", "unknown"))
    sector_id = metadata.get("sector_id")
    sector_id = int(sector_id) if pd.notna(sector_id) else None
    return {
        "schema_version": 1,
        "symbol": str(symbol),
        "asof_date": asof,
        "start_index": int(start),
        "state": {
            "history_120d_normalized": normalized_history_rows(history),
            "summary": state_features(history),
            "sector_id": sector_id,
            "sector_label": sector_label,
            "size_percentile": finite(metadata.get("size_percentile", 0.5)),
        },
        "target": {
            "mfe10": rounded(mfe),
            "mfe10_bucket": bucket(mfe, MFE_BUCKET_EDGES),
            "mfe10_exceedance": mfe_exceedance,
            "mae10": rounded(mae),
            "mae10_bucket": bucket(mae, MAE_BUCKET_EDGES),
            "mae10_exceedance": mae_exceedance,
            "first_touch": first_touch,
            "time_to_mfe": time_to_mfe,
            "d3_return": rounded(future_close[2] / current_close - 1.0),
            "d5_return": rounded(future_close[4] / current_close - 1.0),
            "d10_return": rounded(future_close[9] / current_close - 1.0),
        },
        "analog": None,
    }


def build_split(
    panel: dict[str, pd.DataFrame],
    output: Path,
    signal_start: str | None,
    signal_end: str | None,
    limit: int,
) -> dict[str, Any]:
    count = 0
    symbols = 0
    dates: list[str] = []
    with output.open("w", encoding="utf-8") as handle:
        for symbol in sorted(panel):
            frame = prepare_frame(panel[symbol], str(symbol))
            window = LOOKBACK + HORIZON
            if len(frame) < window:
                continue
            symbols += 1
            for start in range(len(frame) - window + 1):
                asof_date = pd.Timestamp(frame.index[start + LOOKBACK - 1]).date()
                if signal_start and asof_date < pd.Timestamp(signal_start).date():
                    continue
                if signal_end and asof_date > pd.Timestamp(signal_end).date():
                    continue
                record = make_record(str(symbol), frame, start)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                dates.append(record["asof_date"])
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
    return {
        "records": count,
        "symbols": symbols,
        "signal_start": min(dates) if dates else None,
        "signal_end": max(dates) if dates else None,
        "path": str(output),
    }


def load_panel(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise ValueError(f"{path} does not contain a symbol -> DataFrame panel")
    return {str(symbol): frame for symbol, frame in panel.items()}


def load_sector_vocabulary(path: Path) -> tuple[dict[str, int], str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("sector_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path} does not contain sector_labels")
    label_to_id = {str(label): index for index, label in enumerate(labels)}
    unknown_id = int(payload.get("unknown_sector_id", len(labels)))
    return label_to_id, str(payload.get("vocabulary_id", "")), unknown_id


def add_sector_ids(
    panel: dict[str, pd.DataFrame],
    sector_ids: dict[str, int],
    unknown_id: int,
) -> dict[str, pd.DataFrame]:
    enriched = {}
    for symbol, frame in panel.items():
        copy = frame.copy()
        if "sector_id" not in copy.columns:
            labels = copy.get("sector", pd.Series("unknown", index=copy.index)).astype(str)
            copy["sector_id"] = labels.map(sector_ids).fillna(unknown_id).astype(int)
        enriched[symbol] = copy
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict-causal ModernBERT decision records"
    )
    parser.add_argument("--train-panel", required=True, type=Path)
    parser.add_argument("--validation-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-signal-start", default=None)
    parser.add_argument("--train-signal-end", default=None)
    parser.add_argument("--validation-signal-start", default=None)
    parser.add_argument("--validation-signal-end", default=None)
    parser.add_argument(
        "--sector-vocabulary",
        type=Path,
        default=Path("webui/sector_vocabulary.json"),
    )
    parser.add_argument("--limit-per-split", type=int, default=0)
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("--limit-per-split cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sector_ids, vocabulary_id, unknown_sector_id = load_sector_vocabulary(
        args.sector_vocabulary
    )
    train_panel = add_sector_ids(
        load_panel(args.train_panel), sector_ids, unknown_sector_id
    )
    validation_panel = add_sector_ids(
        load_panel(args.validation_panel), sector_ids, unknown_sector_id
    )

    train_stats = build_split(
        train_panel,
        args.output_dir / "train_state_only.jsonl",
        args.train_signal_start,
        args.train_signal_end,
        args.limit_per_split,
    )
    validation_stats = build_split(
        validation_panel,
        args.output_dir / "validation_state_only.jsonl",
        args.validation_signal_start,
        args.validation_signal_end,
        args.limit_per_split,
    )
    manifest = {
        "schema_version": 1,
        "dataset_name": "kronos_modernbert_decision_a_share_v1",
        "input_contract": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "features": list(FEATURES),
            "history_row_format": ["trading_day_offset", *FEATURES],
            "trading_day_offset": "0 is asof_date; -119 is the oldest lookback trading day",
            "serialized_decimals": {
                "history": SERIALIZED_SEQUENCE_DECIMALS,
                "summary_and_target": SERIALIZED_DECIMALS,
            },
            "future_excluded_from_state": True,
            "normalization": "history_only",
            "sector_vocabulary_id": vocabulary_id,
            "unknown_sector_id": unknown_sector_id,
        },
        "target_contract": {
            "mfe10": "max(high[t+1:t+10]) / close[t] - 1",
            "mae10": "min(low[t+1:t+10]) / close[t] - 1",
            "mfe_buckets": ["<0%", "0-3%", "3-5%", "5-8%", "8-12%", ">=12%"],
            "mae_buckets": ["<=-8%", "-8--5%", "-5--3%", "-3-0%", ">0%"],
            "mfe_exceedance_thresholds": list(MFE_EXCEEDANCE_THRESHOLDS),
            "mae_exceedance_thresholds": list(MAE_EXCEEDANCE_THRESHOLDS),
            "first_touch": {
                "labels": list(FIRST_TOUCH_LABELS),
                "upside_threshold": FIRST_TOUCH_UPSIDE_THRESHOLD,
                "downside_threshold": FIRST_TOUCH_DOWNSIDE_THRESHOLD,
                "same_day_both": "downside_first",
            },
            "mfe_is_opportunity_not_realized_pnl": True,
        },
        "analog_contract": {
            "enabled": False,
            "training_index_must_use_past_records_only": True,
            "validation_queries_must_not_enter_training_index": True,
        },
        "source": {
            "train_panel": str(args.train_panel),
            "train_panel_sha256": sha256_file(args.train_panel),
            "validation_panel": str(args.validation_panel),
            "validation_panel_sha256": sha256_file(args.validation_panel),
            "sector_vocabulary": str(args.sector_vocabulary),
            "sector_vocabulary_sha256": sha256_file(args.sector_vocabulary),
        },
        "splits": {"train": train_stats, "validation": validation_stats},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
