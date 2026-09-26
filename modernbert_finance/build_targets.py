"""Build compact numeric labels aligned with Kronos training windows.

The source panels remain the training input. This module only writes a
sidecar Parquet file containing labels keyed by (symbol, start_index).
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
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

from modernbert_finance.build_dataset import (
    FEATURES,
    HORIZON,
    LOOKBACK,
    MAE_EXCEEDANCE_THRESHOLDS,
    MFE_EXCEEDANCE_THRESHOLDS,
    finite,
    prepare_frame,
)


WINDOW = LOOKBACK + HORIZON + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_row(symbol: str, frame: pd.DataFrame, start: int) -> dict[str, Any]:
    asof_position = start + LOOKBACK - 1
    future = frame.iloc[asof_position + 1:asof_position + HORIZON + 1]
    if len(future) != HORIZON:
        raise ValueError(f"{symbol}: incomplete target window at start={start}")

    current_close = float(frame.iloc[asof_position]["close"])
    future_high = future["high"].to_numpy(dtype=np.float64)
    future_low = future["low"].to_numpy(dtype=np.float64)
    mfe10 = finite(np.max(future_high) / current_close - 1.0)
    mae10 = finite(np.min(future_low) / current_close - 1.0)
    row: dict[str, Any] = {
        "symbol": str(symbol),
        "start_index": int(start),
        "asof_date": pd.Timestamp(frame.index[asof_position]).date().isoformat(),
        "mfe10": np.float32(mfe10),
        "mae10": np.float32(mae10),
    }
    for threshold in MFE_EXCEEDANCE_THRESHOLDS:
        key = f"up_{int(threshold * 100):03d}"
        row[key] = np.uint8(mfe10 >= threshold)
    for threshold in MAE_EXCEEDANCE_THRESHOLDS:
        key = f"down_{int(threshold * 100):03d}"
        row[key] = np.uint8(-mae10 >= threshold)
    return row


def build_split(
    panel: dict[str, pd.DataFrame],
    output: Path,
    signal_start: str | None,
    signal_end: str | None,
    limit: int,
    chunk_size: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None
    writer: pq.ParquetWriter | None = None
    columns: list[str] | None = None
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    records = 0
    symbols = 0
    signal_start_seen: str | None = None
    signal_end_seen: str | None = None

    def flush() -> None:
        nonlocal writer, columns, buffer, buffered_rows
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), preserve_index=False)
        if writer is None:
            columns = table.schema.names
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        elif table.schema.names != columns:
            raise ValueError("target chunk schemas do not match")
        writer.write_table(table)
        buffer = []
        buffered_rows = 0

    try:
        for symbol in sorted(panel):
            frame = prepare_frame(panel[symbol], str(symbol))
            if len(frame) < WINDOW:
                continue
            windows = len(frame) - WINDOW + 1
            close = frame["close"].to_numpy(dtype=np.float64)
            high = frame["high"].to_numpy(dtype=np.float64)
            low = frame["low"].to_numpy(dtype=np.float64)
            future_high = sliding_window_view(
                high[LOOKBACK:LOOKBACK + windows + HORIZON - 1], HORIZON
            ).max(axis=1)
            future_low = sliding_window_view(
                low[LOOKBACK:LOOKBACK + windows + HORIZON - 1], HORIZON
            ).min(axis=1)
            current_close = close[LOOKBACK - 1:LOOKBACK - 1 + windows]
            mfe10 = future_high / current_close - 1.0
            mae10 = future_low / current_close - 1.0
            asof_dates = frame.index[LOOKBACK - 1:LOOKBACK - 1 + windows]
            selected = np.ones(windows, dtype=bool)
            if start_date:
                selected &= asof_dates.date >= start_date
            if end_date:
                selected &= asof_dates.date <= end_date
            selected_indices = np.flatnonzero(selected)
            if limit:
                selected_indices = selected_indices[:limit - records]
            if not len(selected_indices):
                if limit and records >= limit:
                    break
                continue

            symbols += 1
            selected_mfe = mfe10[selected_indices]
            selected_mae = mae10[selected_indices]
            data: dict[str, Any] = {
                "symbol": np.repeat(str(symbol), len(selected_indices)),
                "start_index": selected_indices.astype(np.int32, copy=False),
                "asof_date": np.datetime_as_string(
                    asof_dates.values[selected_indices].astype("datetime64[D]"),
                    unit="D",
                ),
                "mfe10": selected_mfe.astype(np.float32),
                "mae10": selected_mae.astype(np.float32),
            }
            for threshold in MFE_EXCEEDANCE_THRESHOLDS:
                data[f"up_{int(threshold * 100):03d}"] = (
                    selected_mfe >= threshold
                ).astype(np.uint8)
            for threshold in MAE_EXCEEDANCE_THRESHOLDS:
                data[f"down_{int(threshold * 100):03d}"] = (
                    -selected_mae >= threshold
                ).astype(np.uint8)
            part = pd.DataFrame(data)
            buffer.append(part)
            part_rows = len(part)
            records += part_rows
            buffered_rows += part_rows
            first_date = str(data["asof_date"][0])
            last_date = str(data["asof_date"][-1])
            if signal_start_seen is None or first_date < signal_start_seen:
                signal_start_seen = first_date
            if signal_end_seen is None or last_date > signal_end_seen:
                signal_end_seen = last_date
            if buffered_rows >= chunk_size:
                flush()
            if limit and records >= limit:
                break
        flush()
    finally:
        if writer is not None:
            writer.close()

    return {
        "records": records,
        "symbols": symbols,
        "signal_start": signal_start_seen,
        "signal_end": signal_end_seen,
        "path": str(output),
    }


def load_panel(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise ValueError(f"{path} does not contain a symbol -> DataFrame panel")
    return {str(symbol): frame for symbol, frame in panel.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build numeric ModernBERT decision labels aligned to Kronos windows"
    )
    parser.add_argument("--train-panel", required=True, type=Path)
    parser.add_argument("--validation-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-signal-start", default=None)
    parser.add_argument("--train-signal-end", default=None)
    parser.add_argument("--validation-signal-start", default=None)
    parser.add_argument("--validation-signal-end", default=None)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("--limit-per-split cannot be negative")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = build_split(
        load_panel(args.train_panel),
        args.output_dir / "train_targets.parquet",
        args.train_signal_start,
        args.train_signal_end,
        args.limit_per_split,
        args.chunk_size,
    )
    validation_stats = build_split(
        load_panel(args.validation_panel),
        args.output_dir / "validation_targets.parquet",
        args.validation_signal_start,
        args.validation_signal_end,
        args.limit_per_split,
        args.chunk_size,
    )
    manifest = {
        "schema_version": 1,
        "dataset_name": "kronos_modernbert_decision_targets_v1",
        "window": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "source_window": WINDOW,
            "asof_position": LOOKBACK - 1,
            "future_rows_used": list(range(LOOKBACK, LOOKBACK + HORIZON)),
        },
        "targets": {
            "mfe10": "max(high[T+1:T+10]) / close[T] - 1",
            "mae10": "min(low[T+1:T+10]) / close[T] - 1",
            "up_thresholds": list(MFE_EXCEEDANCE_THRESHOLDS),
            "down_thresholds": list(MAE_EXCEEDANCE_THRESHOLDS),
            "up_column_format": "up_{threshold_percent:03d}",
            "down_column_format": "down_{threshold_percent:03d}",
        },
        "source": {
            "train_panel": str(args.train_panel),
            "train_panel_sha256": sha256_file(args.train_panel),
            "validation_panel": str(args.validation_panel),
            "validation_panel_sha256": sha256_file(args.validation_panel),
        },
        "splits": {"train": train_stats, "validation": validation_stats},
    }
    (args.output_dir / "targets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
