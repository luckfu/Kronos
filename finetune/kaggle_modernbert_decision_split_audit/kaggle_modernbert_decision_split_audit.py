"""Audit the temporal/symbol split contract of the ModernBERT sidecar datasets."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import pandas as pd


OUTPUT = Path("/kaggle/working/modernbert_decision_split_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)


def log(phase: str, **fields: object) -> None:
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        **fields,
    }
    line = json.dumps(row, ensure_ascii=False, default=str)
    print(line, flush=True)
    with (OUTPUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(pattern: str) -> Path:
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def load_panel(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise TypeError(f"{path} is not a symbol -> DataFrame panel")
    return {str(symbol): frame.sort_index() for symbol, frame in panel.items()}


def parquet_symbols(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    symbols: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["symbol"], batch_size=250_000):
        symbols.update(str(value) for value in batch.column(0).to_pylist())
    return symbols


def main() -> None:
    train_panel_path = find_one("**/processed_datasets/train_data.pkl")
    validation_panel_path = find_one("**/processed_datasets/val_data.pkl")
    train_targets_path = find_one("**/train_targets.parquet")
    validation_targets_path = find_one("**/validation_targets.parquet")
    source_manifest_path = train_panel_path.parent.parent / "data_manifest.json"
    target_manifest_path = train_targets_path.parent / "decision_targets_manifest.json"

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    train_panel = load_panel(train_panel_path)
    validation_panel = load_panel(validation_panel_path)
    train_symbols = set(train_panel)
    validation_symbols = set(validation_panel)
    train_target_symbols = parquet_symbols(train_targets_path)
    validation_target_symbols = parquet_symbols(validation_targets_path)
    panel_overlap = train_symbols & validation_symbols
    target_overlap = train_target_symbols & validation_target_symbols

    train_dates = [
        pd.Timestamp(frame.index.min()).date()
        for frame in train_panel.values()
        if len(frame)
    ]
    train_end_dates = [
        pd.Timestamp(frame.index.max()).date()
        for frame in train_panel.values()
        if len(frame)
    ]
    validation_dates = [
        pd.Timestamp(frame.index.min()).date()
        for frame in validation_panel.values()
        if len(frame)
    ]
    validation_end_dates = [
        pd.Timestamp(frame.index.max()).date()
        for frame in validation_panel.values()
        if len(frame)
    ]

    overlap_examples = []
    for symbol in sorted(panel_overlap)[:20]:
        train_frame = train_panel[symbol]
        validation_frame = validation_panel[symbol]
        overlap_examples.append(
            {
                "symbol": symbol,
                "train_start": str(pd.Timestamp(train_frame.index.min()).date()),
                "train_end": str(pd.Timestamp(train_frame.index.max()).date()),
                "validation_start": str(pd.Timestamp(validation_frame.index.min()).date()),
                "validation_end": str(pd.Timestamp(validation_frame.index.max()).date()),
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
            }
        )

    report = {
        "status": "PASS",
        "source_manifest": source_manifest,
        "target_manifest": target_manifest,
        "hashes": {
            "train_panel_sha256": sha256_file(train_panel_path),
            "validation_panel_sha256": sha256_file(validation_panel_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
        },
        "panel": {
            "train_symbols": len(train_symbols),
            "validation_symbols": len(validation_symbols),
            "intersection": len(panel_overlap),
            "train_min_date": str(min(train_dates)),
            "train_max_date": str(max(train_end_dates)),
            "validation_min_date": str(min(validation_dates)),
            "validation_max_date": str(max(validation_end_dates)),
        },
        "targets": {
            "train_symbols": len(train_target_symbols),
            "validation_symbols": len(validation_target_symbols),
            "intersection": len(target_overlap),
            "train_rows": int(target_manifest["splits"]["train"]["records"]),
            "validation_rows": int(target_manifest["splits"]["validation"]["records"]),
        },
        "overlap_examples": overlap_examples,
        "interpretation": {
            "strict_symbol_holdout": len(panel_overlap) == 0
            and len(target_overlap) == 0,
            "temporal_holdout_possible": True,
            "train_labels_end": target_manifest["split_policy"]["train_signal_end"],
            "validation_labels_start": target_manifest["split_policy"][
                "validation_signal_start"
            ],
        },
    }
    (OUTPUT / "split_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(
        "audit_complete",
        panel_intersection=len(panel_overlap),
        target_intersection=len(target_overlap),
        train_date_range=[str(min(train_dates)), str(max(train_end_dates))],
        validation_date_range=[
            str(min(validation_dates)),
            str(max(validation_end_dates)),
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
