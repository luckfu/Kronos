"""Prepare ModernBERT labels from the exact temporal-symbol holdout panels.

This creates a companion label dataset. It never rewrites or duplicates the
large OHLCVA panel and is intended to run in a Kaggle kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from modernbert_finance.build_targets import build_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_temporal_holdout_root(input_root: Path) -> Path:
    candidates = sorted(
        {
            path.parent.parent
            for path in input_root.glob("**/processed_datasets/train_data.pkl")
            if (path.parent / "val_data.pkl").is_file()
        }
    )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one input dataset with train_data.pkl and val_data.pkl; "
            f"found {len(candidates)}: {candidates}"
        )
    root = candidates[0]
    required = [
        root / "data_manifest.json",
        root / "processed_datasets/train_data.pkl",
        root / "processed_datasets/val_data.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"temporal holdout dataset is incomplete: {missing}")
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    split = manifest.get("split", {})
    contract = manifest.get("window_contract", {})
    if split.get("unit") != "symbol_and_time":
        raise ValueError("input dataset is not a symbol-and-time holdout")
    if int(contract.get("lookback", -1)) != 120:
        raise ValueError("input dataset lookback is not 120")
    if int(contract.get("predict", -1)) != 10:
        raise ValueError("input dataset horizon is not 10")
    return root


def load_panel(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise ValueError(f"{path} is not a symbol -> DataFrame panel")
    return {str(symbol): frame for symbol, frame in panel.items()}


def prepare_labels(input_root: Path, output_dir: Path) -> dict[str, Any]:
    root = find_temporal_holdout_root(input_root)
    source_manifest_path = root / "data_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split = source_manifest["split"]
    contract = source_manifest["window_contract"]
    train_panel_path = root / "processed_datasets/train_data.pkl"
    validation_panel_path = root / "processed_datasets/val_data.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = build_split(
        load_panel(train_panel_path),
        output_dir / "train_targets.parquet",
        signal_start=None,
        signal_end=split["train_cutoff"],
        limit=0,
        chunk_size=50_000,
    )
    validation_stats = build_split(
        load_panel(validation_panel_path),
        output_dir / "validation_targets.parquet",
        signal_start=contract["validation_signal_start"],
        signal_end=contract["validation_signal_end"],
        limit=0,
        chunk_size=50_000,
    )
    train_stats["path"] = "train_targets.parquet"
    validation_stats["path"] = "validation_targets.parquet"
    manifest = {
        "schema_version": 1,
        "dataset_name": "a_share_120d_temporal_symbol_holdout_modernbert_targets",
        "source_dataset": {
            "dataset_id": "luckfu/a-share-120d-temporal-symbol-holdout",
            "root_name": root.name,
            "data_manifest_sha256": sha256_file(source_manifest_path),
            "train_panel_sha256": sha256_file(train_panel_path),
            "validation_panel_sha256": sha256_file(validation_panel_path),
        },
        "window": {
            "lookback": 120,
            "horizon": 10,
            "source_window": 131,
            "asof_position": 119,
            "future_rows_used": list(range(120, 130)),
        },
        "split_policy": {
            "train_signal_end": split["train_cutoff"],
            "validation_signal_start": contract["validation_signal_start"],
            "validation_signal_end": contract["validation_signal_end"],
        },
        "splits": {"train": train_stats, "validation": validation_stats},
        "join_key": ["symbol", "start_index", "asof_date"],
    }
    (output_dir / "decision_targets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/modernbert_decision_targets"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare_labels(args.input_root, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
