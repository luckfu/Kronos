"""Build a leakage-controlled 2026 incremental-training dataset."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd


WINDOW = 101


def load_panel(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_panel(path: Path, panel: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)


def crop_2026(panel: dict) -> dict:
    start = pd.Timestamp("2026-01-01")
    cropped = {}
    for symbol, frame in panel.items():
        selected = frame.loc[frame.index >= start].copy()
        if len(selected) >= WINDOW:
            cropped[symbol] = selected
    return cropped


def describe(panel: dict) -> dict:
    frames = list(panel.values())
    return {
        "symbols": len(panel),
        "rows": sum(len(frame) for frame in frames),
        "windows": sum(max(len(frame) - WINDOW + 1, 0) for frame in frames),
        "start": str(min(frame.index.min() for frame in frames).date()),
        "end": str(max(frame.index.max() for frame in frames).date()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="./data/a_share_v3")
    parser.add_argument(
        "--output-root", default="./data/a_share_v3_2026_incremental"
    )
    args = parser.parse_args()

    source = Path(args.source_root)
    output = Path(args.output_root)
    processed = source / "processed_datasets"

    # The V3 validation panel contains only non-holdout symbols and only dates
    # after the original training cutoff, so it becomes the incremental train set.
    train = load_panel(processed / "val_data.pkl")
    holdout = load_panel(processed / "symbol_holdout_data.pkl")
    validation = crop_2026(holdout)

    overlap = set(train) & set(validation)
    if overlap:
        raise SystemExit(f"Train/validation symbol overlap: {len(overlap)}")
    if min(frame.index.min() for frame in train.values()) < pd.Timestamp("2026-01-01"):
        raise SystemExit("Incremental training data contains pre-2026 rows")

    save_panel(output / "processed_datasets/train_data.pkl", train)
    save_panel(output / "processed_datasets/val_data.pkl", validation)
    save_panel(output / "processed_datasets/symbol_holdout_data.pkl", validation)
    for name in (
        "asset_metadata.csv",
        "size_reference.json",
        "universe_manifest.csv",
        "universe_summary.json",
    ):
        shutil.copy2(source / name, output / name)

    report = {
        "purpose": "2026 incremental training from the completed V3 Last model",
        "training_cutoff_exclusive": "2026-01-01",
        "train": describe(train),
        "validation": describe(validation),
        "train_validation_symbol_overlap": 0,
    }
    (output / "incremental_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
