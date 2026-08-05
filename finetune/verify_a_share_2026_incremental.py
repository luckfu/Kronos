"""Verify the isolated 2026 incremental dataset and V3 Last base model."""

import argparse
import json
import pickle
from pathlib import Path

import torch


EXPECTED = {
    "train": {"symbols": 2312, "rows": 320840, "windows": 89730},
    "validation": {"symbols": 240, "rows": 33307, "windows": 9307},
}


def inspect(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    frames = list(panel.values())
    return panel, {
        "symbols": len(panel),
        "rows": sum(len(frame) for frame in frames),
        "windows": sum(max(len(frame) - 100, 0) for frame in frames),
        "start": str(min(frame.index.min() for frame in frames).date()),
        "end": str(max(frame.index.max() for frame in frames).date()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = data_root / "processed_datasets"
    base = Path(args.base_model)
    required = [
        dataset / "train_data.pkl",
        dataset / "val_data.pkl",
        data_root / "asset_metadata.csv",
        base / "config.json",
        base / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing incremental files: {missing}")

    train, train_report = inspect(dataset / "train_data.pkl")
    validation, validation_report = inspect(dataset / "val_data.pkl")
    report = {"train": train_report, "validation": validation_report}
    for split, expected in EXPECTED.items():
        for key, value in expected.items():
            actual = report[split][key]
            if actual != value:
                raise SystemExit(f"{split}.{key}: expected {value}, found {actual}")
    if train_report["start"] < "2026-01-01":
        raise SystemExit("Training data contains pre-2026 rows")
    if set(train) & set(validation):
        raise SystemExit("Training and validation symbols overlap")

    report.update(
        {
            "train_validation_symbol_overlap": 0,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "base_model": str(base),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run incremental training on Kaggle GPU")


if __name__ == "__main__":
    main()
