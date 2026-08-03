"""Validate the full-market V3 dataset and model before a long Colab run."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch


EXPECTED = {
    "train": {"symbols": 2389, "rows": 5_472_438, "windows": 5_233_538},
    "val": {"symbols": 2312, "rows": 320_840, "windows": 89_730},
    "holdout": {"symbols": 240, "rows": 486_525, "windows": 462_525},
}


def inspect_panel(path):
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


def require_equal(label, actual, expected):
    if actual != expected:
        raise SystemExit(f"{label}: expected {expected!r}, found {actual!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset_root = data_root / "processed_datasets"
    base_model = Path(args.base_model)
    paths = {
        "train": dataset_root / "train_data.pkl",
        "val": dataset_root / "val_data.pkl",
        "holdout": dataset_root / "symbol_holdout_data.pkl",
    }
    required = [
        *paths.values(),
        data_root / "asset_metadata.csv",
        data_root / "size_reference.json",
        data_root / "universe_manifest.csv",
        data_root / "universe_summary.json",
        base_model / "config.json",
        base_model / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing V3 files: {missing}")

    panels = {}
    report = {}
    for split, path in paths.items():
        panels[split], report[split] = inspect_panel(path)
        for key, expected in EXPECTED[split].items():
            require_equal(f"{split}.{key}", report[split][key], expected)

    require_equal("train.start", report["train"]["start"], "2015-01-05")
    require_equal("train.end", report["train"]["end"], "2025-12-31")
    require_equal("val.start", report["val"]["start"], "2026-01-05")
    require_equal("val.end", report["val"]["end"], "2026-07-31")
    overlap = set(panels["train"]) & set(panels["holdout"])
    require_equal("train_holdout_overlap", len(overlap), 0)

    manifest = pd.read_csv(data_root / "universe_manifest.csv", dtype={"symbol": str})
    split_counts = manifest.groupby("split")["symbol"].nunique().to_dict()
    require_equal("manifest.train_symbols", int(split_counts.get("train", 0)), 2389)
    require_equal("manifest.holdout_symbols", int(split_counts.get("holdout", 0)), 240)

    report.update(
        {
            "train_holdout_overlap": 0,
            "metadata_bytes": (data_root / "asset_metadata.csv").stat().st_size,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "base_model": str(base_model),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; V3 long training must run on a GPU runtime.")


if __name__ == "__main__":
    main()
