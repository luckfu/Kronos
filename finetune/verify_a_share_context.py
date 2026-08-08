"""Validate the context-preserving 120-day A-share training bundle."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch


def inspect_panel(path, lookback, predict):
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    window = lookback + predict + 1
    frames = list(panel.values())
    if not frames:
        raise SystemExit(f"{path}: panel is empty")
    invalid = [str(symbol) for symbol, frame in panel.items() if len(frame) < window]
    if invalid:
        raise SystemExit(
            f"{path}: {len(invalid)} symbols have fewer than {window} rows; "
            f"examples={invalid[:5]}"
        )
    return {
        "symbols": len(frames),
        "rows": int(sum(len(frame) for frame in frames)),
        "windows": int(sum(len(frame) - window + 1 for frame in frames)),
        "start": str(min(frame.index.min().date() for frame in frames)),
        "end": str(max(frame.index.max().date() for frame in frames)),
        "context_symbols": int(
            sum(frame.index.min().year <= 2014 for frame in frames)
        ),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--predict", type=int, default=10)
    parser.add_argument("--min-train-windows", type=int, default=1_000_000)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset_root = data_root / "processed_datasets"
    base_model = Path(args.base_model)
    required = [
        dataset_root / "train_data.pkl",
        dataset_root / "val_data.pkl",
        data_root / "asset_metadata.csv",
        data_root / "size_reference.json",
        data_root / "universe_manifest.csv",
        data_root / "v5_context_summary.json",
        data_root / "context_coverage_manifest.csv",
        base_model / "config.json",
        base_model / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing V5 context files: {missing}")

    summary = json.loads((data_root / "v5_context_summary.json").read_text())
    if int(summary.get("lookback", -1)) != args.lookback:
        raise SystemExit("v5_context_summary lookback does not match the requested value")
    if int(summary.get("predict", -1)) != args.predict:
        raise SystemExit("v5_context_summary predict does not match the requested value")
    if int(summary.get("unique_2014_trading_days", 0)) < 120:
        raise SystemExit("V5 bundle has fewer than 120 unique 2014 trading dates")

    report = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "lookback": args.lookback,
        "predict": args.predict,
        "window_rows": args.lookback + args.predict + 1,
        "train": inspect_panel(
            dataset_root / "train_data.pkl", args.lookback, args.predict
        ),
        "val": inspect_panel(
            dataset_root / "val_data.pkl", args.lookback, args.predict
        ),
        "base_model": str(base_model),
        "base_model_sha256": sha256(base_model / "model.safetensors"),
        "unique_2014_trading_days": int(summary["unique_2014_trading_days"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["train"]["windows"] < args.min_train_windows:
        raise SystemExit(
            f"Training dataset is too small: {report['train']['windows']} "
            f"< {args.min_train_windows} windows"
        )
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run V5 training on a GPU runtime")


if __name__ == "__main__":
    main()
