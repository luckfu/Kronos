#!/usr/bin/env python3
"""Turn a roll package (CSV/JSON) into the data root ``train_predictor.py`` reads.

Runs inside the kernel so the pickles are written by the same pandas/numpy that
reads them.  Layout produced under ``--output``:

    processed_datasets/train_data.pkl   {symbol: frame}  train-split symbols
    processed_datasets/val_data.pkl     {symbol: frame}  validation-split symbols
    asset_metadata.csv                  symbol,sector  (+ one row per vocabulary label)
    data_manifest.json                  window contract for the runner

Frames are indexed by ``datetime`` and carry OHLCVA plus an inline
``size_percentile``; the dataset prefers the inline value over metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.package / "roll_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256_file(args.package / name) != expected:
            raise SystemExit(f"package file hash mismatch: {name}")
    contract = manifest["window_contract"]
    raw = pd.read_csv(args.package / "roll_raw.csv", dtype={"symbol": str}, parse_dates=["date"])
    if raw["date"].max() != pd.Timestamp(contract["last_bar"]):
        raise SystemExit("package extends past its declared last bar")
    symbols = pd.read_csv(args.package / "symbols.csv", dtype={"symbol": str})
    vocab = json.loads((args.package / "sector_vocab.json").read_text())

    split = symbols.set_index("symbol")["split"]
    panels = {"train": {}, "validation": {}}
    for symbol, group in raw.groupby("symbol", sort=True):
        frame = group.sort_values("date").set_index("date")[FEATURES + ["size_percentile"]]
        frame.index.name = "datetime"
        panels[split[symbol]][symbol] = frame

    (args.output / "processed_datasets").mkdir(parents=True, exist_ok=True)
    for name, panel in (("train_data.pkl", panels["train"]), ("val_data.pkl", panels["validation"])):
        with (args.output / "processed_datasets" / name).open("wb") as handle:
            pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # AssetMetadata assigns sector ids by sorted unique label.  Pad with one row
    # per vocabulary label so the ids equal the evaluator's even when a sector
    # has no symbol in this package.
    metadata = pd.concat([
        symbols[["symbol", "sector"]],
        pd.DataFrame({"symbol": [f"__vocab_{i:02d}" for i in range(len(vocab))], "sector": vocab}),
    ], ignore_index=True)
    if sorted(metadata["sector"].unique()) != vocab:
        raise SystemExit("sector vocabulary would not reproduce the evaluator ids")
    metadata.to_csv(args.output / "asset_metadata.csv", index=False)

    data_manifest = {
        "schema_version": 1,
        "source_package": manifest,
        "window_contract": {
            "lookback": contract["lookback"], "predict": contract["predict"],
            "train_signal_start": contract["signal_start"], "train_signal_end": contract["signal_end"],
            "validation_signal_start": contract["signal_start"],
            "validation_signal_end": contract["signal_end"],
        },
        "stats": {key: {"symbols": len(panel), "rows": int(sum(len(f) for f in panel.values()))}
                  for key, panel in panels.items()},
    }
    (args.output / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"phase": "panels_ready", "output": str(args.output), **data_manifest["stats"]}), flush=True)


if __name__ == "__main__":
    main()
