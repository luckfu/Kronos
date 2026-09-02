"""Build a temporal validation set on top of a fixed symbol holdout split."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", default="data/a_share_full_market_v1_beta")
    p.add_argument("--symbol-split", default="data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1/symbol_split.csv")
    p.add_argument("--output-root", default="data/a_share_full_market_v1_beta_temporal_symbol_validation_v1")
    p.add_argument("--train-cutoff", default="2024-12-31")
    p.add_argument("--validation-start", default="2025-01-01")
    args = p.parse_args()

    source = Path(args.source_root)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(output)
    with (source / "processed_datasets/train_data.pkl").open("rb") as f:
        panel = pickle.load(f)
    split = pd.read_csv(args.symbol_split, dtype={"symbol": str})
    train_symbols = set(split.loc[split.split == "train", "symbol"])
    val_symbols = set(split.loc[split.split == "validation", "symbol"])
    if train_symbols & val_symbols or train_symbols | val_symbols != set(map(str, panel)):
        raise ValueError("Invalid symbol split")

    cutoff = pd.Timestamp(args.train_cutoff)
    validation_start = pd.Timestamp(args.validation_start)
    train_panel = {}
    val_panel = {}
    for symbol, frame in panel.items():
        symbol = str(symbol)
        if symbol in train_symbols:
            train_panel[symbol] = frame
        else:
            train_frame = frame.loc[frame.index <= cutoff]
            val_frame = frame.loc[frame.index >= validation_start]
            if train_frame.empty:
                # Newly listed validation symbols have no pre-cutoff history.
                pass
            else:
                train_panel[symbol] = train_frame
            if val_frame.empty:
                raise ValueError(f"Empty validation side for {symbol}")
            val_panel[symbol] = val_frame

    output.mkdir(parents=True)
    (output / "processed_datasets").mkdir()
    with (output / "processed_datasets/train_data.pkl").open("wb") as f:
        pickle.dump(dict(sorted(train_panel.items())), f, protocol=pickle.HIGHEST_PROTOCOL)
    with (output / "processed_datasets/val_data.pkl").open("wb") as f:
        pickle.dump(dict(sorted(val_panel.items())), f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(args.symbol_split, output / "symbol_split.csv")
    (output / "asset_metadata.csv").symlink_to((source / "asset_metadata.csv").resolve())

    def stats(data):
        return {"symbols": len(data), "rows": sum(len(x) for x in data.values()),
                "start": str(min(x.index.min() for x in data.values()).date()),
                "end": str(max(x.index.max() for x in data.values()).date())}

    manifest = {
        "schema_version": 1,
        "split": {"unit": "symbol_and_time", "train_cutoff": args.train_cutoff,
                  "validation_start": args.validation_start, "validation_symbols": len(val_symbols),
                  "symbol_intersection": 0},
        "window_contract": {"lookback": 120, "predict": 10,
                            "validation_signal_start": "2025-07-01", "validation_signal_end": "2026-07-02"},
        "stats": {"train": stats(train_panel), "validation": stats(val_panel)},
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
