"""Validate and stage the cleaned train/validation data plus full time-OOS."""

import json
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from finetune.prepare_v1_beta_evaluation import sha256_file


def main():
    source = Path("data/a_share_full_market_v1_beta_temporal_symbol_validation_no_suspension_v1")
    oos = source / "evaluation_oos_full_through_20261009"
    output = Path("scratch/kaggle_clean_full_oos_20261009")
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((source / "data_manifest.json").read_text())
    for entry in audit["files"]:
        assert sha256_file(source / entry["name"]) == entry["sha256"], entry["name"]
    for split in ("train", "val"):
        with (source / f"processed_datasets/{split}_data.pkl").open("rb") as handle:
            panel = pickle.load(handle)
        for symbol, frame in panel.items():
            assert frame.index.is_unique and frame.index.is_monotonic_increasing, symbol
            assert frame["volume"].gt(0).all(), symbol
        print(f"Validated {split}: {len(panel)} symbols", flush=True)
        del panel

    manifest = json.loads((oos / "evaluation_manifest.json").read_text())
    for label in ("panel", "samples", "combined_raw"):
        entry = manifest["artifacts"]
        assert sha256_file(oos / entry[label + "_file"]) == entry[label + "_sha256"]
    with (oos / "evaluation_panel.pkl").open("rb") as handle:
        panel = pickle.load(handle)
    features = ["open", "high", "low", "close", "volume", "amount", "size_percentile"]
    for symbol, frame in panel.items():
        assert frame.index.is_unique and frame.index.is_monotonic_increasing, symbol
        assert frame.volume.gt(0).all(), symbol
        assert np.isfinite(frame[features].to_numpy(dtype=float)).all(), symbol
    seen = set()
    count = 0
    with (oos / "evaluation_samples.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            frame = panel[record["symbol"]]
            start = record["start_index"]
            assert start >= 0 and start + 130 <= len(frame)
            assert str(frame.index[start + 119].date()) == record["asof_date"]
            assert str(frame.index[start + 129].date()) == record["target_date"]
            assert "2026-08-03" <= record["asof_date"] <= "2026-09-17"
            assert record["target_date"] <= "2026-10-09"
            assert np.isclose(frame.close.iloc[start + 129] / frame.close.iloc[start + 119] - 1,
                              record["return_10d"])
            key = record["symbol"], record["asof_date"]
            assert key not in seen
            seen.add(key)
            count += 1
    assert count == manifest["sample_sets"]["incremental_future_all"]["samples"] == 175428
    del panel
    print(f"Validated full OOS: {count} samples", flush=True)

    for name in ("symbol_split.csv", "asset_metadata.csv", "data_manifest.json"):
        shutil.copy2(source / name, output / name)
    for split in ("train", "val"):
        name = f"{split}_data.pkl"
        shutil.copy2(source / "processed_datasets" / name, output / name)
    with zipfile.ZipFile(output / "evaluation_oos.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("evaluation_manifest.json", "evaluation_panel.pkl",
                     "evaluation_samples.jsonl", "combined_raw.csv"):
            archive.write(oos / name, arcname=f"evaluation_oos/{name}")

    inventory = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.name not in {"publication_manifest.json", "dataset-metadata.json", "README.md"}
    ]
    (output / "publication_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "dataset": "smmt315/a-share-clean-train-val-full-oos-20261009",
        "market_data_cutoff": "2026-10-09",
        "training_validation_cleaning": audit["stats"],
        "oos": {
            "signal_start": "2026-08-03", "signal_end": "2026-09-17",
            "latest_target": "2026-10-09", "samples": count,
            "history": 120, "labels": 10, "required_rows": 130,
            "purpose": "evaluation_only_never_train_or_tune",
        },
        "layout": "Train/val pickles are at dataset root; data_manifest.json preserves the original processed_datasets paths for source provenance. Extract evaluation_oos.zip to obtain evaluation_oos/.",
        "files": inventory,
    }, indent=2) + "\n")

    print(f"Staged {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
