"""Prepare TimesFM windows from the existing formal Kronos split, without re-splitting."""

import argparse
import hashlib
import json
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("open", "high", "low", "close", "volume", "amount", "size_percentile")
SHARD_SIZE = 4096


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def discover(root):
    matches = [
        path.parent for path in root.glob("**/data_manifest.json")
        if (path.parent / "processed_datasets/train_data.pkl").is_file()
        and (path.parent / "processed_datasets/val_data.pkl").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one formal split root, found {matches}")
    return matches[0]


def window_indices(frame, start=None, end=None):
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Source dates must be unique and sorted; do not silently reorder")
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    values = frame[list(FEATURES)].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame.index).to_numpy(dtype="datetime64[D]")
    # Match KronosDataset's lookback + predict + 1 eligibility exactly.
    indices = np.arange(max(0, len(frame) - 131 + 1), dtype=np.int64)
    signals = dates[indices + 119]
    mask = np.ones(len(indices), dtype=bool)
    if start is not None:
        mask &= signals >= np.datetime64(start, "D")
    if end is not None:
        mask &= signals <= np.datetime64(end, "D")
    indices = indices[mask]
    for index in indices:
        if not np.isfinite(values[index:index + 130]).all():
            raise ValueError(f"Non-finite formal window at {dates[index + 119]}")
    return values, dates, indices


def prepare(source, output):
    manifest_path = source / "data_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    split, contract = source_manifest["split"], source_manifest["window_contract"]
    if split["unit"] != "symbol_and_time" or (
        contract["lookback"], contract["predict"]
    ) != (120, 10):
        raise ValueError("Expected formal symbol-and-time 120+10 contract")
    assignments = pd.read_csv(source / "symbol_split.csv", dtype={"symbol": str})
    heldout = set(assignments.loc[assignments.split == "validation", "symbol"])
    if not heldout:
        raise ValueError("Validation symbol list is empty")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "source_dataset": "luckfu/a-share-120d-temporal-symbol-holdout",
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256(manifest_path),
        "symbol_split_sha256": sha256(source / "symbol_split.csv"),
        "lookback": 120, "horizon": 10, "eligibility_rows": 131,
        "validation_scope": "all formal eligible windows; no prefix sampling",
        "splits": {},
    }
    for name in ("train", "val"):
        path = source / f"processed_datasets/{name}_data.pkl"
        with path.open("rb") as handle:
            panel = {str(key): value for key, value in pickle.load(handle).items()}
        if name == "val" and not set(panel).issubset(heldout):
            raise ValueError("Validation panel contains non-holdout symbols")
        start = contract["validation_signal_start"] if name == "val" else None
        end = contract["validation_signal_end"] if name == "val" else None
        prepared = []
        count = 0
        for symbol, frame in sorted(panel.items()):
            # Train panel already embodies the original protocol: holdout
            # symbols may occur only in their pre-cutoff historical portion.
            if name == "train" and symbol in heldout and (
                pd.Timestamp(frame.index.max()) > pd.Timestamp(split["train_cutoff"])
            ):
                raise ValueError(f"Holdout symbol crosses training cutoff: {symbol}")
            values, dates, indices = window_indices(frame, start, end)
            prepared.append((symbol, values, dates, indices))
            count += len(indices)
        if count == 0:
            raise ValueError(f"No eligible {name} windows")
        split_output = output / name
        split_output.mkdir()
        shard_rows = []
        shard_index = 0
        written = 0
        shard_manifest = []
        for symbol, values, dates, indices in prepared:
            for index in indices:
                shard_rows.append((
                    values[index:index + 120].copy(),
                    values[index + 120:index + 130, 3].copy(),
                    {"symbol": symbol, "start_index": int(index),
                     "asof_date": str(dates[index + 119]),
                     "label_end_date": str(dates[index + 129])},
                ))
                if len(shard_rows) == SHARD_SIZE or written + len(shard_rows) == count:
                    contexts = np.stack([row[0] for row in shard_rows])
                    targets = np.stack([row[1] for row in shard_rows])
                    filename = f"shard_{shard_index:05d}.npz"
                    shard_path = split_output / filename
                    np.savez_compressed(shard_path, contexts=contexts, targets=targets)
                    with (split_output / "identities.jsonl").open("a") as identities:
                        for row in shard_rows:
                            identities.write(json.dumps(row[2]) + "\n")
                    shard_manifest.append({
                        "file": filename, "samples": len(shard_rows),
                        "sha256": sha256(shard_path),
                    })
                    written += len(shard_rows)
                    shard_index += 1
                    shard_rows.clear()
                    del contexts, targets
        if written != count:
            raise RuntimeError(f"Shard count mismatch: {written} != {count}")
        result["splits"][name] = {
            "samples": count, "symbols": len(panel),
            "signal_start": start, "signal_end": end,
            "source_panel_sha256": sha256(path),
            "artifacts": {
                "shards": shard_manifest,
                "identities": sha256(split_output / "identities.jsonl"),
            },
        }
        print(json.dumps({"phase": "split_complete", "split": name, "samples": count}), flush=True)
        del prepared, panel
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/timesfm3_formal"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.parent / "formal_build_run.log"
    lock = threading.Lock()

    def heartbeat(event):
        line = json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "formal_dataset_build", "event": event,
        })
        with lock:
            print(line, flush=True)
            with log_path.open("a", buffering=1) as handle:
                handle.write(line + "\n")

    def monitor():
        while True:
            time.sleep(30)
            heartbeat("heartbeat")

    heartbeat("started")
    threading.Thread(target=monitor, daemon=True).start()
    prepare(discover(args.input_root), args.output)
    heartbeat("complete")
