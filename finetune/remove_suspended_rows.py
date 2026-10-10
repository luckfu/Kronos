"""Create a versioned panel without zero-volume rows; keep the original dates."""

import argparse
import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_panel(panel):
    cleaned = {}
    stats = {"symbols": len(panel), "rows_before": 0, "rows_after": 0,
             "zero_volume_removed": 0, "symbols_shorter_than_131": 0}
    for symbol, frame in panel.items():
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"Unsorted or duplicate dates: {symbol}")
        volume = pd.to_numeric(frame["volume"], errors="raise")
        if volume.isna().any() or (volume < 0).any():
            raise ValueError(f"Invalid volume: {symbol}")
        result = frame.loc[volume > 0].copy()
        if result.empty:
            raise ValueError(f"No traded rows remain: {symbol}")
        cleaned[symbol] = result
        stats["rows_before"] += len(frame)
        stats["rows_after"] += len(result)
        stats["zero_volume_removed"] += len(frame) - len(result)
        stats["symbols_shorter_than_131"] += len(result) < 131
    return cleaned, stats


def eligible_windows(frame, start, end):
    if len(frame) < 131:
        return 0
    asof = frame.index[119:len(frame) - 11]
    return int(((asof >= start) & (asof <= end)).sum())


def build(source, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "data_manifest.json").read_text())
    split = pd.read_csv(source / "symbol_split.csv", dtype={"symbol": str})
    if split.symbol.duplicated().any():
        raise ValueError("Duplicate symbols in split")
    source_paths = {
        name: source / "processed_datasets" / f"{name}_data.pkl"
        for name in ("train", "val")
    }
    source_paths["symbol_split"] = source / "symbol_split.csv"
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    source_hashes["data_manifest"] = sha256_file(source / "data_manifest.json")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / "processed_datasets").mkdir()
        panels = {}
        stats = {}
        for name in ("train", "val"):
            with source_paths[name].open("rb") as handle:
                panel = pickle.load(handle)
            panels[name], stats[name] = clean_panel(panel)
            del panel
            with (temporary / "processed_datasets" / f"{name}_data.pkl").open("wb") as handle:
                pickle.dump(panels[name], handle, protocol=pickle.HIGHEST_PROTOCOL)

        train_symbols, val_symbols = set(panels["train"]), set(panels["val"])
        if train_symbols | val_symbols != set(split.symbol):
            raise ValueError("Panel symbols do not match split")
        cutoff = pd.Timestamp(manifest["split"]["train_cutoff"])
        val_start = pd.Timestamp(manifest["split"]["validation_start"])
        for symbol in train_symbols & val_symbols:
            if panels["train"][symbol].index.max() > cutoff or panels["val"][symbol].index.min() < val_start:
                raise ValueError(f"Temporal split overlap: {symbol}")

        window = manifest["window_contract"]
        start, end = pd.Timestamp(window["validation_signal_start"]), pd.Timestamp(window["validation_signal_end"])
        rows, windows = [], []
        for symbol in split.symbol:
            parts = [panels[name][symbol] for name in ("train", "val") if symbol in panels[name]]
            combined = pd.concat(parts).sort_index()
            if combined.index.has_duplicates:
                raise ValueError(f"Duplicate dates across splits: {symbol}")
            rows.append(len(combined))
            windows.append(eligible_windows(combined, start, end))
        split["rows"] = rows
        split["eligible_windows"] = windows
        split.to_csv(temporary / "symbol_split.csv", index=False)
        shutil.copy2(source / "asset_metadata.csv", temporary / "asset_metadata.csv")

        paths = [temporary / "processed_datasets" / f"{name}_data.pkl" for name in ("train", "val")]
        paths += [temporary / "symbol_split.csv", temporary / "asset_metadata.csv"]
        audit = {
            "schema_version": 1,
            "source": {"directory": str(source.resolve()), "sha256": source_hashes},
            "filter": "Remove rows with volume == 0 from both panels; preserve DatetimeIndex and all other fields",
            "window_semantics": "131 consecutive traded rows per symbol, not 131 consecutive market sessions; forecast is the next 10 traded rows",
            "split": {**manifest["split"], "symbol_intersection": len(train_symbols & val_symbols)},
            "window_contract": window,
            "stats": stats,
            "files": [
                {"name": str(path.relative_to(temporary)), "size": path.stat().st_size,
                 "sha256": sha256_file(path)}
                for path in paths
            ],
            "notes": [
                "Source symbol_split rows and eligible_windows were recomputed after filtering.",
                "Point-in-time asset_metadata is unchanged; zero-volume dates there are not panel samples.",
                "The original training and validation files and deployed model are unchanged.",
            ],
        }
        (temporary / "data_manifest.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        return audit
    except Exception:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/a_share_full_market_v1_beta_temporal_symbol_validation_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/a_share_full_market_v1_beta_temporal_symbol_validation_no_suspension_v1"))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output)["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
