"""Build a deterministic, materialized symbol-holdout training dataset."""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observed_history_bucket(first_date):
    year = pd.Timestamp(first_date).year
    if year <= 2014:
        return "through_2014"
    if year <= 2017:
        return "2015_2017"
    if year <= 2020:
        return "2018_2020"
    if year <= 2023:
        return "2021_2023"
    return "2024_plus"


def eligible_window_count(frame, lookback, predict, signal_start, signal_end):
    window = int(lookback) + int(predict) + 1
    sample_count = len(frame) - window + 1
    if sample_count <= 0:
        return 0
    starts = np.arange(sample_count, dtype=np.int32)
    asof_positions = starts + int(lookback) - 1
    dates = pd.DatetimeIndex(frame.index).to_numpy(dtype="datetime64[D]")
    asof_dates = dates[asof_positions]
    return int(
        np.count_nonzero(
            (asof_dates >= np.datetime64(signal_start, "D"))
            & (asof_dates <= np.datetime64(signal_end, "D"))
        )
    )


def profile_symbols(panel, reference_date, lookback, predict, signal_start, signal_end):
    reference = pd.Timestamp(reference_date)
    records = []
    for symbol in sorted(panel):
        frame = panel[symbol]
        if frame.empty:
            raise ValueError(f"Empty panel for symbol: {symbol}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"Panel index is not DatetimeIndex: {symbol}")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"Panel index is not sorted: {symbol}")
        reference_rows = frame.loc[frame.index <= reference]
        if reference_rows.empty:
            reference_row = frame.iloc[0]
            effective_reference_date = frame.index[0]
        else:
            reference_row = reference_rows.iloc[-1]
            effective_reference_date = reference_rows.index[-1]
        sector = reference_row.get("sector", "unknown")
        sector = "unknown" if pd.isna(sector) else str(sector)
        size_percentile = pd.to_numeric(
            reference_row.get("size_percentile", np.nan), errors="coerce"
        )
        if pd.isna(size_percentile):
            size_decile = -1
            size_percentile_value = None
        else:
            size_percentile_value = float(np.clip(size_percentile, 0.0, 1.0))
            size_decile = min(int(size_percentile_value * 10), 9)
        first_date = pd.Timestamp(frame.index.min())
        records.append(
            {
                "symbol": str(symbol),
                "sector": sector,
                "size_decile": int(size_decile),
                "size_percentile": size_percentile_value,
                "history_start": first_date.strftime("%Y-%m-%d"),
                "history_start_bucket": observed_history_bucket(first_date),
                "reference_date": pd.Timestamp(effective_reference_date).strftime(
                    "%Y-%m-%d"
                ),
                "rows": int(len(frame)),
                "eligible_windows": eligible_window_count(
                    frame, lookback, predict, signal_start, signal_end
                ),
            }
        )
    return records


def assign_stratified_split(records, seed, train_fraction):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two symbols are required for a holdout split")

    sector_counts = Counter(str(record["sector"]) for record in records)
    allocation_units = defaultdict(list)
    for record in records:
        sector = str(record["sector"])
        if sector_counts[sector] >= 5:
            unit_key = ("sector", sector)
        else:
            major_sector = sector[:1] if sector != "unknown" else "unknown"
            unit_key = ("rare_major_sector", major_sector)
        allocation_units[unit_key].append(record)

    target_train = int(round(len(records) * train_fraction))
    target_train = min(max(target_train, 1), len(records) - 1)
    target_validation = len(records) - target_train
    validation_fraction = 1.0 - train_fraction

    unit_allocations = []
    for key in sorted(allocation_units):
        rows = allocation_units[key]
        ideal_validation = len(rows) * validation_fraction
        validation_count = int(np.floor(ideal_validation))
        if key[0] == "sector":
            validation_count = max(validation_count, 1)
        unit_allocations.append(
            {
                "key": key,
                "rows": rows,
                "validation_count": validation_count,
                "remainder": ideal_validation - np.floor(ideal_validation),
            }
        )

    remaining = target_validation - sum(
        unit["validation_count"] for unit in unit_allocations
    )
    if remaining < 0:
        raise ValueError(
            "Validation target is too small to preserve the minimum sector coverage"
        )
    allocation_order = sorted(
        range(len(unit_allocations)),
        key=lambda position: (
            -unit_allocations[position]["remainder"],
            unit_allocations[position]["key"],
        ),
    )
    if remaining > len(allocation_order):
        raise RuntimeError("Sector allocation could not reach the validation target")
    for position in allocation_order[:remaining]:
        unit_allocations[position]["validation_count"] += 1

    rng = np.random.default_rng(seed)
    train = []
    validation = []
    for unit in unit_allocations:
        strata = defaultdict(list)
        for record in unit["rows"]:
            key = (int(record["size_decile"]), record["history_start_bucket"])
            strata[key].append(record)

        stratum_allocations = []
        for key in sorted(strata):
            rows = sorted(strata[key], key=lambda item: item["symbol"])
            order = rng.permutation(len(rows))
            rows = [rows[int(position)] for position in order]
            ideal_validation = (
                len(rows) * unit["validation_count"] / len(unit["rows"])
            )
            stratum_allocations.append(
                {
                    "key": key,
                    "rows": rows,
                    "validation_count": int(np.floor(ideal_validation)),
                    "remainder": ideal_validation - np.floor(ideal_validation),
                }
            )

        stratum_remaining = unit["validation_count"] - sum(
            stratum["validation_count"] for stratum in stratum_allocations
        )
        stratum_order = sorted(
            range(len(stratum_allocations)),
            key=lambda position: (
                -stratum_allocations[position]["remainder"],
                stratum_allocations[position]["key"],
            ),
        )
        for position in stratum_order[:stratum_remaining]:
            stratum_allocations[position]["validation_count"] += 1

        for stratum in stratum_allocations:
            validation_count = stratum["validation_count"]
            validation.extend(stratum["rows"][:validation_count])
            train.extend(stratum["rows"][validation_count:])

    if len(train) != target_train or len(validation) != len(records) - target_train:
        raise RuntimeError("Stratified allocation did not produce the target split")
    train_symbols = {row["symbol"] for row in train}
    validation_symbols = {row["symbol"] for row in validation}
    if train_symbols & validation_symbols:
        raise RuntimeError("Train and validation symbol sets overlap")
    if train_symbols | validation_symbols != {row["symbol"] for row in records}:
        raise RuntimeError("Split does not cover the source symbol universe")
    return train_symbols, validation_symbols


def value_counts(records, symbols, field):
    return dict(
        sorted(
            Counter(
                str(record[field]) for record in records if record["symbol"] in symbols
            ).items()
        )
    )


def build_audit(records, train_symbols, validation_symbols):
    fields = ("sector", "size_decile", "history_start_bucket")
    distributions = {}
    for field in fields:
        train_counts = value_counts(records, train_symbols, field)
        validation_counts = value_counts(records, validation_symbols, field)
        keys = sorted(set(train_counts) | set(validation_counts))
        distributions[field] = {
            "train": train_counts,
            "validation": validation_counts,
            "max_absolute_count_delta": max(
                (
                    abs(train_counts.get(key, 0) - validation_counts.get(key, 0))
                    for key in keys
                ),
                default=0,
            ),
        }
    by_symbol = {record["symbol"]: record for record in records}
    sector_counts = Counter(str(record["sector"]) for record in records)
    validation_sector_counts = value_counts(records, validation_symbols, "sector")
    required_validation_sectors = {
        sector for sector, count in sector_counts.items() if count >= 5
    }
    return {
        "symbols": {
            "total": len(records),
            "train": len(train_symbols),
            "validation": len(validation_symbols),
            "intersection": len(train_symbols & validation_symbols),
        },
        "rows": {
            "train": sum(by_symbol[symbol]["rows"] for symbol in train_symbols),
            "validation": sum(
                by_symbol[symbol]["rows"] for symbol in validation_symbols
            ),
        },
        "eligible_windows": {
            "train": sum(
                by_symbol[symbol]["eligible_windows"] for symbol in train_symbols
            ),
            "validation": sum(
                by_symbol[symbol]["eligible_windows"]
                for symbol in validation_symbols
            ),
        },
        "sector_coverage": {
            "source": len(sector_counts),
            "validation": len(validation_sector_counts),
            "minimum_source_symbols": 5,
            "required": len(required_validation_sectors),
            "missing_required": sorted(
                required_validation_sectors - set(validation_sector_counts)
            ),
        },
        "distributions": distributions,
    }


def write_pickle(path, panel, symbols):
    subset = {symbol: panel[symbol] for symbol in sorted(symbols)}
    with path.open("wb") as handle:
        pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_dataset(args):
    source_root = Path(args.source_root).resolve()
    source_panel_path = source_root / args.source_panel
    source_manifest_path = source_root / "data_manifest.json"
    source_metadata_path = source_root / "asset_metadata.csv"
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    for path in (source_panel_path, source_manifest_path, source_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = json.loads(source_manifest_path.read_text())
    expected_source_sha = next(
        (
            item["sha256"]
            for item in source_manifest.get("files", [])
            if item.get("name") == Path(args.source_panel).name
        ),
        None,
    )
    actual_source_sha = sha256_file(source_panel_path)
    if expected_source_sha and actual_source_sha != expected_source_sha:
        raise ValueError(
            f"Source panel SHA mismatch: {actual_source_sha} != {expected_source_sha}"
        )

    with source_panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    records = profile_symbols(
        panel,
        args.reference_date,
        args.lookback,
        args.predict,
        args.signal_start,
        args.signal_end,
    )
    train_symbols, validation_symbols = assign_stratified_split(
        records, args.seed, args.train_fraction
    )
    audit = build_audit(records, train_symbols, validation_symbols)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    try:
        dataset_root = temporary_root / "processed_datasets"
        dataset_root.mkdir()
        for record in records:
            record["split"] = (
                "train" if record["symbol"] in train_symbols else "validation"
            )
        split_frame = pd.DataFrame.from_records(records).sort_values(
            ["split", "sector", "size_decile", "history_start_bucket", "symbol"]
        )
        split_path = temporary_root / "symbol_split.csv"
        split_frame.to_csv(split_path, index=False)

        train_path = dataset_root / "train_data.pkl"
        validation_path = dataset_root / "val_data.pkl"
        write_pickle(train_path, panel, train_symbols)
        write_pickle(validation_path, panel, validation_symbols)
        metadata_link = os.path.relpath(source_metadata_path, temporary_root)
        os.symlink(metadata_link, temporary_root / "asset_metadata.csv")

        files = []
        for path in (train_path, validation_path, split_path):
            files.append(
                {
                    "name": str(path.relative_to(temporary_root)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        split_contract = {
            "schema_version": 1,
            "name": args.name,
            "source": {
                "data_manifest_sha256": source_manifest.get("manifest_sha256"),
                "panel": str(source_panel_path.relative_to(source_root)),
                "panel_sha256": actual_source_sha,
                "asset_metadata": os.path.relpath(source_metadata_path, output_root),
                "asset_metadata_sha256": sha256_file(source_metadata_path),
            },
            "split": {
                "unit": "symbol",
                "train_fraction": args.train_fraction,
                "validation_fraction": 1.0 - args.train_fraction,
                "seed": args.seed,
                "allocation": "sector quota followed by within-sector largest-remainder stratification",
                "strata": ["sector", "size_decile", "history_start_bucket"],
                "minimum_validation_sector_size": 5,
                "rare_sector_pool": "first character of the sector code",
                "reference_date": args.reference_date,
                "history_start_semantics": "first date available in the source panel; proxy for listing age",
                "train_label": "train",
                "validation_label": "validation",
            },
            "window_contract": {
                "lookback": args.lookback,
                "predict": args.predict,
                "signal_start": args.signal_start,
                "signal_end": args.signal_end,
            },
            "audit": audit,
            "files": files,
            "limitations": [
                "The source panel begins in 2014, so older listing dates collapse into the through_2014 history bucket.",
                "This split measures cross-symbol transfer during a shared market regime, not unseen-time generalization.",
                "The parent checkpoint may have seen validation symbols in earlier training stages.",
            ],
        }
        split_contract["manifest_sha256"] = canonical_sha256(split_contract)
        (temporary_root / "data_manifest.json").write_text(
            json.dumps(split_contract, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        (temporary_root / "split_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest_sha256": split_contract["manifest_sha256"],
                "audit": audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a deterministic, stratified symbol-holdout dataset"
    )
    parser.add_argument(
        "--source-root", default="data/a_share_full_market_v1_beta"
    )
    parser.add_argument(
        "--source-panel", default="processed_datasets/train_data.pkl"
    )
    parser.add_argument(
        "--output-root",
        default="data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1",
    )
    parser.add_argument(
        "--name", default="a_share_full_market_v1_beta_symbol_holdout_90_10_v1"
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--reference-date", default="2026-06-30")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--predict", type=int, default=10)
    parser.add_argument("--signal-start", default="2015-01-01")
    parser.add_argument("--signal-end", default="2026-07-17")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
