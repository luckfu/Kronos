"""Prepare a signed, temporally isolated v1-beta checkpoint evaluation set."""

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT + 1
FEATURES = ["open", "high", "low", "close", "volume", "amount"]
DIRECTION_NAMES = {-1: "short", 0: "neutral", 1: "long"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_identity(record):
    return "|".join(
        (
            str(record["symbol"]),
            str(int(record["start_index"])),
            str(record["asof_date"]),
            str(record["target_date"]),
        )
    )


def identities_sha256(records):
    digest = hashlib.sha256()
    for identity in sorted(sample_identity(record) for record in records):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def direction_name(value):
    direction = -1 if value < -0.01 else 1 if value > 0.01 else 0
    return DIRECTION_NAMES[direction]


def append_august(base_panel, august_path):
    august = pd.read_csv(august_path, parse_dates=["date"])
    required = {"symbol", "date", "market_cap", *FEATURES}
    missing = sorted(required - set(august.columns))
    if missing:
        raise ValueError(f"August data is missing columns: {missing}")
    if august.duplicated(["symbol", "date"]).any():
        raise ValueError("August data contains duplicate symbol/date rows")

    august["size_percentile"] = august.groupby("date")["market_cap"].rank(
        method="first", pct=True
    )
    august["size_bucket"] = np.minimum(
        np.floor(august["size_percentile"] * 10), 9
    )
    latest_sector = {
        str(symbol): str(frame["sector"].dropna().iloc[-1])
        if "sector" in frame.columns and not frame["sector"].dropna().empty
        else "unknown"
        for symbol, frame in base_panel.items()
    }
    august["sector"] = august["symbol"].astype(str).map(latest_sector).fillna(
        "unknown"
    )

    additions = {
        str(symbol): rows.set_index("date")[
            FEATURES + ["size_bucket", "size_percentile", "sector"]
        ].sort_index()
        for symbol, rows in august.groupby("symbol", sort=False)
    }
    combined = {}
    for symbol, source in base_panel.items():
        symbol = str(symbol)
        frame = source.copy()
        frame.index = pd.to_datetime(frame.index)
        if symbol in additions:
            frame = pd.concat([frame, additions[symbol]])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        combined[symbol] = frame
    return combined, august


def build_candidates(panel, signal_start, signal_end):
    start = np.datetime64(signal_start, "D")
    end = np.datetime64(signal_end, "D")
    records = []
    for symbol, frame in panel.items():
        if len(frame) < WINDOW:
            continue
        dates = frame.index.to_numpy(dtype="datetime64[D]")
        starts = np.arange(len(frame) - WINDOW + 1, dtype=np.int32)
        asof_positions = starts + LOOKBACK - 1
        target_positions = asof_positions + PREDICT
        eligible = (dates[asof_positions] >= start) & (dates[asof_positions] <= end)
        selected = np.flatnonzero(eligible)
        if not len(selected):
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        for offset in selected:
            asof_position = int(asof_positions[offset])
            target_position = int(target_positions[offset])
            realized = float(close[target_position] / close[asof_position] - 1.0)
            records.append(
                {
                    "symbol": str(symbol),
                    "start_index": int(starts[offset]),
                    "asof_date": str(dates[asof_position]),
                    "target_date": str(dates[target_position]),
                    "return_10d": realized,
                    "direction": direction_name(realized),
                    "sector": str(frame["sector"].iloc[asof_position]),
                    "size_decile": int(
                        min(
                            np.floor(float(frame["size_percentile"].iloc[asof_position]) * 10),
                            9,
                        )
                    ),
                }
            )
    if not records:
        raise ValueError(f"No candidates in {signal_start}..{signal_end}")
    return records


def record_distribution(records):
    directions = Counter(record["direction"] for record in records)
    dates = Counter(record["asof_date"] for record in records)
    sectors = Counter(record["sector"] for record in records)
    sizes = Counter(str(record["size_decile"]) for record in records)
    return {
        "samples": len(records),
        "direction_counts": dict(sorted(directions.items())),
        "direction_fractions": {
            key: value / len(records) for key, value in sorted(directions.items())
        },
        "date_start": min(dates),
        "date_end": max(dates),
        "date_count": len(dates),
        "date_min_samples": min(dates.values()),
        "date_max_samples": max(dates.values()),
        "sector_count": len(sectors),
        "size_decile_counts": dict(sorted(sizes.items(), key=lambda item: int(item[0]))),
    }


def deterministic_sample(records, count, seed):
    if len(records) < count:
        raise ValueError(f"Only {len(records):,} records; {count:,} requested")
    positions = np.random.default_rng(seed).choice(len(records), count, replace=False)
    return [records[int(position)] for position in sorted(positions)]


def balanced_sample(records, per_direction, seed):
    selected = []
    for index, direction in enumerate(("short", "neutral", "long")):
        group = [record for record in records if record["direction"] == direction]
        selected.extend(deterministic_sample(group, per_direction, seed + index))
    return sorted(selected, key=sample_identity)


def load_balanced_history(path):
    records = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        record.pop("quick", None)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-panel",
        default="data/a_share_full_market_v1_beta/processed_datasets/val_data.pkl",
    )
    parser.add_argument(
        "--august-data",
        default="data/a_share_v1_beta_eval_20260826/august_raw.csv",
    )
    parser.add_argument(
        "--balanced-history",
        default=(
            "data/a_share_full_market_v1_beta/balanced_validation_v1/"
            "balanced_validation_samples.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir", default="data/a_share_v1_beta_eval_20260826/package"
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--historical-natural-samples", type=int, default=12000)
    parser.add_argument("--future-balanced-per-direction", type=int, default=1000)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.base_panel).open("rb") as handle:
        base_panel = pickle.load(handle)
    panel, august = append_august(base_panel, args.august_data)

    panel_path = output / "evaluation_panel.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    future_all = build_candidates(panel, "2026-08-03", "2026-08-25")
    if min(record["asof_date"] for record in future_all) <= "2026-07-31":
        raise RuntimeError("Future evaluation overlaps the parent's labelled data")
    historical_candidates = build_candidates(panel, "2026-01-05", "2026-06-30")
    historical_natural = deterministic_sample(
        historical_candidates, args.historical_natural_samples, args.seed
    )
    historical_balanced = load_balanced_history(args.balanced_history)
    future_balanced = balanced_sample(
        future_all, args.future_balanced_per_direction, args.seed + 100
    )

    sample_sets = {
        "future_all": future_all,
        "future_balanced": future_balanced,
        "historical_natural": historical_natural,
        "historical_balanced": historical_balanced,
    }
    samples_path = output / "evaluation_samples.jsonl"
    with samples_path.open("w") as handle:
        for set_name, records in sample_sets.items():
            for record in records:
                handle.write(
                    json.dumps(
                        {"set": set_name, **record}, ensure_ascii=False, sort_keys=True
                    )
                    + "\n"
                )

    sectors = sorted(
        {
            str(value)
            for frame in panel.values()
            for value in frame["sector"].dropna().unique()
        }
    )
    if len(sectors) != 86:
        raise RuntimeError(f"Expected 86 sector labels, found {len(sectors)}")
    manifest = {
        "schema_version": 1,
        "name": "kronos_v1_beta_checkpoint_evaluation_20260826",
        "seed": args.seed,
        "model_contract": {
            "lookback": LOOKBACK,
            "predict": PREDICT,
            "window": WINDOW,
            "num_sectors": 86,
            "sector_labels": sectors,
            "use_size_percentile": True,
            "context_layer": 10,
        },
        "temporal_isolation": {
            "parent_training_signal_end": "2026-07-17",
            "parent_latest_training_target": "2026-07-31",
            "future_signal_start": min(record["asof_date"] for record in future_all),
            "future_signal_end": max(record["asof_date"] for record in future_all),
            "future_target_start": min(record["target_date"] for record in future_all),
            "future_target_end": max(record["target_date"] for record in future_all),
            "strictly_after_parent_latest_training_target": True,
        },
        "source": {
            "base_panel": str(Path(args.base_panel)),
            "base_panel_sha256": sha256_file(args.base_panel),
            "august_data": str(Path(args.august_data)),
            "august_data_sha256": sha256_file(args.august_data),
            "august_rows": len(august),
            "august_symbols": int(august["symbol"].nunique()),
            "august_date_start": str(august["date"].min().date()),
            "august_date_end": str(august["date"].max().date()),
        },
        "artifacts": {
            "panel_file": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "samples_file": samples_path.name,
            "samples_sha256": sha256_file(samples_path),
        },
        "sample_sets": {
            name: {
                **record_distribution(records),
                "identities_sha256": identities_sha256(records),
            }
            for name, records in sample_sets.items()
        },
        "limitations": [
            "Future evaluation spans only the newly labelled August trading dates available on 2026-08-26.",
            "Historical natural and balanced sets were visible to the parent training stage and are diagnostics only.",
        ],
    }
    manifest_path = output / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
