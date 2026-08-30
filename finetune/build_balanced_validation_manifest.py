"""Build deterministic direction-balanced validation subsets for v1-beta."""

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DIRECTION_NAMES = {-1: "short", 0: "neutral", 1: "long"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_identity(row):
    return "|".join(
        (
            str(row.symbol),
            str(int(row.start_index)),
            str(row.asof_date),
            str(row.target_date),
        )
    )


def identities_sha256(frame):
    digest = hashlib.sha256()
    for identity in sorted(sample_identity(row) for row in frame.itertuples()):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def training_overlap_audit(
    samples, train_panel, lookback, predict, signal_start, signal_end
):
    """Classify fixed samples by whether the exact as-of window is trainable."""
    requested = defaultdict(list)
    for row in samples.itertuples():
        requested[str(row.symbol)].append(row)
    matched = []
    not_in_training = []
    window = int(lookback) + int(predict) + 1
    start_date = np.datetime64(signal_start, "D")
    end_date = np.datetime64(signal_end, "D")
    for symbol, rows in requested.items():
        source = train_panel.get(symbol)
        if source is None:
            not_in_training.extend(rows)
            continue
        frame = source.reset_index()
        sample_count = len(frame) - window + 1
        available_dates = set()
        if sample_count > 0:
            starts = np.arange(sample_count, dtype=np.int32)
            asof_positions = starts + int(lookback) - 1
            dates = frame["datetime"].to_numpy(dtype="datetime64[D]")[asof_positions]
            eligible = (dates >= start_date) & (dates <= end_date)
            available_dates = set(dates[eligible].astype(str))
        for row in rows:
            if str(row.asof_date) in available_dates:
                matched.append(row)
            else:
                not_in_training.append(row)
    return matched, not_in_training


def build_candidate_frame(panel, lookback, predict, signal_start, signal_end):
    records = []
    window = lookback + predict + 1
    start_date = np.datetime64(signal_start, "D")
    end_date = np.datetime64(signal_end, "D")
    for symbol, source in panel.items():
        frame = source.reset_index()
        sample_count = len(frame) - window + 1
        if sample_count <= 0:
            continue
        starts = np.arange(sample_count, dtype=np.int32)
        asof_positions = starts + lookback - 1
        target_positions = asof_positions + predict
        dates = frame["datetime"].to_numpy(dtype="datetime64[D]")
        eligible = (dates[asof_positions] >= start_date) & (
            dates[asof_positions] <= end_date
        )
        positions = np.flatnonzero(eligible)
        if not len(positions):
            continue

        asof = asof_positions[positions]
        target = target_positions[positions]
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        returns = close[target] / close[asof] - 1.0
        directions = np.where(returns < -0.01, -1, np.where(returns > 0.01, 1, 0))
        percentiles = pd.to_numeric(
            frame["size_percentile"].iloc[asof], errors="coerce"
        ).to_numpy(dtype=np.float64)
        size_deciles = np.full(len(asof), -1, dtype=np.int8)
        known_size = np.isfinite(percentiles)
        size_deciles[known_size] = np.minimum(
            (percentiles[known_size] * 10).astype(np.int8), 9
        )
        sectors = (
            frame["sector"].iloc[asof].fillna("unknown").astype(str).to_numpy()
        )
        asof_dates = dates[asof].astype(str)
        target_dates = dates[target].astype(str)
        for offset, position in enumerate(positions):
            records.append(
                {
                    "symbol": str(symbol),
                    "start_index": int(starts[position]),
                    "asof_date": asof_dates[offset],
                    "target_date": target_dates[offset],
                    "direction": int(directions[offset]),
                    "return_10d": float(returns[offset]),
                    "sector": sectors[offset],
                    "size_decile": int(size_deciles[offset]),
                    "month": asof_dates[offset][:7],
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("No validation candidates matched the configured signal range")
    if result.duplicated(["symbol", "start_index"]).any():
        raise ValueError("Validation candidate identities are not unique")
    return result


def balanced_select(frame, per_direction, seed):
    """Round-robin month/size/sector strata within each direction."""
    chosen = []
    for direction in (-1, 0, 1):
        direction_frame = frame[frame["direction"] == direction]
        if len(direction_frame) < per_direction:
            raise ValueError(
                f"Direction {DIRECTION_NAMES[direction]} has {len(direction_frame):,} "
                f"candidates, fewer than requested {per_direction:,}"
            )
        rng = np.random.default_rng(seed + direction + 1)
        groups = defaultdict(list)
        for row_index, row in direction_frame.iterrows():
            groups[(row["month"], int(row["size_decile"]), row["sector"])].append(
                int(row_index)
            )
        for values in groups.values():
            rng.shuffle(values)
        cursors = {key: 0 for key in groups}
        active = list(groups)
        selected = []
        while len(selected) < per_direction:
            rng.shuffle(active)
            next_active = []
            for key in active:
                cursor = cursors[key]
                values = groups[key]
                if cursor < len(values):
                    selected.append(values[cursor])
                    cursor += 1
                    cursors[key] = cursor
                    if len(selected) == per_direction:
                        break
                if cursor < len(values):
                    next_active.append(key)
            if len(selected) == per_direction:
                break
            if not next_active:
                raise RuntimeError("Balanced validation allocation exhausted unexpectedly")
            active = next_active
        chosen.extend(selected)
    rng = np.random.default_rng(seed + 1000)
    rng.shuffle(chosen)
    return frame.loc[chosen].reset_index(drop=True)


def distribution(frame):
    directions = Counter(DIRECTION_NAMES[int(value)] for value in frame["direction"])
    size_deciles = Counter(str(int(value)) for value in frame["size_decile"])
    sector_counts = Counter(frame["sector"].astype(str))
    date_counts = Counter(frame["asof_date"].astype(str))
    if not len(frame):
        return {
            "samples": 0,
            "direction_counts": {},
            "direction_fractions": {},
            "date_start": None,
            "date_end": None,
            "date_count": 0,
            "date_min_samples": 0,
            "date_max_samples": 0,
            "sector_count": 0,
            "sector_counts": {},
            "size_decile_counts": {},
        }
    return {
        "samples": int(len(frame)),
        "direction_counts": dict(sorted(directions.items())),
        "direction_fractions": {
            key: value / len(frame) for key, value in sorted(directions.items())
        },
        "date_start": min(date_counts),
        "date_end": max(date_counts),
        "date_count": len(date_counts),
        "date_min_samples": min(date_counts.values()),
        "date_max_samples": max(date_counts.values()),
        "sector_count": len(sector_counts),
        "sector_counts": dict(sorted(sector_counts.items())),
        "size_decile_counts": dict(sorted(size_deciles.items(), key=lambda item: int(item[0]))),
    }


def build_manifest(args):
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = data_root / "processed_datasets" / "val_data.pkl"
    train_path = data_root / "processed_datasets" / "train_data.pkl"
    data_manifest_path = data_root / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    with val_path.open("rb") as handle:
        panel = pickle.load(handle)

    candidates = build_candidate_frame(
        panel,
        args.lookback,
        args.predict,
        args.signal_start,
        args.signal_end,
    )
    reserve_start = np.datetime64(args.reserve_start, "D")
    asof_dates = candidates["asof_date"].to_numpy(dtype="datetime64[D]")
    tuning_pool = candidates.loc[asof_dates < reserve_start].reset_index(drop=True)
    reserved = candidates.loc[asof_dates >= reserve_start].reset_index(drop=True)

    large = balanced_select(tuning_pool, args.large_per_direction, args.seed)
    quick = balanced_select(large, args.quick_per_direction, args.seed + 10_000)
    quick_identities = {sample_identity(row) for row in quick.itertuples()}
    large = pd.concat(
        [
            large[large.apply(lambda row: sample_identity(row) in quick_identities, axis=1)],
            large[large.apply(lambda row: sample_identity(row) not in quick_identities, axis=1)],
        ],
        ignore_index=True,
    )

    with train_path.open("rb") as handle:
        train_panel = pickle.load(handle)
    training_overlap, not_in_training = training_overlap_audit(
        large,
        train_panel,
        args.lookback,
        args.predict,
        args.train_signal_start,
        args.train_signal_end,
    )
    if len(training_overlap) + len(not_in_training) != len(large):
        raise RuntimeError("Training overlap audit did not partition the fixed set")

    samples_path = output_dir / "balanced_validation_samples.jsonl"
    with samples_path.open("w") as handle:
        for row in large.itertuples():
            identity = sample_identity(row)
            handle.write(
                json.dumps(
                    {
                        "symbol": row.symbol,
                        "start_index": int(row.start_index),
                        "asof_date": row.asof_date,
                        "target_date": row.target_date,
                        "direction": DIRECTION_NAMES[int(row.direction)],
                        "return_10d": float(row.return_10d),
                        "sector": row.sector,
                        "size_decile": int(row.size_decile),
                        "quick": identity in quick_identities,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    manifest = {
        "schema_version": 1,
        "name": "a_share_v1_beta_fixed_direction_balanced_validation_v1",
        "source": {
            "data_manifest_sha256": data_manifest["manifest_sha256"],
            "val_data_sha256": sha256_file(val_path),
            "train_data_sha256": sha256_file(train_path),
            "lookback": args.lookback,
            "predict": args.predict,
            "signal_start": args.signal_start,
            "signal_end": args.signal_end,
            "candidate_pool_samples": int(len(candidates)),
            "candidate_pool_identities_sha256": identities_sha256(candidates),
        },
        "direction_definition": {
            "return": "close[target_10] / close[input_last] - 1",
            "short": "return < -0.01",
            "neutral": "-0.01 <= return <= 0.01",
            "long": "return > 0.01",
        },
        "selection": {
            "seed": args.seed,
            "strata": ["direction", "month", "size_decile", "sector"],
            "quick_samples": int(len(quick)),
            "large_samples": int(len(large)),
            "quick_is_subset_of_large": True,
            "quick_identities_sha256": identities_sha256(quick),
            "large_identities_sha256": identities_sha256(large),
            "samples_file": samples_path.name,
            "samples_file_sha256": sha256_file(samples_path),
        },
        "training_isolation": {
            "exclusion_key": ["symbol", "asof_date"],
            "train_signal_start": args.train_signal_start,
            "train_signal_end": args.train_signal_end,
            "training_candidate_overlap_samples": len(training_overlap),
            "not_in_training_candidate_samples": len(not_in_training),
            "all_training_overlaps_must_be_excluded": True,
            "not_in_training_candidates": [
                {
                    "symbol": row.symbol,
                    "start_index": int(row.start_index),
                    "asof_date": row.asof_date,
                    "target_date": row.target_date,
                }
                for row in not_in_training
            ],
        },
        "audit": {
            "candidate_pool": distribution(candidates),
            "tuning_pool": distribution(tuning_pool),
            "quick": distribution(quick),
            "large": distribution(large),
            "reserved_tuning_holdout": distribution(reserved),
        },
        "final_test_isolation": {
            "reserved_tuning_holdout_start": args.reserve_start,
            "reserved_tuning_holdout_excluded_from_quick_and_large": True,
            "reserved_tuning_holdout_identities_sha256": identities_sha256(reserved),
            "deployment_final_test_status": "pending_future_data",
            "reason": (
                "All currently labelled 2026 validation windows overlap the historical "
                "initialization training period. The reserved tuning holdout is isolated "
                "from unified-stage model selection but is not a deployment-grade unseen "
                "test. Final deployment evaluation requires newly labelled future data."
            ),
        },
    }
    manifest_path = output_dir / "balanced_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "samples": str(samples_path),
        "candidate_directions": manifest["audit"]["candidate_pool"]["direction_counts"],
        "quick": manifest["audit"]["quick"],
        "large": manifest["audit"]["large"],
        "reserved_samples": len(reserved),
    }, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/a_share_full_market_v1_beta")
    parser.add_argument(
        "--output-dir",
        default="data/a_share_full_market_v1_beta/balanced_validation_v1",
    )
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--predict", type=int, default=10)
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-07-17")
    parser.add_argument("--reserve-start", default="2026-07-01")
    parser.add_argument("--train-signal-start", default="2015-01-01")
    parser.add_argument("--train-signal-end", default="2026-07-17")
    parser.add_argument("--quick-per-direction", type=int, default=1000)
    parser.add_argument("--large-per-direction", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


if __name__ == "__main__":
    build_manifest(parse_args())
