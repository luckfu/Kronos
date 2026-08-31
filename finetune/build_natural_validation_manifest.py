"""Build a deterministic v1-beta validation set with natural daily direction mix."""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_balanced_validation_manifest import (
    DIRECTION_NAMES,
    build_candidate_frame,
    distribution,
    identities_sha256,
    sample_identity,
    sha256_file,
    training_overlap_audit,
)


def daily_quotas(frame, count, seed):
    """Allocate a nearly equal sample count to each as-of date."""
    dates = sorted(frame["asof_date"].astype(str).unique())
    if count > len(frame):
        raise ValueError(f"Requested {count:,} samples from only {len(frame):,}")
    base, extra = divmod(count, len(dates))
    rng = np.random.default_rng(seed)
    extras = set(rng.choice(dates, size=extra, replace=False)) if extra else set()
    quotas = {date: base + int(date in extras) for date in dates}
    exhausted = [date for date, quota in quotas.items()
                 if quota > int((frame["asof_date"] == date).sum())]
    if exhausted:
        raise ValueError(f"Daily quota exceeds candidates for {exhausted[:5]}")
    return quotas


def proportional_direction_counts(frame, count):
    total = len(frame)
    raw = {
        direction: count * int((frame["direction"] == direction).sum()) / total
        for direction in (-1, 0, 1)
    }
    counts = {direction: int(np.floor(value)) for direction, value in raw.items()}
    remainder = count - sum(counts.values())
    ranked = sorted(raw, key=lambda direction: (raw[direction] - counts[direction], direction), reverse=True)
    for direction in ranked[:remainder]:
        counts[direction] += 1
    return counts


def stratified_pick(frame, count, seed):
    """Deterministic round-robin across size and sector within one daily direction."""
    if count == 0:
        return []
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for index, row in frame.iterrows():
        groups[(int(row["size_decile"]), str(row["sector"]))].append(int(index))
    for values in groups.values():
        rng.shuffle(values)
    active = list(groups)
    rng.shuffle(active)
    cursors = {key: 0 for key in active}
    selected = []
    while len(selected) < count:
        next_active = []
        rng.shuffle(active)
        for key in active:
            cursor = cursors[key]
            values = groups[key]
            if cursor < len(values):
                selected.append(values[cursor])
                cursors[key] = cursor + 1
                if len(selected) == count:
                    break
            if cursors[key] < len(values):
                next_active.append(key)
        if len(selected) == count:
            break
        if not next_active:
            raise RuntimeError("Natural-proportion selection exhausted unexpectedly")
        active = next_active
    return selected


def natural_select(frame, count, seed):
    if count == len(frame):
        return frame.sort_values(
            [
                "asof_date",
                "direction",
                "size_decile",
                "sector",
                "symbol",
                "start_index",
            ],
            kind="stable",
        ).reset_index(drop=True)
    quotas = daily_quotas(frame, count, seed)
    selected = []
    for offset, date in enumerate(sorted(quotas)):
        daily = frame.loc[frame["asof_date"] == date]
        for direction, direction_count in proportional_direction_counts(
            daily, quotas[date]
        ).items():
            subset = daily.loc[daily["direction"] == direction]
            if direction_count > len(subset):
                raise ValueError(f"Insufficient {direction} samples on {date}")
            selected.extend(stratified_pick(subset, direction_count, seed + offset * 17 + direction))
    result = frame.loc[selected].reset_index(drop=True)
    if len(result) != count:
        raise RuntimeError(f"Selected {len(result):,}, expected {count:,}")
    return result


def per_date_direction_audit(frame):
    result = {}
    for date, daily in frame.groupby("asof_date", sort=True):
        counts = Counter(DIRECTION_NAMES[int(value)] for value in daily["direction"])
        result[str(date)] = {
            "samples": int(len(daily)),
            "direction_counts": dict(sorted(counts.items())),
            "direction_fractions": {
                direction: count / len(daily)
                for direction, count in sorted(counts.items())
            },
        }
    return result


def validation_period(asof_date, split_date):
    return "2025H2" if str(asof_date) < split_date else "2026H1"


def build_manifest(args):
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = data_root / args.validation_panel
    train_path = data_root / "processed_datasets/train_data.pkl"
    data_manifest = json.loads((data_root / "data_manifest.json").read_text())
    with val_path.open("rb") as handle:
        panel = pickle.load(handle)
    candidates = build_candidate_frame(
        panel, args.lookback, args.predict, args.signal_start, args.signal_end
    )
    reserve_start = np.datetime64(args.reserve_start, "D")
    tuning_pool = candidates.loc[
        candidates["asof_date"].to_numpy(dtype="datetime64[D]") < reserve_start
    ].reset_index(drop=True)
    reserved = candidates.loc[
        candidates["asof_date"].to_numpy(dtype="datetime64[D]") >= reserve_start
    ].reset_index(drop=True)
    large = natural_select(tuning_pool, args.large_samples, args.seed)
    quick = natural_select(large, args.quick_samples, args.seed + 10_000)
    quick_ids = {sample_identity(row) for row in quick.itertuples()}
    # Put quick records first: the dataset loader relies on this invariant.
    large = pd.concat([
        large[large.apply(lambda row: sample_identity(row) in quick_ids, axis=1)],
        large[large.apply(lambda row: sample_identity(row) not in quick_ids, axis=1)],
    ], ignore_index=True)

    if train_path.resolve() == val_path.resolve():
        train_panel = panel
    else:
        with train_path.open("rb") as handle:
            train_panel = pickle.load(handle)
    overlap, absent = training_overlap_audit(
        large, train_panel, args.lookback, args.predict,
        args.train_signal_start, args.train_signal_end,
    )
    if len(overlap) + len(absent) != len(large):
        raise RuntimeError("Training overlap audit did not partition fixed samples")

    samples_path = output_dir / "natural_validation_samples.jsonl"
    with samples_path.open("w") as handle:
        for row in large.itertuples():
            identity = sample_identity(row)
            handle.write(json.dumps({
                "symbol": row.symbol, "start_index": int(row.start_index),
                "asof_date": row.asof_date, "target_date": row.target_date,
                "period": validation_period(row.asof_date, args.period_split),
                "direction": DIRECTION_NAMES[int(row.direction)],
                "return_10d": float(row.return_10d), "sector": row.sector,
                "size_decile": int(row.size_decile), "quick": identity in quick_ids,
            }, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "name": args.name,
        "source": {
            "data_manifest_sha256": data_manifest["manifest_sha256"],
            "val_data_sha256": sha256_file(val_path),
            "train_data_sha256": sha256_file(train_path),
            "lookback": args.lookback, "predict": args.predict,
            "validation_panel": args.validation_panel,
            "signal_start": args.signal_start, "signal_end": args.signal_end,
            "candidate_pool_samples": int(len(candidates)),
            "candidate_pool_identities_sha256": identities_sha256(candidates),
        },
        "direction_definition": {"return": "close[target_10] / close[input_last] - 1", "short": "return < -0.01", "neutral": "-0.01 <= return <= 0.01", "long": "return > 0.01"},
        "selection": {
            "seed": args.seed, "strata": ["asof_date", "direction", "size_decile", "sector"],
            "method": (
                "all_tuning_pool_samples"
                if len(large) == len(tuning_pool)
                else "equal_daily_quota_with_proportional_daily_direction_mix"
            ),
            "large_is_full_tuning_pool": len(large) == len(tuning_pool),
            "period_split": args.period_split,
            "periods": ["2025H2", "2026H1"],
            "quick_samples": len(quick), "large_samples": len(large),
            "quick_is_subset_of_large": True,
            "quick_identities_sha256": identities_sha256(quick),
            "large_identities_sha256": identities_sha256(large),
            "samples_file": samples_path.name, "samples_file_sha256": sha256_file(samples_path),
        },
        "training_isolation": {
            "exclusion_key": ["symbol", "asof_date"],
            "train_signal_start": args.train_signal_start, "train_signal_end": args.train_signal_end,
            "training_candidate_overlap_samples": len(overlap),
            "not_in_training_candidate_samples": len(absent),
            "all_training_overlaps_must_be_excluded": True,
        },
        "audit": {
            "candidate_pool": distribution(candidates), "tuning_pool": distribution(tuning_pool),
            "quick": distribution(quick), "large": distribution(large),
            "quick_by_period": {
                period: distribution(rows)
                for period, rows in quick.groupby(
                    quick["asof_date"].map(
                        lambda value: validation_period(value, args.period_split)
                    )
                )
            },
            "large_by_period": {
                period: distribution(rows)
                for period, rows in large.groupby(
                    large["asof_date"].map(
                        lambda value: validation_period(value, args.period_split)
                    )
                )
            },
            "reserved_tuning_holdout": distribution(reserved),
            "tuning_pool_per_date_direction": per_date_direction_audit(tuning_pool),
            "large_per_date_direction": per_date_direction_audit(large),
        },
        "final_test_isolation": {"reserved_tuning_holdout_start": args.reserve_start, "reserved_tuning_holdout_excluded_from_quick_and_large": True, "reserved_tuning_holdout_identities_sha256": identities_sha256(reserved)},
    }
    manifest_path = output_dir / "natural_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "samples": str(samples_path), "quick": manifest["audit"]["quick"]["direction_counts"], "large": manifest["audit"]["large"]["direction_counts"]}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/a_share_full_market_v1_beta")
    parser.add_argument("--output-dir", default="data/a_share_full_market_v1_beta/natural_validation_v1")
    parser.add_argument("--validation-panel", default="processed_datasets/val_data.pkl")
    parser.add_argument("--name", default="a_share_v1_beta_fixed_natural_direction_validation_v1")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--predict", type=int, default=10)
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-07-17")
    parser.add_argument("--reserve-start", default="2026-07-01")
    parser.add_argument("--train-signal-start", default="2015-01-01")
    parser.add_argument("--train-signal-end", default="2026-07-17")
    parser.add_argument("--quick-samples", type=int, default=3000)
    parser.add_argument("--large-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--period-split", default="2026-01-01")
    return parser.parse_args()


if __name__ == "__main__":
    build_manifest(parse_args())
