"""Build a deterministic ~20K balanced subset of the isolated small validation set."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from build_balanced_validation_manifest import (
    DIRECTION_NAMES,
    balanced_select,
    build_candidate_frame,
    distribution,
    identities_sha256,
    sample_identity,
    sha256_file,
)


def period_name(asof_date: str, split_date: str) -> str:
    return "2025H2" if asof_date < split_date else "2026H1"


def symbol_distribution(frame: pd.DataFrame) -> dict:
    counts = frame.groupby("symbol").size()
    return {
        "symbol_count": int(len(counts)),
        "min_samples": int(counts.min()),
        "median_samples": float(counts.median()),
        "max_samples": int(counts.max()),
        "missing_symbols": [],
    }


def build(args) -> None:
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest_path = data_root / "data_manifest.json"
    val_path = data_root / "processed_datasets" / "val_data.pkl"

    with val_path.open("rb") as handle:
        panel = pickle.load(handle)
    candidates = build_candidate_frame(
        panel,
        args.lookback,
        args.predict,
        args.signal_start,
        args.signal_end,
    )
    selected = balanced_select(candidates, args.per_direction, args.seed)
    candidate_symbols = set(candidates["symbol"].astype(str))
    selected_symbols = set(selected["symbol"].astype(str))
    missing_symbols = sorted(candidate_symbols - selected_symbols)
    if missing_symbols:
        raise RuntimeError(
            f"Balanced subset lost {len(missing_symbols)} eligible symbols: "
            f"{missing_symbols[:10]}"
        )

    selected = selected.copy()
    selected["period"] = selected["asof_date"].map(
        lambda value: period_name(str(value), args.period_split)
    )
    samples_path = output_dir / "balanced_validation_20k_samples.jsonl"
    with samples_path.open("w") as handle:
        for row in selected.itertuples():
            handle.write(
                json.dumps(
                    {
                        "symbol": str(row.symbol),
                        "start_index": int(row.start_index),
                        "asof_date": str(row.asof_date),
                        "target_date": str(row.target_date),
                        "direction": DIRECTION_NAMES[int(row.direction)],
                        "return_10d": float(row.return_10d),
                        "sector": str(row.sector),
                        "size_decile": int(row.size_decile),
                        "period": str(row.period),
                        "quick": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    selected_symbol_audit = symbol_distribution(selected)
    selected_symbol_audit["missing_symbols"] = missing_symbols
    manifest = {
        "schema_version": 1,
        "name": "a_share_120d_temporal_symbol_holdout_balanced_validation_20k_v1",
        "source": {
            "data_manifest_sha256": sha256_file(data_manifest_path),
            "val_data_sha256": sha256_file(val_path),
            "lookback": args.lookback,
            "predict": args.predict,
            "signal_start": args.signal_start,
            "signal_end": args.signal_end,
            "candidate_pool_samples": int(len(candidates)),
            "candidate_pool_symbols": int(len(candidate_symbols)),
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
            "symbol_coverage_required": True,
            "quick_samples": 0,
            "large_samples": int(len(selected)),
            "quick_is_subset_of_large": True,
            "quick_identities_sha256": identities_sha256(selected.iloc[:0]),
            "large_identities_sha256": identities_sha256(selected),
            "samples_file": samples_path.name,
            "samples_file_sha256": sha256_file(samples_path),
            "periods": ["2025H2", "2026H1"],
        },
        "training_isolation": {
            "exclusion_key": ["symbol", "asof_date"],
            "training_candidate_overlap_samples": 0,
            "not_in_training_candidate_samples": int(len(selected)),
            "all_training_overlaps_must_be_excluded": True,
            "basis": "dataset manifest declares zero train/validation symbol intersection",
        },
        "audit": {
            "candidate_pool": distribution(candidates),
            "selected": distribution(selected),
            "selected_symbols": selected_symbol_audit,
            "selected_month_counts": {
                str(key): int(value)
                for key, value in selected.groupby("month").size().items()
            },
            "selected_period_counts": dict(
                sorted(Counter(selected["period"].astype(str)).items())
            ),
        },
    }
    manifest_path = output_dir / "balanced_validation_20k_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "samples": str(samples_path),
                "candidate_samples": len(candidates),
                "selected_samples": len(selected),
                "directions": manifest["audit"]["selected"]["direction_counts"],
                "symbols": selected_symbol_audit,
                "dates": manifest["audit"]["selected"]["date_count"],
                "sectors": manifest["audit"]["selected"]["sector_count"],
                "size_deciles": manifest["audit"]["selected"]["size_decile_counts"],
                "periods": manifest["audit"]["selected_period_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/a_share_full_market_v1_beta_temporal_symbol_validation_v1",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/a_share_full_market_v1_beta_temporal_symbol_validation_v1/"
            "balanced_validation_20k_v1"
        ),
    )
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--predict", type=int, default=10)
    parser.add_argument("--signal-start", default="2025-07-01")
    parser.add_argument("--signal-end", default="2026-07-02")
    parser.add_argument("--period-split", default="2026-01-01")
    parser.add_argument("--per-direction", type=int, default=6666)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
