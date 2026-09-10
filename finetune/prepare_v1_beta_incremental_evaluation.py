#!/usr/bin/env python3
"""Build a versioned evaluation set from newly matured August labels."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_v1_beta_evaluation import (
    FEATURES,
    LOOKBACK,
    PREDICT,
    WINDOW,
    append_august,
    build_candidates,
    identities_sha256,
    record_distribution,
    sample_identity,
    sha256_file,
)


def load_future_records(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("set") == "future_all":
                record.pop("set", None)
                records.append(record)
    if not records:
        raise ValueError(f"No future_all records in {path}")
    return records


def validated_raw(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"symbol", "date", "market_cap", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{label} contains duplicate symbol/date rows")
    numeric = sorted(required - {"symbol", "date"})
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{label} contains non-finite numeric values")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-panel",
        type=Path,
        default=Path(
            "data/a_share_full_market_v1_beta/processed_datasets/val_data.pkl"
        ),
    )
    parser.add_argument("--previous-raw", type=Path, required=True)
    parser.add_argument("--incremental-raw", type=Path, required=True)
    parser.add_argument("--previous-samples", type=Path, required=True)
    parser.add_argument("--source-parity-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-start", default="2026-08-11")
    parser.add_argument("--signal-end", default="2026-08-17")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_raw = validated_raw(args.previous_raw, "previous raw data")
    incremental_raw = validated_raw(args.incremental_raw, "incremental raw data")
    previous_symbols = set(previous_raw["symbol"].astype(str))
    unexpected_symbols = sorted(
        set(incremental_raw["symbol"].astype(str)) - previous_symbols
    )
    if unexpected_symbols:
        raise ValueError(
            f"Incremental data has {len(unexpected_symbols)} symbols outside the fixed universe"
        )
    overlap_rows = previous_raw.merge(
        incremental_raw, on=["symbol", "date"], how="inner"
    )
    if not overlap_rows.empty:
        raise ValueError(
            f"Raw source intervals overlap by {len(overlap_rows)} symbol/date rows"
        )

    combined_raw = pd.concat([previous_raw, incremental_raw], ignore_index=True)
    combined_raw = combined_raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    combined_path = args.output_dir / "august_raw.csv"
    combined_raw.to_csv(combined_path, index=False, date_format="%Y-%m-%d")

    with args.base_panel.open("rb") as handle:
        base_panel = pickle.load(handle)
    panel, august = append_august(base_panel, combined_path)
    panel_path = args.output_dir / "evaluation_panel.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    records = build_candidates(panel, args.signal_start, args.signal_end)
    previous_records = load_future_records(args.previous_samples)
    previous_identities = {sample_identity(record) for record in previous_records}
    current_identities = {sample_identity(record) for record in records}
    identity_overlap = previous_identities & current_identities
    if identity_overlap:
        raise RuntimeError(
            f"Incremental set overlaps the previous future set by {len(identity_overlap)} identities"
        )
    previous_target_end = max(record["target_date"] for record in previous_records)
    current_target_start = min(record["target_date"] for record in records)
    if current_target_start <= previous_target_end:
        raise RuntimeError(
            f"New target start {current_target_start} is not after {previous_target_end}"
        )

    samples_path = args.output_dir / "evaluation_samples.jsonl"
    with samples_path.open("w") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {"set": "incremental_future_all", **record},
                    ensure_ascii=False,
                    sort_keys=True,
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
        "name": "kronos_beta_v2_incremental_time_oos_20260902",
        "purpose": "evaluation_only_never_train_or_tune",
        "model_contract": {
            "lookback": LOOKBACK,
            "predict": PREDICT,
            "window": WINDOW,
            "num_sectors": len(sectors),
            "sector_labels": sectors,
            "use_size_percentile": True,
            "context_layer": 10,
        },
        "temporal_isolation": {
            "beta_v2_training_latest_target": "2026-07-31",
            "previous_future_signal_end": max(
                record["asof_date"] for record in previous_records
            ),
            "previous_future_target_end": previous_target_end,
            "incremental_signal_start": min(record["asof_date"] for record in records),
            "incremental_signal_end": max(record["asof_date"] for record in records),
            "incremental_target_start": current_target_start,
            "incremental_target_end": max(record["target_date"] for record in records),
            "previous_identity_intersection": len(identity_overlap),
            "targets_strictly_after_previous_target_end": True,
            "targets_strictly_after_training_target_end": True,
        },
        "source": {
            "base_panel": str(args.base_panel),
            "base_panel_sha256": sha256_file(args.base_panel),
            "previous_raw": str(args.previous_raw),
            "previous_raw_sha256": sha256_file(args.previous_raw),
            "incremental_raw": str(args.incremental_raw),
            "incremental_raw_sha256": sha256_file(args.incremental_raw),
            "combined_raw_file": combined_path.name,
            "combined_raw_sha256": sha256_file(combined_path),
            "combined_rows": len(august),
            "combined_symbols": int(august["symbol"].nunique()),
            "combined_date_start": str(august["date"].min().date()),
            "combined_date_end": str(august["date"].max().date()),
            **(
                {
                    "source_parity_audit": str(args.source_parity_audit),
                    "source_parity_audit_sha256": sha256_file(
                        args.source_parity_audit
                    ),
                }
                if args.source_parity_audit
                else {}
            ),
        },
        "artifacts": {
            "panel_file": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "samples_file": samples_path.name,
            "samples_sha256": sha256_file(samples_path),
        },
        "sample_sets": {
            "incremental_future_all": {
                **record_distribution(records),
                "identities_sha256": identities_sha256(records),
            }
        },
        "limitations": [
            "This is newly matured time out-of-sample evidence, not proof of trading profitability.",
            "The interval is short and shares the August market regime with the previous future set.",
            "Model and hyperparameter decisions made after this evaluation consume its sealed status.",
        ],
    }
    manifest_path = args.output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    metadata = {
        "id": manifest["name"],
        "title": "Kronos Beta v2 incremental August 2026 time-OOS evaluation",
        "licenses": [{"name": "other"}],
    }
    (args.output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
