#!/usr/bin/env python3
"""Build a sealed OOS package whose rows and labels use actual traded days."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from finetune.prepare_v1_beta_evaluation import (
    FEATURES,
    LOOKBACK,
    PREDICT,
    direction_name,
    identities_sha256,
    record_distribution,
    sample_identity,
    sha256_file,
)

EVALUATION_ROWS = LOOKBACK + PREDICT


def validated_raw(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"symbol", "date", "market_cap", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{label} contains duplicate symbol/date rows")
    numeric = sorted(required - {"symbol", "date"})
    values = frame[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite numeric values")
    if (frame["volume"] < 0).any():
        raise ValueError(f"{label} contains negative volume")
    return frame


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    work = frame.copy()
    work.index = pd.to_datetime(work.index)
    if work.index.has_duplicates or not work.index.is_monotonic_increasing:
        raise ValueError("Panel frame has duplicate or unsorted dates")
    volume = pd.to_numeric(work["volume"], errors="coerce")
    if volume.isna().any() or (volume < 0).any():
        raise ValueError("Panel frame has invalid volume")
    result = work.loc[volume > 0].copy()
    if result.empty:
        raise ValueError("Panel frame has no traded rows")
    return result, int((volume == 0).sum())


def append_actual_traded(base_panel: dict, raw: pd.DataFrame, refresh_history=False):
    raw = raw.copy()
    raw["symbol"] = raw["symbol"].astype(str)
    raw["size_percentile"] = raw.groupby("date")["market_cap"].rank(
        method="first", pct=True
    )
    raw["size_bucket"] = np.minimum(
        np.floor(raw["size_percentile"] * 10), 9
    )
    latest_sector = {}
    for symbol, frame in base_panel.items():
        known = frame["sector"].dropna() if "sector" in frame.columns else pd.Series()
        latest_sector[str(symbol)] = str(known.iloc[-1]) if not known.empty else "unknown"
    raw["sector"] = raw["symbol"].map(latest_sector).fillna("unknown")

    additions = {
        str(symbol): rows.set_index("date")[
            FEATURES + ["size_bucket", "size_percentile", "sector"]
        ].sort_index()
        for symbol, rows in raw.groupby("symbol", sort=False)
    }
    combined = {}
    zero_removed = 0
    for symbol, source in base_panel.items():
        symbol = str(symbol)
        if refresh_history:
            if symbol not in additions:
                continue
            frame = source.iloc[:0].copy()
            removed = int((source["volume"] == 0).sum())
        else:
            frame, removed = clean_frame(source.copy())
        zero_removed += removed
        if symbol in additions:
            frame = pd.concat([frame, additions[symbol]])
        if frame.index.has_duplicates:
            frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.sort_index()
        frame, removed = clean_frame(frame)
        zero_removed += removed
        combined[symbol] = frame
    unexpected = sorted(set(additions) - {str(s) for s in base_panel})
    if unexpected:
        raise ValueError(f"Raw data contains {len(unexpected)} symbols absent from base panel")
    return combined, zero_removed


def build_candidates(panel, signal_start: str, signal_end: str):
    start = np.datetime64(signal_start, "D")
    end = np.datetime64(signal_end, "D")
    records = []
    for symbol, frame in panel.items():
        if len(frame) < EVALUATION_ROWS:
            continue
        dates = frame.index.to_numpy(dtype="datetime64[D]")
        starts = np.arange(len(frame) - EVALUATION_ROWS + 1, dtype=np.int32)
        asof_positions = starts + LOOKBACK - 1
        target_positions = asof_positions + PREDICT
        eligible = (dates[asof_positions] >= start) & (
            dates[asof_positions] <= end
        )
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        for offset in np.flatnonzero(eligible):
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
                        min(np.floor(float(frame["size_percentile"].iloc[asof_position]) * 10), 9)
                    ),
                }
            )
    if not records:
        raise ValueError(f"No candidates in {signal_start}..{signal_end}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-panel", type=Path, required=True)
    parser.add_argument("--previous-raw", type=Path, required=True)
    parser.add_argument("--incremental-raw", type=Path, required=True)
    parser.add_argument("--previous-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-start", default="2026-09-04")
    parser.add_argument("--signal-end", default="2026-10-09")
    parser.add_argument("--refresh-history", action="store_true",
                        help="Replace all OHLCVA history with a single consistent qfq snapshot.")
    parser.add_argument("--full-oos", action="store_true",
                        help="Rebuild the full OOS range, superseding earlier sample packages.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    previous_raw = validated_raw(args.previous_raw, "previous raw")
    incremental_raw = validated_raw(args.incremental_raw, "incremental raw")
    overlap = previous_raw.merge(incremental_raw, on=["symbol", "date"], suffixes=("_old", "_new"))
    changed_overlap = 0
    if not overlap.empty:
        differences = np.max(
            np.abs(overlap[[f"{col}_old" for col in FEATURES]].to_numpy()
                   - overlap[[f"{col}_new" for col in FEATURES]].to_numpy()),
            axis=1,
        )
        changed_overlap = int((differences > 1e-10).sum())
        if changed_overlap and not args.refresh_history:
            raise ValueError("Overlapping raw rows disagree between source downloads")
    if not overlap.empty and not args.refresh_history:
        incremental_raw = incremental_raw.merge(
            previous_raw[["symbol", "date"]],
            on=["symbol", "date"],
            how="left",
            indicator=True,
        )
        incremental_raw = incremental_raw.loc[
            incremental_raw["_merge"].eq("left_only")
        ].drop(columns="_merge")

    combined_raw = (incremental_raw.copy() if args.refresh_history else
                    pd.concat([previous_raw, incremental_raw], ignore_index=True))
    combined_raw = combined_raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    if combined_raw.duplicated(["symbol", "date"]).any():
        raise ValueError("Combined raw data contains duplicate symbol/date rows")
    raw_zero_removed = int(combined_raw["volume"].eq(0).sum())
    combined_raw = combined_raw.loc[combined_raw["volume"] > 0].reset_index(drop=True)

    with args.base_panel.open("rb") as handle:
        base_panel = pickle.load(handle)
    panel, zero_removed = append_actual_traded(base_panel, combined_raw, args.refresh_history)
    records = build_candidates(panel, args.signal_start, args.signal_end)

    previous_samples = args.previous_package / "evaluation_samples.jsonl"
    previous_records = []
    with previous_samples.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("set") in {"future_all", "incremental_future_all"}:
                previous_records.append(record)
    old_ids = {sample_identity(record) for record in previous_records}
    new_ids = {sample_identity(record) for record in records}
    if old_ids & new_ids and not args.full_oos:
        raise ValueError("New actual-traded OOS overlaps the previous OOS")
    semantic_overlap = ({(r["symbol"], r["asof_date"]) for r in previous_records}
                        & {(r["symbol"], r["asof_date"]) for r in records})
    if semantic_overlap and not args.full_oos:
        raise ValueError("New OOS signal identities overlap the previous OOS")

    output = args.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with (temporary / "evaluation_panel.pkl").open("wb") as handle:
            pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        combined_path = temporary / "combined_raw.csv"
        combined_raw.to_csv(combined_path, index=False, date_format="%Y-%m-%d")
        samples_path = temporary / "evaluation_samples.jsonl"
        with samples_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(
                    {"set": "incremental_future_all", **record},
                    ensure_ascii=False, sort_keys=True
                ) + "\n")

        sectors = sorted({
            str(value) for frame in panel.values()
            for value in frame["sector"].dropna().unique()
        })
        signal_end = max(record["asof_date"] for record in records)
        target_end = max(record["target_date"] for record in records)
        manifest = {
            "schema_version": 2,
            "name": f"kronos_beta_v2_actual_traded_oos_through_{signal_end.replace('-', '')}",
            "purpose": "evaluation_only_never_train_or_tune",
            "model_contract": {
                "lookback": LOOKBACK, "predict": PREDICT, "window": EVALUATION_ROWS,
                "window_semantics": "120 actual traded history rows followed by 10 actual traded label rows per symbol",
                "num_sectors": len(sectors), "sector_labels": sectors,
                "use_size_percentile": True, "context_layer": 10,
            },
            "temporal_isolation": {
                "previous_future_signal_end": max(r["asof_date"] for r in previous_records),
                "previous_future_target_end": max(r["target_date"] for r in previous_records),
                "incremental_signal_start": min(r["asof_date"] for r in records),
                "incremental_signal_end": signal_end,
                "incremental_target_end": target_end,
                "latest_raw_date": str(combined_raw["date"].max().date()),
                "previous_identity_intersection": len(old_ids & new_ids),
                "previous_signal_identity_intersection": len(semantic_overlap),
                "supersedes_previous_oos_samples": args.full_oos,
            },
            "source": {
                "base_panel": str(args.base_panel),
                "base_panel_sha256": sha256_file(args.base_panel),
                "previous_raw": str(args.previous_raw),
                "previous_raw_sha256": sha256_file(args.previous_raw),
                "incremental_raw": str(args.incremental_raw),
                "incremental_raw_sha256": sha256_file(args.incremental_raw),
                "previous_package": str(args.previous_package),
                "combined_raw_rows": len(combined_raw),
                "combined_raw_symbols": int(combined_raw["symbol"].nunique()),
                "combined_raw_date_start": str(combined_raw["date"].min().date()),
                "combined_raw_date_end": str(combined_raw["date"].max().date()),
                "overlap_rows_audited" if args.refresh_history else "overlap_rows_deduplicated": len(overlap),
                "overlap_rows_with_changed_ohlcva": changed_overlap,
                "history_refreshed_single_qfq_snapshot": args.refresh_history,
                "symbols_without_returned_history": sorted(set(base_panel) - set(panel)),
            },
            "cleaning": {
                "panel_zero_volume_rows_removed": 0 if args.refresh_history else zero_removed,
                "discarded_base_zero_volume_rows": zero_removed if args.refresh_history else 0,
                "raw_zero_volume_rows_removed": raw_zero_removed,
                "policy": "drop volume==0 rows; preserve actual date index; no calendar reindexing",
            },
            "artifacts": {
                "panel_file": "evaluation_panel.pkl",
                "panel_sha256": sha256_file(temporary / "evaluation_panel.pkl"),
                "samples_file": "evaluation_samples.jsonl",
                "samples_sha256": sha256_file(samples_path),
                "combined_raw_file": "combined_raw.csv",
                "combined_raw_sha256": sha256_file(combined_path),
            },
            "sample_sets": {
                "incremental_future_all": {
                    **record_distribution(records),
                    "identities_sha256": identities_sha256(records),
                }
            },
            "limitations": [
                "Latest raw market data is through 2026-10-09; the latest mature signal is earlier because ten actual future traded rows are required.",
                "Sector labels for appended rows inherit the latest sector in the base panel.",
                "Market-cap proxy is amount/turnover from the current vendor snapshot; historical float revisions are not independently point-in-time verified.",
                "Percentiles rank the fixed universe with valid returned bars each date, not stocks without returned bars.",
                "Legacy evaluators hardcoding 131 rows must use this package's 130-row window contract for the final mature signals.",
                "No stock-symbol holdout against training is claimed; this package extends time-OOS only.",
                "This package is sealed evaluation evidence and must not be used for training or tuning.",
            ],
        }
        (temporary / "evaluation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        (temporary / "dataset-metadata.json").write_text(json.dumps({
            "id": manifest["name"],
            "title": f"Kronos Beta v2 actual-traded-day OOS through {signal_end}",
            "licenses": [{"name": "other"}],
        }, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
