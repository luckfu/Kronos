"""Time-disjoint + symbol-disjoint validation panel for the Beta v2.1 heavy-reg retrain.

Why: temporal_symbol_validation_v1 ``val_data.pkl`` is symbol-disjoint (520 holdout
symbols) but NOT time-disjoint: its signals (2025-07-01..2026-07-02) overlap the
training period (train_data.pkl bars to 2026-07-17). Val->sealed-OOS drops of
-0.06..-0.18 IC suggest it rewards memorising the training period.

This module builds the cleanest time-disjoint selection set available without
touching the sealed window or downloading new data:

- Symbols: the same 520 holdout symbols (keys of temporal_symbol_validation_v1
  ``val_data.pkl``, sha 4cce31bc...). The retrain only sees the 4,678 train symbols,
  so the set stays symbol-disjoint w.r.t. this retrain.
- Signal dates: 2026-07-17 .. 2026-08-10 (17 trading days), strictly after the
  retrain's last training signal (train_data.pkl last full-window asof 2026-07-02;
  last bar 2026-07-17). 10d labels end 2026-07-31 .. 2026-08-24.
- Bars: from the OLD (2026-08-26) package panel
  ``kronos_small_0_1_time_oos_evaluation_20260826/evaluation/evaluation_panel.pkl``
  (manifest ``kronos_v1_beta_checkpoint_evaluation_20260826``, panel sha 6827d0eb...),
  truncated at 2026-08-25 (QlibDataset window = 120 + 10 + 1 rows, so the 08-10
  window needs the 08-25 bar). This package predates the sealed evaluation and
  is NOT the sealed root package.
- NEVER read: the dataset-root sealed files (``evaluation_manifest.json`` named
  ``kronos_beta_v2_time_oos_through_20260903``, its evaluation_panel.pkl /
  evaluation_samples.jsonl / august_raw.csv). Inputs are pinned by SHA-256 and the
  sealed manifest name is refused explicitly.

Caveats (documented in finetune/docs/beta_v21_heavy_reg_time_val_cn.md):
- Ancestors of Best@475 (V5/V6 lineage) trained with labels up to 2026-07-31, so
  labels of signals 07-17..07-21 sit inside the lineage's label horizon and
  07-17..07-31 bars were seen by ancestors (on train symbols and, for the
  full-pool v1.1+ ancestors, possibly on these symbols). Only signals with
  targets after 2026-07-31 (asof >= 2026-07-31) are strictly after every ancestor.
- 2026-08-03..08-10 were already used in earlier post-hoc audits of Best@475/C2.
- 17 overlapping 10d windows ~ 2 independent label periods: paired t is optimistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT + 1

SOURCE_PACKAGE_DIR = "kronos_small_0_1_time_oos_evaluation_20260826"
SOURCE_MANIFEST_NAME = "kronos_v1_beta_checkpoint_evaluation_20260826"
SOURCE_PANEL_SHA256 = "6827d0ebcc1ca90d9d5326b4a1dbbe78ac8e21a11d8549aa542cb5dc5c510bb6"
SEALED_MANIFEST_NAMES = frozenset({"kronos_beta_v2_time_oos_through_20260903"})
# Sealed root artifacts (refused even if someone points the builder at them).
SEALED_FILE_NAMES = frozenset({"august_raw.csv"})
HOLDOUT_VAL_SHA256 = "4cce31bc3e70eab83d5b7ea05f19fce04aa57a87f3acf00b882ddfbac4219bf7"
HOLDOUT_SYMBOLS = 520

SIGNAL_START = "2026-07-17"
SIGNAL_END = "2026-08-10"
MAX_BAR_DATE = "2026-08-25"
TRAIN_LAST_SIGNAL = "2026-07-02"
TRAIN_LAST_BAR = "2026-07-17"
LINEAGE_LATEST_TARGET = "2026-07-31"
SEALED_SIGNAL_START = "2026-08-11"
EXPECTED_SIGNAL_DATES = 17
COLUMNS = [
    "open", "high", "low", "close", "volume", "amount",
    "size_bucket", "size_percentile", "sector",
]
PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "size_percentile"]
OUTPUT_NAME = "time_disjoint_val_20260717_20260810.pkl"
MANIFEST_NAME = "time_disjoint_val_manifest.json"
CONTRACT_NAME = "holdout520_time_disjoint_20260717_20260810_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source_manifest(manifest: dict) -> None:
    name = str(manifest.get("name", ""))
    if name in SEALED_MANIFEST_NAMES:
        raise RuntimeError(f"Refusing sealed evaluation package {name}")
    if name != SOURCE_MANIFEST_NAME:
        raise RuntimeError(f"Unexpected source manifest {name!r} != {SOURCE_MANIFEST_NAME}")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("panel_sha256") != SOURCE_PANEL_SHA256:
        raise RuntimeError("Source manifest panel_sha256 is not the pinned 2026-08-26 panel")
    contract = manifest.get("model_contract", {})
    if int(contract.get("lookback", -1)) != LOOKBACK or int(contract.get("predict", -1)) != PREDICT:
        raise RuntimeError("Source manifest lookback/predict mismatch")


def find_source(input_root: Path) -> tuple[Path, Path]:
    """Locate (panel, manifest) of the 2026-08-26 package; never the sealed root package."""
    matches = sorted(
        path for path in Path(input_root).glob(f"**/{SOURCE_PACKAGE_DIR}/evaluation/evaluation_panel.pkl")
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {SOURCE_PACKAGE_DIR} evaluation_panel.pkl, found {matches}")
    panel = matches[0]
    manifest = panel.parent / "evaluation_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"Missing {manifest}")
    return panel, manifest


def find_holdout_val(input_root: Path) -> Path:
    matches = sorted(
        Path(input_root).glob("**/*temporal_symbol_validation_v1/processed_datasets/val_data.pkl")
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one temporal_symbol_validation_v1 val_data.pkl, found {matches}")
    return matches[0]


def verify_inputs(source_panel: Path, source_manifest: Path, holdout_val: Path) -> dict:
    for path in (source_panel, source_manifest, holdout_val):
        if Path(path).name in SEALED_FILE_NAMES:
            raise RuntimeError(f"Refusing sealed artifact {path}")
    manifest = json.loads(Path(source_manifest).read_text())
    check_source_manifest(manifest)
    panel_sha = sha256_file(source_panel)
    if panel_sha != SOURCE_PANEL_SHA256:
        raise RuntimeError(f"Source panel SHA mismatch: {panel_sha} != {SOURCE_PANEL_SHA256}")
    holdout_sha = sha256_file(holdout_val)
    if holdout_sha != HOLDOUT_VAL_SHA256:
        raise RuntimeError(f"Holdout val_data SHA mismatch: {holdout_sha} != {HOLDOUT_VAL_SHA256}")
    return {"source_panel_sha256": panel_sha, "holdout_val_sha256": holdout_sha,
            "source_manifest_name": manifest["name"],
            "source_manifest_sha256": sha256_file(source_manifest)}


def build_panel(
    source: dict,
    holdout: dict,
    *,
    max_bar_date: str = MAX_BAR_DATE,
    overlap_end: str = TRAIN_LAST_BAR,
    expected_symbols: int | None = HOLDOUT_SYMBOLS,
) -> tuple[dict, dict]:
    """Restrict ``source`` to the holdout symbols, truncate bars, check overlap equality."""
    symbols = sorted(str(symbol) for symbol in holdout)
    if expected_symbols is not None and len(symbols) != expected_symbols:
        raise RuntimeError(f"Expected {expected_symbols} holdout symbols, found {len(symbols)}")
    missing = [symbol for symbol in symbols if symbol not in source]
    if missing:
        raise RuntimeError(f"{len(missing)} holdout symbols missing from source panel: {missing[:5]}")
    cutoff = pd.Timestamp(max_bar_date)
    overlap_cut = pd.Timestamp(overlap_end)
    panel: dict = {}
    overlap_rows = 0
    for symbol in symbols:
        frame = source[symbol]
        frame = frame.loc[frame.index <= cutoff, COLUMNS].copy()
        frame.index = pd.DatetimeIndex(frame.index, name="datetime")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise RuntimeError(f"{symbol}: non-monotonic or duplicate dates")
        reference = holdout[symbol]
        reference = reference.loc[reference.index <= overlap_cut]
        shared = frame.index.intersection(reference.index)
        if len(shared) != len(reference.loc[reference.index >= frame.index.min()]):
            raise RuntimeError(f"{symbol}: source panel misses holdout bars in the overlap")
        left = frame.loc[shared, PRICE_COLUMNS].to_numpy(dtype=np.float64)
        right = reference.loc[shared, PRICE_COLUMNS].to_numpy(dtype=np.float64)
        if not np.allclose(left, right, rtol=0.0, atol=1e-9, equal_nan=True):
            raise RuntimeError(f"{symbol}: source panel disagrees with holdout val_data on overlap")
        if not (frame.loc[shared, "sector"].astype(str).to_numpy()
                == reference.loc[shared, "sector"].astype(str).to_numpy()).all():
            raise RuntimeError(f"{symbol}: sector disagrees with holdout val_data on overlap")
        overlap_rows += len(shared)
        panel[symbol] = frame
    return panel, {"symbols": len(panel), "overlap_rows_checked": overlap_rows,
                   "max_bar_date": str(cutoff.date())}


def build_records(panel: dict, signal_start: str = SIGNAL_START, signal_end: str = SIGNAL_END) -> list[dict]:
    """QlibDataset('val') eligibility: every full 131-row window with asof in range."""
    start = np.datetime64(signal_start, "D")
    end = np.datetime64(signal_end, "D")
    records = []
    for symbol in panel:
        frame = panel[symbol]
        count = len(frame) - WINDOW + 1
        if count <= 0:
            continue
        dates = frame.index.to_numpy(dtype="datetime64[D]")
        for position in range(count):
            asof = dates[position + LOOKBACK - 1]
            if start <= asof <= end:
                records.append({
                    "symbol": str(symbol),
                    "start_index": int(position),
                    "asof_date": str(asof),
                    "target_date": str(dates[position + LOOKBACK - 1 + PREDICT]),
                })
    return records


def identities_sha256(records: list[dict]) -> str:
    """Order-free fingerprint of (symbol, asof, target) identities (pandas-version stable)."""
    rows = sorted(f"{r['symbol']}|{r['asof_date']}|{r['target_date']}" for r in records)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def summarize(records: list[dict]) -> dict:
    by_date: dict[str, int] = {}
    for record in records:
        by_date[record["asof_date"]] = by_date.get(record["asof_date"], 0) + 1
    targets = sorted({record["target_date"] for record in records})
    dates = sorted(by_date)
    return {
        "windows": len(records),
        "signal_dates": len(dates),
        "signal_date_list": dates,
        "windows_by_date": by_date,
        "target_range": [targets[0], targets[-1]] if targets else None,
        "identities_sha256": identities_sha256(records),
        # Label path (asof, target] entirely after the lineage's latest training
        # target 2026-07-31 -> the strictly clean subset (diagnostic only).
        "strict_subset_signal_dates": [d for d in dates if d >= LINEAGE_LATEST_TARGET],
    }


def build(
    source_panel: Path,
    source_manifest: Path,
    holdout_val: Path,
    output_dir: Path,
    *,
    expected_dates: int | None = EXPECTED_SIGNAL_DATES,
) -> dict:
    hashes = verify_inputs(source_panel, source_manifest, holdout_val)
    with Path(holdout_val).open("rb") as handle:
        holdout = pickle.load(handle)
    with Path(source_panel).open("rb") as handle:
        source = pickle.load(handle)
    panel, audit = build_panel(source, holdout)
    del source
    records = build_records(panel)
    summary = summarize(records)
    if expected_dates is not None and summary["signal_dates"] != expected_dates:
        raise RuntimeError(f"Expected {expected_dates} signal dates, got {summary['signal_dates']}")
    if summary["signal_date_list"][0] < SIGNAL_START or summary["signal_date_list"][-1] >= SEALED_SIGNAL_START:
        raise RuntimeError("Signal dates leak outside the time-disjoint window")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / OUTPUT_NAME
    with output.open("wb") as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "contract": CONTRACT_NAME,
        "panel_file": OUTPUT_NAME,
        "panel_sha256": sha256_file(output),
        "signal_range": [SIGNAL_START, SIGNAL_END],
        "max_bar_date": MAX_BAR_DATE,
        "lookback": LOOKBACK,
        "predict": PREDICT,
        "window": WINDOW,
        "inputs": {
            "source_package": SOURCE_PACKAGE_DIR,
            **hashes,
            "holdout_symbols": audit["symbols"],
            "overlap_rows_checked_equal": audit["overlap_rows_checked"],
        },
        "isolation": {
            "retrain_train_last_signal": TRAIN_LAST_SIGNAL,
            "retrain_train_last_bar": TRAIN_LAST_BAR,
            "symbol_disjoint_from_retrain_training": True,
            "lineage_latest_training_target": LINEAGE_LATEST_TARGET,
            "sealed_signal_start_never_used": SEALED_SIGNAL_START,
            "sealed_root_files_read": False,
        },
        **summary,
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", type=Path, help="Discover inputs under this root (Kaggle)")
    parser.add_argument("--source-panel", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--holdout-val", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.input_root:
        source_panel, source_manifest = find_source(args.input_root)
        holdout_val = find_holdout_val(args.input_root)
    else:
        source_panel, source_manifest, holdout_val = args.source_panel, args.source_manifest, args.holdout_val
        if not (source_panel and source_manifest and holdout_val):
            parser.error("give --input-root or all of --source-panel/--source-manifest/--holdout-val")
    manifest = build(source_panel, source_manifest, holdout_val, args.output_dir)
    printable = {k: v for k, v in manifest.items() if k != "windows_by_date"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
