"""Prepare a context-preserving A-share panel for long lookback experiments.

This helper is intentionally separate from ``prepare_a_share.py``.  It accepts
the existing 2015--2026 raw panel plus a non-overlapping 2014 context file,
keeps the required rows immediately before each split's signal range, and
reports the actual signal-window coverage.  The training code must still set
``KRONOS_*_SIGNAL_START/END`` so that context rows are not treated as targets.

The helper never edits an input CSV and does not remove an existing output
directory.  A duplicate ``symbol,date`` pair is an error by default because
silently choosing one row would make a resumed experiment non-reproducible.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .prepare_a_share import add_size_buckets
except ImportError:  # pragma: no cover - direct script execution
    from prepare_a_share import add_size_buckets


REQUIRED_COLUMNS = {
    "symbol", "date", "open", "high", "low", "close", "volume",
}
PANEL_COLUMNS = [
    "open", "high", "low", "close", "volume", "amount",
    "size_bucket", "size_percentile",
]


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    """Expand repeatable file/directory arguments deterministically."""

    paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(Path(value) for value in sorted(
                glob.glob(str(path / "*.csv"))
            ))
        else:
            paths.append(path)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No raw CSV inputs were found")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Raw CSV input(s) missing: {missing}")
    return unique


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Raw input is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", *numeric])
    frame = frame[(frame["close"] > 0) & (frame["volume"] >= 0)]
    if "amount" not in frame.columns:
        frame["amount"] = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
    else:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
        frame["amount"] = frame["amount"].fillna(
            frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
        )
    return frame


def load_raw_inputs(paths: Iterable[Path], allow_duplicates: bool = False):
    """Load raw files and return the frame plus per-file audit rows."""

    frames = []
    audit = []
    for path in paths:
        frame = _normalise_frame(pd.read_csv(path))
        duplicates = int(frame.duplicated(["symbol", "date"], keep=False).sum())
        audit.append({
            "path": str(path),
            "rows": int(len(frame)),
            "symbols": int(frame["symbol"].nunique()),
            "start": str(frame["date"].min().date()) if len(frame) else None,
            "end": str(frame["date"].max().date()) if len(frame) else None,
            "duplicate_rows_within_file": duplicates,
        })
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    duplicate_mask = frame.duplicated(["symbol", "date"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = int(frame.loc[duplicate_mask, ["symbol", "date"]].drop_duplicates().shape[0])
    if duplicate_rows and not allow_duplicates:
        examples = frame.loc[duplicate_mask, ["symbol", "date"]].drop_duplicates().head(5)
        values = [f"{row.symbol}:{row.date.date()}" for row in examples.itertuples()]
        raise ValueError(
            "Duplicate symbol,date pairs found across raw inputs "
            f"({duplicate_keys} keys/{duplicate_rows} rows), examples: {values}. "
            "Use --allow-duplicate-inputs only after auditing the overlap."
        )
    if duplicate_rows:
        frame = frame.sort_values(["symbol", "date"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame, audit, duplicate_keys, duplicate_rows


def _panel_frame(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    columns = list(PANEL_COLUMNS)
    if "sector" in rows.columns:
        columns.append("sector")
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"Prepared rows are missing columns: {missing}")
    result = rows[["date", *columns]].copy().set_index("date")
    result.index.name = "datetime"
    return result


def make_signal_panel(
    frame: pd.DataFrame,
    signal_start: str,
    signal_end: str,
    lookback: int,
    predict: int,
):
    """Keep context plus all rows needed for signal dates in one range.

    The panel deliberately includes context rows before ``signal_start``.  A
    Dataset instance should set signal env vars to the same range; otherwise
    those rows can become unintended training targets.
    """

    start = pd.Timestamp(signal_start)
    end = pd.Timestamp(signal_end)
    if start > end:
        raise ValueError(f"Signal start {start.date()} is after end {end.date()}")
    panel = {}
    coverage = []
    skipped = []
    window = lookback + predict + 1
    for symbol, rows in frame.groupby("symbol", sort=True):
        rows = rows.sort_values("date").drop_duplicates("date", keep="last")
        dates = rows["date"].to_numpy(dtype="datetime64[D]")
        eligible = np.flatnonzero((dates >= np.datetime64(start.date())) & (dates <= np.datetime64(end.date())))
        if len(eligible) == 0:
            skipped.append({"symbol": str(symbol), "reason": "no_signal_dates"})
            continue
        first = max(int(eligible[0]), lookback - 1)
        # Dataset windows contain ``lookback + predict + 1`` rows.  Therefore
        # the as-of row needs ``predict + 1`` rows after it, not merely
        # ``predict`` future rows.  The requested signal end is an as-of
        # boundary; the extra label rows may extend beyond it.  The caller
        # must choose a boundary whose extra rows stay outside the next split.
        last = min(
            int(eligible[-1]),
            len(rows) - predict - 2,
        )
        if first > last:
            skipped.append({
                "symbol": str(symbol),
                "reason": "insufficient_context_or_future_rows",
                "raw_rows": int(len(rows)),
                "raw_start": str(rows["date"].iloc[0].date()),
                "raw_end": str(rows["date"].iloc[-1].date()),
            })
            continue
        # Since ``first`` and ``last`` are valid as-of positions, this slice
        # contains exactly lookback rows before the first signal and all rows
        # through the final in-range signal (including its future labels and
        # the extra token endpoint required by the next-token loss).
        context_start = first - lookback + 1
        selected = rows.iloc[context_start:last + predict + 2].copy()
        panel[str(symbol)] = _panel_frame(selected)
        signal_dates = dates[first:last + 1]
        coverage.append({
            "symbol": str(symbol),
            "raw_rows": int(len(rows)),
            "raw_start": str(rows["date"].iloc[0].date()),
            "raw_end": str(rows["date"].iloc[-1].date()),
            "context_start": str(selected["date"].iloc[0].date()),
            "first_signal": str(pd.Timestamp(signal_dates[0]).date()),
            "last_signal": str(pd.Timestamp(signal_dates[-1]).date()),
            "context_rows": int(lookback),
            "rows": int(len(selected)),
            "windows": int(len(signal_dates)),
            "window": int(window),
        })
    return panel, coverage, skipped


def make_full_panel(frame: pd.DataFrame, start: str, end: str, lookback: int, predict: int):
    """Build a broad holdout panel while retaining a bounded date range."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected = frame[frame["date"].between(start_ts, end_ts)].copy()
    panel = {}
    for symbol, rows in selected.groupby("symbol", sort=True):
        if len(rows) >= lookback + predict + 1:
            panel[str(symbol)] = _panel_frame(rows.sort_values("date"))
    return panel


def describe(panel, lookback: int, predict: int, coverage=None):
    frames = list(panel.values())
    report = {
        "symbols": len(panel),
        "rows": int(sum(len(item) for item in frames)),
        "windows": int(sum(max(len(item) - lookback - predict, 0) for item in frames)),
        "window": int(lookback + predict + 1),
    }
    if frames:
        report.update({
            "start": str(min(frame.index.min() for frame in frames).date()),
            "end": str(max(frame.index.max() for frame in frames).date()),
        })
    else:
        report.update({"start": None, "end": None})
    if coverage is not None:
        report["eligible_windows"] = int(sum(item["windows"] for item in coverage))
        report["signal_start"] = min((item["first_signal"] for item in coverage), default=None)
        report["signal_end"] = max((item["last_signal"] for item in coverage), default=None)
    return report


def save_panel(path: Path, panel: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_size_reference(frame: pd.DataFrame, path: Path, universe_label: str) -> None:
    if "market_cap" not in frame.columns:
        return
    caps = pd.to_numeric(frame["market_cap"], errors="coerce")
    valid = frame.loc[caps > 0, ["date"]].copy()
    valid["market_cap"] = caps[caps > 0]
    if valid.empty:
        return
    date = valid["date"].max()
    values = sorted(float(value) for value in valid.loc[valid["date"] == date, "market_cap"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reference_date": date.strftime("%Y-%m-%d"),
        "market_caps": values,
        "count": len(values),
        "method": "point-in-time market_cap from raw input",
        "universe": universe_label,
    }, separators=(",", ":")) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", action="append", default=[], help="Raw CSV or directory; repeatable")
    parser.add_argument("--raw-inputs", nargs="+", default=[], help="Raw CSVs/directories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--predict", type=int, default=10)
    parser.add_argument("--train-start", default="2015-01-01")
    parser.add_argument("--train-end", default="2026-12-31")
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument(
        "--val-from-train", action="store_true",
        help="Use the complete train signal range as the validation pool; validation sampling is handled by the trainer.",
    )
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--holdout-start", default=None)
    parser.add_argument("--holdout-end", default=None)
    parser.add_argument("--num-size-buckets", type=int, default=10)
    parser.add_argument(
        "--min-2014-trading-days", type=int, default=120,
        help="Refuse the bundle unless the merged input has this many unique 2014 dates.",
    )
    parser.add_argument("--universe-label", default="A-share context experiment")
    parser.add_argument("--allow-duplicate-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.lookback <= 0 or args.predict <= 0:
        raise SystemExit("--lookback and --predict must be positive")
    input_items = [*args.raw_input, *args.raw_inputs]
    paths = expand_inputs(input_items)
    frame, input_audit, duplicate_keys, duplicate_rows = load_raw_inputs(
        paths, allow_duplicates=args.allow_duplicate_inputs
    )
    frame = add_size_buckets(frame, args.num_size_buckets)
    context_dates = frame.loc[
        frame["date"].between("2014-01-01", "2014-12-31"), "date"
    ].dt.normalize().nunique()
    if context_dates < args.min_2014_trading_days:
        raise SystemExit(
            f"Merged input has only {context_dates} unique 2014 trading dates; "
            f"need at least {args.min_2014_trading_days}."
        )

    manifest = None
    manifest_report = None
    holdout_frame = frame.iloc[0:0].copy()
    train_frame = frame
    if args.universe_manifest:
        manifest_path = Path(args.universe_manifest).expanduser().resolve()
        manifest = pd.read_csv(manifest_path, dtype={"symbol": str})
        required = {"symbol", "split"} - set(manifest.columns)
        if required:
            raise ValueError(f"Universe manifest is missing columns: {sorted(required)}")
        split_by_symbol = manifest.drop_duplicates("symbol").set_index("symbol")["split"]
        raw_symbols = set(frame["symbol"].astype(str).unique())
        manifest_symbols = set(split_by_symbol.index.astype(str))
        missing_manifest_symbols = sorted(manifest_symbols - raw_symbols)
        if missing_manifest_symbols:
            raise ValueError(
                f"{len(missing_manifest_symbols)} manifest symbols have no raw rows; "
                f"examples: {missing_manifest_symbols[:10]}"
            )
        frame_split = frame["symbol"].map(split_by_symbol)
        missing = frame_split.isna()
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} raw rows are absent from the universe manifest"
            )
        holdout_frame = frame.loc[frame_split == "holdout"].copy()
        train_frame = frame.loc[frame_split == "train"].copy()
        invalid = set(manifest["split"].dropna().astype(str)) - {"train", "holdout"}
        if invalid:
            raise ValueError(f"Unsupported manifest split labels: {sorted(invalid)}")
        manifest_report = {
            "path": str(manifest_path),
            "symbols": int(len(manifest_symbols)),
            "split_counts": {
                str(key): int(value)
                for key, value in manifest.drop_duplicates("symbol")["split"].value_counts().items()
            },
            "missing_raw_symbols": 0,
            "raw_symbols_absent_from_manifest": 0,
        }

    output_root = Path(args.output_root).expanduser()
    dataset_root = output_root / "processed_datasets"
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_columns = ["symbol", "date", "size_bucket", "size_percentile"]
    if "sector" in frame.columns:
        metadata_columns.insert(2, "sector")
    metadata = frame[metadata_columns].copy()
    metadata["date"] = metadata["date"].dt.strftime("%Y-%m-%d")
    metadata.drop_duplicates(["symbol", "date"], keep="last").to_csv(
        output_root / "asset_metadata.csv", index=False
    )
    write_size_reference(frame, output_root / "size_reference.json", args.universe_label)
    if args.universe_manifest:
        shutil.copy2(args.universe_manifest, output_root / "universe_manifest.csv")

    split_specs = {"train": (args.train_start, args.train_end, train_frame)}
    if args.val_from_train and (args.val_start or args.val_end):
        raise SystemExit("--val-from-train cannot be combined with --val-start/--val-end")
    if bool(args.val_start) != bool(args.val_end):
        raise SystemExit("--val-start and --val-end must be provided together")
    if args.val_from_train:
        split_specs["val"] = (args.train_start, args.train_end, train_frame)
    elif args.val_start:
        split_specs["val"] = (args.val_start, args.val_end, train_frame)
    if bool(args.test_start) != bool(args.test_end):
        raise SystemExit("--test-start and --test-end must be provided together")
    if args.test_start:
        split_specs["test"] = (args.test_start, args.test_end, train_frame)

    summary = {
        "lookback": args.lookback,
        "predict": args.predict,
        "window": args.lookback + args.predict + 1,
        "raw_inputs": input_audit,
        "raw_rows": int(len(frame)),
        "raw_symbols": int(frame["symbol"].nunique()),
        "raw_start": str(frame["date"].min().date()),
        "raw_end": str(frame["date"].max().date()),
        "unique_2014_trading_days": int(context_dates),
        "min_2014_trading_days": int(args.min_2014_trading_days),
        "duplicate_symbol_date_keys": duplicate_keys,
        "duplicate_symbol_date_rows": duplicate_rows,
        "allow_duplicate_inputs": bool(args.allow_duplicate_inputs),
        "splits": {},
    }
    if manifest_report is not None:
        summary["universe_manifest"] = manifest_report
    coverage_manifest = []
    for split, (signal_start, signal_end, source) in split_specs.items():
        panel, coverage, skipped = make_signal_panel(
            source, signal_start, signal_end, args.lookback, args.predict
        )
        save_panel(dataset_root / f"{split}_data.pkl", panel)
        summary["splits"][split] = {
            **describe(panel, args.lookback, args.predict, coverage),
            "skipped_symbols": len(skipped),
            "skipped_examples": skipped[:20],
            "signal_range_requested": [signal_start, signal_end],
        }
        if split == "val" and args.val_from_train:
            summary["splits"][split]["source"] = "train_pool_random_sample"
        coverage_manifest.extend(
            {"split": split, **item} for item in coverage
        )

    if args.universe_manifest and not holdout_frame.empty:
        holdout_start = args.holdout_start or args.train_start
        holdout_end = args.holdout_end or args.val_end or args.train_end
        holdout, holdout_coverage, holdout_skipped = make_signal_panel(
            holdout_frame, holdout_start, holdout_end, args.lookback, args.predict
        )
        save_panel(dataset_root / "symbol_holdout_data.pkl", holdout)
        summary["holdout"] = {
            **describe(holdout, args.lookback, args.predict, holdout_coverage),
            "skipped_symbols": len(holdout_skipped),
            "skipped_examples": holdout_skipped[:20],
            "signal_range_requested": [holdout_start, holdout_end],
        }
        coverage_manifest.extend(
            {"split": "holdout", **item} for item in holdout_coverage
        )

    (output_root / "context_coverage_manifest.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_manifest).to_csv(
        output_root / "context_coverage_manifest.csv", index=False
    )
    (output_root / "v5_context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
