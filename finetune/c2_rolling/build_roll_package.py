#!/usr/bin/env python3
"""Build the *new-data* training package for the C2 rolling-retrain experiment.

The question: does continuing C2 on bars it has never seen (signal dates after
its 2026-07-02 training cutoff) improve out-of-sample ranking?  This script cuts
a raw panel that physically ends at ``--last-bar`` so nothing from the later
evaluation window can reach the trainer, and records exactly which signal dates
the trainer may use.

Outputs (CSV/JSON only -- the kernel builds the pickles itself, so the package
does not depend on the local pandas/numpy version):

    roll_raw.csv        symbol,date,open,high,low,close,volume,amount,size_percentile
    symbols.csv         symbol,sector,split        (split: train | validation)
    sector_vocab.json   the 86 sector labels, in the evaluator's id order
    roll_manifest.json  window contract, counts, file hashes

Window contract mirrors ``finetune/dataset.py``: a sample is 131 consecutive
traded rows of one symbol (120 history + 10 label + 1 trailing row), and its
signal date is row 120.  ``size_percentile`` mirrors the evaluator: the daily
cross-sectional percentile rank of market cap over traded rows.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOOKBACK, PREDICT = 120, 10
WINDOW = LOOKBACK + PREDICT + 1          # finetune/dataset.py: lookback + predict + 1
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_sector_labels() -> list[str]:
    """The evaluator's id order; training must map sectors to the same ids."""
    source = (ROOT / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py").read_text()
    labels = ast.literal_eval(re.search(r"^SECTOR_LABELS = (\[.*\])$", source, re.M).group(1))
    # finetune/asset_metadata.py assigns ids by sorted unique label, so the
    # evaluator order must equal the sorted order or ids would silently differ.
    if len(labels) != 86 or labels != sorted(labels):
        raise SystemExit("Evaluator sector order is not the sorted order; ids would not match training")
    return labels


def count_windows(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Samples per symbol with a signal date in [start, end] (dataset.py logic)."""
    lo, hi = np.datetime64(start, "D"), np.datetime64(end, "D")
    counts = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        dates = group["date"].to_numpy(dtype="datetime64[D]")
        n = len(dates) - WINDOW + 1
        if n <= 0:
            continue
        asof = dates[np.arange(n) + LOOKBACK - 1]
        counts[symbol] = int(((asof >= lo) & (asof <= hi)).sum())
    return pd.Series(counts, dtype="int64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sector-source", type=Path, required=True, help="evaluation_samples.jsonl")
    parser.add_argument("--symbol-split", type=Path, required=True, help="symbol_split.csv (symbol, split)")
    parser.add_argument("--signal-start", required=True, help="first signal date the trainer may use")
    parser.add_argument("--last-bar", required=True, help="last bar kept in the package")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")

    raw = pd.read_csv(args.raw, dtype={"symbol": str})
    raw = raw[raw["volume"] > 0].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if raw.duplicated(["symbol", "date"]).any() or not np.isfinite(raw[FEATURES + ["market_cap"]].to_numpy()).all():
        raise SystemExit("raw panel has duplicate or non-finite rows")
    raw["size_percentile"] = raw.groupby("date")["market_cap"].rank(pct=True, method="average")

    sessions = pd.DatetimeIndex(sorted(raw["date"].unique()))
    last_bar, signal_start = pd.Timestamp(args.last_bar), pd.Timestamp(args.signal_start)
    if last_bar not in sessions or signal_start not in sessions:
        raise SystemExit("--last-bar and --signal-start must be traded sessions")
    last_index = int(sessions.get_loc(last_bar))
    # A sample needs its signal row plus PREDICT label rows plus one trailing row.
    signal_end = sessions[last_index - (PREDICT + 1)]
    first_bar = sessions[max(0, int(sessions.get_loc(signal_start)) - (LOOKBACK + 15))]
    kept = raw[(raw["date"] >= first_bar) & (raw["date"] <= last_bar)].sort_values(["symbol", "date"])

    sectors = pd.read_json(args.sector_source, lines=True, dtype={"symbol": str}).sort_values("asof_date")
    sector_by_symbol = sectors.groupby("symbol")["sector"].last()
    vocab = evaluator_sector_labels()
    split = pd.read_csv(args.symbol_split, dtype={"symbol": str}).set_index("symbol")["split"]
    symbols = pd.DataFrame({"symbol": sorted(kept["symbol"].unique())})
    symbols["sector"] = symbols["symbol"].map(sector_by_symbol).fillna("unknown")
    symbols["split"] = symbols["symbol"].map(split).fillna("train")   # symbols listed after the split
    if not set(symbols["sector"]) <= set(vocab) or not set(symbols["split"]) <= {"train", "validation"}:
        raise SystemExit("unexpected sector label or split value")

    windows = count_windows(kept, str(signal_start.date()), str(signal_end.date()))
    symbols["windows"] = symbols["symbol"].map(windows).fillna(0).astype(int)

    args.output_dir.mkdir(parents=True)
    out = kept[["symbol", "date", *FEATURES, "size_percentile"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(args.output_dir / "roll_raw.csv", index=False)
    symbols.to_csv(args.output_dir / "symbols.csv", index=False)
    (args.output_dir / "sector_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False) + "\n")
    by_split = symbols.groupby("split")
    manifest = {
        "schema_version": 1,
        "purpose": "C2 rolling-retrain experiment: training bars the model has never seen",
        "window_contract": {
            "lookback": LOOKBACK, "predict": PREDICT, "rows_per_sample": WINDOW,
            "signal_start": str(signal_start.date()), "signal_end": str(signal_end.date()),
            "first_bar": str(first_bar.date()), "last_bar": str(last_bar.date()),
            "last_label_bar": str(sessions[int(sessions.get_loc(signal_end)) + PREDICT].date()),
            "signal_sessions": int(((sessions >= signal_start) & (sessions <= signal_end)).sum()),
        },
        "counts": {
            "rows": int(len(out)), "symbols": int(len(symbols)),
            "train_symbols": int((symbols["split"] == "train").sum()),
            "validation_symbols": int((symbols["split"] == "validation").sum()),
            "train_windows": int(by_split["windows"].sum().get("train", 0)),
            "validation_windows": int(by_split["windows"].sum().get("validation", 0)),
            "unknown_sector_symbols": int((symbols["sector"] == "unknown").sum()),
            "symbols_absent_from_split_file": int((~symbols["symbol"].isin(split.index)).sum()),
        },
        "source": {"raw": args.raw.name, "raw_sha256": sha256_file(args.raw)},
        "files": {name: sha256_file(args.output_dir / name)
                  for name in ("roll_raw.csv", "symbols.csv", "sector_vocab.json")},
    }
    (args.output_dir / "roll_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
