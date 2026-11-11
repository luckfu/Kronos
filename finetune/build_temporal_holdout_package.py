#!/usr/bin/env python3
"""Build a sealed **temporal** holdout evaluation package from a raw price panel.

Why this exists
---------------
``docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md`` §2.2 forbids selecting
checkpoints on the symbol-holdout ``val_data.pkl`` and requires a split by
*time*.  The frozen window ``2026-04-01 → 2026-06-30`` assumes
``train_time_upper_bound = 2024-12-31``; a run trained through 2026-07-02 has
already seen those dates, so that window is contaminated for it.

This script materialises a new temporal window out of the frozen OOS raw panel
(``combined_raw.csv``, 2025-07-01 onward) so the *same* evaluator and the *same*
metric (daily D10 Rank IC) can be applied to signal dates that the model has
never produced a target for.

It only builds the package: no GPU, no Kaggle push, no training data touched.
Signal dates must be chosen from the un-consumed part of the calendar; see
``--signal-start``/``--signal-end`` and the guard below.

Schema is byte-compatible with ``a-share-120d-oos-20260803-20260930`` so
``evaluate_base_sealed.py`` can score it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT  # 130 rows: 120 history bars + 10 label bars
EVALUATION_ROWS = WINDOW
FEATURES = ["open", "high", "low", "close", "volume", "amount"]
DIRECTION_NAMES = {-1: "short", 0: "neutral", 1: "long"}

# Signal windows already spent by acceptance reports.  They stay read-only: a
# newer model must not be tuned against them again, and a temporal *selection*
# window must not overlap them.
CONSUMED_SIGNAL_WINDOWS = (
    ("package_34d", "2026-08-03", "2026-09-17"),
    ("sealed_8d", "2026-09-18", "2026-09-30"),
    ("section_5_2", "2026-10-09", "2026-10-23"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction_name(value: float) -> str:
    direction = -1 if value < -0.01 else 1 if value > 0.01 else 0
    return DIRECTION_NAMES[direction]


def overlapping_consumed(start: str, end: str) -> list[str]:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    hits = []
    for label, win_start, win_end in CONSUMED_SIGNAL_WINDOWS:
        a, b = pd.Timestamp(win_start), pd.Timestamp(win_end)
        if lo <= b and hi >= a:
            hits.append(label)
    return hits


def check_window_is_free(start: str, end: str, allowed: set[str]) -> list[str]:
    """Refuse consumed windows unless the caller declares an acceptance-only build.

    A consumed window may still be *built* (it is the designated acceptance
    window), it just may never be used for checkpoint selection.  The caller
    must name it explicitly with ``--allow-consumed`` so the intent is on the
    command line and lands in the manifest.
    """
    hits = overlapping_consumed(start, end)
    unexpected = [label for label in hits if label not in allowed]
    if unexpected:
        raise SystemExit(
            f"Refusing: signal window {start}..{end} overlaps the consumed "
            f"window(s) {unexpected}. Pass --allow-consumed <label> to build it as "
            "an acceptance-only package, or write a new protocol revision first."
        )
    return hits


def read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"symbol", "date", "market_cap", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    if frame.duplicated(["symbol", "date"]).any():
        raise SystemExit(f"{path} contains duplicate symbol/date rows")
    numeric = sorted(required - {"symbol", "date"})
    values = frame[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise SystemExit(f"{path} contains non-finite numeric values")
    if (frame["volume"] < 0).any():
        raise SystemExit(f"{path} contains negative volume")
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["symbol", "date"])


def sector_lookup(samples_path: Path) -> dict[str, str]:
    """Latest known sector per symbol, taken from an existing sealed package."""
    samples = pd.read_json(samples_path, lines=True, dtype={"symbol": str})
    if "sector" not in samples.columns:
        raise SystemExit(f"{samples_path} has no sector column")
    samples = samples.dropna(subset=["sector"])
    samples = samples.sort_values("asof_date")
    latest = samples.groupby("symbol")["sector"].last()
    return {str(k): str(v) for k, v in latest.items()}


def build_panel(raw: pd.DataFrame, sectors: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Per-symbol frames: FEATURES + size_percentile + sector, traded days only."""
    traded = raw[raw["volume"] > 0].copy()
    if traded.empty:
        raise SystemExit("raw panel has no traded rows")
    traded["size_percentile"] = traded.groupby("date")["market_cap"].rank(
        pct=True, method="average"
    )
    traded["sector"] = traded["symbol"].map(sectors).fillna("unknown")
    panel: dict[str, pd.DataFrame] = {}
    for symbol, group in traded.groupby("symbol", sort=False):
        frame = group.set_index("date")[FEATURES + ["size_percentile", "sector"]].sort_index()
        if frame.index.has_duplicates:
            raise SystemExit(f"{symbol} has duplicate dates after filtering")
        panel[str(symbol)] = frame
    return panel


def build_candidates(
    panel: dict[str, pd.DataFrame], signal_start: str, signal_end: str
) -> list[dict]:
    start = np.datetime64(signal_start, "D")
    end = np.datetime64(signal_end, "D")
    records: list[dict] = []
    for symbol, frame in panel.items():
        if len(frame) < EVALUATION_ROWS:
            continue
        dates = frame.index.to_numpy(dtype="datetime64[D]")
        starts = np.arange(len(frame) - EVALUATION_ROWS + 1, dtype=np.int32)
        asof_positions = starts + LOOKBACK - 1
        target_positions = asof_positions + PREDICT
        eligible = (dates[asof_positions] >= start) & (dates[asof_positions] <= end)
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        percentiles = frame["size_percentile"].to_numpy()
        sector_values = frame["sector"].to_numpy()
        for offset in np.flatnonzero(eligible):
            asof_position = int(asof_positions[offset])
            target_position = int(target_positions[offset])
            realized = float(close[target_position] / close[asof_position] - 1.0)
            records.append(
                {
                    "asof_date": str(dates[asof_position]),
                    "symbol": symbol,
                    "sector": str(sector_values[asof_position]),
                    "size_decile": int(
                        min(np.floor(float(percentiles[asof_position]) * 10), 9)
                    ),
                    "return_10d": round(realized, 10),
                    "direction": direction_name(realized),
                    "set": "temporal_holdout_all",
                    "start_index": int(starts[offset]),
                    "target_date": str(dates[target_position]),
                }
            )
    if not records:
        raise SystemExit(f"No candidates in {signal_start}..{signal_end}")
    records.sort(key=lambda row: (row["asof_date"], row["symbol"]))
    return records


def verify_windows(panel: dict[str, pd.DataFrame], records: list[dict]) -> None:
    """Fail here, not on Kaggle: every row must reconstruct a full 130-row window."""
    for record in records:
        frame = panel.get(record["symbol"])
        if frame is None:
            raise SystemExit(f"missing symbol {record['symbol']}")
        asof = pd.Timestamp(record["asof_date"])
        if asof not in frame.index:
            raise SystemExit(f"missing asof {record['symbol']} {record['asof_date']}")
        position = int(frame.index.get_loc(asof))
        if position - (LOOKBACK - 1) < 0:
            raise SystemExit(f"short history {record['symbol']} {record['asof_date']}")
        window = frame.iloc[position - (LOOKBACK - 1) : position + PREDICT + 1]
        if len(window) != EVALUATION_ROWS:
            raise SystemExit(f"incomplete window {record['symbol']} {record['asof_date']}")
        dates = pd.DatetimeIndex(window.index)
        if (str(dates[LOOKBACK - 1].date()), str(dates[WINDOW - 1].date())) != (
            record["asof_date"],
            record["target_date"],
        ):
            raise SystemExit(f"window identity drift {record['symbol']} {record['asof_date']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="combined_raw.csv with lookback+label bars")
    parser.add_argument("--sector-source", type=Path, required=True, help="existing evaluation_samples.jsonl")
    parser.add_argument("--signal-start", required=True)
    parser.add_argument("--signal-end", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", default="evaluation_only_never_train_or_tune")
    parser.add_argument(
        "--allow-consumed",
        action="append",
        default=[],
        metavar="LABEL",
        help="build a consumed window anyway, as acceptance-only (never for selection)",
    )
    parser.add_argument(
        "--role",
        default="",
        help="override the manifest role (default derives from the consumed-window guard)",
    )
    parser.add_argument(
        "--train-signal-end",
        default="2026-07-02",
        help="last signal date the model was trained on; its D10 label sets the clean boundary",
    )
    parser.add_argument("--trim-raw", action="store_true", help="keep only rows needed for the windows")
    args = parser.parse_args()

    allowed = {label.strip() for label in args.allow_consumed if label.strip()}
    consumed = check_window_is_free(args.signal_start, args.signal_end, allowed)
    role = args.role or (
        "acceptance_only_never_selection" if consumed else "selection_eligible"
    )
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")

    raw = read_raw(args.raw)
    traded = pd.DatetimeIndex(sorted(raw.loc[raw["volume"] > 0, "date"].unique()))
    anchor = pd.Timestamp(args.train_signal_end)
    position = int(traded.get_indexer([anchor])[0])
    if position < 0:
        raise SystemExit(f"train signal end {args.train_signal_end} is not a traded session")
    if position + PREDICT >= len(traded):
        raise SystemExit("panel is too short to resolve the last training label bar")
    train_label_end = traded[position + PREDICT]
    if pd.Timestamp(args.signal_start) <= train_label_end:
        raise SystemExit(
            f"Refusing: signal start {args.signal_start} is not after the last training "
            f"label bar {train_label_end.date()} (train signal end {anchor.date()} + "
            f"{PREDICT} sessions). Its D10 labels would overlap training labels."
        )

    dropped_short = 0
    if args.trim_raw:
        # Trim BEFORE building candidates.  A session-based trim can still leave
        # gap-prone symbols without 120 of their own traded bars, so the samples
        # that cannot reconstruct a 130-row window are dropped here and counted,
        # instead of failing on Kaggle two hours into an evaluation.
        anchor_index = int(traded.get_indexer([pd.Timestamp(args.signal_start)])[0])
        keep_from = pd.Timestamp(traded[max(0, anchor_index - (LOOKBACK + 10))])
        before = int(raw["symbol"].nunique())
        raw = raw[raw["date"] >= keep_from].copy()
        dropped_short = before - int(raw["symbol"].nunique())

    sectors = sector_lookup(args.sector_source)
    panel = build_panel(raw, sectors)
    records = build_candidates(panel, args.signal_start, args.signal_end)
    verify_windows(panel, records)

    signal_dates = sorted({row["asof_date"] for row in records})
    target_end = max(row["target_date"] for row in records)

    args.output_dir.mkdir(parents=True)
    samples_path = args.output_dir / "evaluation_samples.jsonl"
    with samples_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    raw_out = raw
    raw_path = args.output_dir / "combined_raw.csv"
    raw_out.to_csv(raw_path, index=False)

    manifest = {
        "schema_version": 2,
        "name": args.name,
        "purpose": args.purpose,
        "role": role,
        "model_contract": {
            "lookback": LOOKBACK,
            "predict": PREDICT,
            "window": WINDOW,
            "window_semantics": (
                "120 actual traded history bars followed by 10 actual traded label bars per symbol"
            ),
        },
        "label": {"formula": "close[t+10]/close[t]-1", "target": "t+10 trading day close"},
        "temporal_isolation": {
            "signal_start": signal_dates[0],
            "signal_end": signal_dates[-1],
            "target_end": target_end,
            "latest_raw_date": str(raw["date"].max().date()),
            "train_signal_end": str(anchor.date()),
            "train_label_end": str(train_label_end.date()),
            "boundary_rule": (
                "first signal date must be strictly after the last bar a training "
                "label covers (train signal end + predict sessions)"
            ),
        },
        "counts": {
            "samples": len(records),
            "signal_dates": len(signal_dates),
            "symbols": len({row["symbol"] for row in records}),
            "symbols_dropped_by_trim": dropped_short,
            "sectors": len({row["sector"] for row in records}),
        },
        "files": {
            "evaluation_samples.jsonl": sha256_file(samples_path),
            "combined_raw.csv": sha256_file(raw_path),
        },
        "consumed_windows_excluded": [
            {"label": label, "start": start, "end": end}
            for label, start, end in CONSUMED_SIGNAL_WINDOWS
        ],
        "consumed_windows_matched": consumed,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    readme = (
        f"# A-share 120D temporal holdout ({signal_dates[0]} to {signal_dates[-1]})\n\n"
        "Evaluation-only **time** out-of-sample panel for A-share daily models with a\n"
        "120-bar context and 10-day target. Unlike the symbol-holdout validation set,\n"
        "the split here is by calendar time: no signal date in this window is inside a\n"
        "training target.\n\n"
        f"- **Signal dates**: {len(signal_dates)} trading days, {signal_dates[0]} to {signal_dates[-1]}\n"
        f"- **Samples**: {len(records):,} rows (one per asof_date x symbol)\n"
        f"- **Target end**: {target_end}\n"
        f"- **Raw data**: `combined_raw.csv`, qfq daily OHLCVA + market cap, "
        f"{raw['date'].min().date()} to {raw['date'].max().date()}\n\n"
        "`evaluation_samples.jsonl` fields: `asof_date, symbol, sector, size_decile, "
        "return_10d, direction, set, start_index, target_date`.\n\n"
        "**Do not train or tune on this window.** It exists so mid-training checkpoint\n"
        "selection can be measured on unseen time instead of on the symbol holdout.\n"
    )
    (args.output_dir / "README.md").write_text(readme)

    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "samples": len(records),
                "signal_dates": len(signal_dates),
                "symbols": manifest["counts"]["symbols"],
                "target_end": target_end,
                "samples_sha": manifest["files"]["evaluation_samples.jsonl"][:16],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
