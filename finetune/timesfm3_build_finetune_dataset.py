"""Build leak-free TimesFM-3 fine-tuning samples from the A-share panel."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "size_percentile",
]


def encode_sectors(panel: dict[str, pd.DataFrame]) -> dict[str, int]:
    names = sorted(
        {
            str(value)
            for frame in panel.values()
            for value in frame["sector"].dropna().unique()
        }
    )
    return {name: index for index, name in enumerate(names)}


def build_split(
    panel: dict[str, pd.DataFrame],
    sector_ids: dict[str, int],
    start: str,
    end: str,
    context: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts: list[float] = []
    sectors: list[int] = []
    symbols: list[str] = []
    asofs: list[str] = []

    for symbol in sorted(panel):
        frame = panel[symbol].copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        if not all(column in frame.columns for column in FEATURES + ["sector"]):
            continue
        values = frame[FEATURES].astype("float32").to_numpy()
        close = frame["close"].astype("float32").to_numpy()
        dates = frame.index[
            (frame.index >= pd.Timestamp(start))
            & (frame.index <= pd.Timestamp(end))
        ]
        for asof in dates:
            end_index = frame.index.get_loc(asof)
            context_start = end_index - context + 1
            future_end = end_index + horizon + 1
            if context_start < 0 or future_end > len(frame):
                continue
            context_values = values[context_start : end_index + 1]
            target_values = close[end_index + 1 : future_end]
            if not np.isfinite(context_values).all() or not np.isfinite(target_values).all():
                continue
            starts.append(float(close[end_index]))
            contexts.append(context_values)
            targets.append(target_values)
            sector = str(frame.iloc[end_index]["sector"])
            sectors.append(sector_ids[sector])
            symbols.append(symbol)
            asofs.append(asof.strftime("%Y-%m-%d"))

    return {
        "contexts": np.asarray(contexts, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "start_close": np.asarray(starts, dtype=np.float32),
        "sector_id": np.asarray(sectors, dtype=np.int64),
        "symbol": np.asarray(symbols),
        "asof": np.asarray(asofs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-start", default="2025-07-01")
    parser.add_argument("--train-end", default="2026-07-02")
    parser.add_argument("--val-start", default="2026-07-17")
    parser.add_argument("--val-end", default="2026-08-10")
    parser.add_argument("--test-start", default="2026-08-11")
    parser.add_argument("--test-end", default="2026-08-15")
    args = parser.parse_args()

    with Path(args.panel).open("rb") as handle:
        panel = pickle.load(handle)
    sector_ids = encode_sectors(panel)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (args.train_start, args.train_end),
        "val": (args.val_start, args.val_end),
        "test": (args.test_start, args.test_end),
    }
    manifest = {
        "context": args.context,
        "horizon": args.horizon,
        "features": FEATURES,
        "target": "future close",
        "sector_encoding": sector_ids,
        "splits": {},
    }
    for name, (start, end) in splits.items():
        data = build_split(
            panel,
            sector_ids,
            start,
            end,
            args.context,
            args.horizon,
        )
        np.savez_compressed(output / f"{name}.npz", **data)
        manifest["splits"][name] = {
            "signal_start": start,
            "signal_end": end,
            "samples": int(len(data["targets"])),
            "symbols": int(len(set(data["symbol"].tolist()))),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
