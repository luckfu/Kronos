#!/usr/bin/env python3
"""Download a short A-share interval from Sina with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "market_cap",
]


def fetch_symbol(symbol: str, start: str, end: str, retries: int) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            frame = ak.stock_zh_a_daily(
                symbol=symbol.replace(".", ""),
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            if frame.empty:
                return []
            required = {
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turnover",
            }
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"missing Sina columns: {missing}")
            frame = frame.copy()
            numeric = sorted(required - {"date"})
            frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["market_cap"] = frame["amount"] / frame["turnover"]
            frame["market_cap"] = frame["market_cap"].replace(
                [np.inf, -np.inf], np.nan
            )
            frame["symbol"] = symbol
            frame = frame.dropna(subset=OUTPUT_COLUMNS)
            frame = frame[
                (frame["close"] > 0)
                & (frame["volume"] >= 0)
                & (frame["amount"] >= 0)
                & (frame["market_cap"] > 0)
            ]
            frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
            return frame[OUTPUT_COLUMNS].to_dict("records")
        except Exception as exc:  # Network failures need bounded retries.
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8) + random.random())
    assert last_error is not None
    raise last_error


def write_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-file", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1 or args.checkpoint_every < 1:
        parser.error("workers, retries, and checkpoint-every must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.state or args.output.with_suffix(".state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = sorted(
        set(pd.read_csv(args.symbols_file, usecols=["symbol"])["symbol"].astype(str))
    )
    state = {
        "schema_version": 1,
        "source": "Sina stock_zh_a_daily via AkShare",
        "adjustment": "qfq",
        "market_cap_formula": "amount / turnover_fraction",
        "start": args.start,
        "end": args.end,
        "symbols_file": str(args.symbols_file),
        "requested_symbols": len(symbols),
        "completed": [],
        "failures": {},
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        contract = ("start", "end", "symbols_file", "requested_symbols")
        if any(previous.get(key) != state[key] for key in contract):
            raise ValueError(f"Resume state contract mismatch: {state_path}")
        state = previous

    completed = set(state["completed"])
    pending = [symbol for symbol in symbols if symbol not in completed]
    buffered_rows: list[dict] = []
    buffered_symbols = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_symbol, symbol, args.start, args.end, args.retries): symbol
            for symbol in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                buffered_rows.extend(future.result())
                completed.add(symbol)
                state["failures"].pop(symbol, None)
            except Exception as exc:
                state["failures"][symbol] = f"{type(exc).__name__}: {exc}"
            buffered_symbols += 1
            if buffered_symbols >= args.checkpoint_every or index == len(futures):
                append_rows(args.output, buffered_rows)
                buffered_rows.clear()
                buffered_symbols = 0
                state["completed"] = sorted(completed)
                state["completed_symbols"] = len(completed)
                state["failed_symbols"] = len(state["failures"])
                write_state(state_path, state)
                print(
                    f"processed={len(completed) + len(state['failures'])}/"
                    f"{len(symbols)} completed={len(completed)} "
                    f"failed={len(state['failures'])}",
                    flush=True,
                )

    if args.output.exists():
        result = pd.read_csv(args.output)
        duplicates = int(result.duplicated(["symbol", "date"]).sum())
        if duplicates:
            raise RuntimeError(f"Output contains {duplicates} duplicate symbol/date rows")
        if not np.isfinite(result.select_dtypes("number").to_numpy()).all():
            raise RuntimeError("Output contains non-finite numeric values")
    if state["failures"]:
        raise RuntimeError(f"Download ended with {len(state['failures'])} failed symbols")


if __name__ == "__main__":
    main()
