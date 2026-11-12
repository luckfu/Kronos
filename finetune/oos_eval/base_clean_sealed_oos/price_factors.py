"""Classic price/volume factors per (symbol, signal date), for the C2 ceiling test T4.

Writes one row per (symbol, signal date) on which the symbol traded, with 11
factors built from daily bars: past 1/5/10/20/60-bar returns, 20-bar return
volatility, 20-bar mean amount, amount relative to that mean, market-cap
percentile, intraday range and close relative to the 60-bar high.

Windows are row-based: "past N bars" are the symbol's own previous N traded
rows (volume > 0), the same convention as evaluate_base_sealed.py and
fusion_analysis.reversal_10d.  A factor is NaN until its window is full.

No-lookahead rule: a factor for signal date t uses only rows dated <= t.
Rows after --date-to are discarded right after the file is parsed, before
anything is computed; every time-series window ends at the signal row, and
size_pct ranks only that date's cross-section.  --date-to later than
LATEST_SIGNAL_DATE is refused.

    python3 finetune/oos_eval/base_clean_sealed_oos/price_factors.py \
        --raw scratch/oos65/combined_raw.csv \
        --date-from 2026-07-17 --date-to 2026-10-23 \
        --out scratch/c2_ceiling/price_factors.csv.gz
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

# Pre-registration 20261112_c2_ceiling_tests_prereg_cn.md: no sample with a
# later signal date is built or read.
LATEST_SIGNAL_DATE = "2026-10-23"
RAW_COLUMNS = ["symbol", "date", "high", "low", "close", "volume", "amount", "market_cap"]
RETURN_WINDOWS = [1, 5, 10, 20, 60]
FACTORS = [*(f"ret_{n}" for n in RETURN_WINDOWS),
           "vol_20", "amt_20", "amt_ratio", "size_pct", "range_1", "pos_60"]


def iso_date(text: str) -> str:
    """Validate YYYY-MM-DD so that string comparison orders dates correctly."""
    return date.fromisoformat(text).isoformat()


def load_traded_bars(raw_path: Path, date_to: str) -> pd.DataFrame:
    """Traded rows dated <= date_to, sorted by symbol then date."""
    if date_to > LATEST_SIGNAL_DATE:
        raise SystemExit(f"--date-to {date_to} is later than the sealed limit {LATEST_SIGNAL_DATE}")
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS, dtype={"symbol": str, "date": str})
    bars = raw[(raw["volume"] > 0) & (raw["date"] <= date_to)]
    bars = bars.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    if bars.duplicated(["symbol", "date"]).any():
        raise SystemExit(f"Duplicate (symbol, date) rows in {raw_path}")
    return bars


def compute_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """All factors for every row of `bars` (sorted traded rows from load_traded_bars)."""
    symbol, close = bars["symbol"], bars["close"]

    def trailing(series: pd.Series, window: int):
        """Per-symbol window over the current row and the window - 1 rows before it."""
        return series.groupby(symbol, sort=False).rolling(window, min_periods=window)

    out = pd.DataFrame({"symbol": symbol, "asof_date": bars["date"]})
    past_close = close.groupby(symbol, sort=False)
    for n in RETURN_WINDOWS:
        out[f"ret_{n}"] = close / past_close.shift(n) - 1
    out["vol_20"] = trailing(out["ret_1"], 20).std(ddof=1).droplevel(0)
    out["amt_20"] = trailing(bars["amount"], 20).mean().droplevel(0)
    out["amt_ratio"] = bars["amount"] / out["amt_20"]
    out["size_pct"] = bars.groupby("date")["market_cap"].rank(pct=True, method="average")
    out["range_1"] = (bars["high"] - bars["low"]) / close
    out["pos_60"] = close / trailing(bars["high"], 60).max().droplevel(0)
    return out[["symbol", "asof_date", *FACTORS]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--date-from", type=iso_date, required=True)
    parser.add_argument("--date-to", type=iso_date, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.date_from > args.date_to:
        raise SystemExit(f"--date-from {args.date_from} is after --date-to {args.date_to}")

    factors = compute_factors(load_traded_bars(args.raw, args.date_to))
    factors = factors[factors["asof_date"] >= args.date_from]
    if factors.empty:
        raise SystemExit(f"No traded rows in [{args.date_from}, {args.date_to}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(args.out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{args.out}: {len(factors)} rows, {factors['asof_date'].nunique()} dates "
          f"({factors['asof_date'].min()}..{factors['asof_date'].max()}), "
          f"{factors['symbol'].nunique()} symbols")
    print("NaN share:", factors[FACTORS].isna().mean().round(6).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
