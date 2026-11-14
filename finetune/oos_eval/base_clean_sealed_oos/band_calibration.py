"""Calibrated forecast bands: make the displayed 10%-90% interval cover what it claims (CPU only).

The production page shows the 10%/90% quantiles of 16 sampled paths.  That is
the spread between samples, not a price range: about a quarter of realised
prices fall inside it.  This script replaces it with

    forecast return + [q10, q90] of past forecast errors (actual - forecast),

estimated only from forecasts whose label had matured by the signal day, and
checks the out-of-sample coverage walk-forward.

Methods: RAW (sample quantiles, as shown today), G (one set of error quantiles
for the whole market), V (per 20-day-volatility bucket), S (per forecast-size
bucket).  Calibration changes how wide the band is; it does not make the
forecast more accurate and does not change any ranking.

Frozen in ``finetune/reports/20261112_forecast_band_calibration_prereg_cn.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 11))
REPORT_HORIZONS = (1, 5, 10)
NOMINAL = 0.80
LOW_Q, HIGH_Q = 0.10, 0.90
BUCKETS = 5
MIN_BUCKET_ROWS = 500
MIN_HISTORY_DATES = 5
VOL_WINDOW = 20
PRICE_BASIS_TOLERANCE = 0.005
SEAL_AFTER = "2026-10-23"
BOOTSTRAP = {"block": 3, "replicates": 2000, "seed": 20261112}
METHODS = ("RAW", "G", "V", "S")
RANK_GROUPS = (("top10", 1, 10), ("rank11_100", 11, 100), ("rest", 101, 10**9))


def load_market(raw_path: Path) -> pd.DataFrame:
    """Per traded bar: forward returns, the date each horizon's label matures, trailing volatility."""
    raw = pd.read_csv(raw_path, usecols=["symbol", "date", "close", "volume"], dtype={"symbol": str})
    raw = raw[raw["volume"] > 0].sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = raw.groupby("symbol")
    for h in HORIZONS:                                # the h-th traded bar after the signal bar
        raw[f"actual_{h}"] = grouped["close"].shift(-h) / raw["close"] - 1.0
        raw[f"label_{h}"] = grouped["date"].shift(-h)
    raw["vol"] = grouped["close"].transform(lambda c: c.pct_change().rolling(VOL_WINDOW).std())
    return raw


def load_forecasts(spec: str) -> pd.DataFrame:
    """Long frame: signal, symbol, h, forecast return and the sample 10%/90% returns."""
    kind, path = spec.split(":", 1)
    if kind == "production":                          # extract of the archive's predictions.jsonl
        frame = pd.read_csv(path, dtype={"code": str}).rename(columns={"code": "symbol"})
        for column in ("p10", "p50", "p90"):
            frame[column] = frame[column] / frame["close_asof"] - 1.0
        frame = frame.rename(columns={"p50": "forecast", "p10": "raw_low", "p90": "raw_high"})
    elif kind == "shards":                            # evaluator shards with per-path D10 returns
        parts = []
        for shard in sorted(glob.glob(str(Path(path) / "*.csv.gz"))):
            wide = pd.read_csv(shard, dtype={"symbol": str})
            paths = wide[[c for c in wide.columns if c.startswith("path") and c[4:6].isdigit()]].to_numpy()
            for h in HORIZONS:
                part = pd.DataFrame({"signal": wide["asof_date"].astype(str), "symbol": wide["symbol"], "h": h,
                                     "forecast": wide[f"path_median_d{h}"], "close_asof": wide["last_close"]})
                if h == 10:                           # only D10 keeps every path
                    part["raw_low"] = np.quantile(paths, LOW_Q, axis=1)
                    part["raw_high"] = np.quantile(paths, HIGH_Q, axis=1)
                parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
    else:
        raise SystemExit(f"unknown forecast source {kind}")
    if frame["signal"].max() > SEAL_AFTER:
        raise SystemExit(f"Refusing: signal dates after {SEAL_AFTER} are the sealed acceptance window")
    return frame


def attach_outcomes(forecasts: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add the realised return, its maturity date and the signal-day volatility; drop unusable rows."""
    keys = market.rename(columns={"date": "signal"})
    merged = forecasts.merge(keys, on=["signal", "symbol"], how="inner")
    basis_off = (merged["close_asof"] / merged["close"] - 1.0).abs() > PRICE_BASIS_TOLERANCE
    merged = merged[~basis_off].copy()
    horizon = merged["h"].to_numpy()
    rows = np.arange(len(merged))
    merged["actual"] = merged[[f"actual_{h}" for h in HORIZONS]].to_numpy()[rows, horizon - 1]
    merged["matures"] = merged[[f"label_{h}" for h in HORIZONS]].to_numpy()[rows, horizon - 1]
    dropped = {"no_bar_on_signal_day": int(len(forecasts) - len(basis_off)),
               "price_basis_mismatch": int(basis_off.sum()), "label_not_available": int(merged["actual"].isna().sum())}
    merged = merged.dropna(subset=["actual", "matures"])
    keep = ["signal", "symbol", "h", "forecast", "actual", "matures", "vol"] + \
        [c for c in ("raw_low", "raw_high") if c in merged.columns]
    merged = merged[keep].reset_index(drop=True)
    merged["error"] = merged["actual"] - merged["forecast"]
    merged["rank"] = merged.groupby(["signal", "h"])["forecast"].rank(ascending=False, method="first")
    return merged, dropped


def bucket_edges(values: np.ndarray) -> np.ndarray:
    """Inner cut points of BUCKETS equal-count buckets (use with np.searchsorted)."""
    return np.quantile(values, np.linspace(0, 1, BUCKETS + 1)[1:-1])


def fit_offsets(history: pd.DataFrame, key: str | None) -> dict:
    """Error quantiles per bucket of ``key`` (None = one global set)."""
    offsets = {"global": (float(history["error"].quantile(LOW_Q)), float(history["error"].quantile(HIGH_Q)))}
    if key is None:
        return offsets
    usable = history.dropna(subset=[key])
    if len(usable) < MIN_BUCKET_ROWS:
        return offsets
    edges = bucket_edges(usable[key].to_numpy())
    which = np.searchsorted(edges, usable[key].to_numpy())
    offsets["edges"] = edges
    for b in range(BUCKETS):
        errors = usable["error"].to_numpy()[which == b]
        if len(errors) >= MIN_BUCKET_ROWS:
            offsets[b] = (float(np.quantile(errors, LOW_Q)), float(np.quantile(errors, HIGH_Q)))
    return offsets


def apply_offsets(today: pd.DataFrame, offsets: dict, key: str | None) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(len(today), offsets["global"][0])
    high = np.full(len(today), offsets["global"][1])
    if key is not None and "edges" in offsets:
        values = today[key].to_numpy()
        which = np.searchsorted(offsets["edges"], values)
        for b in range(BUCKETS):
            if b in offsets:
                inside = (which == b) & np.isfinite(values)
                low[inside], high[inside] = offsets[b]
    return today["forecast"].to_numpy() + low, today["forecast"].to_numpy() + high


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample bands for every signal day with enough matured history."""
    out = []
    for h, frame in data.groupby("h"):
        for signal in sorted(frame["signal"].unique()):
            history = frame[frame["matures"] <= signal]      # labels known at the close of the signal day
            if history["signal"].nunique() < MIN_HISTORY_DATES:
                continue
            today = frame[frame["signal"] == signal].copy()
            today["history_dates"] = history["signal"].nunique()
            for method, key in (("G", None), ("V", "vol"), ("S", "forecast")):
                today[f"{method}_low"], today[f"{method}_high"] = apply_offsets(today, fit_offsets(history, key), key)
            out.append(today)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def block_ci(daily: np.ndarray) -> list[float] | None:
    n, block = len(daily), BOOTSTRAP["block"]
    if n < 2 * block:
        return None
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = np.empty(BOOTSTRAP["replicates"])
    for r in range(BOOTSTRAP["replicates"]):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        draws[r] = daily[idx[:n]].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def score_band(frame: pd.DataFrame, method: str) -> dict | None:
    low, high, actual = f"{method}_low", f"{method}_high", frame["actual"]
    if method == "RAW":
        low, high = "raw_low", "raw_high"
    if low not in frame.columns or frame[low].isna().all():
        return None
    inside = (actual >= frame[low]) & (actual <= frame[high])
    below, above = actual < frame[low], actual > frame[high]
    width = frame[high] - frame[low]
    alpha = 1.0 - NOMINAL                              # interval score: width plus a penalty for misses
    interval_score = width + (2 / alpha) * ((frame[low] - actual).clip(lower=0) + (actual - frame[high]).clip(lower=0))
    daily = inside.groupby(frame["signal"]).mean()
    return {"rows": int(len(frame)), "signal_days": int(daily.size), "coverage": float(inside.mean()),
            "coverage_ci95": block_ci(daily.to_numpy()), "below": float(below.mean()), "above": float(above.mean()),
            "mean_width": float(width.mean()), "interval_score": float(interval_score.mean()),
            "daily_coverage_min": float(daily.min()), "daily_coverage_max": float(daily.max())}


def evaluate(data: pd.DataFrame) -> dict:
    bands = walk_forward(data)
    result = {"evaluated_rows": int(len(bands))}
    if bands.empty:
        return result
    for h in REPORT_HORIZONS:
        frame = bands[bands["h"] == h]
        if frame.empty:
            continue
        entry = {"first_signal": frame["signal"].min(), "last_signal": frame["signal"].max(),
                 "history_dates_range": [int(frame["history_dates"].min()), int(frame["history_dates"].max())],
                 "mean_forecast": float(frame["forecast"].mean()), "mean_actual": float(frame["actual"].mean()),
                 "methods": {m: s for m in METHODS if (s := score_band(frame, m)) is not None}, "by_rank": {}}
        for name, low, high in RANK_GROUPS:
            group = frame[(frame["rank"] >= low) & (frame["rank"] <= high)]
            entry["by_rank"][name] = {m: s for m in METHODS if (s := score_band(group, m)) is not None}
        result[f"d{h}"] = entry
    return result


def latest_offsets(data: pd.DataFrame) -> dict:
    """What production would use today: global error quantiles from every matured forecast in the dataset."""
    out = {}
    for h, frame in data.groupby("h"):
        out[str(h)] = {"low": float(frame["error"].quantile(LOW_Q)), "high": float(frame["error"].quantile(HIGH_Q)),
                       "median_error": float(frame["error"].median()), "rows": int(len(frame)),
                       "signal_days": int(frame["signal"].nunique())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--dataset", action="append", required=True,
                        help="name=production:<extract.csv.gz> | name=shards:<dir>")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    market = load_market(Path(args.raw))
    report = {"rules": {"nominal": NOMINAL, "quantiles": [LOW_Q, HIGH_Q], "buckets": BUCKETS,
                        "min_bucket_rows": MIN_BUCKET_ROWS, "min_history_dates": MIN_HISTORY_DATES,
                        "volatility_window": VOL_WINDOW, "bootstrap": BOOTSTRAP, "sealed_after": SEAL_AFTER,
                        "primary": {"dataset": args.dataset[0].split("=", 1)[0], "method": "G", "horizon": 10}},
              "datasets": {}}
    for item in args.dataset:
        name, spec = item.split("=", 1)
        data, dropped = attach_outcomes(load_forecasts(spec), market)
        result = evaluate(data)
        report["datasets"][name] = {"source": spec, "signal_days": int(data["signal"].nunique()),
                                    "signal_start": data["signal"].min(), "signal_end": data["signal"].max(),
                                    "rows": int(len(data)), "dropped": dropped, "walk_forward": result,
                                    "offsets_from_all_matured": latest_offsets(data)}
        print(f"\n== {name}: {data['signal'].nunique()} signal days {data['signal'].min()}..{data['signal'].max()}, dropped {dropped}")
        for h in REPORT_HORIZONS:
            entry = result.get(f"d{h}")
            if entry is None:
                print(f"-- day {h}: not enough matured history")
                continue
            print(f"-- day {h}: evaluated {entry['first_signal']}..{entry['last_signal']} "
                  f"(history {entry['history_dates_range'][0]}-{entry['history_dates_range'][1]} signal days); "
                  f"mean forecast {entry['mean_forecast']:+.2%}, mean actual {entry['mean_actual']:+.2%}")
            for method, s in entry["methods"].items():
                ci = "n/a" if s["coverage_ci95"] is None else f"[{s['coverage_ci95'][0]:.0%}, {s['coverage_ci95'][1]:.0%}]"
                print(f"   {method:4s} coverage {s['coverage']:.1%} {ci:>12s} | below {s['below']:.0%} above {s['above']:.0%} | "
                      f"width {s['mean_width']:.1%} | score {s['interval_score']:.4f} | per day {s['daily_coverage_min']:.0%}-"
                      f"{s['daily_coverage_max']:.0%} over {s['signal_days']} days")
            for group, methods in entry["by_rank"].items():
                text = ", ".join(f"{m} {s['coverage']:.0%} (below {s['below']:.0%})" for m, s in methods.items())
                rows = next(iter(methods.values()))["rows"] if methods else 0
                print(f"      {group:11s} n={rows:6d}: {text}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
