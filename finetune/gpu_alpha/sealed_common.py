"""Shared sealed-window batch builder for 2026-10-26..2026-11-20 acceptance scoring on Kaggle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEAT_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _future_weekdays(last_date: str, n_days: int = 12) -> list[pd.Timestamp]:
    ts = pd.Timestamp(last_date)
    out = []
    cur = ts + pd.Timedelta(days=1)
    while len(out) < n_days:
        if cur.weekday() < 5:
            out.append(cur)
        cur += pd.Timedelta(days=1)
    return out


def build_sealed_batches(
    kline_parquet: Path,
    ranks_parquet: Path,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[tuple[str, list[tuple]]]]:
    """Build exact (sym_arrays, sealed_batches) compatible with all G24-G34 OOS workers.

    - Adjusts OHLC continuously per symbol by adj_factor / adj_factor[-1] (scale-invariant under
      norm_window and bit-aligned with prediction_drill.py on the final date).
    - Appends 12 future weekday timestamps to stamps so G32 AR (y_st = st[t+1:t+11]) never overflows.
    - Sets identity = "{asof}|{code}" for deterministic 1-to-1 alignment back to shadow rankings.
    """
    kline = pd.read_parquet(kline_parquet)
    ranks = pd.read_parquet(ranks_parquet)
    kline["date"] = kline["date"].astype(str)
    ranks["asof"] = ranks["asof"].astype(str)
    kline["code"] = kline["code"].astype(str)
    ranks["code"] = ranks["code"].astype(str)

    kline = kline.sort_values(["code", "date"]).reset_index(drop=True)
    last_cal_date = str(kline["date"].max())
    fut_dts = _future_weekdays(last_cal_date, n_days=12)
    fut_st = np.column_stack([
        np.zeros(len(fut_dts), dtype=np.float32),
        np.zeros(len(fut_dts), dtype=np.float32),
        np.asarray([d.weekday() for d in fut_dts], dtype=np.float32),
        np.asarray([d.day for d in fut_dts], dtype=np.float32),
        np.asarray([d.month for d in fut_dts], dtype=np.float32),
    ])

    sym_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    sym_to_idx: dict[str, tuple[int, dict[str, int]]] = {}

    for code, grp in kline.groupby("code", sort=True):
        raw_vals = grp[FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
        adjs = grp["adj_factor"].to_numpy(dtype=np.float64)
        ratio = (adjs / float(adjs[-1])).astype(np.float32)[:, None]
        raw_vals[:, :4] *= ratio

        dt = pd.to_datetime(grp["date"])
        st = np.column_stack([
            dt.dt.minute.to_numpy(dtype=np.float32),
            dt.dt.hour.to_numpy(dtype=np.float32),
            dt.dt.weekday.to_numpy(dtype=np.float32),
            dt.dt.day.to_numpy(dtype=np.float32),
            dt.dt.month.to_numpy(dtype=np.float32),
        ])
        st_ext = np.vstack([st, fut_st])
        pcts = np.full(len(raw_vals), 0.5, dtype=np.float32)
        dpos = {str(d): i for i, d in enumerate(grp["date"].tolist())}
        sym_arrays.append((raw_vals, st_ext, pcts))
        sym_to_idx[str(code)] = (len(sym_arrays) - 1, dpos)

    date_to_recs: dict[str, list[tuple]] = {}
    for r in ranks.itertuples(index=False):
        s_i, dpos = sym_to_idx[r.code]
        t = dpos[r.asof]
        if t < 119:
            raise RuntimeError(f"Insufficient history for {r.code} on {r.asof}: t={t}")
        identity = f"{r.asof}|{r.code}"
        date_to_recs.setdefault(r.asof, []).append(
            (s_i, t, int(r.sector_id), float(r.size_percentile), identity, r.code)
        )

    sealed_batches = [(d, date_to_recs[d]) for d in sorted(date_to_recs.keys())]
    return sym_arrays, sealed_batches
