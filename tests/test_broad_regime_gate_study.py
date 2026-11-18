"""Unit & causal-leakage tests for broad_regime_gate_study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"))

import broad_regime_gate_study as brg  # noqa: E402


def _make_raw_and_idx(tmp_path: Path, cutoff_date: str | None = None):
    dates = pd.bdate_range("2026-01-05", "2026-04-15").strftime("%Y-%m-%d").tolist()
    if cutoff_date is not None:
        dates = [d for d in dates if d <= cutoff_date]
    rng = np.random.default_rng(7)
    rows = []
    for s_idx in range(25):
        sym = f"sz.{300001 + s_idx:06d}"
        px = 10.0 + s_idx * 0.2
        for d in dates:
            px *= float(np.exp(rng.normal(0.001, 0.02)))
            rows.append({
                "date": d, "symbol": sym,
                "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
                "volume": 1e5, "amount": 1e7, "turnover": 0.02, "market_cap": 5e8,
            })
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    nonpv = tmp_path / "non_pv"
    nonpv.mkdir(exist_ok=True)
    for fname in ("idx_sh000300.parquet", "idx_sh000852.parquet"):
        px = 3000.0
        idx_rows = []
        for d in dates:
            px *= float(np.exp(rng.normal(0.0005, 0.01)))
            idx_rows.append({"date": d, "close": px})
        pd.DataFrame(idx_rows).to_parquet(nonpv / fname, index=False)
    return raw_path, nonpv


def test_broad_regime_table_has_no_future_leakage(tmp_path: Path) -> None:
    (tmp_path / "full").mkdir(exist_ok=True)
    raw_full, nonpv_full = _make_raw_and_idx(tmp_path / "full")
    (tmp_path / "trunc").mkdir(exist_ok=True)
    raw_trunc = (tmp_path / "trunc") / "raw.csv"
    nonpv_trunc = (tmp_path / "trunc") / "non_pv"
    nonpv_trunc.mkdir(exist_ok=True)

    # Ensure identical truncated prefix by writing identical raw CSV prefix instead of distinct RNG calls
    df_full = pd.read_csv(raw_full)
    df_full[df_full["date"] <= "2026-03-10"].to_csv(raw_trunc, index=False)
    for fname in ("idx_sh000300.parquet", "idx_sh000852.parquet"):
        idx_f = pd.read_parquet(nonpv_full / fname)
        idx_f[idx_f["date"] <= "2026-03-10"].to_parquet(nonpv_trunc / fname, index=False)

    reg_full = brg.build_broad_regime_table(raw_full, nonpv_full)
    reg_trunc = brg.build_broad_regime_table(raw_trunc, nonpv_trunc)
    common = reg_full.loc[reg_trunc.index]
    pd.testing.assert_frame_equal(common, reg_trunc)
    assert set(reg_full["G_EW"].unique()) <= {0, 1, 2}
    assert set(reg_full["G_BREADTH"].unique()) <= {0, 1, 2}
