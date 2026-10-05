"""Unit tests for Phase F industry-neutral forward residuals (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modernbert_finance.ablations.industry_neutral_ablations import _industry_demean


def test_industry_mean_demean_zeros_within_group() -> None:
    meta = pd.DataFrame(
        {
            "asof_date": ["2025-01-01"] * 4,
            "sector": ["A", "A", "B", "B"],
        }
    )
    vals = np.array([0.10, 0.00, -0.04, 0.04], dtype=np.float64)
    resid, stats = _industry_demean(meta, vals, how="mean", min_industry_n=2)
    # A mean=0.05 → resid +0.05, -0.05; B mean=0 → resid -0.04, +0.04
    np.testing.assert_allclose(resid, [0.05, -0.05, -0.04, 0.04], atol=1e-12)
    assert stats["n_fallback_market"] == 0
    assert abs(resid.mean()) < 1e-12


def test_singleton_falls_back_to_market() -> None:
    meta = pd.DataFrame(
        {
            "asof_date": ["d1", "d1", "d1"],
            "sector": ["A", "A", "solo"],
        }
    )
    vals = np.array([0.10, 0.00, 0.30], dtype=np.float64)
    resid, stats = _industry_demean(meta, vals, how="mean", min_industry_n=2)
    assert stats["n_fallback_market"] == 1
    # solo uses market mean = 0.1333...
    mkt = vals.mean()
    np.testing.assert_allclose(resid[2], vals[2] - mkt, atol=1e-12)
    # A pair still industry-demeaned
    np.testing.assert_allclose(resid[0], 0.05, atol=1e-12)
    np.testing.assert_allclose(resid[1], -0.05, atol=1e-12)


def test_median_demean() -> None:
    meta = pd.DataFrame(
        {
            "asof_date": ["d"] * 3,
            "sector": ["X"] * 3,
        }
    )
    vals = np.array([0.0, 0.1, 0.9], dtype=np.float64)
    resid, _ = _industry_demean(meta, vals, how="median", min_industry_n=2)
    # median=0.1
    np.testing.assert_allclose(resid, [-0.1, 0.0, 0.8], atol=1e-12)
