"""Unit tests for G1 GPU multi-scale math, strict date bounds, and Ridge sufficiency equivalence."""
from __future__ import annotations

import numpy as np
import torch

from finetune.gpu_alpha.g1_multiscale_extract_and_fit import (
    BLOCK_SLICES,
    TOTAL_DIM,
    lookup_conditions_for_symbol,
    orthogonalize_target,
    solve_ridge_subblock,
    zscore_cross_section_torch,
)


def test_sufficient_stats_ridge_matches_direct_solve():
    rng = np.random.default_rng(42)
    n, d = 600, 32
    x = rng.normal(size=(n, d)).astype(np.float64)
    y = rng.normal(size=n).astype(np.float64)
    gram = x.T @ x
    rhs = x.T @ y
    cols = slice(8, 20)
    alpha = 150.0
    w_sub = solve_ridge_subblock(gram, rhs, cols, alpha)
    x_sub = x[:, cols]
    w_direct = np.linalg.solve(x_sub.T @ x_sub + alpha * np.eye(12), x_sub.T @ y)
    np.testing.assert_allclose(w_sub, w_direct, atol=1e-10)


def test_zscore_cross_section_torch_clips_and_centers():
    x = torch.tensor([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [100.0, 40.0]], dtype=torch.float32)
    z = zscore_cross_section_torch(x, clip_val=2.0)
    assert z.shape == (4, 2)
    assert float(z.abs().max()) <= 2.0 + 1e-6
    assert abs(float(z[:, 1].mean())) < 1e-5


def test_orthogonalize_target_removes_linear_confounders():
    rng = np.random.default_rng(7)
    n = 300
    sz = rng.uniform(0, 1, size=n)
    rv = rng.normal(0, 0.05, size=n)
    sec = rng.integers(0, 5, size=n)
    ret10 = 0.5 * sz - 0.3 * rv + 0.02 * sec + rng.normal(0, 0.01, size=n)
    y_raw, y_orth = orthogonalize_target(ret10, sz, rv, sec, num_sectors=5)
    assert abs(float(np.mean(y_orth))) < 1e-10
    assert abs(float(np.std(y_orth)) - 1.0) < 1e-5
    # Correlation with pure confounder `sz` should drop sharply after neutralization
    assert abs(np.corrcoef(y_orth, sz)[0, 1]) < 0.08 < abs(np.corrcoef(y_raw, sz)[0, 1])


def test_lookup_conditions_for_symbol_asof_binary_search():
    dates = np.array(["2025-01-02", "2025-06-01", "2026-01-05"], dtype="datetime64[ns]").view(np.int64)
    sec = np.array([3, 7, 11], dtype=np.int32)
    pct = np.array([0.2, 0.4, 0.9], dtype=np.float32)
    lookup = {"000001.SZ": (dates, sec, pct)}
    q = np.array(["2024-12-31", "2025-05-01", "2025-06-01", "2026-06-01"], dtype="datetime64[ns]").view(np.int64)
    s_out, p_out = lookup_conditions_for_symbol(lookup, "000001.SZ", q, default_sector=85)
    assert s_out.tolist() == [85, 3, 7, 11]
    np.testing.assert_allclose(p_out, [0.5, 0.2, 0.4, 0.9], atol=1e-6)


def test_block_slices_cover_total_dim():
    covered = []
    for sl in BLOCK_SLICES.values():
        covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(TOTAL_DIM))
