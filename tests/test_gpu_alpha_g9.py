"""Unit tests for G9 anisotropic block-regularized ridge solver and bundle."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from g9_aniso_multiscale_entry import BASE_DIM, JOINT_DIM, SMALL_DIM, solve_aniso_ridge  # noqa: E402
from push_g9 import source_bundle  # noqa: E402


def test_solve_aniso_ridge_matches_block_penalties():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 12))
    y = rng.normal(size=64)
    gram = x.T @ x
    rhs = x.T @ y
    w_aniso = solve_aniso_ridge(gram, rhs, small_dim=4, alpha_small=10.0, alpha_base=1000.0)
    pen = np.diag([10.0] * 4 + [1000.0] * 8)
    w_ref = np.linalg.solve(gram + pen, rhs)
    assert np.allclose(w_aniso, w_ref, atol=1e-10)
    # Stronger penalty shrinks base slice weights relative to isotropic small alpha
    w_iso_small = solve_aniso_ridge(gram, rhs, small_dim=4, alpha_small=10.0, alpha_base=10.0)
    assert np.linalg.norm(w_aniso[4:]) < np.linalg.norm(w_iso_small[4:])


def test_g9_constants_and_bundle():
    assert SMALL_DIM == 512
    assert BASE_DIM == 2496
    assert JOINT_DIM == 3008
    assert len(source_bundle()) > 5000
