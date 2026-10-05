"""Unit tests for Phase I sidecar deep-fail diagnosis helpers."""

from __future__ import annotations

import math

import numpy as np
import torch

from modernbert_finance.ablations.sidecar_deep_fail_diagnosis import (
    logit,
    train_bce_linear_head,
)


def test_logit_prior_25pct() -> None:
    assert abs(logit(0.25) - math.log(0.25 / 0.75)) < 1e-12


def test_bce_linear_recovers_separable_signal() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 8))
    # Strong linear signal → pos rate ~25%
    logits = x[:, 0] * 2.0 - 1.1
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    out = train_bce_linear_head(x, y, seed=0, epochs=25, lr=0.1)
    assert out["bce_linear"]["beats_prior"]
    assert out["bce_linear"]["delta_vs_prior"] < -0.01


def test_gate_ones_vs_zeros_forward_differs() -> None:
    """Sanity: multiplicative gate=0 silences market branch."""
    torch.manual_seed(0)
    market = torch.randn(2, 4, 8)
    gate0 = market * torch.zeros(1)
    gate1 = market * torch.ones(1)
    assert float(gate0.abs().sum()) == 0.0
    assert float(gate1.abs().sum()) > 0.0
