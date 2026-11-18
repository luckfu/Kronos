"""Unit tests for G10 FrozenSkipPeerRanker."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from g10_peer_resid_entry import FrozenSkipPeerRanker  # noqa: E402
from push_g10 import source_bundle  # noqa: E402


def test_frozen_skip_peer_ranker_freezes_skip_and_starts_exact():
    w = np.linspace(-0.1, 0.1, 64, dtype=np.float32)
    model = FrozenSkipPeerRanker(w_skip=w, proj_dim=16, dropout=0.1, resid_scale=0.25)
    assert not model.core.skip.weight.requires_grad
    assert not model.core.skip.bias.requires_grad
    x = torch.randn(10, 64)
    sec = torch.randint(0, 86, (10,))
    out = model(x, sec)
    expected = x @ torch.as_tensor(w)
    assert torch.allclose(out, expected, atol=1e-6)
    loss = out.sum()
    loss.backward()
    assert model.core.skip.weight.grad is None
    assert len(source_bundle()) > 5000
