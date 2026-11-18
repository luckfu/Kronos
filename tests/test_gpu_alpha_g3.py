"""Unit tests for G3 CrossStockPeerRanker."""
from __future__ import annotations

import torch

from finetune.gpu_alpha.g3_cross_stock_context import CrossStockPeerRanker


def test_cross_stock_peer_ranker_shape_and_equivariance():
    torch.manual_seed(0)
    model = CrossStockPeerRanker(in_dim=64, proj_dim=32, num_sectors=12, num_heads=4)
    x = torch.randn(40, 64)
    sec = torch.randint(0, 12, (40,))
    out = model(x, sec)
    assert out.shape == (40,)
    # Initialized at exact linear skip
    torch.testing.assert_close(out, model.skip(x).squeeze(-1))

    # Permutation equivariance across stocks within the same date
    perm = torch.randperm(40)
    out_perm = model(x[perm], sec[perm])
    torch.testing.assert_close(out_perm, out[perm], atol=1e-5, rtol=1e-5)

    # Backward passes cleanly into attention + sector_emb
    loss = (out ** 2).mean()
    loss.backward()
    assert model.skip.weight.grad is not None
