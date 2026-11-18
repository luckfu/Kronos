"""Unit tests for G2 TopBlockRankAdapter freeze mask, KL/Cosine-IC losses, and gradients."""
from __future__ import annotations

import torch

from model.kronos import Kronos, KronosTokenizer
from finetune.gpu_alpha.g2_rank_finetune import (
    TopBlockRankAdapter,
    cosine_ic_loss_vec,
    kl_logits_anchor,
)


def _tiny_kronos():
    tok = KronosTokenizer(
        d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    )
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=4, d_model=32, n_heads=2, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False, num_sectors=10, num_size_buckets=0, context_layer=2,
        use_size_percentile=True, size_mlp_hidden_dim=16,
    )
    return tok, model


def test_top_block_adapter_freezes_lower_and_trains_top():
    tok, model = _tiny_kronos()
    adapter = TopBlockRankAdapter(model, freeze_until=2)
    for idx in range(2):
        for p in adapter.model.transformer[idx].parameters():
            assert not p.requires_grad
    for idx in range(2, 4):
        for p in adapter.model.transformer[idx].parameters():
            assert p.requires_grad
    assert all(not p.requires_grad for p in adapter.model.embedding.parameters())
    assert all(not p.requires_grad for p in adapter.model.sector_emb.parameters())

    raw = torch.randn(12, 60, 6).abs() + 10.0
    stamp = torch.zeros(12, 60, 5)
    sec = torch.randint(0, 10, (12,))
    pct = torch.rand(12)
    h_low = adapter.encode_frozen_lower(tok, raw, stamp)
    assert not h_low.requires_grad
    score, s1_ft, s1_c2 = adapter.forward_top(h_low, sec, pct, with_teacher=True)
    assert score.shape == (12,)
    assert s1_c2 is not None and s1_ft.shape == s1_c2.shape

    y = torch.randn(12)
    loss = cosine_ic_loss_vec(score, y) + 0.05 * kl_logits_anchor(s1_c2, s1_ft) + 1e-3 * adapter.weight_l2_anchor()
    loss.backward()
    assert adapter.rank_head.weight.grad is not None
    assert float(adapter.rank_head.weight.grad.abs().sum()) > 0.0
    # Lower layer received zero gradient
    assert all(p.grad is None for p in adapter.model.transformer[0].parameters())


def test_varlen_lookback_forward_works_for_60_and_120():
    tok, model = _tiny_kronos()
    adapter = TopBlockRankAdapter(model, freeze_until=2)
    raw = torch.randn(8, 120, 6).abs() + 10.0
    stamp = torch.zeros(8, 120, 5)
    sec = torch.randint(0, 10, (8,))
    pct = torch.rand(8)
    for lb in (60, 90, 120):
        h_low = adapter.encode_frozen_lower(tok, raw[:, -lb:, :], stamp[:, -lb:, :])
        sc, _, _ = adapter.forward_top(h_low, sec, pct, with_teacher=False)
        assert sc.shape == (8,)
