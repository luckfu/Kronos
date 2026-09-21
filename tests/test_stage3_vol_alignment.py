import torch

from finetune.stage3_vol_alignment import (
    daily_returns_from_close, expected_path_vol, mixture_mean_path_vol,
)


def test_jensen_gap_expected_vol_exceeds_vol_of_mean():
    last = torch.ones(2)
    low = last[:, None, None] * (1.0 + torch.linspace(-0.01, 0.01, 10)).view(1, 10, 1)
    high = last[:, None, None] * (1.0 + torch.linspace(-0.08, 0.08, 10)).view(1, 10, 1)
    candidates = torch.cat([low.expand(2, 10, 1), high.expand(2, 10, 1)], dim=-1)
    weights = torch.full((2, 10, 2), 0.5)
    expected, _, _ = expected_path_vol(candidates, weights, last)
    mixed = mixture_mean_path_vol(candidates, weights, last)
    assert torch.all(expected > mixed)


def test_daily_returns_match_close_to_close():
    last = torch.tensor([10.0, 20.0])
    future = torch.tensor([[10.2, 10.1, 10.5], [20.0, 21.0, 19.0]])
    daily = daily_returns_from_close(future, last)
    assert torch.allclose(daily[:, 0], future[:, 0] / last - 1.0)
    assert torch.allclose(daily[:, 1], (future[:, 1] / last) / (future[:, 0] / last) - 1.0)


def test_detached_decode_weights_have_no_vol_gradient():
    last = torch.ones(1)
    candidates = (1.0 + 0.1 * torch.randn(1, 10, 2)).abs() + 0.5
    logits = torch.tensor([[[0.0, 1.0]]]).repeat(1, 10, 1).requires_grad_(True)
    live = logits.softmax(-1)
    detached = live.detach()
    pred_live, _, _ = expected_path_vol(candidates, live, last)
    pred_dead, _, _ = expected_path_vol(candidates, detached, last)
    assert pred_live.requires_grad
    assert not pred_dead.requires_grad
    pred_live.mean().backward()
    assert logits.grad.abs().sum() > 0


def test_expected_vol_has_weight_gradient():
    last = torch.ones(1)
    low = (1.0 + 0.01 * torch.randn(1, 10, 1)).abs() + 0.5
    high = (1.0 + 0.2 * torch.randn(1, 10, 1)).abs() + 0.5
    candidates = torch.cat([low, high], dim=-1).detach()
    logits = torch.tensor([[[0.0, 0.0]]]).repeat(1, 10, 1).requires_grad_(True)
    weights = logits.softmax(-1)
    pred, _, _ = expected_path_vol(candidates, weights, last)
    pred.mean().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
