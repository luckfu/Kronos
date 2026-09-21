import torch

from finetune.stage3_path_alignment import PathAlignmentConfig
from finetune.stage3_training_model import Stage3TrainingModel
from finetune.stage3_vol_alignment import (
    VOL_METRIC_KEYS, VolAlignmentConfig, apply_temperature_nucleus,
    daily_returns_from_close, expected_path_vol, mixture_mean_path_vol,
)
from model import Kronos, KronosTokenizer


def test_lower_temperature_peaks_the_distribution():
    logits = torch.tensor([[0.0, 1.0, 0.2]])
    hot = apply_temperature_nucleus(logits, 1.0, 1.0).softmax(-1)
    cold = apply_temperature_nucleus(logits, 0.65, 1.0).softmax(-1)
    assert cold.max() > hot.max()


def test_nucleus_masks_the_tail():
    logits = torch.tensor([[4.0, 1.0, -8.0, -9.0]])
    filtered = apply_temperature_nucleus(logits, 1.0, 0.8)
    probs = filtered.softmax(-1)
    assert probs[0, -1] < 1e-6
    assert probs[0, -2] < 1e-6
    assert probs[0, 0] > 0.5


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


def test_production_decode_vol_metrics_include_evaluate_keys():
    """evaluate() and train logging both index these keys on the vol path."""
    torch.manual_seed(21)
    predictor = Kronos(4, 4, 2, 32, 4, 64, 0.0, 0.0, 0.0, 0.0, False)
    tokenizer = KronosTokenizer(6, 32, 4, 64, 2, 2, 0.0, 0.0, 0.0, 4, 4, 0.25, 1., 1., 1., 4)
    core = Stage3TrainingModel(
        predictor, tokenizer, lookback=7, horizon=10, synchronize_ema=False,
        config=PathAlignmentConfig(weight=0.0),
        vol_config=VolAlignmentConfig(
            weight=0.15, temperature=0.65, top_p=0.8, candidates=5,
        ),
    )
    generator = torch.Generator().manual_seed(617)
    x = torch.randn(2, 18, 6, generator=generator)
    stamps = torch.zeros(2, 18, 5)
    means = torch.zeros(2, 6)
    stds = torch.ones(2, 6)
    assert 'uniform_path_vol' in VOL_METRIC_KEYS
    for training in (False, True):
        core.train(training)
        _, metrics = core(x, stamps, feature_means=means, feature_stds=stds)
        missing = [key for key in VOL_METRIC_KEYS if key not in metrics]
        assert missing == [], missing
        stacked = torch.stack([metrics[key].double() * len(x) for key in VOL_METRIC_KEYS])
        assert torch.isfinite(stacked).all()


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
