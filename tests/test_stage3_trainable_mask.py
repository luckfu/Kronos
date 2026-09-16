"""CPU tests for Stage3 freeze-mask probes."""
import pytest
import torch

from finetune.stage3_ce_rank import Stage3CERankConfig
from finetune.stage3_path_alignment import PathAlignmentConfig
from finetune.stage3_trainable_mask import (
    apply_trainable_mask, assert_frozen_parameters_unchanged, snapshot_frozen_parameters,
)
from finetune.stage3_training_model import Stage3TrainingModel
from model import Kronos, KronosTokenizer


def make_model():
    torch.manual_seed(715)
    predictor = Kronos(4, 4, 2, 32, 4, 64, 0.0, 0.0, 0.0, 0.0, False)
    tokenizer = KronosTokenizer(6, 32, 4, 64, 2, 2, 0.0, 0.0, 0.0, 4, 4, 0.25, 1., 1., 1., 4)
    return Stage3TrainingModel(predictor, tokenizer, lookback=7, horizon=10, synchronize_ema=False)


def make_conditioned_model():
    torch.manual_seed(715)
    predictor = Kronos(
        4, 4, 2, 32, 4, 64, 0.0, 0.0, 0.0, 0.0, False,
        num_sectors=3, use_size_percentile=True,
    )
    tokenizer = KronosTokenizer(6, 32, 4, 64, 2, 2, 0.0, 0.0, 0.0, 4, 4, 0.25, 1., 1., 1., 4)
    return Stage3TrainingModel(predictor, tokenizer, lookback=7, horizon=10, synchronize_ema=False)


def batch(n=4):
    generator = torch.Generator().manual_seed(617)
    x = torch.randn(n, 18, 6, generator=generator)
    stamps = torch.zeros(n, 18, 5)
    return x, stamps


def named_requires_grad(predictor):
    return {name: bool(parameter.requires_grad) for name, parameter in predictor.named_parameters()}


def test_unknown_mask_is_rejected():
    predictor = make_model().predictor
    with pytest.raises(ValueError, match='Unknown trainable mask'):
        apply_trainable_mask(predictor, 'not_a_mask')


def test_dependency_layer_mask_freezes_the_rest_and_survives_a_step():
    core = make_model().train()
    core.config = PathAlignmentConfig(weight=0.0)
    core.ce_rank_config = Stage3CERankConfig(enabled=True, lambda_rank=0.0)
    audit = apply_trainable_mask(core.predictor, 'dependency_layer')
    assert audit['mask'] == 'dependency_layer'
    assert audit['trainable_names']
    assert all(name.startswith('dep_layer.') for name in audit['trainable_names'])
    flags = named_requires_grad(core.predictor)
    assert any(flags.values())
    for name, trainable in flags.items():
        if name.startswith('dep_layer.'):
            assert trainable is True, name
        else:
            assert trainable is False, name
    snapshot = snapshot_frozen_parameters(core.predictor)
    before = {name: parameter.detach().clone() for name, parameter in core.predictor.named_parameters()}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in core.predictor.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    x, stamps = batch()
    loss, _ = core(x, stamps)
    loss.backward()
    optimizer.step()
    assert_frozen_parameters_unchanged(core.predictor, snapshot)
    moved = any(
        not torch.equal(parameter.detach().cpu(), before[name].cpu())
        for name, parameter in core.predictor.named_parameters()
        if parameter.requires_grad
    )
    assert moved


def test_forecast_head_and_condition_masks_select_the_right_prefixes():
    head_audit = apply_trainable_mask(make_model().predictor, 'forecast_head')
    assert head_audit['trainable_names']
    assert all(name.startswith('head.') for name in head_audit['trainable_names'])
    condition_audit = apply_trainable_mask(make_conditioned_model().predictor, 'condition_branch')
    assert condition_audit['trainable_names']
    assert all(
        name.startswith(('sector_emb.', 'size_emb.', 'size_mlp.'))
        for name in condition_audit['trainable_names']
    )
    with pytest.raises(ValueError, match='selected no parameters'):
        apply_trainable_mask(make_model().predictor, 'blocks_7_8')


def test_weighted_ce_skips_rank_when_lambda_rank_is_zero():
    core = make_model().train()
    core.config = PathAlignmentConfig(weight=0.0)
    core.ce_rank_config = Stage3CERankConfig(enabled=True, lambda_rank=0.0)
    x, stamps = batch()
    loss, metrics = core(x, stamps)
    assert torch.isfinite(loss)
    assert float(metrics['rank_loss']) == 0.0
    loss.backward()
    assert any(parameter.grad is not None for parameter in core.predictor.parameters())
