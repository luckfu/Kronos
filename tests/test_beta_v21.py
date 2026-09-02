import numpy as np
import torch

from finetune.beta_v21 import (
    DetachedEMANormalizer,
    compose_beta_v21_objective,
    consistency_statistics,
    same_date_return_bias_loss,
)
from finetune.dataset import QlibDataset, build_beta_v21_labels
from model import Kronos


def tiny_model(use_auxiliary):
    return Kronos(
        s1_bits=2,
        s2_bits=2,
        n_layers=2,
        d_model=8,
        n_heads=2,
        ff_dim=16,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=True,
        use_beta_v21_auxiliary=use_auxiliary,
    )


def test_return_bias_compares_same_date_cross_section_means():
    targets = torch.tensor([
        [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [2.0, 0.0]
    ])
    predictions = torch.tensor([
        [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 0.0]
    ])
    date_ids = torch.tensor([10, 10, 11, 11])

    loss = same_date_return_bias_loss(predictions, targets, date_ids)

    assert torch.isclose(loss, torch.tensor(0.125))


def test_ambiguous_same_day_barrier_is_masked_but_backtests_as_stop_loss():
    raw = np.ones((131, 6), dtype=np.float32) * 100.0
    raw[:, 3] = np.linspace(90.0, 100.0, 131)
    raw[120, 0] = 100.0
    raw[120, 1] = 106.0
    raw[120, 2] = 96.0

    labels = build_beta_v21_labels(raw, 120, np.datetime64('2026-08-03'))

    assert labels['barrier_target'].item() == 1
    assert not labels['barrier_valid'].item()
    assert torch.isclose(labels['utility'], torch.tensor(-0.034))
    assert labels['return_targets'].shape == (4,)


def test_v21_segment_order_groups_dates_without_changing_coverage():
    dataset = QlibDataset.__new__(QlibDataset)
    dataset.total_samples = 6
    dataset.n_samples = 6
    dataset.data_type = 'train'
    dataset.use_beta_v21_auxiliary = True
    dataset.coverage_order = np.asarray([4, 0, 3, 1, 5, 2])
    dataset.signal_date_ids = np.asarray([2, 1, 2, 1, 3, 3])

    dataset.set_epoch_seed(0)

    assert set(dataset.active_positions.tolist()) == set(range(6))
    ordered_dates = dataset.signal_date_ids[dataset.active_positions]
    assert ordered_dates.tolist() == sorted(ordered_dates.tolist())


def test_auxiliary_state_is_backward_compatible_and_causal_at_asof(tmp_path):
    base = tiny_model(use_auxiliary=False)
    auxiliary = tiny_model(use_auxiliary=True)
    incompatible = auxiliary.load_state_dict(base.state_dict(), strict=False)

    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        'return_head.weight', 'return_head.bias',
        'barrier_head.weight', 'barrier_head.bias',
    }
    base.save_pretrained(tmp_path, config=base._hub_mixin_config)
    loaded = Kronos.from_pretrained(tmp_path, use_beta_v21_auxiliary=True)
    assert loaded.return_head is not None
    assert loaded.barrier_head is not None

    auxiliary.eval()
    s1 = torch.zeros(2, 8, dtype=torch.long)
    s2 = torch.zeros(2, 8, dtype=torch.long)
    changed_s1 = s1.clone()
    changed_s2 = s2.clone()
    changed_s1[:, 5:] = 2
    changed_s2[:, 5:] = 3
    targets = torch.zeros_like(s1)
    _, first = auxiliary(
        s1, s2, use_teacher_forcing=True, s1_targets=targets,
        return_auxiliary=True, asof_index=4,
    )
    _, second = auxiliary(
        changed_s1, changed_s2, use_teacher_forcing=True, s1_targets=targets,
        return_auxiliary=True, asof_index=4,
    )

    assert torch.allclose(first['return'], second['return'], atol=1e-6)
    assert torch.allclose(first['barrier'], second['barrier'], atol=1e-6)


def test_composite_loss_warms_auxiliary_heads_without_changing_total_scale():
    scalar = torch.tensor(1.0, requires_grad=True)
    auxiliary = {
        'return': scalar,
        'barrier': scalar,
        'ranking': scalar,
    }
    normalizer = DetachedEMANormalizer(decay=0.99)

    cold, cold_metrics = compose_beta_v21_objective(
        scalar, scalar, auxiliary, normalizer, global_step=0
    )
    warm, warm_metrics = compose_beta_v21_objective(
        scalar, scalar, auxiliary, normalizer, global_step=1000
    )

    assert torch.isclose(cold, torch.tensor(1.0))
    assert torch.isclose(warm, torch.tensor(1.0))
    assert cold_metrics['ramp'] == 0.0
    assert warm_metrics['ramp'] == 1.0


def test_validation_consistency_reports_bias_and_distribution_drift():
    actual = torch.zeros(3, 4)
    auxiliary = torch.ones(3, 4)
    generated = torch.full((3, 4), -1.0)

    stats = consistency_statistics(auxiliary, generated, actual)

    assert stats['samples'] == 3
    assert stats['mae'] == [2.0] * 4
    assert stats['sign_agreement'] == [0.0] * 4
    assert stats['auxiliary_bias'] == [1.0] * 4
    assert stats['generated_bias'] == [-1.0] * 4
