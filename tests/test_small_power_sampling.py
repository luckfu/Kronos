import ast
import copy
import math
import pickle
import runpy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from finetune.dataset import QlibDataset
from finetune.kaggle_kronos_small_v21 import swanlab_chart_step
from finetune.power_law_sampler import PowerLawPlan, window_amplitudes


ROOT = Path(__file__).resolve().parents[1]


def test_amplitude_includes_reversal_and_uses_only_h1_through_h10():
    closes = np.ones(131) * 100
    closes[122] = 120
    closes[130] = 1000
    assert window_amplitudes(closes, [0])[0] == pytest.approx(0.2)


def test_mixture_is_capped_deterministic_and_increases_tail_exposure():
    amplitudes = np.repeat([0.01, 0.1, 0.2, 1.0], 100)
    plan = PowerLawPlan(amplitudes, 10000, 5, 13)
    repeated = PowerLawPlan(amplitudes, 10000, 5, 13)
    counts = np.bincount(plan.order // 100, minlength=4) / len(plan.order)
    weights = np.array([1, 2, 4, 4])
    expected = 0.4 / 4 + 0.6 * weights / weights.sum()
    assert counts == pytest.approx(expected, abs=0.01)
    assert plan.sha256 == repeated.sha256
    assert plan.report['sampled_amplitude_ge_5pct'] > plan.report['natural_amplitude_ge_5pct']
    assert np.array_equal(plan.take_segment(3), repeated.take_segment(3))
    assert plan.unique_through(2) == len(np.unique(plan.order[:20000]))
    with pytest.raises(ValueError):
        plan.take_segment(5)


def test_real_dataset_uses_sampled_plan_and_leaves_validation_natural(tmp_path, monkeypatch):
    dates = pd.bdate_range('2025-01-01', periods=180, name='datetime')
    closes = 100 * np.exp(np.sin(np.arange(180) / 8) * 0.1)
    panel = {'sh.600001': pd.DataFrame({
        'open': closes, 'high': closes + 1, 'low': closes - 1, 'close': closes,
        'volume': np.ones(180) * 100, 'amount': closes * 100,
    }, index=dates)}
    path = tmp_path / 'panel.pkl'
    with path.open('wb') as handle:
        pickle.dump(panel, handle)
    values = {
        'KRONOS_TRAIN_DATA_PATHS': str(path), 'KRONOS_VAL_DATA_PATHS': str(path),
        'KRONOS_METADATA_PATH': '', 'KRONOS_LOOKBACK_WINDOW': '120',
        'KRONOS_PREDICT_WINDOW': '10', 'KRONOS_TRAIN_SAMPLES_PER_SEGMENT': '20',
        'KRONOS_TRAIN_SAMPLING': 'capped_power_mixture_v1',
        'KRONOS_SAMPLING_START_SEGMENT': '520', 'KRONOS_SAMPLING_SEGMENTS': '3',
        'KRONOS_VALIDATION_SAMPLES': '0', 'KRONOS_COVERAGE_SEED': '20261013',
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    train = QlibDataset('train')
    val = QlibDataset('val')
    train.set_epoch_seed(520)
    first = train.active_positions.copy()
    train.set_epoch_seed(522)
    third = train.active_positions.copy()
    resumed = QlibDataset('train')
    resumed.set_epoch_seed(522)
    assert np.array_equal(third, resumed.active_positions)
    assert not np.array_equal(first, third)
    assert len(train[0][0]) == 131
    assert train.coverage_state(522)['unique_samples_covered'] == len(np.unique(train.sampling_plan.order))
    assert val.sampling_plan is None
    assert len(val) == val.total_samples
    assert len(np.unique(val.active_positions)) == val.total_samples
    monkeypatch.setenv('KRONOS_TRAIN_SAMPLING', 'coverage')
    baseline = QlibDataset('val')
    assert np.array_equal(val.active_positions, baseline.active_positions)


def trainer_helpers():
    names = {'restore_sampling_transition', 'validate_scheduler_transition_state',
             'validate_resume_guard', 'optimizer_to', 'segment_sample_count',
             'completed_coverage_windows', 'sampling_scheduler_steps'}
    tree = ast.parse((ROOT / 'finetune/train_predictor.py').read_text())
    module = ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {'math': math, 'torch': torch,
                 'refuse_low_precision_checkpoint': lambda *args: None,
                 'restore_rng_state': lambda state: None}
    exec(compile(module, 'trainer_helpers', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('parent_segment,parent_step', [(520, 162724), (720, 225324)])
def test_sampling_transition_preserves_optimizer_scheduler_and_global_axes(parent_segment, parent_step):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    scheduler.last_epoch = parent_step
    config = {'sampling_start_segment': parent_segment, 'sampling_parent_global_step': parent_step, 'predictor_loss_mode': 'forecast',
              'history_loss_weight': 0.02, 'use_amp': False}
    source = {'next_epoch': parent_segment, 'resume_step': 0, 'batch_idx_global': parent_step,
              'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
              'scheduler': scheduler.state_dict(), 'scheduler_type': 'warmup_constant',
              'scheduler_warmup_steps': 1627, 'resume_guard': {'batch_size': 32, 'effective_epochs': 520},
              'predictor_loss_mode': 'forecast', 'history_loss_weight': 0.02,
              'use_amp': False, 'amp_dtype': 'disabled', 'rng_state': {}}
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-5)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
    guard = {'batch_size': 32, 'effective_epochs': 720, 'train_sampling': 'capped_power_mixture_v1'}
    helpers = trainer_helpers()
    boundary, global_step = helpers['restore_sampling_transition'](
        source, config, guard, 1627, restored, restored_optimizer, restored_scheduler,
        None, torch.device('cpu'), 'disabled',
    )
    assert (boundary, global_step) == (parent_segment, parent_step)
    assert restored_scheduler.last_epoch == parent_step
    for parameter, saved in zip(restored.parameters(), model.parameters()):
        assert torch.equal(parameter, saved)
    for state in restored_optimizer.state.values():
        assert state['step'] == 1
        assert torch.count_nonzero(state['exp_avg']) > 0
    env = {'KRONOS_TRAIN_SAMPLING': 'capped_power_mixture_v1',
           'KRONOS_SAMPLING_START_SEGMENT': '520', 'KRONOS_SAMPLING_PARENT_GLOBAL_STEP': '162724'}
    assert swanlab_chart_step(521, 1, 313, env) == 162725
    assert swanlab_chart_step(521, 313, 313, env) == 163037
    assert swanlab_chart_step(522, 1, 313, env) == 163038
    with pytest.raises(ValueError):
        swanlab_chart_step(520, 277, 277, env)
    third_env = {**env, 'KRONOS_SAMPLING_START_SEGMENT': '720',
                 'KRONOS_SAMPLING_PARENT_GLOBAL_STEP': '225324'}
    assert swanlab_chart_step(721, 1, 313, third_env) == 225325
    assert swanlab_chart_step(920, 313, 313, third_env) == 287924
    with pytest.raises(ValueError, match='batch_size'):
        helpers['restore_sampling_transition'](
            source, config, {**guard, 'batch_size': 64}, 1627, restored,
            restored_optimizer, restored_scheduler, None, torch.device('cpu'), 'disabled',
        )


def test_round_environment_keeps_dashboard_and_12_hour_budget():
    ns = runpy.run_path(str(ROOT / 'finetune/small_power_relay/launch.py'))
    env = ns['training_environment'](Path('/runtime'), Path('/data'), Path('/parent'),
                                     Path('/tokenizer'), False, 520)
    assert env['SWANLAB_RUN_ID'] == 'smcl1011'
    assert env['KRONOS_EPOCHS'] == '720'
    assert env['KRONOS_MAX_RUNTIME_SECONDS'] == '40500'
    assert env['KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT'] == '520'
    assert env['KRONOS_SAMPLING_TRANSITION_STATE'] == '/parent/checkpoints/last_state.pt'
    assert env['KRONOS_COVERAGE_SEED'] == '20261011'
    assert env['KRONOS_SAMPLING_SEED'] == '20261013'


def test_global_scheduler_budget_includes_short_parent_final_segment():
    helpers = trainer_helpers()
    dataset = SimpleNamespace(total_samples=10397679, n_samples=20000,
                              sampling_plan=object(), config=SimpleNamespace(sampling_start_segment=520))
    steps = [math.ceil(helpers['segment_sample_count'](dataset, segment) / 64)
             for segment in range(720)]
    assert sum(steps[:520]) == 162724
    assert steps[519] == 277
    assert steps[520:] == [313] * 200
    assert sum(steps) == 225324


def test_gpu_quota_counts_days_in_duration():
    ns = runpy.run_path(str(ROOT / 'finetune/small_power_relay/push_chunk.py'))
    quota = SimpleNamespace(total_time_allowed=timedelta(days=1, hours=6),
                            time_used=timedelta(minutes=6), time_reserved=timedelta(0))
    assert ns['available_gpu_seconds'](quota) == 107640


def test_new_round_keeps_parent_step_after_multiple_sampling_rounds():
    ns = runpy.run_path(str(ROOT / 'finetune/small_power_relay/launch.py'))
    ns['configure_stage']({'start_segment': 720, 'parent_global_step': 225324,
                          'round_segments': 200, 'sampling_plan_sha256': 'new-plan',
                          'round': 3, 'seed': 20261018, 'first_chunk': 5,
                          'parent_kernel': 'user281434/kronos-small-clean-wc-dual-t4-c04'})
    env = ns['training_environment'](Path('/runtime'), Path('/data'), Path('/parent'),
                                     Path('/tokenizer'), False, 720)
    assert env['KRONOS_EPOCHS'] == '920'
    assert env['KRONOS_SAMPLING_START_SEGMENT'] == '720'
    assert env['KRONOS_SAMPLING_PARENT_GLOBAL_STEP'] == '225324'
    assert env['KRONOS_SAMPLING_SEED'] == '20261018'
    assert env['SWANLAB_RUN_ID'] == 'smcl1011'
    dataset = SimpleNamespace(total_samples=10397679, n_samples=20000,
                              sampling_plan=object(), config=SimpleNamespace(sampling_start_segment=720))
    config = {'sampling_parent_global_step': 225324, 'sampling_start_segment': 720, 'batch_size': 32}
    assert trainer_helpers()['sampling_scheduler_steps'](dataset, config, 920, 2) == 287924
