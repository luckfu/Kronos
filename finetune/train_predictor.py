import os
import sys
import json
import math
import random
import signal
import threading
import time
from time import gmtime, strftime
import numpy as np
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    try:
        import torch_xla.runtime as xr
    except ImportError:
        xr = None
except ImportError:
    xm = None
    xr = None
    MpDeviceLoader = None


def get_xla_world_size():
    """Safely get TPU world size across legacy XRT and modern PJRT runtimes."""
    if xr is not None and hasattr(xr, 'addressable_device_count'):
        try:
            count = int(xr.addressable_device_count())
            if count > 1:
                return count
        except Exception:
            pass
    if xr is not None and hasattr(xr, 'world_size'):
        try:
            count = int(xr.world_size())
            if count > 1:
                return count
        except Exception:
            pass
    if xm is not None and hasattr(xm, 'xrt_world_size'):
        try:
            return int(xm.xrt_world_size())
        except Exception:
            pass
    return int(os.getenv('KRONOS_TPU_CORES', '1'))

try:
    import comet_ml
except ImportError:
    comet_ml = None

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset, build_same_day_ranking_batches
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference
from beta_v21 import (
    DetachedEMANormalizer,
    compose_beta_v21_objective,
    compute_auxiliary_losses,
    consistency_statistics,
    expected_utility_score,
    generated_return_targets,
    mean_within_date_spearman,
    same_date_pairwise_accuracy,
)
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)
from drive_cleanup import cleanup_drive_conflict_files
from validation_precision import (
    ValidationLossAccumulator,
    all_reduce_validation_sums,
    to_float32,
    weighted_means,
)
try:
    import tpu_self_checks
except ImportError:  # GPU/dual-T4 overlay may omit TPU-only helpers
    tpu_self_checks = None


STOP_REQUESTED = False

# PJRT's thread-per-device launcher invokes ``main`` concurrently from eight
# threads in one Python process.  Constructing QlibDataset independently in
# every thread reloads both full pandas panels eight times (16 copies in
# total), which can exhaust Kaggle host RAM before the first segment finishes.
# Dataset access is read-only during a segment; share one instance per data
# split within the process and keep the cache scoped to this training run.
_DATASET_CACHE = {}
_DATASET_CACHE_LOCK = threading.Lock()


def _get_shared_dataset(data_type):
    dataset = _DATASET_CACHE.get(data_type)
    if dataset is not None:
        return dataset
    with _DATASET_CACHE_LOCK:
        dataset = _DATASET_CACHE.get(data_type)
        if dataset is None:
            dataset = QlibDataset(data_type)
            _DATASET_CACHE[data_type] = dataset
    return dataset


def request_safe_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        "Stop requested; training will stop after the current batch and resume "
        "from the last completed segment."
    )


def write_progress(save_dir, **payload):
    if not save_dir:
        return
    path = os.path.join(save_dir, 'progress.json')
    temporary = f'{path}.tmp'
    document = {
        'updated_at': strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        **payload,
    }
    with open(temporary, 'w') as handle:
        json.dump(document, handle, indent=2)
    os.replace(temporary, path)


def append_metric(save_dir, **payload):
    if not save_dir:
        return
    document = {
        'updated_at': strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        **payload,
    }
    with open(os.path.join(save_dir, 'metrics.jsonl'), 'a') as handle:
        handle.write(json.dumps(document) + '\n')


def distributed_barrier(device, tag):
    """Synchronize either a torch.distributed group or PJRT replicas."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    elif device.type == 'xla' and xm is not None:
        if os.getenv('KRONOS_XLA_SINGLE_PROCESS', '0').strip().lower() in {
            '1', 'true', 'yes', 'on'
        }:
            return
        try:
            if get_xla_world_size() > 1:
                xm.rendezvous(tag)
        except Exception:
            # Preserve the original exception at the operation following the
            # barrier when a legacy one-core runtime lacks topology helpers.
            if os.getenv('KRONOS_XLA_SINGLE_PROCESS', '0') != '1':
                raise


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def mps_available():
    return bool(
        hasattr(torch, 'backends')
        and hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
    )


def capture_rng_state(include_xla=False):
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if mps_available() and hasattr(torch.mps, 'get_rng_state'):
        state['mps'] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if include_xla and xm is not None and hasattr(xm, 'get_rng_state'):
        state['xla'] = xm.get_rng_state()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'mps' in state and mps_available() and hasattr(torch.mps, 'set_rng_state'):
        torch.mps.set_rng_state(state['mps'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'xla' in state and xm is not None and hasattr(xm, 'set_rng_state'):
        xm.set_rng_state(state['xla'])


def save_resume_state(path, model, optimizer, scheduler, **metadata):
    cleanup_drive_conflict_files()
    temporary = f'{path}.tmp'
    model_device = next(model.parameters()).device
    is_xla = model_device.type == 'xla'
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': capture_rng_state(include_xla=is_xla),
        **metadata,
    }
    if is_xla:
        if xm is None:
            raise RuntimeError('Cannot save XLA checkpoint without torch_xla')
        xm.save(payload, temporary)
        # ``xm.save`` is master-only by default.  Non-master replicas must not
        # attempt the replace: their temporary file does not exist.
        is_master = (
            xm.is_master_ordinal() if hasattr(xm, 'is_master_ordinal')
            else int(os.getenv('ORDINAL', '0')) == 0
        )
        if is_master:
            os.replace(temporary, path)
        # No barrier here: every caller runs inside a ``rank == 0`` block, and
        # with real PJRT replication a rank-0-only rendezvous would hang (or
        # pair with the next all-rank barrier).  Callers synchronize after.
        cleanup_drive_conflict_files()
        return
    else:
        torch.save(payload, temporary)
    os.replace(temporary, path)
    cleanup_drive_conflict_files()


def save_pretrained_with_retry(model, path, config, attempts=3, retry_delay=2):
    """Retry checkpoint export when a mounted filesystem briefly loses the directory."""
    cleanup_drive_conflict_files()
    for attempt in range(1, attempts + 1):
        try:
            os.makedirs(path, exist_ok=True)
            model_device = next(model.parameters()).device
            if model_device.type == 'xla':
                _save_xla_pretrained(model, path, config)
            else:
                model.save_pretrained(path, config=config)
            cleanup_drive_conflict_files()
            return
        except FileNotFoundError:
            if attempt == attempts:
                raise
            print(
                f"Checkpoint directory disappeared while saving {path}; "
                f"retrying ({attempt}/{attempts})..."
            )
            time.sleep(retry_delay)


def _save_xla_pretrained(model, path, config):
    """Export an XLA model through a CPU state dict before safetensors writes.

    ``PyTorchModelHubMixin`` serializes each XLA tensor directly.  That can
    trigger unsupported ``contiguous``/CPU conversions during a filesystem
    export, so first synchronize the state with ``xm.save`` and then use the
    normal hub serializer on the CPU copy.
    """
    if xm is None:
        raise RuntimeError('Cannot export an XLA model without torch_xla')
    from pathlib import Path
    from huggingface_hub.hub_mixin import save_model_as_safetensor

    state_path = f'{path}.xla_state.pt'
    xm.save(model.state_dict(), state_path)
    if hasattr(xm, 'is_master_ordinal') and not xm.is_master_ordinal():
        # xm.save wrote nothing on non-master replicas (weights are identical).
        return
    try:
        state_dict = torch.load(state_path, map_location='cpu')

        class _StateDictProxy:
            def __init__(self, state):
                self._state = state

            def state_dict(self):
                return self._state

        save_model_as_safetensor(
            _StateDictProxy(state_dict),
            str(Path(path) / 'model.safetensors'),
        )
        with open(Path(path) / 'config.json', 'w') as handle:
            json.dump(config, handle, sort_keys=True, indent=2)
    finally:
        if os.path.exists(state_path):
            os.unlink(state_path)


def model_export_config(core_model, config):
    """Return the conditioning configuration embedded in Best/Last exports."""
    model_config = dict(getattr(core_model, '_hub_mixin_config', {}) or {})
    model_config.update({
        'num_sectors': int(config.get('num_sectors', 0)),
        'num_size_buckets': int(config.get('num_size_buckets', 0)),
        'context_layer': int(config.get('context_layer', 0)),
        'use_size_percentile': bool(config.get('use_size_percentile', False)),
        'size_mlp_hidden_dim': int(config.get('size_mlp_hidden_dim', 64)),
        'use_beta_v21_auxiliary': bool(
            config.get('use_beta_v21_auxiliary', False)
        ),
    })
    return model_config


def build_resume_guard(config, effective_epochs, segments_per_coverage):
    """Values that must remain identical when an output tree is continued."""
    keys = (
        'lookback_window', 'predict_window', 'batch_size', 'use_amp', 'n_train_iter',
        'coverage_seed',
        'n_val_iter', 'coverage_passes', 'effective_epochs',
        'segments_per_coverage', 'scheduler_type', 'scheduler_min_learning_rate',
        'predictor_min_learning_rate', 'condition_min_learning_rate',
        'scheduler_warmup_ratio', 'predictor_warmup_start_learning_rate',
        'condition_warmup_start_learning_rate', 'predictor_learning_rate',
        'condition_learning_rate', 'condition_fast_decay_ratio',
        'condition_fast_decay_learning_rate', 'adam_weight_decay',
        'gradient_clip_norm', 'condition_monitor_interval_steps',
        'condition_ablation_interval_segments',
        'predictor_loss_mode', 'history_loss_weight', 'forecast_horizon_weights',
        'best_selection_metric',
        'trainable_transformer_layers',
        'use_sector_features', 'use_size_features', 'use_size_percentile',
        'disable_condition_inputs',
        'num_sectors', 'num_size_buckets', 'context_layer',
        'train_signal_start', 'train_signal_end', 'val_signal_start', 'val_signal_end',
        'dataset_manifest_sha256', 'bootstrap_completed_segments',
        'fixed_validation_manifest_sha256', 'validation_quick_samples',
        'validation_large_samples', 'validation_large_interval_segments',
        'validation_full_only',
        'use_beta_v21_auxiliary', 'beta_v21_auxiliary_warmup_steps',
        'beta_v21_ranking_weight',
        'beta_v21_ema_decay', 'beta_v21_validation_denominators',
        'beta_v21_auto_calibrate', 'collect_validation_auxiliary',
        'beta_v21_consistency_samples', 'beta_v21_consistency_sample_count',
        'exclude_fixed_validation_from_training',
    )
    guard = {}
    for key in keys:
        if key in config:
            value = config[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                guard[key] = value
    guard['effective_epochs'] = int(effective_epochs)
    guard['segments_per_coverage'] = int(segments_per_coverage)
    return guard


def validate_resume_guard(saved, current, ignore_keys=None):
    if not isinstance(saved, dict):
        raise ValueError(
            'Resume checkpoint has no complete resume_guard; refusing unsafe continuation'
        )
    ignored = set(ignore_keys or ())
    current = {key: value for key, value in current.items() if key not in ignored}
    missing = sorted(set(current) - set(saved))
    if missing:
        raise ValueError(
            f'Resume checkpoint is missing guard fields: {missing}'
        )
    for key, expected in current.items():
        actual = saved.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f'Resume guard mismatch for {key}: {actual!r} != {expected!r}'
                )
        elif actual != expected:
            raise ValueError(
                f'Resume guard mismatch for {key}: {actual!r} != {expected!r}'
            )


def restore_optimizer_for_scheduler_transition(
    optimizer, source_state, target_group_plan, device
):
    """Restore AdamW moments while retaining the new stage's LR group plan."""
    source_optimizer = source_state.get('optimizer')
    if not isinstance(source_optimizer, dict):
        raise ValueError('Scheduler transition checkpoint has no optimizer state')
    source_groups = source_optimizer.get('param_groups', [])
    if len(source_groups) != len(target_group_plan):
        raise ValueError(
            'Scheduler transition optimizer group count mismatch: '
            f'{len(source_groups)} != {len(target_group_plan)}'
        )
    source_names = [group.get('name') for group in source_groups]
    target_names = [group.get('name') for group in target_group_plan]
    if source_names != target_names:
        raise ValueError(
            'Scheduler transition optimizer groups do not match: '
            f'{source_names} != {target_names}'
        )

    optimizer.load_state_dict(source_optimizer)
    optimizer_to(optimizer, device)
    plan_keys = (
        'name', 'family', 'lr', 'initial_lr', 'peak_lr', 'warmup_start_lr',
        'min_lr', 'weight_decay',
    )
    for group, target in zip(optimizer.param_groups, target_group_plan):
        for key in plan_keys:
            if key in target:
                group[key] = target[key]


def validate_scheduler_transition_state(source_state, config, amp_dtype_name):
    """Reject a cross-stage transition that changes the training contract."""
    if int(source_state.get('resume_step', -1)) != 0:
        raise ValueError('Scheduler transition requires a segment-boundary checkpoint')
    if source_state.get('predictor_loss_mode') != config.get('predictor_loss_mode'):
        raise ValueError('Scheduler transition predictor loss mode mismatch')
    if not math.isclose(
        float(source_state.get('history_loss_weight', float('nan'))),
        float(config.get('history_loss_weight', 0.0)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError('Scheduler transition history loss weight mismatch')
    if bool(source_state.get('use_amp', False)) != bool(config.get('use_amp', False)):
        raise ValueError('Scheduler transition AMP mode mismatch')
    if source_state.get('amp_dtype') != amp_dtype_name:
        raise ValueError('Scheduler transition AMP dtype mismatch')



class SameDayRankingBatchSampler:
    """Yield same-day index batches from the dataset's current coverage slice.

    Recomputed every iteration so set_epoch_seed can advance the shuffled
    segment without sorting it. Does not touch active_positions.
    """

    def __init__(self, dataset, batch_size, rank, world_size):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def _batches(self):
        positions = self.dataset.active_positions
        date_ids = self.dataset.signal_date_ids[positions]
        return build_same_day_ranking_batches(
            date_ids, self.batch_size, self.rank, self.world_size,
        )

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: training, quick-validation and large-validation loaders and datasets.
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = _get_shared_dataset('train')
    valid_dataset = _get_shared_dataset('val')
    validation_full_only = bool(config.get('validation_full_only', False))
    quick_dataset = None
    if not validation_full_only:
        quick_count = int(
            getattr(valid_dataset, 'quick_validation_count', len(valid_dataset))
        )
        quick_dataset = Subset(valid_dataset, range(quick_count))
        print(
            f"[Rank {rank}] Train dataset size: {len(train_dataset)}, "
            f"Quick validation size: {len(quick_dataset)}, "
            f"Large validation size: {len(valid_dataset)}"
        )
    else:
        print(
            f"[Rank {rank}] Train dataset size: {len(train_dataset)}, "
            f"Full-only validation size: {len(valid_dataset)}"
        )

    use_same_day_batches = bool(config.get('same_day_ranking_batches', False))
    use_ddp = (dist.is_available() and dist.is_initialized()) or world_size > 1
    if use_same_day_batches:
        if rank == 0:
            print(
                f"[Rank {rank}] Same-day ranking batches enabled "
                f"(batch_size={config['batch_size']}, world_size={world_size}). "
                "Segment order stays shuffled coverage_order; pairs are packed "
                "inside batches only and are not a signal_date sort."
            )

        def _same_day_loader(dataset):
            return DataLoader(
                dataset,
                batch_sampler=SameDayRankingBatchSampler(
                    dataset, config['batch_size'], rank, world_size,
                ),
                num_workers=config.get('num_workers', 2),
                pin_memory=torch.cuda.is_available(),
            )

        train_loader = _same_day_loader(train_dataset)
        quick_val_loader = (
            _same_day_loader(quick_dataset) if quick_dataset is not None else None
        )
        large_val_loader = _same_day_loader(valid_dataset)
        return (
            train_loader, quick_val_loader, large_val_loader,
            train_dataset, valid_dataset,
        )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    quick_val_sampler = (
        DistributedSampler(
            quick_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if use_ddp and quick_dataset is not None else None
    )
    rank0_only_validation = bool(
        config.get('validation_rank0_only', False)
        and world_size > 1
        and os.getenv('KRONOS_DEVICE', '').lower() == 'xla'
    )
    # Non-zero TPU workers do not iterate validation in rank-0-only mode. Do
    # not shard rank 0: it must see the complete fixed validation set.
    large_val_sampler = (
        None if rank0_only_validation else
        DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if use_ddp else None
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        shuffle=False, num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=False
    )
    quick_val_loader = None
    if quick_dataset is not None:
        quick_val_loader = DataLoader(
            quick_dataset, batch_size=config['batch_size'], sampler=quick_val_sampler,
            shuffle=False, num_workers=config.get('num_workers', 2),
            pin_memory=torch.cuda.is_available(), drop_last=False
        )
    large_val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=large_val_sampler,
        shuffle=False, num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=False
    )
    return (
        train_loader, quick_val_loader, large_val_loader,
        train_dataset, valid_dataset,
    )


def configure_trainable_parameters(model, config):
    """Freeze the pretrained trunk and train only the adaptation layers."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    heads_only = bool(config.get('train_beta_v21_heads_only', False))
    if heads_only:
        # Frozen trunk: only Beta v2.1 return_head + barrier_head train.
        # Size/sector condition modules feed forecast, so they stay frozen.
        trainable_modules = [
            getattr(model, 'return_head', None),
            getattr(model, 'barrier_head', None),
        ]
        if any(module is None for module in trainable_modules):
            raise ValueError(
                'train_beta_v21_heads_only requires use_beta_v21_auxiliary '
                'with return_head and barrier_head present'
            )
    else:
        trainable_modules = [
            model.sector_emb, model.size_emb, model.size_mlp,
            model.norm, model.dep_layer, model.head,
            getattr(model, 'return_head', None), getattr(model, 'barrier_head', None),
        ]

    for module in trainable_modules:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    layer_count = int(config.get('trainable_transformer_layers', 0))
    if heads_only:
        # Ignore KRONOS_TRAINABLE_TRANSFORMER_LAYERS=-1 full-train when freezing.
        layer_count = 0
    if layer_count < 0:
        # Explicit full-Predictor incremental fine-tuning mode.  This mirrors
        # finetune_csv/finetune_base_model.py, while retaining separate LR
        # groups for the conditioning branches below.
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif layer_count > 0:
        for layer in model.transformer[-layer_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable predictor parameters: {trainable:,}/{total:,} ({trainable / total:.1%})")
    if heads_only:
        print(
            "Frozen trunk heads-only mode: trainable modules = "
            + ", ".join(trainable_names)
        )


def reset_conditioning(model, config):
    """Reset selected condition adapters before starting a fresh optimizer."""
    if config.get('reset_sector_embedding', False) and model.sector_emb is not None:
        torch.nn.init.zeros_(model.sector_emb.weight)
        print('Reset sector_emb to zero for fresh industry conditioning.')
    if config.get('reset_size_embedding', False):
        if model.size_emb is not None:
            torch.nn.init.zeros_(model.size_emb.weight)
            print('Reset size_emb to zero for the new full-market bucket definition.')
        if model.size_mlp is not None:
            torch.nn.init.zeros_(model.size_mlp[-1].weight)
            torch.nn.init.zeros_(model.size_mlp[-1].bias)
            print('Reset size percentile output layer to zero.')


def completed_coverage_windows(dataset, completed_segments, coverage_passes):
    segments_per_pass = math.ceil(dataset.total_samples / dataset.n_samples)
    complete_passes, remaining_segments = divmod(
        int(completed_segments), segments_per_pass
    )
    covered = complete_passes * dataset.total_samples + min(
        remaining_segments * dataset.n_samples, dataset.total_samples
    )
    return min(covered, dataset.total_samples * coverage_passes)


def segment_sample_count(dataset, segment_index):
    segments_per_pass = math.ceil(dataset.total_samples / dataset.n_samples)
    segment_in_pass = int(segment_index) % segments_per_pass
    start = segment_in_pass * dataset.n_samples
    return min(dataset.n_samples, dataset.total_samples - start)


def optimizer_steps_for_completed_segments(
    dataset, completed_segments, world_size, batch_size
):
    """Return the exact global scheduler position after complete segments."""
    return sum(
        math.ceil(
            math.ceil(segment_sample_count(dataset, segment) / world_size)
            / batch_size
        )
        for segment in range(int(completed_segments))
    )


def segment_run_limit_reached(start_segment, next_segment, max_segments_per_run):
    """Return whether this invocation has completed its configured chunk."""
    limit = max(0, int(max_segments_per_run or 0))
    return limit > 0 and int(next_segment) - int(start_segment) >= limit


def is_condition_parameter(name):
    return name.startswith(('sector_emb.', 'size_emb.', 'size_mlp.'))


# Forecast / auxiliary heads. These are not the transformer trunk. With
# KRONOS_SPLIT_TRUNK_HEAD_LR they share the condition adapter learning rate
# instead of predictor_learning_rate.
ADAPTATION_HEAD_PREFIXES = (
    'norm.', 'dep_layer.', 'head.', 'return_head.', 'barrier_head.',
)


def is_adaptation_head_parameter(name):
    return name.startswith(ADAPTATION_HEAD_PREFIXES)


def parameter_optimizer_family(name, config):
    """Map a parameter to an AdamW learning-rate family.

    Historical default: sector/size are ``condition``; the transformer trunk
    and every head share ``adaptation`` (predictor LR).

    ``split_trunk_head_learning_rate`` (KRONOS_SPLIT_TRUNK_HEAD_LR): trunk
    stays ``adaptation``. norm / dep_layer / head / return_head / barrier_head
    join ``condition`` so they use condition_learning_rate with sector/size.
    """
    if is_condition_parameter(name):
        return 'condition'
    if (
        config.get('split_trunk_head_learning_rate')
        and is_adaptation_head_parameter(name)
    ):
        return 'condition'
    return 'adaptation'


def parameter_uses_weight_decay(name, parameter):
    lower_name = name.lower()
    return bool(
        parameter.ndim >= 2
        and not name.endswith('.bias')
        and 'norm' not in lower_name
        and not name.startswith(('sector_emb.', 'size_emb.'))
    )


def warmup_cosine_multiplier(step, total_steps, warmup_steps, start_lr, peak_lr, min_lr):
    step = max(0, min(int(step), int(total_steps)))
    if step <= warmup_steps:
        progress = step / max(1, warmup_steps)
        learning_rate = start_lr + (peak_lr - start_lr) * progress
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        learning_rate = min_lr + 0.5 * (peak_lr - min_lr) * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )
    return learning_rate / peak_lr


def warmup_constant_multiplier(step, warmup_steps, start_lr, peak_lr):
    """Warm up linearly, then hold the configured peak learning rate."""
    step = max(0, int(step))
    if step <= warmup_steps:
        progress = step / max(1, int(warmup_steps))
        learning_rate = start_lr + (peak_lr - start_lr) * progress
    else:
        learning_rate = peak_lr
    return learning_rate / peak_lr


def two_speed_multiplier(
    step, total_steps, warmup_steps, start_lr, peak_lr, min_lr,
    family, condition_fast_decay_steps, condition_fast_decay_lr,
):
    """Continuous monotonic schedule with an accelerated condition decay."""
    if family != 'condition':
        return warmup_cosine_multiplier(
            step, total_steps, warmup_steps, start_lr, peak_lr, min_lr
        )

    step = max(0, min(int(step), int(total_steps)))
    fast_decay_steps = max(int(warmup_steps) + 1, int(condition_fast_decay_steps))
    fast_decay_steps = min(fast_decay_steps, int(total_steps))
    if step <= warmup_steps:
        progress = step / max(1, warmup_steps)
        learning_rate = start_lr + (peak_lr - start_lr) * progress
    elif step <= fast_decay_steps:
        progress = (step - warmup_steps) / max(1, fast_decay_steps - warmup_steps)
        learning_rate = condition_fast_decay_lr + 0.5 * (
            peak_lr - condition_fast_decay_lr
        ) * (1.0 + math.cos(math.pi * progress))
    else:
        progress = (step - fast_decay_steps) / max(1, total_steps - fast_decay_steps)
        learning_rate = min_lr + 0.5 * (
            condition_fast_decay_lr - min_lr
        ) * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return learning_rate / peak_lr


def build_optimizer_groups(model, config):
    grouped = {
        ('adaptation', True): [],
        ('adaptation', False): [],
        ('condition', True): [],
        ('condition', False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = parameter_optimizer_family(name, config)
        grouped[(family, parameter_uses_weight_decay(name, parameter))].append(parameter)

    learning_rates = {
        'adaptation': float(config['predictor_learning_rate']),
        'condition': float(config['condition_learning_rate']),
    }
    warmup_start_lrs = {
        'adaptation': float(config['predictor_warmup_start_learning_rate']),
        'condition': float(config['condition_warmup_start_learning_rate']),
    }
    minimum_lrs = {
        'adaptation': float(config.get(
            'predictor_min_learning_rate',
            config.get('scheduler_min_learning_rate', 1e-6),
        )),
        'condition': float(config.get(
            'condition_min_learning_rate',
            config.get('scheduler_min_learning_rate', 1e-6),
        )),
    }
    optimizer_groups = []
    for family in ('adaptation', 'condition'):
        for decay in (True, False):
            parameters = grouped[(family, decay)]
            if not parameters:
                continue
            optimizer_groups.append({
                'params': parameters,
                'name': f"{family}_{'decay' if decay else 'no_decay'}",
                'family': family,
                'lr': learning_rates[family],
                'peak_lr': learning_rates[family],
                'warmup_start_lr': warmup_start_lrs[family],
                'min_lr': minimum_lrs[family],
                'weight_decay': float(config['adam_weight_decay']) if decay else 0.0,
            })
    return optimizer_groups


def validate_uniform_learning_rate_config(config):
    if config.get('scheduler_type') != 'uniform_cosine':
        return
    pairs = (
        ('peak LR', 'predictor_learning_rate', 'condition_learning_rate'),
        (
            'warmup start LR',
            'predictor_warmup_start_learning_rate',
            'condition_warmup_start_learning_rate',
        ),
        ('minimum LR', 'predictor_min_learning_rate', 'condition_min_learning_rate'),
    )
    for label, adaptation_key, condition_key in pairs:
        adaptation_value = float(config[adaptation_key])
        condition_value = float(config[condition_key])
        if adaptation_value != condition_value:
            raise ValueError(
                'uniform_cosine requires identical adaptation and condition '
                f'{label}: {adaptation_value} != {condition_value}'
            )


def parameter_family_statistics(named_parameters, learning_rate):
    """Return size-normalized gradient and update diagnostics for one family."""
    grad_square_sum = None
    weight_square_sum = None
    parameter_count = 0
    gradient_count = 0
    for _, parameter in named_parameters:
        values = parameter.detach().float()
        weight_term = torch.sum(values * values)
        weight_square_sum = (
            weight_term if weight_square_sum is None else weight_square_sum + weight_term
        )
        parameter_count += parameter.numel()
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            grad_term = torch.sum(gradient * gradient)
            grad_square_sum = (
                grad_term if grad_square_sum is None else grad_square_sum + grad_term
            )
            gradient_count += parameter.numel()
    grad_total_l2 = math.sqrt(
        0.0 if grad_square_sum is None else float(grad_square_sum.item())
    )
    weight_total_l2 = math.sqrt(
        0.0 if weight_square_sum is None else float(weight_square_sum.item())
    )
    grad_rms = grad_total_l2 / math.sqrt(max(1, gradient_count))
    weight_rms = weight_total_l2 / math.sqrt(max(1, parameter_count))
    return {
        'grad_total_l2': grad_total_l2,
        'grad_rms': grad_rms,
        'weight_total_l2': weight_total_l2,
        'weight_rms': weight_rms,
        'update_weight_ratio': float(learning_rate) * grad_rms / (weight_rms + 1e-12),
        'parameter_count': parameter_count,
    }


def learning_rates_by_family(optimizer):
    """Return peak applied LR per family; missing families (zero params) are 0.0.

    Frozen-trunk / heads-only builds omit the empty adaptation (or condition)
    AdamW groups, so callers must not KeyError on absent families when printing
    or logging split LRs.
    """
    result = {'adaptation': 0.0, 'condition': 0.0}
    for group in optimizer.param_groups:
        family = group['family']
        result[family] = max(result.get(family, 0.0), float(group['lr']))
    return result


def objective_token_slices(sequence_length, lookback_window, predict_window):
    """Return next-token positions for history reconstruction and forecasting."""
    history_stop = int(lookback_window) - 1
    forecast_stop = history_stop + int(predict_window)
    if history_stop < 1:
        raise ValueError('lookback_window must provide at least one history target')
    if forecast_stop > int(sequence_length):
        raise ValueError(
            f'Need {forecast_stop} target positions for the forecast objective, '
            f'but only {sequence_length} are available'
        )
    return slice(0, history_stop), slice(history_stop, forecast_stop)


def compute_predictor_losses(head, logits, targets, config):
    """Compute compatible full loss and the V6 history/forecast objectives."""
    full_loss, full_s1, full_s2 = head.compute_loss(
        logits[0], logits[1], targets[0], targets[1]
    )
    history_slice, forecast_slice = objective_token_slices(
        targets[0].shape[1],
        config['lookback_window'],
        config['predict_window'],
    )
    history_loss, history_s1, history_s2 = head.compute_loss(
        logits[0][:, history_slice], logits[1][:, history_slice],
        targets[0][:, history_slice], targets[1][:, history_slice],
    )
    forecast_loss, forecast_s1, forecast_s2 = head.compute_loss(
        logits[0][:, forecast_slice], logits[1][:, forecast_slice],
        targets[0][:, forecast_slice], targets[1][:, forecast_slice],
    )
    forecast_weights = torch.as_tensor(
        config.get(
            'forecast_horizon_weights',
            (1.0,) * (forecast_slice.stop - forecast_slice.start),
        ),
        device=logits[0].device,
        dtype=logits[0].dtype,
    )
    if forecast_weights.numel() != forecast_slice.stop - forecast_slice.start:
        raise ValueError('forecast_horizon_weights length does not match predict_window')
    forecast_weights = forecast_weights / forecast_weights.sum()
    if torch.all(forecast_weights == forecast_weights[0]):
        weighted_forecast_loss = forecast_loss
        weighted_forecast_s1 = forecast_s1
        weighted_forecast_s2 = forecast_s2
    else:
        weighted_s1 = F.cross_entropy(
            logits[0][:, forecast_slice].transpose(1, 2),
            targets[0][:, forecast_slice],
            reduction='none',
        ).mean(0)
        weighted_s2 = F.cross_entropy(
            logits[1][:, forecast_slice].transpose(1, 2),
            targets[1][:, forecast_slice],
            reduction='none',
        ).mean(0)
        weighted_forecast_s1 = torch.sum(weighted_s1 * forecast_weights)
        weighted_forecast_s2 = torch.sum(weighted_s2 * forecast_weights)
        weighted_forecast_loss = (weighted_forecast_s1 + weighted_forecast_s2) / 2
    if config.get('predictor_loss_mode', 'full_sequence') == 'forecast':
        history_weight = float(config.get('history_loss_weight', 0.0))
        objective = weighted_forecast_loss + history_weight * history_loss
        objective_s1 = weighted_forecast_s1 + history_weight * history_s1
        objective_s2 = weighted_forecast_s2 + history_weight * history_s2
    else:
        objective, objective_s1, objective_s2 = full_loss, full_s1, full_s2
    return {
        'objective': objective,
        'objective_s1': objective_s1,
        'objective_s2': objective_s2,
        'full_sequence': full_loss,
        'history': history_loss,
        'forecast': forecast_loss,
        'weighted_forecast': weighted_forecast_loss,
    }


def should_run_large_validation(segment, total_segments, interval):
    """Run the audit set at the first, periodic, and final milestones."""
    segment = int(segment)
    total_segments = int(total_segments)
    interval = int(interval)
    if interval <= 0:
        raise ValueError('Large validation interval must be positive')
    return segment == 1 or segment % interval == 0 or segment == total_segments


def best_selection_value(metric, quick_metrics, large_metrics=None):
    """Return the configured checkpoint-selection loss, or None if not evaluated."""
    values = {
        'objective': quick_metrics['objective_loss'],
        'full_sequence': quick_metrics['full_sequence_loss'],
        # Horizon-weighted forecast (the training objective's forecast term),
        # not the unweighted window mean stored as forecast_loss.
        'forecast': quick_metrics.get(
            'weighted_forecast_loss', quick_metrics['forecast_loss']
        ),
        'history': quick_metrics['history_loss'],
        'validation_large_objective': (
            large_metrics['objective_loss'] if large_metrics is not None else None
        ),
        'beta_v21_score': (large_metrics or quick_metrics).get('beta_v21_score'),
        # Lower is better. Not beta_v21_score: that composite is 50% forecast
        # and its ranking denominator was calibrated on sparse shuffled pairs
        # (~0.37 pairs/batch). Dense same-day pairs change the ranking scale,
        # so the score is not a ranking metric.
        'ranking': (large_metrics or quick_metrics).get('ranking_loss'),
        # Higher is better. Same pair rule as ranking_loss, but ties in score
        # count as wrong. Not the training loss.
        'pairwise_accuracy': (large_metrics or quick_metrics).get('pairwise_accuracy'),
    }
    if metric not in values:
        raise ValueError(f'Unsupported best selection metric: {metric}')
    return values[metric]



HIGHER_IS_BETTER_SELECTION_METRICS = frozenset({'pairwise_accuracy'})


def selection_metric_higher_is_better(metric):
    return str(metric or '') in HIGHER_IS_BETTER_SELECTION_METRICS


def selection_is_improvement(metric, value, best):
    """True when value beats best. pairwise_accuracy is higher-better."""
    if value is None:
        return False
    value = float(value)
    if not math.isfinite(value):
        return False
    if selection_metric_higher_is_better(metric):
        return value > float(best)
    return value < float(best)


def resolve_kept_best_loss(
    saved_metric_name, saved_loss, current_metric, calibration_loss,
):
    """Do not treat a foreign selection_loss as the new metric's threshold.

    weighted_forecast_loss (~2.3) and same-day ranking loss (~0.7) are not
    the same number. When the checkpoint objective switches to ranking, the
    bar is the parent checkpoint's calibration ranking_loss (lower better),
    or +inf if that calibration was not run.

    pairwise_accuracy is higher-better. A saved ranking_loss or forecast
    loss is not an accuracy. Switching into it uses the parent calibration
    pairwise accuracy, or -inf when that calibration was not run, so the
    first evaluated segment can become best. A saved pairwise_accuracy is
    kept. Other metrics keep the saved threshold.
    """
    if current_metric == 'pairwise_accuracy' and str(saved_metric_name or '') != 'pairwise_accuracy':
        if calibration_loss is not None and math.isfinite(float(calibration_loss)):
            return float(calibration_loss)
        return float('-inf')
    if current_metric == 'ranking' and str(saved_metric_name or '') != 'ranking':
        if calibration_loss is not None and math.isfinite(float(calibration_loss)):
            return float(calibration_loss)
        return float('inf')
    return float(saved_loss)


def resolve_amp_dtype(config, device):
    """Return the predictor autocast dtype, or None when AMP is disabled.

    CUDA: the configured dtype (fp16 + GradScaler, or bf16).  XLA/TPU: always
    bfloat16 autocast over fp32 master weights (fp16 autocast is not a TPU
    mixed-precision mode and needs no GradScaler).
    """
    if not bool(config.get('use_amp', False)):
        return None
    if device.type == 'xla':
        return torch.bfloat16
    if device.type != 'cuda':
        return None
    dtype_name = str(config.get('amp_dtype', 'float16')).strip().lower()
    if dtype_name in {'float16', 'fp16'}:
        return torch.float16
    if dtype_name in {'bfloat16', 'bf16'}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA device does not support bfloat16 AMP')
        return torch.bfloat16
    raise ValueError(f'Unsupported AMP dtype: {dtype_name}')


def move_auxiliary_labels(labels, device):
    if labels is None:
        return None
    return {
        key: value.to(device, non_blocking=True)
        for key, value in labels.items()
    }


def format_validation_value(value):
    """Format validation scalars with 8 decimals (None stays 'None')."""
    return 'None' if value is None else f'{float(value):.8f}'


def beta_v21_validation_score(metrics, config):
    raw = str(config.get('beta_v21_validation_denominators', '')).strip()
    if not raw:
        return None
    path, _history, returns, barrier, ranking = (
        float(value.strip()) for value in raw.split(',')
    )
    safe_path = max(path, 1e-6)
    safe_returns = max(returns, 1e-6)
    safe_barrier = max(barrier, 1e-6)
    safe_ranking = max(ranking, 1e-6)
    return (
        0.50 * metrics['weighted_forecast_loss'] / safe_path
        + 0.20 * metrics['return_loss'] / safe_returns
        + 0.20 * metrics['barrier_loss'] / safe_barrier
        + 0.10 * metrics['ranking_loss'] / safe_ranking
    )



DEFAULT_BETA_V21_SCORE_FEEDING_MODE = 'shuffled_no_segment_date_sort'


def resolve_beta_v21_score_feeding_mode(config):
    raw = str(config.get('beta_v21_score_feeding_mode', '') or '').strip()
    return raw or DEFAULT_BETA_V21_SCORE_FEEDING_MODE


def should_reuse_saved_beta_v21_denominators(saved, feeding_mode, force_recalibrate):
    """Reuse dens.json only when feeding paradigm matches and force is off.

    Chronological within-segment date-sort dens inflate ranking_loss; after
    removing that sort, same-day pairs are sparse and old dens make
    beta_v21_score look ~0.07–0.08 better than reality. When a same-day
    ranking batch path lands later, bump feeding_mode so dens recalibrate.
    """
    if force_recalibrate:
        return False
    if not isinstance(saved, dict):
        return False
    csv = str(saved.get('csv') or '').strip()
    if not csv:
        return False
    return str(saved.get('feeding_mode') or '').strip() == feeding_mode


def prepare_model_for_validation(model, device, optimizer=None):
    """Clear grads and release training scratch before validation.

    After a training segment the optimizer (AdamW) fp32 state is live and the
    previous step's graph may still be held. Combined with AR consistency decode
    this was exhausting TPU/host memory (SIGKILL with no [VAL] progress). On
    CUDA dual-T4 the same prep frees AMP/allocator fragmentation before the
    full ~123k-sample validation pass.
    """
    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
    else:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad = None
    if device.type == 'xla' and xm is not None:
        xm.mark_step()
    elif device.type == 'cuda' and torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def _release_xla_validation_scratch(device):
    """Best-effort host/device scratch release between scalar val and AR pass."""
    import gc
    if device.type == 'xla' and xm is not None:
        gc.collect()
        xm.mark_step()
        return
    if device.type == 'cuda' and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def consistency_ar_batch_cap(config, device):
    """Per-step AR microbatch size. Cap on XLA/CUDA; elsewhere no default cap."""
    configured = int(config.get('beta_v21_consistency_ar_batch', 0) or 0)
    if configured <= 0:
        configured = int(
            os.getenv('KRONOS_BETA_V21_CONSISTENCY_AR_BATCH', '0') or '0'
        )
    if configured > 0:
        return configured
    if device.type in {'xla', 'cuda'}:
        # Dual-T4 AR decode of 2048 consistency samples without a microbatch
        # can OOM the 15GB cards after a full teacher-forcing val.
        return 8
    return 0  # 0 => take the full remaining slice of each loader batch


def run_return_path_consistency(
    model, tokenizer, loader, device, config, amp_dtype, rank,
    consistency_limit, distributed_world_size, rank0_only_validation,
):
    """Second-pass return-path consistency (AR decode), after scalar validation.

    Teacher-forcing scalar metrics (including beta_v21_score) are already final
    when this runs, so a later AR OOM cannot erase the score. On XLA the AR
    microbatch is capped and ``auto_regressive_inference`` mark_steps each
    decode step.
    """
    consistency_samples = int(config.get('beta_v21_consistency_samples', 0))
    empty = (torch.empty(0, 4), torch.empty(0, 4), torch.empty(0, 4))
    if consistency_samples <= 0 or consistency_limit <= 0:
        return consistency_statistics(*empty)

    core_model = model.module if isinstance(model, DDP) else model
    ar_cap = consistency_ar_batch_cap(config, device)
    if rank == 0:
        print(
            "[VAL] Consistency pass starting: "
            f"target_samples={consistency_samples} "
            f"(per_rank_limit={consistency_limit}, "
            f"ar_batch_cap={ar_cap if ar_cap > 0 else 'full'})",
            flush=True,
        )

    _release_xla_validation_scratch(device)

    consistency_auxiliary = []
    consistency_generated = []
    consistency_actual = []
    consistency_seen = 0
    lookback = int(config['lookback_window'])
    predict_window = int(config['predict_window'])
    float_tokenizer = Float32Tokenizer(tokenizer)

    with torch.no_grad():
        for batch in loader:
            if consistency_seen >= consistency_limit:
                break
            batch_x, batch_x_stamp = batch[0], batch[1]
            auxiliary_labels = batch[-1]
            batch_sector = (
                batch[2]
                if len(batch) > 2 and config.get('use_sector_features', True)
                else None
            )
            batch_size_bucket = (
                batch[3]
                if len(batch) > 3 and config.get('use_size_features', True)
                else None
            )
            batch_size_percentile = (
                batch[4]
                if len(batch) > 4 and config.get('use_size_percentile', False)
                else None
            )
            if config.get('disable_condition_inputs', False):
                batch_sector = batch_size_bucket = batch_size_percentile = None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            if batch_sector is not None:
                batch_sector = batch_sector.to(device, non_blocking=True)
            if batch_size_bucket is not None:
                batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
            if batch_size_percentile is not None:
                batch_size_percentile = batch_size_percentile.to(
                    device, non_blocking=True
                )
            auxiliary_labels = move_auxiliary_labels(auxiliary_labels, device)
            batch_samples = int(batch_x.shape[0])
            offset = 0
            while offset < batch_samples and consistency_seen < consistency_limit:
                take = min(
                    batch_samples - offset,
                    consistency_limit - consistency_seen,
                    ar_cap if ar_cap > 0 else batch_samples,
                )
                sl = slice(offset, offset + take)
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x[sl], half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                sector_sl = None if batch_sector is None else batch_sector[sl]
                bucket_sl = (
                    None if batch_size_bucket is None else batch_size_bucket[sl]
                )
                percentile_sl = (
                    None if batch_size_percentile is None
                    else batch_size_percentile[sl]
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float16,
                    enabled=amp_dtype is not None,
                ):
                    model_output = model(
                        token_in[0], token_in[1], batch_x_stamp[sl, :-1, :],
                        sector_id=sector_sl, size_bucket=bucket_sl,
                        size_percentile=percentile_sl,
                        use_teacher_forcing=True, s1_targets=token_out[0],
                        return_auxiliary=True,
                        asof_index=int(config['lookback_window']) - 1,
                    )
                    _logits, auxiliary_predictions = model_output
                    generated = auto_regressive_inference(
                        float_tokenizer,
                        core_model,
                        batch_x[sl, :lookback],
                        batch_x_stamp[sl, :lookback],
                        batch_x_stamp[
                            sl, lookback:lookback + predict_window
                        ],
                        int(config.get('max_context', 512)),
                        predict_window,
                        clip=float(config.get('clip', 5.0)),
                        T=1.0,
                        top_k=0,
                        top_p=1.0,
                        sample_count=int(
                            config.get('beta_v21_consistency_sample_count', 1)
                        ),
                        verbose=False,
                        sample_logits=False,
                        sector_id=sector_sl,
                        size_bucket=bucket_sl,
                        size_percentile=percentile_sl,
                    )
                auxiliary_predictions = to_float32(auxiliary_predictions)
                generated_path = torch.as_tensor(
                    generated[:, -predict_window:, :], device=device
                )
                generated_returns = generated_return_targets(
                    generated_path,
                    auxiliary_labels['feature_means'][sl],
                    auxiliary_labels['feature_stds'][sl],
                    auxiliary_labels['return_scales'][sl],
                )
                consistency_auxiliary.append(
                    auxiliary_predictions['return'].detach().cpu()
                )
                consistency_generated.append(generated_returns.detach().cpu())
                consistency_actual.append(
                    auxiliary_labels['return_targets'][sl].detach().cpu()
                )
                consistency_seen += take
                offset += take
                if device.type == 'xla' and xm is not None:
                    xm.mark_step()

    if rank == 0:
        print(
            f"[VAL] Consistency pass finished: local_samples={consistency_seen}",
            flush=True,
        )

    local_consistency = (
        torch.cat(consistency_auxiliary),
        torch.cat(consistency_generated),
        torch.cat(consistency_actual),
    ) if consistency_auxiliary else empty
    gathered = [local_consistency]
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_consistency)
    if rank0_only_validation:
        combined = list(local_consistency)
    elif (
        device.type == 'xla' and xm is not None
        and distributed_world_size > 1
    ):
        combined = [
            xm.all_gather(value.to(device), dim=0).cpu()[:consistency_samples]
            for value in local_consistency
        ]
    else:
        combined = [
            torch.cat([item[index] for item in gathered], dim=0)[
                :consistency_samples
            ]
            for index in range(3)
        ]
    return consistency_statistics(*combined)


class Float32Tokenizer:
    """Run a KronosTokenizer's encode/decode outside any active autocast.

    The training loop tokenizes in float32 (outside the predictor autocast).
    The Beta v2.1 return-path consistency check calls
    ``auto_regressive_inference`` inside the predictor autocast, which would
    otherwise encode the context and decode the generated path in bf16 and
    yield different tokens from the float32 teacher-forced path.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    @staticmethod
    def _device_type(value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        return value.device.type

    def encode(self, x, half=False):
        with torch.autocast(device_type=self._device_type(x), enabled=False):
            return self._tokenizer.encode(x, half=half)

    def decode(self, x, half=False):
        with torch.autocast(device_type=self._device_type(x), enabled=False):
            return self._tokenizer.decode(x, half=half)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def _pad_gather_1d(local_cpu, device):
    """All-gather a 1d CPU tensor across torch.distributed ranks."""
    local = local_cpu.reshape(-1)
    world = dist.get_world_size()
    reduce_device = device if getattr(device, 'type', 'cpu') != 'mps' else 'cpu'
    count = torch.tensor([local.numel()], dtype=torch.long, device=reduce_device)
    counts = [torch.zeros(1, dtype=torch.long, device=reduce_device) for _ in range(world)]
    dist.all_gather(counts, count)
    sizes = [int(item.item()) for item in counts]
    width = max(sizes) if sizes else 0
    if width == 0:
        return local
    padded = torch.zeros(width, dtype=local.dtype, device=reduce_device)
    if local.numel():
        padded[:local.numel()] = local.to(reduce_device)
    gathered = [torch.zeros(width, dtype=local.dtype, device=reduce_device) for _ in range(world)]
    dist.all_gather(gathered, padded)
    parts = []
    for piece, size in zip(gathered, sizes):
        if size:
            parts.append(piece[:size].cpu())
    if not parts:
        return torch.empty(0, dtype=local.dtype)
    return torch.cat(parts)


def _xla_pad_gather_1d(local_cpu, device):
    """All-gather a 1d tensor across XLA workers. Shapes are padded."""
    local = local_cpu.reshape(-1).to(device)
    count = torch.tensor([local.numel()], dtype=torch.long, device=device)
    max_count = int(xm.all_reduce(xm.REDUCE_MAX, count).cpu().item())
    if max_count == 0:
        return local_cpu.reshape(-1)
    padded = torch.zeros(max_count, dtype=local.dtype, device=device)
    if local.numel():
        padded[:local.numel()] = local
    gathered = xm.all_gather(padded, dim=0).cpu()
    sizes = [int(item) for item in xm.all_gather(count, dim=0).cpu().tolist()]
    parts = []
    for index, size in enumerate(sizes):
        if size:
            parts.append(gathered[index * max_count:index * max_count + size])
    if not parts:
        return torch.empty(0, dtype=local_cpu.dtype)
    return torch.cat(parts)


def gather_ranking_column(local_cpu, device):
    """Return the global 1d column. No-op when this process is alone."""
    local_cpu = local_cpu.reshape(-1).cpu()
    if dist.is_available() and dist.is_initialized():
        return _pad_gather_1d(local_cpu, device)
    if (
        getattr(device, 'type', 'cpu') == 'xla'
        and xm is not None
        and get_xla_world_size() > 1
    ):
        return _xla_pad_gather_1d(local_cpu, device)
    return local_cpu


def attach_pairwise_ranking_metrics(result, scores, utilities, date_ids):
    """Write validation pairwise accuracy and mean within-date Spearman."""
    accuracy, pair_count = same_date_pairwise_accuracy(scores, utilities, date_ids)
    rank_ic, rank_ic_dates = mean_within_date_spearman(scores, utilities, date_ids)
    result['pairwise_accuracy'] = accuracy
    result['pairwise_pairs'] = int(pair_count)
    result['rank_ic'] = rank_ic
    result['rank_ic_dates'] = int(rank_ic_dates)
    return result


def evaluate_validation(
    model, tokenizer, loader, device, config, amp_dtype, run_condition_ablation=False,
    period_names=None, rank=0,
):
    """Evaluate a fixed validation set and report its named date periods."""
    model.eval()
    core_model = model.module if isinstance(model, DDP) else model
    sums = {
        'objective_loss': 0.0,
        'full_sequence_loss': 0.0,
        'history_loss': 0.0,
        'forecast_loss': 0.0,
        # Horizon-weighted forecast is the forecast-mode training term and the
        # best-checkpoint metric. compute_predictor_losses always returns it,
        # including when auxiliary heads are off (KRONOS_USE_BETA_V21_AUXILIARY=0).
        # Keeping it out of the accumulator keys KeyError'd the first large
        # validation on the dual-T4 phase-1 run.
        'weighted_forecast_loss': 0.0,
        'condition_none_forecast_loss': 0.0,
        'condition_shuffled_forecast_loss': 0.0,
    }
    use_beta_v21 = bool(config.get('use_beta_v21_auxiliary', False))
    if use_beta_v21:
        sums.update({
            'return_loss': 0.0,
            'return_huber_loss': 0.0,
            'return_bias_loss': 0.0,
            'barrier_loss': 0.0,
            'ranking_loss': 0.0,
        })
    single_process_xla = (
        device.type == 'xla'
        and os.getenv('KRONOS_XLA_SINGLE_PROCESS', '0').strip().lower()
        in {'1', 'true', 'yes', 'on'}
    )
    if single_process_xla:
        distributed_world_size = 1
    elif dist.is_available() and dist.is_initialized():
        distributed_world_size = dist.get_world_size()
    elif device.type == 'xla' and xm is not None:
        distributed_world_size = get_xla_world_size()
    else:
        distributed_world_size = 1
    consistency_limit = math.ceil(
        int(config.get('beta_v21_consistency_samples', 0))
        / distributed_world_size
    )
    rank0_only_validation = bool(
        config.get('validation_rank0_only', False)
        and device.type == 'xla'
        and distributed_world_size > 1
    )
    if rank0_only_validation and rank != 0:
        # Keep all workers in the collective reductions below, but avoid
        # materializing validation batches/model outputs on seven workers.
        loader = []
    if rank0_only_validation and rank == 0:
        consistency_limit = int(config.get('beta_v21_consistency_samples', 0))
    # A full validation can contain 100k+ samples (~123k on A-share holdout).
    # Keeping auxiliary predictions/labels for every batch and then
    # all_gather_object-ing them duplicates ~60k+ sample tensors on each DDP
    # rank and has SIGKILL'd dual-T4 Kaggle hosts right after "[VAL] Processed
    # 1935/1935 batches..." (calibration).  Scalar auxiliary losses are already
    # accumulated below, so sample-level collection is opt-in only
    # (KRONOS_COLLECT_VALIDATION_AUXILIARY=1 / collect_validation_auxiliary).
    collect_validation_auxiliary = (
        use_beta_v21
        and bool(config.get('collect_validation_auxiliary', False))
    )
    validation_auxiliary = []
    log_pairwise_ranking_metrics = bool(
        config.get('log_pairwise_ranking_metrics', False)
    ) or str(config.get('best_selection_metric', '')) == 'pairwise_accuracy'
    ranking_score_parts = []
    ranking_utility_parts = []
    ranking_date_parts = []
    period_names = dict(period_names or {})
    # fp32 losses -> float64 sample-weighted sums (see validation_precision).
    loss_accumulator = ValidationLossAccumulator(sums, device)
    period_accumulators = {
        int(code): ValidationLossAccumulator(sums, device)
        for code in period_names
    }
    period_samples = {int(code): 0 for code in period_names}
    batches = 0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_x_stamp = batch[0], batch[1]
            auxiliary_labels = batch[-1] if use_beta_v21 else None
            batch_sector = (
                batch[2]
                if len(batch) > 2 and config.get('use_sector_features', True)
                else None
            )
            batch_size_bucket = (
                batch[3]
                if len(batch) > 3 and config.get('use_size_features', True)
                else None
            )
            batch_size_percentile = (
                batch[4]
                if len(batch) > 4 and config.get('use_size_percentile', False)
                else None
            )
            batch_period = batch[5] if len(batch) > 5 and period_names else None
            if config.get('disable_condition_inputs', False):
                batch_sector = batch_size_bucket = batch_size_percentile = None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            if batch_sector is not None:
                batch_sector = batch_sector.to(device, non_blocking=True)
            if batch_size_bucket is not None:
                batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
            if batch_size_percentile is not None:
                batch_size_percentile = batch_size_percentile.to(
                    device, non_blocking=True
                )
            auxiliary_labels = move_auxiliary_labels(auxiliary_labels, device)

            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float16,
                enabled=amp_dtype is not None,
            ):
                model_output = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    sector_id=batch_sector, size_bucket=batch_size_bucket,
                    size_percentile=batch_size_percentile,
                    use_teacher_forcing=True, s1_targets=token_out[0],
                    return_auxiliary=use_beta_v21,
                    asof_index=int(config['lookback_window']) - 1,
                )
            if use_beta_v21:
                logits, auxiliary_predictions = model_output
            else:
                logits = model_output
            # Validation losses are computed outside autocast from fp32
            # logits/aux outputs so small checkpoint differences stay visible.
            logits = to_float32(logits)
            losses = compute_predictor_losses(
                core_model.head, logits, token_out, config
            )
            if use_beta_v21:
                auxiliary_predictions = to_float32(auxiliary_predictions)
                auxiliary_losses = compute_auxiliary_losses(
                    auxiliary_predictions['return'],
                    auxiliary_predictions['barrier'],
                    to_float32(auxiliary_labels),
                )
                if log_pairwise_ranking_metrics:
                    # Same fp32 scores the ranking loss just used. Kept as
                    # three 1d columns (~1.5MB for the full holdout), not the
                    # sample-level aux gather that SIGKILL'd dual-T4.
                    ranking_scores = expected_utility_score(
                        auxiliary_predictions['return'],
                        auxiliary_predictions['barrier'],
                        auxiliary_labels['return_scales'],
                    )
                    ranking_score_parts.append(ranking_scores.detach().float().cpu())
                    ranking_utility_parts.append(
                        auxiliary_labels['utility'].detach().float().cpu()
                    )
                    ranking_date_parts.append(
                        auxiliary_labels['date_id'].detach().to(dtype=torch.int64).cpu()
                    )
                if collect_validation_auxiliary:
                    validation_auxiliary.append((
                        auxiliary_predictions['return'].detach().float().cpu(),
                        auxiliary_predictions['barrier'].detach().float().cpu(),
                        {
                            key: auxiliary_labels[key].detach().cpu()
                            for key in (
                                'return_targets', 'return_scales',
                                'barrier_target', 'barrier_valid',
                                'utility', 'date_id',
                            )
                        },
                    ))
            batch_samples = int(batch_x.shape[0])
            samples += batch_samples
            batch_values = {
                'objective_loss': losses['objective'],
                'full_sequence_loss': losses['full_sequence'],
                'history_loss': losses['history'],
                'forecast_loss': losses['forecast'],
                'weighted_forecast_loss': losses['weighted_forecast'],
            }
            if use_beta_v21:
                batch_values.update({
                    'return_loss': auxiliary_losses['return'],
                    'return_huber_loss': auxiliary_losses['return_huber'],
                    'return_bias_loss': auxiliary_losses['return_bias'],
                    'barrier_loss': auxiliary_losses['barrier'],
                    'ranking_loss': auxiliary_losses['ranking'],
                })
            batches += 1
            if rank == 0 and (batches % 50 == 0 or batches == len(loader)):
                print(
                    f"[VAL] Processed {batches}/{len(loader)} batches...",
                    flush=True,
                )

            if run_condition_ablation:
                shuffled_sector = (
                    torch.roll(batch_sector, shifts=1, dims=0)
                    if batch_sector is not None and batch_sector.shape[0] > 1
                    else batch_sector
                )
                shuffled_bucket = (
                    torch.roll(batch_size_bucket, shifts=1, dims=0)
                    if batch_size_bucket is not None and batch_size_bucket.shape[0] > 1
                    else batch_size_bucket
                )
                shuffled_percentile = (
                    torch.roll(batch_size_percentile, shifts=1, dims=0)
                    if batch_size_percentile is not None
                    and batch_size_percentile.shape[0] > 1
                    else batch_size_percentile
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float16,
                    enabled=amp_dtype is not None,
                ):
                    none_logits = model(
                        token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                        sector_id=None, size_bucket=None, size_percentile=None,
                        use_teacher_forcing=True, s1_targets=token_out[0],
                    )
                    shuffled_logits = model(
                        token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                        sector_id=shuffled_sector,
                        size_bucket=shuffled_bucket,
                        size_percentile=shuffled_percentile,
                        use_teacher_forcing=True, s1_targets=token_out[0],
                    )
                none_logits = to_float32(none_logits)
                shuffled_logits = to_float32(shuffled_logits)
                none_losses = compute_predictor_losses(
                    core_model.head, none_logits, token_out, config
                )
                shuffled_losses = compute_predictor_losses(
                    core_model.head, shuffled_logits, token_out, config
                )
                batch_values['condition_none_forecast_loss'] = none_losses['forecast']
                batch_values['condition_shuffled_forecast_loss'] = (
                    shuffled_losses['forecast']
                )
            # One fp32 vector per batch: device float64 sums on CUDA/CPU, a
            # single host copy on XLA/MPS (was one .item() per loss term).
            loss_accumulator.add(batch_values, batch_samples)

            if batch_period is not None:
                batch_period = batch_period.to(device)
                for code in period_names:
                    mask = batch_period == int(code)
                    count = int(mask.sum().item())
                    if not count:
                        continue
                    period_losses = compute_predictor_losses(
                        core_model.head,
                        [value[mask] for value in logits],
                        [value[mask] for value in token_out],
                        config,
                    )
                    values = {
                        'objective_loss': period_losses['objective'],
                        'full_sequence_loss': period_losses['full_sequence'],
                        'history_loss': period_losses['history'],
                        'forecast_loss': period_losses['forecast'],
                        'weighted_forecast_loss': period_losses['weighted_forecast'],
                    }
                    if run_condition_ablation:
                        period_none = compute_predictor_losses(
                            core_model.head,
                            [value[mask] for value in none_logits],
                            [value[mask] for value in token_out],
                            config,
                        )
                        period_shuffled = compute_predictor_losses(
                            core_model.head,
                            [value[mask] for value in shuffled_logits],
                            [value[mask] for value in token_out],
                            config,
                        )
                        values['condition_none_forecast_loss'] = period_none['forecast']
                        values['condition_shuffled_forecast_loss'] = (
                            period_shuffled['forecast']
                        )
                    period_accumulators[int(code)].add(values, count)
                    period_samples[int(code)] += count

    ordered_keys = tuple(sums)
    # Reduce float64 sums (exact int64 fixed-point on XLA) across all replicas
    # and divide on the host: global mean = sum(loss*n) / sum(n).
    global_totals, global_counts = all_reduce_validation_sums(
        loss_accumulator.totals(), [batches, samples], device,
        xm=xm, xla_world_size=distributed_world_size,
    )
    result = weighted_means(ordered_keys, global_totals, global_counts[1])
    result['batches'] = int(global_counts[0])
    result['samples'] = int(global_counts[1])
    if use_beta_v21:
        local_validation = None
        if collect_validation_auxiliary and validation_auxiliary:
            local_validation = (
                torch.cat([item[0] for item in validation_auxiliary]),
                torch.cat([item[1] for item in validation_auxiliary]),
                {
                    key: torch.cat([item[2][key] for item in validation_auxiliary])
                    for key in validation_auxiliary[0][2]
                },
            )
        if collect_validation_auxiliary:
            gathered_validation = [local_validation]
            if dist.is_available() and dist.is_initialized():
                gathered_validation = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_validation, local_validation)
            nonempty_validation = [
                item for item in gathered_validation if item is not None
            ]
            global_returns = (
                torch.cat([item[0] for item in nonempty_validation])
                if nonempty_validation else None
            )
            global_barriers = (
                torch.cat([item[1] for item in nonempty_validation])
                if nonempty_validation else None
            )
            global_labels = (
                {
                    key: torch.cat([
                        item[2][key] for item in nonempty_validation
                    ])
                    for key in nonempty_validation[0][2]
                }
                if nonempty_validation else None
            )
            if global_returns is not None:
                global_auxiliary_losses = compute_auxiliary_losses(
                    global_returns, global_barriers, global_labels
                )
                result.update({
                    'return_loss': float(global_auxiliary_losses['return'].item()),
                    'return_huber_loss': float(
                        global_auxiliary_losses['return_huber'].item()
                    ),
                    'return_bias_loss': float(
                        global_auxiliary_losses['return_bias'].item()
                    ),
                    'barrier_loss': float(global_auxiliary_losses['barrier'].item()),
                    'ranking_loss': float(global_auxiliary_losses['ranking'].item()),
                })
        result['beta_v21_score'] = beta_v21_validation_score(result, config)
        if log_pairwise_ranking_metrics:
            if ranking_score_parts:
                local_scores = torch.cat(ranking_score_parts)
                local_utilities = torch.cat(ranking_utility_parts)
                local_dates = torch.cat(ranking_date_parts)
            else:
                local_scores = torch.empty(0)
                local_utilities = torch.empty(0)
                local_dates = torch.empty(0, dtype=torch.int64)
            attach_pairwise_ranking_metrics(
                result,
                gather_ranking_column(local_scores, device),
                gather_ranking_column(local_utilities, device),
                gather_ranking_column(local_dates, device),
            )
            if rank == 0:
                print(
                    "Validation Pairwise Accuracy/RankIC: "
                    f"{format_validation_value(result.get('pairwise_accuracy'))} / "
                    f"{format_validation_value(result.get('rank_ic'))} "
                    f"(pairs={int(result.get('pairwise_pairs') or 0)}, "
                    f"rank_ic_dates={int(result.get('rank_ic_dates') or 0)})",
                    flush=True,
                )
        if rank == 0:
            print(
                "[VAL] Scalar validation complete: "
                f"samples={result['samples']}, "
                f"beta_v21_score={format_validation_value(result.get('beta_v21_score'))}",
                flush=True,
            )
        # Decoupled second pass: AR consistency cannot kill beta_v21_score.
        result['return_path_consistency'] = run_return_path_consistency(
            model, tokenizer, loader, device, config, amp_dtype, rank,
            consistency_limit=consistency_limit,
            distributed_world_size=distributed_world_size,
            rank0_only_validation=rank0_only_validation,
        )
    if run_condition_ablation:
        result['condition_full_minus_none_forecast_loss'] = (
            result['forecast_loss'] - result['condition_none_forecast_loss']
        )
        result['condition_full_minus_shuffled_forecast_loss'] = (
            result['forecast_loss'] - result['condition_shuffled_forecast_loss']
        )
    else:
        for key in (
            'condition_none_forecast_loss',
            'condition_shuffled_forecast_loss',
        ):
            result.pop(key)
    result['periods'] = {}
    for code, name in sorted(period_names.items()):
        period_totals, period_counts = all_reduce_validation_sums(
            period_accumulators[int(code)].totals(),
            [period_samples[int(code)]], device,
            xm=xm, xla_world_size=distributed_world_size,
        )
        metrics = weighted_means(ordered_keys, period_totals, period_counts[0])
        metrics['samples'] = int(period_counts[0])
        if run_condition_ablation:
            metrics['condition_full_minus_none_forecast_loss'] = (
                metrics['forecast_loss']
                - metrics['condition_none_forecast_loss']
            )
            metrics['condition_full_minus_shuffled_forecast_loss'] = (
                metrics['forecast_loss']
                - metrics['condition_shuffled_forecast_loss']
            )
        else:
            metrics.pop('condition_none_forecast_loss')
            metrics.pop('condition_shuffled_forecast_loss')
        result['periods'][name] = metrics
    return result


def reset_cuda_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory(device):
    if device.type != 'cuda':
        return {}
    return {
        'cuda_peak_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        'cuda_peak_reserved_gb': torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


def is_cuda_out_of_memory(exc):
    return torch.cuda.is_available() and 'out of memory' in str(exc).lower()


def write_oom_marker(config, exc):
    save_dir = os.path.join(
        config['save_path'], config['predictor_save_folder_name']
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'oom.json'), 'w') as handle:
        json.dump({
            'error': str(exc),
            'batch_size': config['batch_size'],
            'predictor_loss_mode': config.get('predictor_loss_mode'),
        }, handle, indent=2)


EVAL_ONLY_TRUE_VALUES = {'1', 'true', 'yes', 'on'}


def eval_only_requested():
    """KRONOS_EVAL_ONLY=1 validates the untouched parent once and exits.

    Default off.  The check happens after Beta v2.1 denominator calibration
    and before the optimizer/scheduler exist, so zero optimizer steps run.
    """
    return os.getenv('KRONOS_EVAL_ONLY', '0').strip().lower() in EVAL_ONLY_TRUE_VALUES


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run_eval_only_validation(
    model, tokenizer, device, config, save_dir, rank, loader, valid_dataset,
    amp_dtype, calibration_metrics=None, start_time=None,
):
    """Run one full validation of the pretrained model; no training."""
    started = time.time()
    if rank == 0:
        print(
            "EVAL-ONLY: Running Base Model Validation on "
            f"{len(valid_dataset):,} samples with zero optimizer steps."
        )
    metrics = evaluate_validation(
        model, tokenizer, loader, device, config, amp_dtype,
        run_condition_ablation=False,
        period_names=getattr(valid_dataset, 'validation_period_names', {}),
        rank=rank,
    )
    elapsed = time.time() - started
    denominators_csv = str(
        config.get('beta_v21_validation_denominators', '') or ''
    ).strip()
    denominator_source = (
        'auto_calibrated_same_run_parent_checkpoint'
        if calibration_metrics is not None
        else ('fixed_env_or_saved' if denominators_csv else None)
    )
    use_beta_v21 = bool(config.get('use_beta_v21_auxiliary', False))
    document = {
        'mode': 'eval_only_base_model',
        'optimizer_steps': 0,
        'pretrained_predictor_path': config.get('pretrained_predictor_path'),
        'validation_seconds': elapsed,
        'total_seconds': None if start_time is None else time.time() - start_time,
        'beta_v21_validation_denominators': denominators_csv or None,
        'beta_v21_denominator_source': denominator_source,
        'beta_v21_score_definition': (
            '0.50*weighted_forecast/path + 0.20*return/return + '
            '0.20*barrier/barrier + 0.10*ranking/ranking '
            '(denominators = path,history,return,barrier,ranking)'
        ),
        'metrics': _json_safe(metrics),
        'calibration_metrics': _json_safe(calibration_metrics),
    }
    if rank == 0:
        print("\n--- Base Model Validation Summary (eval-only, 0 optimizer steps) ---")
        print(f"Base Model Validation Loss: {metrics['objective_loss']:.8f}")
        print(
            "Base Model Validation Forecast/History/Full: "
            f"{metrics['forecast_loss']:.8f} / "
            f"{metrics['history_loss']:.8f} / "
            f"{metrics['full_sequence_loss']:.8f}"
        )
        if use_beta_v21:
            print(
                "Base Model Validation v2.1 Score/Return/Bias/Barrier/Rank: "
                f"{format_validation_value(metrics.get('beta_v21_score'))} / "
                f"{metrics['return_loss']:.8f} / "
                f"{metrics['return_bias_loss']:.8f} / "
                f"{metrics['barrier_loss']:.8f} / "
                f"{metrics['ranking_loss']:.8f}"
            )
            print(
                "Base Model Validation Weighted Forecast (score path term): "
                f"{metrics['weighted_forecast_loss']:.8f}"
            )
            print(
                "Base Model beta_v21_score denominators "
                f"(path,history,return,barrier,ranking; {denominator_source}): "
                f"{denominators_csv}"
            )
            if 'return_path_consistency' in metrics:
                print(
                    "Base Model Validation Return-Path Consistency JSON: "
                    + json.dumps(
                        _json_safe(metrics['return_path_consistency']),
                        sort_keys=True,
                    )
                )
        for period, period_metrics in metrics.get('periods', {}).items():
            print(
                f"Base Model Validation {period} Objective/Forecast/History/Full: "
                f"{period_metrics['objective_loss']:.8f} / "
                f"{period_metrics['forecast_loss']:.8f} / "
                f"{period_metrics['history_loss']:.8f} / "
                f"{period_metrics['full_sequence_loss']:.8f} "
                f"({period_metrics['samples']:,} samples)"
            )
        print(
            f"Base Model Validation samples/batches: {metrics['samples']:,} / "
            f"{metrics['batches']:,}; validation time {elapsed:.1f}s"
        )
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'base_model_validation.json')
        with open(f'{path}.tmp', 'w') as handle:
            json.dump(document, handle, indent=2)
        os.replace(f'{path}.tmp', path)
        append_metric(
            save_dir,
            type='base_model_validation',
            segment=0,
            step=0,
            loss=metrics['objective_loss'],
            forecast_loss=metrics['forecast_loss'],
            history_loss=metrics['history_loss'],
            full_sequence_loss=metrics['full_sequence_loss'],
            beta_v21_score=metrics.get('beta_v21_score'),
            return_loss=metrics.get('return_loss'),
            barrier_loss=metrics.get('barrier_loss'),
            ranking_loss=metrics.get('ranking_loss'),
            pairwise_accuracy=metrics.get('pairwise_accuracy'),
            rank_ic=metrics.get('rank_ic'),
            samples=metrics['samples'],
            batches=metrics['batches'],
        )
        write_progress(
            save_dir,
            status='eval_only_completed',
            current_segment=0,
            optimizer_steps=0,
            base_model_validation=path,
            device=str(device),
        )
        print(f"EVAL-ONLY: wrote {path}; exiting without training.")
    distributed_barrier(device, 'kronos_eval_only_done')
    return document


def master_weight_dtype(model):
    """Storage dtype of the trainable weights (bf16 under XLA_USE_BF16)."""
    dtypes = {str(p.dtype).removeprefix('torch.') for p in model.parameters()}
    dtype = dtypes.pop() if len(dtypes) == 1 else 'mixed'
    if tpu_self_checks is not None and tpu_self_checks.bf16_env_flags():
        return 'bfloat16'
    return dtype


def refuse_low_precision_checkpoint(state, device, path):
    """Never continue TPU training from a bf16-storage (pre-fix) checkpoint."""
    if device.type != 'xla':
        return
    saved = state.get('master_weight_dtype')
    if saved != 'float32' or not state.get('xla_gradient_all_reduce', False):
        raise ValueError(
            f'Refusing to resume TPU training from {path}: it was written by the '
            'pre-fix TPU trainer (XLA_USE_BF16 storage and/or no gradient '
            f'all-reduce; master_weight_dtype={saved!r}, '
            f"xla_gradient_all_reduce={state.get('xla_gradient_all_reduce')!r}). "
            'Start fresh from the fp32 parent in a new output directory.'
        )


def _xla_all_reduce_sum_tensor(value):
    return xm.all_reduce(xm.REDUCE_SUM, value)


def run_xla_startup_checks(model, device, rank, world_size):
    """Fail fast before training: topology, real collectives, fp32 storage.

    Every replica must call this (it issues collectives).
    """
    if tpu_self_checks is None:
        raise ImportError(
            "tpu_self_checks is required on XLA/TPU; include it in the kernel overlay"
        )
    tpu_self_checks.assert_no_bf16_storage_env()
    messages = []
    if world_size > 1:
        runtime_world_size = (
            xr.world_size() if xr is not None and hasattr(xr, 'world_size')
            else get_xla_world_size()
        )
        messages.append(tpu_self_checks.check_world_size(
            runtime_world_size, world_size, context=f'(rank {rank})'
        ))
        timeout = float(os.getenv('KRONOS_COLLECTIVE_CHECK_TIMEOUT_SECONDS', '900'))
        with tpu_self_checks.Watchdog(timeout, 'xla_collective_check'):
            def reduce_scalar(value):
                tensor = torch.tensor([float(value)], dtype=torch.float32).to(device)
                return float(_xla_all_reduce_sum_tensor(tensor).cpu().item())

            def gather_scalar(value):
                tensor = torch.tensor([float(value)], dtype=torch.float32).to(device)
                rows = tpu_self_checks.gather_rows_via_all_reduce(
                    tensor, rank, world_size, device, _xla_all_reduce_sum_tensor
                )
                return rows.cpu().reshape(-1).tolist()

            messages.append(tpu_self_checks.check_collectives(
                rank, world_size, reduce_scalar, gather_scalar
            ))
    parameters = [p for p in model.parameters()]
    probe = tpu_self_checks.parameter_probe(parameters).cpu()
    storage = tpu_self_checks.check_fp32_storage(
        probe,
        [p.dtype for p in parameters],
        allow_bf16_parent=os.getenv('KRONOS_ALLOW_BF16_PARENT', '0').strip().lower()
        in tpu_self_checks.TRUE_VALUES,
    )
    messages.append('fp32_master_weight_check_passed ' + json.dumps(storage, sort_keys=True))
    if rank == 0:
        for message in messages:
            print(message, flush=True)
    return storage


def begin_first_step_sync_check(model, batch_x, device, rank, world_size):
    """Capture pre-reduction state for the first optimizer step (all ranks)."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    local_rows = int(batch_x.shape[0])
    return {
        'parameters': parameters,
        'params_before': tpu_self_checks.parameter_probe(parameters).clone(),
        'grads_before_rows': tpu_self_checks.gather_rows_via_all_reduce(
            tpu_self_checks.gradient_probe(parameters), rank, world_size, device,
            _xla_all_reduce_sum_tensor,
        ),
        'global_rows': _xla_all_reduce_sum_tensor(
            torch.tensor([float(local_rows)], dtype=torch.float32).to(device)
        ),
        'local_rows': local_rows,
    }


def finish_first_step_sync_check(state, model, optimizer, device, rank, world_size):
    """After step 1: identical weights/grads on every rank, fp32 updates applied."""
    timeout = float(os.getenv('KRONOS_FIRST_STEP_CHECK_TIMEOUT_SECONDS', '1800'))
    with tpu_self_checks.Watchdog(timeout, 'grad_sync_check'):
        parameters = state['parameters']
        params_after = tpu_self_checks.parameter_probe(parameters)
        changed = (params_after != state['params_before']).float().mean()
        params_rows = tpu_self_checks.gather_rows_via_all_reduce(
            params_after, rank, world_size, device, _xla_all_reduce_sum_tensor
        )
        grads_after_rows = tpu_self_checks.gather_rows_via_all_reduce(
            tpu_self_checks.gradient_probe(parameters), rank, world_size, device,
            _xla_all_reduce_sum_tensor,
        )
        xm.mark_step()
        result = tpu_self_checks.evaluate_first_step(
            world_size=world_size,
            gathered_params_after=params_rows.cpu().tolist(),
            gathered_grads_before_reduce=state['grads_before_rows'].cpu().tolist(),
            gathered_grads_after_reduce=grads_after_rows.cpu().tolist(),
            changed_fraction=float(changed.cpu().item()),
            global_batch_rows=int(round(float(state['global_rows'].cpu().item()))),
            local_batch_rows=state['local_rows'],
            optimizer_state_info=tpu_self_checks.check_optimizer_state_dtypes(optimizer),
        )
    result['param_dtype'] = master_weight_dtype(model)
    if rank == 0:
        print(tpu_self_checks.format_passed(result), flush=True)
    return result


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    amp_dtype = resolve_amp_dtype(config, device)
    use_amp = amp_dtype is not None
    amp_dtype_name = str(amp_dtype).removeprefix('torch.') if use_amp else 'disabled'
    scale_gradients = amp_dtype == torch.float16
    # GradScaler is a CUDA-only mechanism.  Keeping it out of XLA/CPU runs
    # avoids initializing a CUDA backend just to hold a disabled scaler.
    scaler = torch.amp.GradScaler('cuda') if scale_gradients else None
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")
        print(
            f"Predictor AMP: {amp_dtype_name}"
            f"{' autocast' if use_amp else ''}; gradient scaling: "
            f"{'enabled' if scale_gradients else 'disabled'}; "
            "master weights/optimizer state: float32; "
            "tokenizer encoding and RoPE remain float32."
        )
        print(
            f"Predictor loss mode: {config.get('predictor_loss_mode', 'full_sequence')}; "
            f"history weight: {float(config.get('history_loss_weight', 0.0)):.4f}"
        )
        print(
            "Best checkpoint selection metric: "
            f"{config.get('best_selection_metric', 'objective')}"
        )

    (
        train_loader, val_loader, large_val_loader,
        train_dataset, valid_dataset,
    ) = create_dataloaders(config, rank, world_size)
    if device.type == 'xla':
        if MpDeviceLoader is None:
            raise RuntimeError('TPU requested but torch_xla is not installed')
        train_loader = MpDeviceLoader(train_loader, device)
        if val_loader is not None:
            val_loader = MpDeviceLoader(val_loader, device)
        large_val_loader = MpDeviceLoader(large_val_loader, device)
    validation_full_only = bool(config.get('validation_full_only', False))
    denominator_path = os.path.join(
        save_dir, 'beta_v21_validation_denominators.json'
    )
    calibration_metrics = None
    denominators_recalibrated_this_run = False
    if config.get('use_beta_v21_auxiliary', False):
        feeding_mode = resolve_beta_v21_score_feeding_mode(config)
        force_recalibrate = bool(config.get('beta_v21_force_recalibrate', False))
        if os.path.exists(denominator_path):
            with open(denominator_path) as handle:
                saved_denominators = json.load(handle)
            if should_reuse_saved_beta_v21_denominators(
                saved_denominators, feeding_mode, force_recalibrate
            ):
                if not config.get('beta_v21_validation_denominators'):
                    config['beta_v21_validation_denominators'] = saved_denominators['csv']
            else:
                old_mode = (
                    None if not isinstance(saved_denominators, dict)
                    else saved_denominators.get('feeding_mode')
                )
                if rank == 0:
                    print(
                        'Wiping Beta v2.1 validation denominators for recalibration '
                        f'(saved_feeding_mode={old_mode!r}, '
                        f'current_feeding_mode={feeding_mode!r}, '
                        f'force={force_recalibrate}).',
                        flush=True,
                    )
                    try:
                        os.remove(denominator_path)
                    except FileNotFoundError:
                        pass
                config['beta_v21_validation_denominators'] = ''
                force_recalibrate = True
        if (
            not config.get('beta_v21_validation_denominators')
            and (
                config.get('beta_v21_auto_calibrate', False) or force_recalibrate
            )
        ):
            if rank == 0:
                print(
                    'Calibrating fixed Beta v2.1 validation denominators from '
                    'the untrained auxiliary-head initialization '
                    f'(feeding_mode={feeding_mode}).',
                    flush=True,
                )
            prepare_model_for_validation(model, device, optimizer=None)
            calibration_config = dict(config)
            calibration_config['beta_v21_consistency_samples'] = 0
            calibration_metrics = evaluate_validation(
                model,
                tokenizer,
                large_val_loader,
                device,
                calibration_config,
                amp_dtype,
                run_condition_ablation=False,
                period_names={},
                rank=rank,
            )
            denominator_values = (
                max(float(calibration_metrics['weighted_forecast_loss']), 1e-5),
                max(float(calibration_metrics['history_loss']), 1e-5),
                max(float(calibration_metrics['return_loss']), 1e-5),
                max(float(calibration_metrics['barrier_loss']), 1e-5),
                max(float(calibration_metrics['ranking_loss']), 1e-5),
            )
            denominator_csv = ','.join(
                f'{value:.17g}' for value in denominator_values
            )
            config['beta_v21_validation_denominators'] = denominator_csv
            denominators_recalibrated_this_run = True
            if rank == 0:
                print(
                    'Pre-train calibration baseline: '
                    f"pairwise_accuracy={format_validation_value(calibration_metrics.get('pairwise_accuracy'))} "
                    f"rank_ic={format_validation_value(calibration_metrics.get('rank_ic'))} "
                    f"weighted_forecast_loss={float(calibration_metrics['weighted_forecast_loss']):.8f} "
                    f"ranking_loss={float(calibration_metrics['ranking_loss']):.8f}",
                    flush=True,
                )
                config['_freeze_forecast_baseline'] = float(
                    calibration_metrics['weighted_forecast_loss']
                )
                with open(f'{denominator_path}.tmp', 'w') as handle:
                    json.dump({
                        'source': 'untrained_beta_v21_heads_on_parent_checkpoint',
                        'parent_path': config['pretrained_predictor_path'],
                        'feeding_mode': feeding_mode,
                        'note': (
                            'Dens are tied to feeding_mode. Sparse shuffled batches '
                            '(~0.37 same-day pairs/batch) shrink the ranking denominator '
                            'and make beta_v21_score incomparable once same-day batches '
                            'are dense. Do not select checkpoints on that score; use '
                            'ranking_loss. Bump feeding_mode when the batch path changes.'
                        ),
                        'path': denominator_values[0],
                        'history': denominator_values[1],
                        'return': denominator_values[2],
                        'barrier': denominator_values[3],
                        'ranking': denominator_values[4],
                        'csv': denominator_csv,
                    }, handle, indent=2)
                os.replace(f'{denominator_path}.tmp', denominator_path)
            distributed_barrier(device, 'kronos_denominator_saved')
            if os.path.exists(denominator_path):
                with open(denominator_path) as handle:
                    saved_denominators = json.load(handle)
                config['beta_v21_validation_denominators'] = saved_denominators['csv']
        if not config.get('beta_v21_validation_denominators'):
            raise ValueError(
                'Beta v2.1 requires fixed validation denominators or auto calibration'
            )

    if eval_only_requested():
        # Base-model measurement: the parent weights are untouched here (no
        # optimizer, no resume/transition load has happened yet).
        return run_eval_only_validation(
            model, tokenizer, device, config, save_dir, rank,
            large_val_loader, valid_dataset, amp_dtype,
            calibration_metrics=calibration_metrics, start_time=start_time,
        )

    segments_per_coverage = max(
        1, math.ceil(train_dataset.total_samples / train_dataset.n_samples)
    )
    coverage_passes = max(1, int(config.get('coverage_passes', 1)))
    minimum_coverage_segments = segments_per_coverage * coverage_passes
    patience = max(0, int(config.get('early_stopping_patience', 0)))
    required_segments = (
        minimum_coverage_segments + patience
        if config.get('require_full_coverage', True)
        else 0
    )
    effective_epochs = max(int(config['epochs']), required_segments)
    max_segments_per_run = max(0, int(config.get('max_segments_per_run', 0)))
    max_runtime_seconds = max(
        0.0, float(config.get('max_runtime_seconds', 0.0) or 0.0)
    )
    resume_guard = build_resume_guard(
        config, effective_epochs, segments_per_coverage
    )
    if rank == 0:
        print(
            f"Coverage plan: {train_dataset.total_samples:,} windows, "
            f"{train_dataset.n_samples:,}/segment, {segments_per_coverage} segments/pass, "
            f"{coverage_passes} pass(es), up to {effective_epochs} segments."
        )
        if max_segments_per_run:
            print(
                f"Invocation limit: stop safely after {max_segments_per_run} "
                "completed segment(s); the global schedule is unchanged."
            )

    core_model = model.module if isinstance(model, DDP) else model
    v21_normalizer = DetachedEMANormalizer(
        decay=float(config.get('beta_v21_ema_decay', 0.99))
    )
    validate_uniform_learning_rate_config(config)
    optimizer_groups = build_optimizer_groups(core_model, config)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(config['adam_beta1'], config['adam_beta2']),
    )
    family_named_parameters = {
        'condition': [],
        'adaptation': [],
    }
    for parameter_name, parameter in core_model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = parameter_optimizer_family(parameter_name, config)
        family_named_parameters[family].append((parameter_name, parameter))
    scheduler_steps = sum(
        math.ceil(
            math.ceil(segment_sample_count(train_dataset, segment) / world_size)
            / config['batch_size']
        )
        for segment in range(effective_epochs)
    )
    scheduler_type = config.get('scheduler_type', 'warmup_cosine')
    if scheduler_type not in {'warmup_cosine', 'warmup_constant', 'two_speed', 'uniform_cosine', 'fixed', 'one_cycle'}:
        raise ValueError('Unsupported v1-beta scheduler type')
    warmup_steps = max(
        0,
        int(round(scheduler_steps * float(config['scheduler_warmup_ratio']))),
    )
    condition_fast_decay_steps = max(
        warmup_steps + 1,
        int(round(
            scheduler_steps * float(config.get('condition_fast_decay_ratio', 0.075))
        )),
    )
    if scheduler_type == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[float(group['peak_lr']) for group in optimizer.param_groups],
            total_steps=scheduler_steps,
            pct_start=0.03,
            div_factor=10.0,
            final_div_factor=1e4,
        )
        scheduler_lambdas = None
    else:
        scheduler_lambdas = [
        (
            lambda step: 1.0
        ) if scheduler_type == 'fixed' else (
            lambda step, group=group: warmup_constant_multiplier(
                step,
                warmup_steps,
                float(group['warmup_start_lr']),
                float(group['peak_lr']),
            )
        ) if scheduler_type == 'warmup_constant' else (
            lambda step, group=group: two_speed_multiplier(
                step,
                scheduler_steps,
                warmup_steps,
                float(group['warmup_start_lr']),
                float(group['peak_lr']),
                float(group['min_lr']),
                group['family'],
                condition_fast_decay_steps,
                float(config.get('condition_fast_decay_learning_rate', 1e-5)),
            )
        ) if scheduler_type == 'two_speed' else (
            lambda step, group=group: warmup_cosine_multiplier(
                step,
                scheduler_steps,
                warmup_steps,
                float(group['warmup_start_lr']),
                float(group['peak_lr']),
                float(group['min_lr']),
            )
        )
            for group in optimizer.param_groups
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler_lambdas
        )
    target_optimizer_group_plan = [
        {
            key: value
            for key, value in group.items()
            if key != 'params'
        }
        for group in optimizer.param_groups
    ]
    if rank == 0:
        print(
            f"Learning-rate plan: {scheduler_steps:,} global optimizer steps; "
            f"{warmup_steps:,} warmup steps "
            f"({float(config['scheduler_warmup_ratio']):.2%}); "
            f"scheduler={scheduler_type}."
        )
        if scheduler_type == 'two_speed':
            print(
                "Condition fast-decay milestone: "
                f"step {condition_fast_decay_steps:,} "
                f"({float(config['condition_fast_decay_ratio']):.2%}) -> "
                f"{float(config['condition_fast_decay_learning_rate']):.10e}, "
                "then monotonic cosine tail."
            )
        for family_name, named in family_named_parameters.items():
            modules = sorted({name.split('.')[0] for name, _ in named})
            parameter_count = sum(parameter.numel() for _, parameter in named)
            print(
                f"LR family {family_name}: params={parameter_count:,}, "
                f"modules={modules}, "
                f"split_trunk_head_lr="
                f"{bool(config.get('split_trunk_head_learning_rate', False))}"
            )
        for group in optimizer.param_groups:
            parameter_count = sum(parameter.numel() for parameter in group['params'])
            print(
                f"Optimizer group {group['name']}: params={parameter_count:,}, "
                f"warmup_start_lr={float(group['warmup_start_lr']):.10e}, "
                f"peak_lr={float(group['peak_lr']):.10e}, "
                f"min_lr={float(group['min_lr']):.10e}, "
                f"weight_decay={float(group['weight_decay']):.4f}"
            )

    if selection_metric_higher_is_better(
        config.get('best_selection_metric', 'objective')
    ):
        best_val_loss = float('-inf')
    else:
        best_val_loss = float('inf')
    epochs_without_improvement = 0
    post_coverage_without_improvement = 0
    dt_result = {}
    batch_idx_global = 0
    start_epoch = 0
    resume_path = os.path.join(save_dir, 'checkpoints', 'last_state.pt')
    scheduler_transition_source = ''
    scheduler_transition_parent_segment = 0
    bootstrap_completed_segments = int(
        config.get('bootstrap_completed_segments', 0)
    )
    if bootstrap_completed_segments > effective_epochs:
        raise ValueError(
            'Bootstrap completed segments exceed the global coverage plan: '
            f'{bootstrap_completed_segments} > {effective_epochs}'
        )

    def persist_resume_checkpoint(next_epoch):
        save_resume_state(
            resume_path,
            core_model,
            optimizer,
            scheduler,
            next_epoch=next_epoch,
            resume_step=0,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            post_coverage_without_improvement=post_coverage_without_improvement,
            batch_idx_global=batch_idx_global,
            amp_scaler=scaler.state_dict() if scaler is not None else {},
            use_amp=use_amp,
            amp_dtype=amp_dtype_name,
            master_weight_dtype=master_weight_dtype(core_model),
            xla_gradient_all_reduce=bool(device.type == 'xla' and world_size > 1),
            effective_epochs=effective_epochs,
            segments_per_coverage=segments_per_coverage,
            coverage_passes=coverage_passes,
            predictor_loss_mode=config.get('predictor_loss_mode', 'full_sequence'),
            history_loss_weight=float(config.get('history_loss_weight', 0.0)),
            resume_guard=resume_guard,
            scheduler_type=config.get('scheduler_type', 'warmup_cosine'),
            scheduler_min_learning_rate=float(
                config.get('scheduler_min_learning_rate', 1e-6)
            ),
            predictor_min_learning_rate=float(
                config.get('predictor_min_learning_rate', 1e-6)
            ),
            condition_min_learning_rate=float(
                config.get('condition_min_learning_rate', 1e-6)
            ),
            scheduler_warmup_ratio=float(config['scheduler_warmup_ratio']),
            scheduler_warmup_steps=warmup_steps,
            scheduler_total_steps=scheduler_steps,
            condition_fast_decay_ratio=float(
                config.get('condition_fast_decay_ratio', 0.075)
            ),
            condition_fast_decay_steps=condition_fast_decay_steps,
            condition_fast_decay_learning_rate=float(
                config.get('condition_fast_decay_learning_rate', 1e-5)
            ),
            predictor_warmup_start_learning_rate=float(
                config['predictor_warmup_start_learning_rate']
            ),
            condition_warmup_start_learning_rate=float(
                config['condition_warmup_start_learning_rate']
            ),
            predictor_learning_rate=float(config['predictor_learning_rate']),
            condition_learning_rate=float(
                config.get('condition_learning_rate', 1e-4)
            ),
            optimizer_group_plan=[
                {
                    'name': group['name'],
                    'family': group['family'],
                    'peak_lr': float(group['peak_lr']),
                    'warmup_start_lr': float(group['warmup_start_lr']),
                    'min_lr': float(group['min_lr']),
                    'weight_decay': float(group['weight_decay']),
                }
                for group in optimizer.param_groups
            ],
            beta_v21_loss_ema=v21_normalizer.state_dict(),
            scheduler_transition_source=scheduler_transition_source,
            scheduler_transition_parent_segment=scheduler_transition_parent_segment,
        )

    transition_path = str(config.get('scheduler_transition_state', '')).strip()
    if transition_path:
        if os.path.exists(resume_path):
            raise ValueError(
                'Scheduler transition requires a new output tree without last_state.pt'
            )
        if scheduler_type == 'one_cycle':
            raise ValueError('Scheduler transition does not support one_cycle')
        if not os.path.isfile(transition_path):
            raise FileNotFoundError(
                f'Scheduler transition checkpoint not found: {transition_path}'
            )
        transition_state = torch.load(
            transition_path, map_location='cpu', weights_only=False
        )
        validate_scheduler_transition_state(
            transition_state, config, amp_dtype_name
        )
        refuse_low_precision_checkpoint(transition_state, device, transition_path)
        core_model.load_state_dict(transition_state['model'])
        restore_optimizer_for_scheduler_transition(
            optimizer, transition_state, target_optimizer_group_plan, device
        )
        if config.get('use_beta_v21_auxiliary', False):
            saved_ema = transition_state.get('beta_v21_loss_ema')
            if saved_ema is None:
                raise ValueError(
                    'Beta v2.1 scheduler transition checkpoint has no loss EMA state'
                )
            v21_normalizer.load_state_dict(saved_ema)
        if scale_gradients:
            if 'amp_scaler' not in transition_state:
                raise ValueError(
                    'Scheduler transition checkpoint has no AMP scaler state'
                )
            scaler.load_state_dict(transition_state['amp_scaler'])
        scheduler.last_epoch = 0
        scheduler._step_count = 1
        scheduler._last_lr = [
            float(group['lr']) for group in optimizer.param_groups
        ]
        batch_idx_global = 0
        start_epoch = 0
        scheduler_transition_source = os.path.abspath(transition_path)
        scheduler_transition_parent_segment = int(
            transition_state.get('next_epoch', 0)
        )
        if rank == 0:
            family_lrs = learning_rates_by_family(optimizer)
            print(
                'Started a new scheduler stage while preserving model and AdamW '
                f'state from parent Segment {scheduler_transition_parent_segment}; '
                f'new stage Segment 1/{effective_epochs}, scheduler={scheduler_type}, '
                f'warmup_steps={warmup_steps}, optimizer_steps={scheduler_steps}, '
                f'adaptation_lr={family_lrs["adaptation"]:.10e}, '
                f'condition_lr={family_lrs["condition"]:.10e}.'
            )
    elif config.get('resume_training', False) and os.path.exists(resume_path):
        resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
        ignore_guard_keys = (
            ('beta_v21_validation_denominators',)
            if denominators_recalibrated_this_run
            else ()
        )
        validate_resume_guard(
            resume_state.get('resume_guard'),
            resume_guard,
            ignore_keys=ignore_guard_keys,
        )
        if denominators_recalibrated_this_run and rank == 0:
            print(
                'Resume guard: ignored beta_v21_validation_denominators after '
                'explicit feeding-mode recalibration this run.',
                flush=True,
            )
        saved_effective_epochs = int(resume_state.get('effective_epochs', effective_epochs))
        if saved_effective_epochs != effective_epochs:
            raise ValueError(
                f'Resume plan has {saved_effective_epochs} segments but current plan has {effective_epochs}'
            )
        saved_loss_mode = resume_state.get(
            'predictor_loss_mode', config.get('predictor_loss_mode', 'full_sequence')
        )
        saved_history_weight = float(resume_state.get(
            'history_loss_weight', config.get('history_loss_weight', 0.0)
        ))
        if saved_loss_mode != config.get('predictor_loss_mode', 'full_sequence'):
            raise ValueError(
                f'Resume loss mode is {saved_loss_mode}, current mode is '
                f"{config.get('predictor_loss_mode', 'full_sequence')}"
            )
        if not math.isclose(
            saved_history_weight, float(config.get('history_loss_weight', 0.0))
        ):
            raise ValueError('Resume history loss weight does not match current config')
        saved_scheduler_type = resume_state.get(
            'scheduler_type', config.get('scheduler_type', 'warmup_cosine')
        )
        if saved_scheduler_type != config.get('scheduler_type', 'warmup_cosine'):
            raise ValueError('Resume scheduler type does not match current config')
        saved_min_lr = float(resume_state.get(
            'scheduler_min_learning_rate',
            config.get('scheduler_min_learning_rate', 1e-6),
        ))
        if not math.isclose(
            saved_min_lr,
            float(config.get('scheduler_min_learning_rate', 1e-6)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('Resume scheduler minimum learning rate does not match current config')
        if int(resume_state.get('scheduler_total_steps', -1)) != scheduler_steps:
            raise ValueError('Resume scheduler total steps do not match current plan')
        if int(resume_state.get('scheduler_warmup_steps', -1)) != warmup_steps:
            raise ValueError('Resume scheduler warmup steps do not match current plan')
        refuse_low_precision_checkpoint(resume_state, device, resume_path)
        core_model.load_state_dict(resume_state['model'])
        if config.get('use_beta_v21_auxiliary', False):
            saved_ema = resume_state.get('beta_v21_loss_ema')
            if saved_ema is None:
                raise ValueError('Beta v2.1 resume checkpoint has no loss EMA state')
            v21_normalizer.load_state_dict(saved_ema)
        optimizer.load_state_dict(resume_state['optimizer'])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(resume_state['scheduler'])
        if bool(resume_state.get('use_amp', False)) != use_amp:
            raise ValueError(
                f"Resume AMP mismatch: {resume_state.get('use_amp')} != {use_amp}"
            )
        saved_amp_dtype = resume_state.get(
            'amp_dtype', 'float16' if use_amp else 'disabled'
        )
        if saved_amp_dtype != amp_dtype_name:
            raise ValueError(
                f"Resume AMP dtype mismatch: {saved_amp_dtype} != {amp_dtype_name}"
            )
        if scale_gradients:
            if 'amp_scaler' not in resume_state:
                raise ValueError('AMP continuation checkpoint has no scaler state')
            scaler.load_state_dict(resume_state['amp_scaler'])
        restore_rng_state(resume_state.get('rng_state'))
        start_epoch = int(resume_state['next_epoch'])
        resume_step = int(resume_state.get('resume_step', 0))
        if resume_step != 0:
            raise ValueError(
                'v1-beta uses segment-level resume only; resume_step must be zero'
            )
        best_val_loss = float(resume_state.get('best_val_loss', best_val_loss))
        epochs_without_improvement = int(
            resume_state.get('epochs_without_improvement', 0)
        )
        post_coverage_without_improvement = int(
            resume_state.get('post_coverage_without_improvement', 0)
        )
        batch_idx_global = int(resume_state.get('batch_idx_global', 0))
        scheduler_transition_source = str(
            resume_state.get('scheduler_transition_source', '')
        )
        scheduler_transition_parent_segment = int(
            resume_state.get('scheduler_transition_parent_segment', 0)
        )
        if int(scheduler.last_epoch) != batch_idx_global:
            raise ValueError(
                'Resume scheduler step does not match persisted global batch step: '
                f'{scheduler.last_epoch} != {batch_idx_global}'
            )
        best_metric_path = os.path.join(
            save_dir, 'checkpoints', 'best_model', 'best_metric.json'
        )
        if start_epoch > 0:
            if not os.path.isfile(best_metric_path):
                raise ValueError(
                    'Completed continuation has no best_model/best_metric.json; '
                    'refusing a checkpoint whose Best contract cannot be verified'
                )
            with open(best_metric_path) as handle:
                best_metric = json.load(handle)
            exported_best_loss = float(best_metric.get(
                'selection_loss', best_metric['objective_loss']
            ))
            resume_metric = str(
                best_metric.get('selection_metric')
                or config.get('best_selection_metric', 'objective')
            )
            if selection_metric_higher_is_better(resume_metric):
                if (
                    selection_is_improvement(
                        resume_metric, best_val_loss, exported_best_loss
                    )
                    and not math.isclose(
                        exported_best_loss, best_val_loss, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise ValueError(
                        'last_state.pt claims a better validation metric than best_model: '
                        f'{best_val_loss} > {exported_best_loss}'
                    )
                if selection_is_improvement(
                    resume_metric, exported_best_loss, best_val_loss
                ):
                    print(
                        'Recovered a Best export committed immediately before an '
                        'interrupted State update: '
                        f'{exported_best_loss:.6f} > {best_val_loss:.6f}'
                    )
                    best_val_loss = exported_best_loss
            elif exported_best_loss > best_val_loss and not math.isclose(
                exported_best_loss, best_val_loss, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    'last_state.pt claims a better validation loss than best_model: '
                    f'{best_val_loss} < {exported_best_loss}'
                )
            elif exported_best_loss < best_val_loss:
                print(
                    'Recovered a Best export committed immediately before an '
                    'interrupted State update: '
                    f'{exported_best_loss:.6f} < {best_val_loss:.6f}'
                )
                best_val_loss = exported_best_loss
        if rank == 0:
            print(
                f'Resumed training from the last completed segment; '
                f'next coverage segment is {start_epoch + 1}.'
            )
    elif bootstrap_completed_segments:
        start_epoch = bootstrap_completed_segments
        batch_idx_global = optimizer_steps_for_completed_segments(
            train_dataset,
            bootstrap_completed_segments,
            world_size,
            config['batch_size'],
        )
        scheduler.last_epoch = batch_idx_global
        scheduler._step_count = batch_idx_global + 1
        scheduler._last_lr = []
        for group, schedule in zip(optimizer.param_groups, scheduler_lambdas):
            group['lr'] = float(group['initial_lr']) * float(
                schedule(batch_idx_global)
            )
            scheduler._last_lr.append(group['lr'])
        best_val_loss = float(config.get('bootstrap_best_val_loss', float('inf')))
        if not math.isfinite(best_val_loss):
            raise ValueError(
                'A finite KRONOS_BOOTSTRAP_BEST_VAL_LOSS is required when '
                'bootstrapping from completed segments'
            )
        best_metric_path = os.path.join(
            save_dir, 'checkpoints', 'best_model', 'best_metric.json'
        )
        if not os.path.isfile(best_metric_path):
            raise ValueError(
                'Bootstrap output has no best_model/best_metric.json'
            )
        with open(best_metric_path) as handle:
            bootstrap_best_metric = json.load(handle)
        if int(bootstrap_best_metric.get('segment', -1)) != start_epoch:
            raise ValueError(
                'Bootstrap Best segment does not match the requested position: '
                f'{bootstrap_best_metric} vs {start_epoch}'
            )
        if not math.isclose(
            float(bootstrap_best_metric.get(
                'selection_loss', bootstrap_best_metric.get('objective_loss', float('inf'))
            )),
            best_val_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                'Bootstrap Best objective does not match the requested value'
            )
        if rank == 0:
            family_lrs = learning_rates_by_family(optimizer)
            print(
                'Bootstrapped fresh optimizer from historical Best; '
                f'completed_segments={start_epoch}, '
                f'global_step={batch_idx_global}, '
                f'next coverage segment={start_epoch + 1}, '
                f'adaptation_lr={family_lrs["adaptation"]:.10e}, '
                f'condition_lr={family_lrs["condition"]:.10e}.'
            )
    elif config.get('keep_existing_best', False):
        best_metric_path = os.path.join(
            save_dir, 'checkpoints', 'best_model', 'best_metric.json'
        )
        if not os.path.isfile(best_metric_path):
            raise ValueError(
                'KRONOS_KEEP_EXISTING_BEST requires best_model/best_metric.json; '
                'refusing to train without the historical Best threshold'
            )
        with open(best_metric_path) as handle:
            existing_best_metric = json.load(handle)
        saved_metric_name = str(existing_best_metric.get('selection_metric') or '')
        saved_loss = float(existing_best_metric.get(
            'selection_loss', existing_best_metric.get('objective_loss', float('inf'))
        ))
        if not math.isfinite(saved_loss):
            raise ValueError(
                'Existing best_metric.json has no finite selection loss'
            )
        current_metric = str(config.get('best_selection_metric', 'objective'))
        calibration_value = None
        if calibration_metrics is not None:
            if current_metric == 'pairwise_accuracy':
                calibration_value = calibration_metrics.get('pairwise_accuracy')
            elif current_metric == 'ranking':
                calibration_value = calibration_metrics.get('ranking_loss')
        best_val_loss = resolve_kept_best_loss(
            saved_metric_name, saved_loss, current_metric, calibration_value,
        )
        if rank == 0:
            family_lrs = learning_rates_by_family(optimizer)
            print(
                'Fresh AdamW on existing best_model weights; '
                'last_state optimizer moments were not loaded. '
                f'best_segment={int(existing_best_metric.get("segment", -1))}, '
                f'saved_selection_metric={saved_metric_name or "unset"}, '
                f'saved_selection_loss={saved_loss:.8f}, '
                f'active_metric={current_metric}, '
                f'active_best_threshold={best_val_loss:.8f}, '
                f'adaptation_lr={family_lrs["adaptation"]:.10e}, '
                f'condition_lr={family_lrs["condition"]:.10e}.'
            )
            if current_metric == 'ranking' and saved_metric_name != 'ranking':
                print(
                    'Historical forecast best was not reused as the ranking '
                    f'threshold ({saved_metric_name or "unset"} '
                    f'{saved_loss:.8f} -> {current_metric}). '
                    'beta_v21_score is not the selection metric.'
                )
            if (
                current_metric == 'pairwise_accuracy'
                and saved_metric_name != 'pairwise_accuracy'
            ):
                print(
                    'Historical selection loss was not reused as the pairwise '
                    f'accuracy threshold ({saved_metric_name or "unset"} '
                    f'{saved_loss:.8f} -> {current_metric}). '
                    'Higher accuracy is better. rank_ic is logged only.'
                )

    if rank == 0:
        # Keep the output contract valid even if Kaggle interrupts before the
        # first validation pass. This is the V6 base plus the configured heads;
        # a real validation winner replaces it after the first segment.
        best_path = os.path.join(save_dir, 'checkpoints', 'best_model')
        if not os.path.isfile(os.path.join(best_path, 'model.safetensors')):
            save_pretrained_with_retry(
                core_model, best_path, model_export_config(core_model, config)
            )
            print('Initialized best_model from the configured parent model before first validation.')
        if not os.path.isfile(resume_path):
            persist_resume_checkpoint(start_epoch)
            print(
                f'Initialized Segment {start_epoch} resume checkpoint before training.'
            )
        write_progress(
            save_dir,
            status='running',
            phase='initializing',
            current_segment=start_epoch + 1,
            current_step=0,
            observed_step=0,
            total_steps=len(train_loader),
            total_segments=effective_epochs,
            segments_per_coverage=segments_per_coverage,
            coverage_passes=coverage_passes,
            total_train_windows=train_dataset.total_samples,
            samples_per_segment=train_dataset.n_samples,
            validation_samples=valid_dataset.n_samples,
            best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
            device=str(device),
        )

    first_step_check_done = False
    last_completed_segment = start_epoch
    for epoch_idx in range(start_epoch, effective_epochs):
        epoch_start_time = time.time()
        reset_cuda_peak_memory(device)
        model.train()
        coverage_epoch = epoch_idx + int(config.get('coverage_epoch_offset', 0))
        train_dataset.set_epoch_seed(coverage_epoch)
        valid_dataset.set_epoch_seed(0)
        train_sampler = getattr(train_loader, 'sampler', None)
        if train_sampler is None and hasattr(train_loader, '_loader'):
            train_sampler = getattr(train_loader._loader, 'sampler', None)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.num_samples = math.ceil(len(train_dataset) / world_size)
            train_sampler.total_size = train_sampler.num_samples * world_size
            train_sampler.set_epoch(epoch_idx)
        if rank == 0 and config.get('same_day_ranking_batches', False):
            segment_dates = train_dataset.signal_date_ids[
                train_dataset.active_positions
            ]
            packed = build_same_day_ranking_batches(
                segment_dates, int(config['batch_size']), 0, world_size,
            )
            with_pairs = sum(1 for batch in packed if len(set(
                int(segment_dates[index]) for index in batch
            )) == 1 and len(batch) >= 2)
            print(
                f"Same-day batch pack rank0: batches={len(packed)}, "
                f"with_pairs={with_pairs}, segment_len={len(segment_dates)}",
                flush=True,
            )

        if rank == 0:
            write_progress(
                save_dir,
                status='stopping' if STOP_REQUESTED else 'running',
                phase='training',
                current_segment=epoch_idx + 1,
                total_segments=effective_epochs,
                current_step=0,
                observed_step=0,
                total_steps=len(train_loader),
                segments_per_coverage=segments_per_coverage,
                coverage_passes=coverage_passes,
                total_train_windows=train_dataset.total_samples,
                samples_per_segment=len(train_dataset),
                unique_windows_covered=completed_coverage_windows(
                    train_dataset, epoch_idx, coverage_passes
                ),
                best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
                device=str(device),
            )

        epoch_loss_sum = 0.0
        epoch_full_loss_sum = 0.0
        epoch_history_loss_sum = 0.0
        epoch_forecast_loss_sum = 0.0
        epoch_batches = 0
        interrupted_step = 0
        for i, batch in enumerate(train_loader):
            monitor_interval = int(config.get('condition_monitor_interval_steps', 0))
            monitor_due = bool(
                rank == 0
                and monitor_interval > 0
                and (batch_idx_global + 1) % monitor_interval == 0
            )
            core_model.collect_condition_stats = monitor_due
            batch_x, batch_x_stamp = batch[0], batch[1]
            use_beta_v21 = bool(config.get('use_beta_v21_auxiliary', False))
            auxiliary_labels = batch[-1] if use_beta_v21 else None
            batch_sector = batch[2] if len(batch) > 2 and config.get('use_sector_features', True) else None
            batch_size_bucket = batch[3] if len(batch) > 3 and config.get('use_size_features', True) else None
            batch_size_percentile = batch[4] if len(batch) > 4 and config.get('use_size_percentile', False) else None
            if config.get('disable_condition_inputs', False):
                batch_sector = batch_size_bucket = batch_size_percentile = None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            if batch_sector is not None:
                batch_sector = batch_sector.to(device, non_blocking=True)
            if batch_size_bucket is not None:
                batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
            if batch_size_percentile is not None:
                batch_size_percentile = batch_size_percentile.to(device, non_blocking=True)
            auxiliary_labels = move_auxiliary_labels(auxiliary_labels, device)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Keep tokenization in float32; AMP applies only to the predictor.
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float16,
                enabled=use_amp,
            ):
                model_output = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    sector_id=batch_sector, size_bucket=batch_size_bucket,
                    size_percentile=batch_size_percentile,
                    return_auxiliary=use_beta_v21,
                    asof_index=int(config['lookback_window']) - 1,
                )
                if use_beta_v21:
                    logits, auxiliary_predictions = model_output
                else:
                    logits = model_output
                core_model.collect_condition_stats = False
            # Losses in float32 outside autocast (bf16 autocast covers only the
            # predictor forward); matches the fp32 validation loss path.
            logits = to_float32(logits)
            if use_beta_v21:
                auxiliary_predictions = to_float32(auxiliary_predictions)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = compute_predictor_losses(core_model.head, logits, token_out, config)
                auxiliary_losses = None
                normalized_losses = None
                if use_beta_v21:
                    auxiliary_losses = compute_auxiliary_losses(
                        auxiliary_predictions['return'],
                        auxiliary_predictions['barrier'],
                        auxiliary_labels,
                    )
                    loss, normalized_losses = compose_beta_v21_objective(
                        losses['weighted_forecast'],
                        losses['history'],
                        auxiliary_losses,
                        v21_normalizer,
                        batch_idx_global,
                        warmup_steps=int(
                            config.get('beta_v21_auxiliary_warmup_steps', 1000)
                        ),
                        ranking_weight=float(
                            config.get('beta_v21_ranking_weight', 0.05)
                        ),
                    )
                else:
                    loss = losses['objective']
                s1_loss = losses['objective_s1']
                s2_loss = losses['objective_s2']

            # Backward pass and optimization
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            applied_family_lrs = learning_rates_by_family(optimizer)
            monitoring_stats = {}
            if auxiliary_losses is not None:
                monitoring_stats.update({
                    'return_loss': float(auxiliary_losses['return'].item()),
                    'return_huber_loss': float(
                        auxiliary_losses['return_huber'].item()
                    ),
                    'return_bias_loss': float(
                        auxiliary_losses['return_bias'].item()
                    ),
                    'barrier_loss': float(auxiliary_losses['barrier'].item()),
                    'ranking_loss': float(auxiliary_losses['ranking'].item()),
                    'auxiliary_ramp': float(normalized_losses['ramp']),
                })
            if monitor_due:
                monitoring_stats.update(getattr(core_model, 'last_condition_stats', {}))
                if core_model.sector_emb is not None:
                    monitoring_stats['sector_embedding_weight_norm'] = float(
                        core_model.sector_emb.weight.detach().float().norm().item()
                    )
                if core_model.size_mlp is not None:
                    monitoring_stats['size_mlp_output_weight_norm'] = float(
                        core_model.size_mlp[-1].weight.detach().float().norm().item()
                    )
                for family in ('condition', 'adaptation'):
                    named = family_named_parameters.get(family) or []
                    if not named:
                        continue
                    family_stats = parameter_family_statistics(
                        named, applied_family_lrs.get(family, 0.0)
                    )
                    monitoring_stats.update({
                        f'{family}_{name}': value
                        for name, value in family_stats.items()
                    })
            run_first_step_check = bool(
                device.type == 'xla' and world_size > 1 and not first_step_check_done
            )
            if run_first_step_check:
                first_step_state = begin_first_step_sync_check(
                    core_model, batch_x, device, rank, world_size
                )
            if device.type == 'xla':
                # Same as xm.optimizer_step (reduce_gradients + step + sync),
                # but clipping sees the all-reduced (global-mean) gradient,
                # like DDP on GPU.  reduce_gradients is a real all-reduce only
                # when runtime.world_size() == 8 (checked at startup).
                xm.reduce_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config.get('gradient_clip_norm', 3.0))
            )
            if device.type == 'xla':
                optimizer.step()
                xm.mark_step()
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if run_first_step_check:
                finish_first_step_sync_check(
                    first_step_state, core_model, optimizer, device, rank, world_size
                )
                first_step_check_done = True
            epoch_loss_sum += float(loss.item())
            epoch_full_loss_sum += float(losses['full_sequence'].item())
            epoch_history_loss_sum += float(losses['history'].item())
            epoch_forecast_loss_sum += float(losses['forecast'].item())
            epoch_batches += 1

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                family_lrs = learning_rates_by_family(optimizer)
                adaptation_lr = family_lrs['adaptation']
                condition_lr = family_lrs['condition']
                print(
                    f"[Rank {rank}, Segment {epoch_idx + 1}/{effective_epochs}, Step {i + 1}/{len(train_loader)}] "
                    f"Adaptation LR {adaptation_lr:.10e}, "
                    f"Condition LR {condition_lr:.10e}, Loss: {loss.item():.4f}, "
                    f"Forecast: {losses['forecast'].item():.4f}, "
                    f"History: {losses['history'].item():.4f}, "
                    f"GlobalBatch: {int(batch_x.shape[0]) * world_size} "
                    f"({int(batch_x.shape[0])} x {world_size} replicas)"
                )
                if monitoring_stats:
                    print(
                        "Condition Monitor JSON: "
                        + json.dumps(monitoring_stats, sort_keys=True)
                    )
                write_progress(
                    save_dir,
                    status='stopping' if STOP_REQUESTED else 'running',
                    phase='training',
                    current_segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    current_step=0,
                    observed_step=i + 1,
                    total_steps=len(train_loader),
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                    total_train_windows=train_dataset.total_samples,
                    samples_per_segment=len(train_dataset),
                    unique_windows_covered=completed_coverage_windows(
                        train_dataset, epoch_idx, coverage_passes
                    ),
                    observed_windows_covered=min(
                        completed_coverage_windows(train_dataset, epoch_idx, coverage_passes)
                        + min((i + 1) * config['batch_size'] * world_size, len(train_dataset)),
                        train_dataset.total_samples * coverage_passes,
                    ),
                    train_loss=epoch_loss_sum / epoch_batches,
                    train_full_sequence_loss=epoch_full_loss_sum / epoch_batches,
                    train_history_loss=epoch_history_loss_sum / epoch_batches,
                    train_forecast_loss=epoch_forecast_loss_sum / epoch_batches,
                    best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
                    device=str(device),
                )
                append_metric(
                    save_dir,
                    type='train',
                    segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    step=i + 1,
                    total_steps=len(train_loader),
                    loss=float(loss.item()),
                    average_loss=epoch_loss_sum / epoch_batches,
                    full_sequence_loss=float(losses['full_sequence'].item()),
                    history_loss=float(losses['history'].item()),
                    forecast_loss=float(losses['forecast'].item()),
                    learning_rate=adaptation_lr,
                    adaptation_learning_rate=adaptation_lr,
                    condition_learning_rate=condition_lr,
                    **monitoring_stats,
                )
            if rank == 0 and logger:
                family_lrs = learning_rates_by_family(optimizer)
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric(
                    'predictor_learning_rate', family_lrs['adaptation'],
                    step=batch_idx_global,
                )
                logger.log_metric(
                    'condition_learning_rate', family_lrs['condition'],
                    step=batch_idx_global,
                )

            batch_idx_global += 1

            if STOP_REQUESTED:
                interrupted_step = i + 1
                break

        if interrupted_step:
            if rank == 0:
                print(
                    f"Stopping after segment {epoch_idx + 1}, step "
                    f"{interrupted_step}/{len(train_loader)}; this incomplete "
                    "segment will be replayed from step 1."
                )
                write_progress(
                    save_dir,
                    status='stopping',
                    phase='stopping',
                    current_segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    current_step=0,
                    observed_step=interrupted_step,
                    total_steps=len(train_loader),
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                    total_train_windows=train_dataset.total_samples,
                    samples_per_segment=len(train_dataset),
                    unique_windows_covered=completed_coverage_windows(
                        train_dataset, epoch_idx, coverage_passes
                    ),
                    train_loss=epoch_loss_sum / max(epoch_batches, 1),
                    best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
                    device=str(device),
                )
                write_progress(
                    save_dir,
                    status='stopped',
                    phase='complete',
                    current_segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    current_step=0,
                    observed_step=interrupted_step,
                    total_steps=len(train_loader),
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                    total_train_windows=train_dataset.total_samples,
                    samples_per_segment=len(train_dataset),
                    unique_windows_covered=completed_coverage_windows(
                        train_dataset, epoch_idx, coverage_passes
                    ),
                    train_loss=epoch_loss_sum / max(epoch_batches, 1),
                    best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
                    device=str(device),
                )
            distributed_barrier(device, 'kronos_training_interrupted')
            dt_result.update({
                'best_val_loss': best_val_loss,
                'status': 'stopped',
                'completed_segments': epoch_idx,
                'partial_segment': epoch_idx + 1,
                'partial_step': interrupted_step,
                'train_loss': epoch_loss_sum / max(epoch_batches, 1),
                'total_segments': effective_epochs,
            })
            break

        # --- Validation Loop ---
        model.eval()
        if rank == 0:
            write_progress(
                save_dir,
                status='stopping' if STOP_REQUESTED else 'running',
                phase='validation',
                current_segment=epoch_idx + 1,
                total_segments=effective_epochs,
                current_step=0,
                observed_step=len(train_loader),
                total_steps=len(train_loader),
                validation_total_steps=len(
                    large_val_loader if validation_full_only else val_loader
                ),
                segments_per_coverage=segments_per_coverage,
                coverage_passes=coverage_passes,
                total_train_windows=train_dataset.total_samples,
                samples_per_segment=len(train_dataset),
                unique_windows_covered=completed_coverage_windows(
                    train_dataset, epoch_idx, coverage_passes
                ),
                observed_windows_covered=completed_coverage_windows(
                    train_dataset, epoch_idx + 1, coverage_passes
                ),
                train_loss=epoch_loss_sum / max(epoch_batches, 1),
                best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
                device=str(device),
            )
        ablation_interval = int(config.get('condition_ablation_interval_segments', 0))
        run_condition_ablation = bool(
            ablation_interval > 0
            and (epoch_idx == start_epoch or (epoch_idx + 1) % ablation_interval == 0)
        )
        selection_metric = config.get('best_selection_metric', 'objective')
        large_interval = int(
            config.get('validation_large_interval_segments', 10)
        )
        run_large_validation = validation_full_only or should_run_large_validation(
            epoch_idx + 1, effective_epochs, large_interval
        )
        large_metrics = None
        quick_metrics = None
        if not validation_full_only:
            prepare_model_for_validation(model, device, optimizer)
            quick_metrics = evaluate_validation(
                model, tokenizer, val_loader, device, config, amp_dtype,
                run_condition_ablation=run_condition_ablation,
                period_names=getattr(valid_dataset, 'validation_period_names', {}),
                rank=rank,
            )
        if run_large_validation:
            if rank == 0:
                print(
                    f"Running fixed large validation at Segment {epoch_idx + 1}: "
                    f"{len(valid_dataset):,} samples."
                )
            prepare_model_for_validation(model, device, optimizer)
            large_metrics = evaluate_validation(
                model, tokenizer, large_val_loader, device, config, amp_dtype,
                run_condition_ablation=False,
                period_names=getattr(valid_dataset, 'validation_period_names', {}),
                rank=rank,
            )

        primary_metrics = large_metrics if validation_full_only else quick_metrics
        if primary_metrics is None:
            raise RuntimeError("No validation metrics were produced")
        avg_val_loss = primary_metrics['objective_loss']
        avg_val_full_loss = primary_metrics['full_sequence_loss']
        avg_val_history_loss = primary_metrics['history_loss']
        avg_val_forecast_loss = primary_metrics['forecast_loss']
        ablation_metrics = {
            key: value
            for key, value in primary_metrics.items()
            if key.startswith('condition_')
        }

        selection_val_loss = best_selection_value(
            selection_metric, primary_metrics, large_metrics
        )

        improved = selection_is_improvement(
            selection_metric, selection_val_loss, best_val_loss
        )
        if improved:
            best_val_loss = selection_val_loss
            epochs_without_improvement = 0
            if epoch_idx + 1 > minimum_coverage_segments:
                post_coverage_without_improvement = 0
        elif selection_val_loss is not None:
            epochs_without_improvement += 1
            if epoch_idx + 1 > minimum_coverage_segments:
                post_coverage_without_improvement += 1

        next_segment = epoch_idx + 1

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Coverage Segment {epoch_idx + 1}/{effective_epochs} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.8f}")
            print(
                f"Validation Forecast/History/Full: {avg_val_forecast_loss:.8f} / "
                f"{avg_val_history_loss:.8f} / {avg_val_full_loss:.8f}"
            )
            weighted_val_forecast = primary_metrics.get('weighted_forecast_loss')
            if weighted_val_forecast is not None:
                print(
                    "Validation Weighted Forecast: "
                    f"{float(weighted_val_forecast):.8f}"
                )
                monitor_base = config.get('forecast_monitor_base')
                if monitor_base is not None:
                    red_line = float(monitor_base) + float(
                        config.get('forecast_monitor_margin') or 0.0
                    )
                    exceeded = float(weighted_val_forecast) > red_line
                    # Monitoring only. Do not stop the run for this line.
                    print(
                        "Forecast monitor (not a stop): "
                        f"weighted_forecast_loss={float(weighted_val_forecast):.8f} "
                        f"red_line={red_line:.8f} exceeded={int(exceeded)}"
                    )
                freeze_baseline = config.get('_freeze_forecast_baseline')
                if (
                    freeze_baseline is not None
                    and config.get('train_beta_v21_heads_only', False)
                ):
                    delta = abs(float(weighted_val_forecast) - float(freeze_baseline))
                    print(
                        "Frozen-trunk forecast sanity: "
                        f"weighted_forecast_loss={float(weighted_val_forecast):.8f} "
                        f"baseline={float(freeze_baseline):.8f} "
                        f"abs_delta={delta:.8e}"
                    )
                    if delta > 1e-4:
                        print(
                            "WARNING: FROZEN TRUNK FORECAST DRIFT "
                            f"|weighted_forecast_loss - baseline|={delta:.8e} > 1e-4. "
                            "Trunk/forecast heads should be frozen; investigate.",
                            flush=True,
                        )
            if config.get('use_beta_v21_auxiliary', False):
                print(
                    "Validation v2.1 Score/Return/Bias/Barrier/Rank: "
                    f"{format_validation_value(primary_metrics.get('beta_v21_score'))} / "
                    f"{primary_metrics['return_loss']:.8f} / "
                    f"{primary_metrics['return_bias_loss']:.8f} / "
                    f"{primary_metrics['barrier_loss']:.8f} / "
                    f"{primary_metrics['ranking_loss']:.8f}"
                )
                if (
                    'pairwise_accuracy' in primary_metrics
                    or 'rank_ic' in primary_metrics
                ):
                    print(
                        "Validation Pairwise Accuracy/RankIC: "
                        f"{format_validation_value(primary_metrics.get('pairwise_accuracy'))} / "
                        f"{format_validation_value(primary_metrics.get('rank_ic'))}"
                    )
                print(
                    "Validation Return-Path Consistency JSON: "
                    + json.dumps(
                        primary_metrics['return_path_consistency'], sort_keys=True
                    )
                )
            for period, metrics in primary_metrics.get('periods', {}).items():
                print(
                    f"Validation {period} Objective/Forecast/History/Full: "
                    f"{metrics['objective_loss']:.8f} / "
                    f"{metrics['forecast_loss']:.8f} / "
                    f"{metrics['history_loss']:.8f} / "
                    f"{metrics['full_sequence_loss']:.8f} "
                    f"({metrics['samples']:,} samples)"
                )
            print(
                "Validation Train Average/Best: "
                f"{epoch_loss_sum / max(epoch_batches, 1):.8f} / "
                f"{best_val_loss:.8f}"
            )
            if selection_val_loss is None:
                print(f"Best selection metric: {selection_metric}=not evaluated")
            else:
                print(
                    f"Best selection metric: {selection_metric}="
                    f"{selection_val_loss:.8f}"
                )
            if large_metrics is not None:
                print(
                    "Large Validation Objective/Forecast/History/Full: "
                    f"{large_metrics['objective_loss']:.8f} / "
                    f"{large_metrics['forecast_loss']:.8f} / "
                    f"{large_metrics['history_loss']:.8f} / "
                    f"{large_metrics['full_sequence_loss']:.8f}"
                )
                for period, metrics in large_metrics.get('periods', {}).items():
                    print(
                        f"Large Validation {period} Objective/Forecast/History/Full: "
                        f"{metrics['objective_loss']:.8f} / "
                        f"{metrics['forecast_loss']:.8f} / "
                        f"{metrics['history_loss']:.8f} / "
                        f"{metrics['full_sequence_loss']:.8f} "
                        f"({metrics['samples']:,} samples)"
                    )
            if ablation_metrics:
                print(
                    "Validation Condition Full/None/Shuffled Forecast: "
                    f"{avg_val_forecast_loss:.8f} / "
                    f"{ablation_metrics['condition_none_forecast_loss']:.8f} / "
                    f"{ablation_metrics['condition_shuffled_forecast_loss']:.8f}; "
                    "Delta Full-None/Full-Shuffled: "
                    f"{ablation_metrics['condition_full_minus_none_forecast_loss']:.8f} / "
                    f"{ablation_metrics['condition_full_minus_shuffled_forecast_loss']:.8f}"
                )
            memory_metrics = cuda_peak_memory(device)
            if memory_metrics:
                print(
                    "CUDA Peak Allocated/Reserved: "
                    f"{memory_metrics['cuda_peak_allocated_gb']:.2f} / "
                    f"{memory_metrics['cuda_peak_reserved_gb']:.2f} GB"
                )
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            if improved:
                save_path = f"{save_dir}/checkpoints/best_model"
                save_pretrained_with_retry(
                    core_model, save_path, model_export_config(core_model, config)
                )
                best_metric_path = os.path.join(save_path, 'best_metric.json')
                best_metric_temporary = f'{best_metric_path}.tmp'
                with open(best_metric_temporary, 'w') as handle:
                    json.dump({
                        'objective_loss': float(avg_val_loss),
                        'forecast_loss': float(avg_val_forecast_loss),
                        'history_loss': float(avg_val_history_loss),
                        'full_sequence_loss': float(avg_val_full_loss),
                        'selection_metric': selection_metric,
                        'selection_loss': float(selection_val_loss),
                        'period_metrics': primary_metrics.get('periods', {}),
                        'selection_period_metrics': (
                            large_metrics.get('periods', {})
                            if selection_metric == 'validation_large_objective'
                            else primary_metrics.get('periods', {})
                        ),
                        'large_metrics': large_metrics,
                        'segment': int(next_segment),
                    }, handle, indent=2)
                os.replace(best_metric_temporary, best_metric_path)
                print(
                    f"Best model saved to {save_path} "
                    f"({selection_metric}: {best_val_loss:.8f})"
                )
            # Best is committed before State. If Kaggle interrupts between the
            # two, best_metric.json lets the next run reconcile that transaction.
            persist_resume_checkpoint(next_segment)
            completed_coverage_segments = next_segment
            last_completed_segment = completed_coverage_segments
            write_progress(
                save_dir,
                status='stopping' if STOP_REQUESTED else 'running',
                phase='checkpointing',
                current_segment=completed_coverage_segments,
                total_segments=effective_epochs,
                current_step=len(train_loader),
                total_steps=len(train_loader),
                segments_per_coverage=segments_per_coverage,
                coverage_passes=coverage_passes,
                total_train_windows=train_dataset.total_samples,
                samples_per_segment=len(train_dataset),
                unique_windows_covered=completed_coverage_windows(
                    train_dataset, completed_coverage_segments, coverage_passes
                ),
                train_loss=epoch_loss_sum / max(epoch_batches, 1),
                val_loss=avg_val_loss,
                val_full_sequence_loss=avg_val_full_loss,
                val_history_loss=avg_val_history_loss,
                val_forecast_loss=avg_val_forecast_loss,
                val_large_objective_loss=(
                    large_metrics['objective_loss']
                    if large_metrics is not None else None
                ),
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                device=str(device),
                **memory_metrics,
            )
            if not validation_full_only:
                append_metric(
                    save_dir,
                    type='validation',
                    segment=completed_coverage_segments,
                    total_segments=effective_epochs,
                    step=0,
                    loss=float(avg_val_loss),
                    full_sequence_loss=float(avg_val_full_loss),
                    history_loss=float(avg_val_history_loss),
                    forecast_loss=float(avg_val_forecast_loss),
                    best_loss=float(best_val_loss),
                    best_selection_metric=selection_metric,
                    selection_loss=(
                        float(selection_val_loss)
                        if selection_val_loss is not None else None
                    ),
                    train_average=epoch_loss_sum / max(epoch_batches, 1),
                    period_metrics=primary_metrics.get('periods', {}),
                    **ablation_metrics,
                    **memory_metrics,
                )
            if large_metrics is not None:
                append_metric(
                    save_dir,
                    type='validation_large',
                    segment=completed_coverage_segments,
                    total_segments=effective_epochs,
                    step=0,
                    total_steps=len(train_loader),
                    loss=large_metrics['objective_loss'],
                    full_sequence_loss=large_metrics['full_sequence_loss'],
                    history_loss=large_metrics['history_loss'],
                    forecast_loss=large_metrics['forecast_loss'],
                    weighted_forecast_loss=large_metrics.get('weighted_forecast_loss'),
                    samples=len(valid_dataset),
                    batches=large_metrics['batches'],
                    train_average=epoch_loss_sum / max(epoch_batches, 1),
                    best_loss=float(best_val_loss),
                    best_selection_metric=selection_metric,
                    selection_loss=(
                        float(selection_val_loss)
                        if selection_val_loss is not None else None
                    ),
                    manifest_sha256=getattr(
                        valid_dataset, 'fixed_validation_manifest_sha256', None
                    ),
                    period_metrics=large_metrics.get('periods', {}),
                    beta_v21_score=large_metrics.get('beta_v21_score'),
                    return_loss=large_metrics.get('return_loss'),
                    return_huber_loss=large_metrics.get('return_huber_loss'),
                    return_bias_loss=large_metrics.get('return_bias_loss'),
                    barrier_loss=large_metrics.get('barrier_loss'),
                    ranking_loss=large_metrics.get('ranking_loss'),
                    pairwise_accuracy=large_metrics.get('pairwise_accuracy'),
                    pairwise_pairs=large_metrics.get('pairwise_pairs'),
                    rank_ic=large_metrics.get('rank_ic'),
                    rank_ic_dates=large_metrics.get('rank_ic_dates'),
                    return_path_consistency=large_metrics.get(
                        'return_path_consistency'
                    ),
                )
        distributed_barrier(device, f'kronos_segment_{next_segment}_saved')

        # A wall-clock stop is evaluated only after validation/checkpointing,
        # preserving a complete segment and a resumable last_state.pt.
        runtime_limit_reached = bool(
            max_runtime_seconds > 0
            and (time.time() - start_time) >= max_runtime_seconds
        )
        if dist.is_available() and dist.is_initialized():
            decision = torch.tensor(
                [int(runtime_limit_reached)], device=device, dtype=torch.int32
            )
            dist.broadcast(decision, src=0)
            runtime_limit_reached = bool(decision.item())
        elif device.type == 'xla' and xm is not None and world_size > 1:
            decision = torch.tensor(
                [int(runtime_limit_reached)], device=device, dtype=torch.int32
            )
            decision = xm.all_reduce(xm.REDUCE_MAX, decision)
            runtime_limit_reached = bool(decision.item())
        if runtime_limit_reached:
            if rank == 0:
                print(
                    f"Runtime limit reached after segment {next_segment}; "
                    "stopping at a durable segment boundary."
                )
            dt_result.update({
                'best_val_loss': best_val_loss,
                'status': 'stopped',
                'stop_reason': 'runtime_limit',
                'completed_segments': next_segment,
                'resume_segment': next_segment + 1,
                'total_segments': effective_epochs,
            })
            break

        if (
            next_segment < effective_epochs
            and segment_run_limit_reached(
                start_epoch, next_segment, max_segments_per_run
            )
        ):
            if rank == 0:
                print(
                    f"Chunk limit reached after {next_segment - start_epoch} "
                    f"segment(s); resume from coverage segment {next_segment + 1}."
                )
            dt_result.update({
                'best_val_loss': best_val_loss,
                'status': 'stopped',
                'stop_reason': 'segment_limit',
                'completed_segments': next_segment,
                'resume_segment': next_segment + 1,
                'total_segments': effective_epochs,
            })
            break

        coverage_requirement_met = epoch_idx + 1 >= minimum_coverage_segments
        if (
            coverage_requirement_met
            and patience > 0
            and post_coverage_without_improvement >= patience
        ):
            if rank == 0:
                print(f"Early stopping after {epoch_idx + 1} coverage segments.")
            break

    dt_result.setdefault('best_val_loss', best_val_loss)
    dt_result.setdefault('status', 'completed')
    dt_result.setdefault('completed_segments', last_completed_segment)
    dt_result.setdefault('total_segments', effective_epochs)
    dt_result.setdefault('train_selection', train_dataset.selection_report)
    dt_result.setdefault('validation_selection', valid_dataset.selection_report)
    if rank == 0:
        partial_segment = dt_result.get('partial_segment')
        if partial_segment is not None:
            display_segment = partial_segment
            observed_step = dt_result.get('partial_step', 0)
            durable_step = 0
        elif dt_result['completed_segments']:
            display_segment = dt_result['completed_segments']
            observed_step = len(train_loader)
            durable_step = len(train_loader)
        else:
            display_segment = 1
            observed_step = 0
            durable_step = 0
        unique_windows_covered = completed_coverage_windows(
            train_dataset, dt_result['completed_segments'], coverage_passes
        )
        write_progress(
            save_dir,
            status=dt_result['status'],
            phase='complete',
            current_segment=display_segment,
            total_segments=effective_epochs,
            current_step=durable_step,
            observed_step=observed_step,
            total_steps=len(train_loader),
            segments_per_coverage=segments_per_coverage,
            coverage_passes=coverage_passes,
            total_train_windows=train_dataset.total_samples,
            samples_per_segment=segment_sample_count(
                train_dataset, max(display_segment - 1, 0)
            ),
            unique_windows_covered=unique_windows_covered,
            train_loss=dt_result.get('train_loss'),
            best_val_loss=None if not math.isfinite(best_val_loss) else best_val_loss,
            device=str(device),
        )
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, request_safe_stop)
            signal.signal(signal.SIGTERM, request_safe_stop)
    except (ValueError, AttributeError):
        pass
    rank, world_size, local_rank = setup_ddp()
    if os.getenv('KRONOS_DEVICE', '').lower() in {'xla', 'tpu'}:
        if xm is None:
            raise RuntimeError('KRONOS_DEVICE=xla requires torch_xla')
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Rank {rank}] Using device: {device}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_comet'] and comet_ml is not None:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    distributed_barrier(device, 'kronos_model_initialized')

    # Model Initialization
    tokenizer_path = config['finetuned_tokenizer_path']
    if not os.path.exists(tokenizer_path):
        tokenizer_path = config['pretrained_tokenizer_path']
    # Newer huggingface_hub releases do not implicitly pass a local
    # config.json into this project's non-standard model constructors.
    with open(os.path.join(tokenizer_path, 'config.json')) as handle:
        tokenizer_kwargs = json.load(handle)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    tokenizer.eval().to(device)

    with open(os.path.join(config['pretrained_predictor_path'], 'config.json')) as handle:
        model_kwargs = json.load(handle)
    model_kwargs.update({
        'num_sectors': int(config.get('num_sectors', 0)),
        'num_size_buckets': int(config.get('num_size_buckets', 0)),
        'context_layer': int(config.get('context_layer', 0)),
        'use_size_percentile': bool(config.get('use_size_percentile', False)),
        'size_mlp_hidden_dim': int(config.get('size_mlp_hidden_dim', 64)),
        'use_beta_v21_auxiliary': bool(
            config.get('use_beta_v21_auxiliary', False)
        ),
    })
    model = Kronos.from_pretrained(config['pretrained_predictor_path'], **model_kwargs)
    reset_conditioning(model, config)
    configure_trainable_parameters(model, config)
    model.to(device)
    if device.type == 'xla':
        run_xla_startup_checks(model, device, rank, world_size)
    if dist.is_available() and dist.is_initialized():
        find_unused = bool(config.get('train_beta_v21_heads_only', False))
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused,
        )
        if find_unused and rank == 0:
            print(
                'DDP find_unused_parameters=True for beta v2.1 heads-only freeze',
                flush=True,
            )

    if rank == 0:
        core_model = model.module if isinstance(model, DDP) else model
        print(f"Predictor Model Size: {get_model_size(core_model)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    config_instance = Config()
    try:
        main(config_instance.__dict__)
    except RuntimeError as exc:
        if is_cuda_out_of_memory(exc):
            write_oom_marker(config_instance.__dict__, exc)
            print('CUDA out of memory; wrote oom.json for the Kaggle fallback.')
        raise
