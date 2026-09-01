import os
import sys
import json
import math
import random
import signal
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
    import comet_ml
except ImportError:
    comet_ml = None

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)
from drive_cleanup import cleanup_drive_conflict_files


STOP_REQUESTED = False


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


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if mps_available() and hasattr(torch.mps, 'get_rng_state'):
        state['mps'] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
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


def save_resume_state(path, model, optimizer, scheduler, **metadata):
    cleanup_drive_conflict_files()
    temporary = f'{path}.tmp'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': capture_rng_state(),
        **metadata,
    }, temporary)
    os.replace(temporary, path)
    cleanup_drive_conflict_files()


def save_pretrained_with_retry(model, path, config, attempts=3, retry_delay=2):
    """Retry checkpoint export when a mounted filesystem briefly loses the directory."""
    cleanup_drive_conflict_files()
    for attempt in range(1, attempts + 1):
        try:
            os.makedirs(path, exist_ok=True)
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


def model_export_config(core_model, config):
    """Return the conditioning configuration embedded in Best/Last exports."""
    model_config = dict(getattr(core_model, '_hub_mixin_config', {}) or {})
    model_config.update({
        'num_sectors': int(config.get('num_sectors', 0)),
        'num_size_buckets': int(config.get('num_size_buckets', 0)),
        'context_layer': int(config.get('context_layer', 0)),
        'use_size_percentile': bool(config.get('use_size_percentile', False)),
        'size_mlp_hidden_dim': int(config.get('size_mlp_hidden_dim', 64)),
        'use_size_path': bool(config.get('use_size_path', False)),
        'size_path_input_dim': int(config.get('size_path_input_dim', 4)),
    })
    return model_config


def build_resume_guard(config, effective_epochs, segments_per_coverage):
    """Values that must remain identical when an output tree is continued."""
    keys = (
        'lookback_window', 'predict_window', 'batch_size', 'use_amp', 'n_train_iter',
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
        'use_size_path', 'size_path_input_dim',
        'disable_condition_inputs',
        'num_sectors', 'num_size_buckets', 'context_layer',
        'train_signal_start', 'train_signal_end', 'val_signal_start', 'val_signal_end',
        'dataset_manifest_sha256', 'bootstrap_completed_segments',
        'fixed_validation_manifest_sha256', 'validation_quick_samples',
        'validation_large_samples', 'validation_large_interval_segments',
        'validation_full_only',
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


def validate_resume_guard(saved, current):
    if not isinstance(saved, dict):
        raise ValueError(
            'Resume checkpoint has no complete resume_guard; refusing unsafe continuation'
        )
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
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
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

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    quick_val_sampler = (
        DistributedSampler(
            quick_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if use_ddp and quick_dataset is not None else None
    )
    large_val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

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

    for module in [
        model.sector_emb, model.size_emb, model.size_mlp, model.size_path_mlp,
        model.norm, model.dep_layer, model.head,
    ]:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    layer_count = int(config.get('trainable_transformer_layers', 0))
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

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable predictor parameters: {trainable:,}/{total:,} ({trainable / total:.1%})")


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
        if model.size_path_mlp is not None:
            torch.nn.init.zeros_(model.size_path_mlp[-1].weight)
            torch.nn.init.zeros_(model.size_path_mlp[-1].bias)
            print('Reset dynamic size-path output layer to zero.')


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
    return name.startswith((
        'sector_emb.', 'size_emb.', 'size_mlp.', 'size_path_mlp.'
    ))


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
        family = 'condition' if is_condition_parameter(name) else 'adaptation'
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
    result = {}
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
        'forecast': quick_metrics['forecast_loss'],
        'history': quick_metrics['history_loss'],
        'validation_large_objective': (
            large_metrics['objective_loss'] if large_metrics is not None else None
        ),
    }
    if metric not in values:
        raise ValueError(f'Unsupported best selection metric: {metric}')
    return values[metric]


def resolve_amp_dtype(config, device):
    """Return the configured CUDA autocast dtype, or None when AMP is disabled."""
    if not bool(config.get('use_amp', False)) or device.type != 'cuda':
        return None
    dtype_name = str(config.get('amp_dtype', 'float16')).strip().lower()
    if dtype_name in {'float16', 'fp16'}:
        return torch.float16
    if dtype_name in {'bfloat16', 'bf16'}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA device does not support bfloat16 AMP')
        return torch.bfloat16
    raise ValueError(f'Unsupported AMP dtype: {dtype_name}')


def evaluate_validation(
    model, tokenizer, loader, device, config, amp_dtype, run_condition_ablation=False,
    period_names=None,
):
    """Evaluate a fixed validation set and report its named date periods."""
    model.eval()
    core_model = model.module if isinstance(model, DDP) else model
    sums = {
        'objective_loss': 0.0,
        'full_sequence_loss': 0.0,
        'history_loss': 0.0,
        'forecast_loss': 0.0,
        'condition_none_forecast_loss': 0.0,
        'condition_shuffled_forecast_loss': 0.0,
    }
    period_names = dict(period_names or {})
    period_sums = {
        int(code): {key: 0.0 for key in sums}
        for code in period_names
    }
    period_samples = {int(code): 0 for code in period_names}
    batches = 0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_x_stamp = batch[0], batch[1]
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
            batch_size_path = (
                batch[5]
                if len(batch) > 5 and config.get('use_size_path', False)
                else None
            )
            period_index = 6 if config.get('use_size_path', False) else 5
            batch_period = (
                batch[period_index]
                if len(batch) > period_index and period_names else None
            )
            if config.get('disable_condition_inputs', False):
                batch_sector = batch_size_bucket = batch_size_percentile = None
                batch_size_path = None
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
            if batch_size_path is not None:
                batch_size_path = batch_size_path.to(device, non_blocking=True)

            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float16,
                enabled=amp_dtype is not None,
            ):
                logits = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    sector_id=batch_sector, size_bucket=batch_size_bucket,
                    size_percentile=batch_size_percentile,
                    size_path=(
                        batch_size_path[:, :token_in[0].shape[1], :]
                        if batch_size_path is not None else None
                    ),
                    use_teacher_forcing=True, s1_targets=token_out[0],
                )
                losses = compute_predictor_losses(
                    core_model.head, logits, token_out, config
                )
            batch_samples = int(batch_x.shape[0])
            samples += batch_samples
            sums['objective_loss'] += losses['objective'].item() * batch_samples
            sums['full_sequence_loss'] += losses['full_sequence'].item() * batch_samples
            sums['history_loss'] += losses['history'].item() * batch_samples
            sums['forecast_loss'] += losses['forecast'].item() * batch_samples
            batches += 1

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
                shuffled_size_path = (
                    torch.roll(batch_size_path, shifts=1, dims=0)
                    if batch_size_path is not None and batch_size_path.shape[0] > 1
                    else batch_size_path
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float16,
                    enabled=amp_dtype is not None,
                ):
                    none_logits = model(
                        token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                        sector_id=None, size_bucket=None, size_percentile=None,
                        size_path=None,
                        use_teacher_forcing=True, s1_targets=token_out[0],
                    )
                    none_losses = compute_predictor_losses(
                        core_model.head, none_logits, token_out, config
                    )
                    shuffled_logits = model(
                        token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                        sector_id=shuffled_sector,
                        size_bucket=shuffled_bucket,
                        size_percentile=shuffled_percentile,
                        size_path=(
                            shuffled_size_path[:, :token_in[0].shape[1], :]
                            if shuffled_size_path is not None else None
                        ),
                        use_teacher_forcing=True, s1_targets=token_out[0],
                    )
                    shuffled_losses = compute_predictor_losses(
                        core_model.head, shuffled_logits, token_out, config
                    )
                sums['condition_none_forecast_loss'] += (
                    none_losses['forecast'].item() * batch_samples
                )
                sums['condition_shuffled_forecast_loss'] += (
                    shuffled_losses['forecast'].item() * batch_samples
                )

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
                    values = period_sums[int(code)]
                    values['objective_loss'] += period_losses['objective'].item() * count
                    values['full_sequence_loss'] += period_losses['full_sequence'].item() * count
                    values['history_loss'] += period_losses['history'].item() * count
                    values['forecast_loss'] += period_losses['forecast'].item() * count
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
                        values['condition_none_forecast_loss'] += (
                            period_none['forecast'].item() * count
                        )
                        values['condition_shuffled_forecast_loss'] += (
                            period_shuffled['forecast'].item() * count
                        )
                    period_samples[int(code)] += count

    ordered_keys = tuple(sums)
    totals = torch.tensor([sums[key] for key in ordered_keys], device=device)
    counts = torch.tensor([batches, samples], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    divisor = max(1, int(counts[1].item()))
    result = {
        key: float(value)
        for key, value in zip(ordered_keys, (totals / divisor).tolist())
    }
    result['batches'] = int(counts[0].item())
    result['samples'] = int(counts[1].item())
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
        values = period_sums[int(code)]
        period_totals = torch.tensor(
            [values[key] for key in ordered_keys], device=device
        )
        period_count = torch.tensor(
            period_samples[int(code)], device=device, dtype=torch.long
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(period_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(period_count, op=dist.ReduceOp.SUM)
        period_divisor = max(1, int(period_count.item()))
        metrics = {
            key: float(value)
            for key, value in zip(
                ordered_keys, (period_totals / period_divisor).tolist()
            )
        }
        metrics['samples'] = int(period_count.item())
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


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    amp_dtype = resolve_amp_dtype(config, device)
    use_amp = amp_dtype is not None
    amp_dtype_name = str(amp_dtype).removeprefix('torch.') if use_amp else 'disabled'
    scale_gradients = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=scale_gradients)
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")
        print(
            f"Predictor AMP: {amp_dtype_name}; gradient scaling: "
            f"{'enabled' if scale_gradients else 'disabled'}; "
            "tokenizer encoding remains float32."
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
    validation_full_only = bool(config.get('validation_full_only', False))

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
        family = 'condition' if is_condition_parameter(parameter_name) else 'adaptation'
        family_named_parameters[family].append((parameter_name, parameter))
    scheduler_steps = sum(
        math.ceil(
            math.ceil(segment_sample_count(train_dataset, segment) / world_size)
            / config['batch_size']
        )
        for segment in range(effective_epochs)
    )
    scheduler_type = config.get('scheduler_type', 'warmup_cosine')
    if scheduler_type not in {'warmup_cosine', 'two_speed', 'uniform_cosine', 'fixed', 'one_cycle'}:
        raise ValueError('Unsupported v1-beta scheduler type')
    warmup_steps = max(
        1,
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
        for group in optimizer.param_groups:
            parameter_count = sum(parameter.numel() for parameter in group['params'])
            print(
                f"Optimizer group {group['name']}: params={parameter_count:,}, "
                f"warmup_start_lr={float(group['warmup_start_lr']):.10e}, "
                f"peak_lr={float(group['peak_lr']):.10e}, "
                f"min_lr={float(group['min_lr']):.10e}, "
                f"weight_decay={float(group['weight_decay']):.4f}"
            )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    post_coverage_without_improvement = 0
    dt_result = {}
    batch_idx_global = 0
    start_epoch = 0
    resume_path = os.path.join(save_dir, 'checkpoints', 'last_state.pt')
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
            amp_scaler=scaler.state_dict(),
            use_amp=use_amp,
            amp_dtype=amp_dtype_name,
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
        )

    if config.get('resume_training', False) and os.path.exists(resume_path):
        resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
        validate_resume_guard(resume_state.get('resume_guard'), resume_guard)
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
        core_model.load_state_dict(resume_state['model'])
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
            if exported_best_loss > best_val_loss and not math.isclose(
                exported_best_loss, best_val_loss, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    'last_state.pt claims a better validation loss than best_model: '
                    f'{best_val_loss} < {exported_best_loss}'
                )
            if exported_best_loss < best_val_loss:
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
            best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
            device=str(device),
        )

    last_completed_segment = start_epoch
    for epoch_idx in range(start_epoch, effective_epochs):
        epoch_start_time = time.time()
        reset_cuda_peak_memory(device)
        model.train()
        train_dataset.set_epoch_seed(epoch_idx)
        valid_dataset.set_epoch_seed(0)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.num_samples = math.ceil(len(train_dataset) / world_size)
            train_loader.sampler.total_size = train_loader.sampler.num_samples * world_size
            train_loader.sampler.set_epoch(epoch_idx)

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
                best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
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
            batch_sector = batch[2] if len(batch) > 2 and config.get('use_sector_features', True) else None
            batch_size_bucket = batch[3] if len(batch) > 3 and config.get('use_size_features', True) else None
            batch_size_percentile = batch[4] if len(batch) > 4 and config.get('use_size_percentile', False) else None
            batch_size_path = batch[5] if len(batch) > 5 and config.get('use_size_path', False) else None
            if config.get('disable_condition_inputs', False):
                batch_sector = batch_size_bucket = batch_size_percentile = None
                batch_size_path = None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            if batch_sector is not None:
                batch_sector = batch_sector.to(device, non_blocking=True)
            if batch_size_bucket is not None:
                batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
            if batch_size_percentile is not None:
                batch_size_percentile = batch_size_percentile.to(device, non_blocking=True)
            if batch_size_path is not None:
                batch_size_path = batch_size_path.to(device, non_blocking=True)

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
                logits = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    sector_id=batch_sector, size_bucket=batch_size_bucket,
                    size_percentile=batch_size_percentile,
                    size_path=(
                        batch_size_path[:, :token_in[0].shape[1], :]
                        if batch_size_path is not None else None
                    ),
                )
                core_model.collect_condition_stats = False
                losses = compute_predictor_losses(core_model.head, logits, token_out, config)
                loss = losses['objective']
                s1_loss = losses['objective_s1']
                s2_loss = losses['objective_s2']

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            applied_family_lrs = learning_rates_by_family(optimizer)
            monitoring_stats = {}
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
                if core_model.size_path_mlp is not None:
                    monitoring_stats['size_path_mlp_output_weight_norm'] = float(
                        core_model.size_path_mlp[-1].weight.detach().float().norm().item()
                    )
                for family in ('condition', 'adaptation'):
                    family_stats = parameter_family_statistics(
                        family_named_parameters[family], applied_family_lrs[family]
                    )
                    monitoring_stats.update({
                        f'{family}_{name}': value
                        for name, value in family_stats.items()
                    })
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config.get('gradient_clip_norm', 3.0))
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
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
                    f"History: {losses['history'].item():.4f}"
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
                    best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
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
                    best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
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
                    best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
                    device=str(device),
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
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
                best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
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
        has_distinct_large_validation = (
            int(config.get('validation_large_samples', 0))
            > int(config.get('validation_quick_samples', 0))
        )
        run_large_validation = validation_full_only or (
            has_distinct_large_validation
            and should_run_large_validation(
                epoch_idx + 1, effective_epochs, large_interval
            )
        )
        large_metrics = None
        quick_metrics = None
        if not validation_full_only:
            quick_metrics = evaluate_validation(
                model, tokenizer, val_loader, device, config, amp_dtype,
                run_condition_ablation=run_condition_ablation,
                period_names=getattr(valid_dataset, 'validation_period_names', {}),
            )
        if run_large_validation:
            if rank == 0:
                print(
                    f"Running fixed large validation at Segment {epoch_idx + 1}: "
                    f"{len(valid_dataset):,} samples."
                )
            large_metrics = evaluate_validation(
                model, tokenizer, large_val_loader, device, config, amp_dtype,
                run_condition_ablation=False,
                period_names=getattr(valid_dataset, 'validation_period_names', {}),
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

        improved = bool(
            selection_val_loss is not None and selection_val_loss < best_val_loss
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
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(
                f"Validation Forecast/History/Full: {avg_val_forecast_loss:.4f} / "
                f"{avg_val_history_loss:.4f} / {avg_val_full_loss:.4f}"
            )
            for period, metrics in primary_metrics.get('periods', {}).items():
                print(
                    f"Validation {period} Objective/Forecast/History/Full: "
                    f"{metrics['objective_loss']:.6f} / "
                    f"{metrics['forecast_loss']:.6f} / "
                    f"{metrics['history_loss']:.6f} / "
                    f"{metrics['full_sequence_loss']:.6f} "
                    f"({metrics['samples']:,} samples)"
                )
            print(
                "Validation Train Average/Best: "
                f"{epoch_loss_sum / max(epoch_batches, 1):.6f} / "
                f"{best_val_loss:.6f}"
            )
            if selection_val_loss is None:
                print(f"Best selection metric: {selection_metric}=not evaluated")
            else:
                print(
                    f"Best selection metric: {selection_metric}="
                    f"{selection_val_loss:.6f}"
                )
            if large_metrics is not None:
                print(
                    "Large Validation Objective/Forecast/History/Full: "
                    f"{large_metrics['objective_loss']:.6f} / "
                    f"{large_metrics['forecast_loss']:.6f} / "
                    f"{large_metrics['history_loss']:.6f} / "
                    f"{large_metrics['full_sequence_loss']:.6f}"
                )
                for period, metrics in large_metrics.get('periods', {}).items():
                    print(
                        f"Large Validation {period} Objective/Forecast/History/Full: "
                        f"{metrics['objective_loss']:.6f} / "
                        f"{metrics['forecast_loss']:.6f} / "
                        f"{metrics['history_loss']:.6f} / "
                        f"{metrics['full_sequence_loss']:.6f} "
                        f"({metrics['samples']:,} samples)"
                    )
            if ablation_metrics:
                print(
                    "Validation Condition Full/None/Shuffled Forecast: "
                    f"{avg_val_forecast_loss:.6f} / "
                    f"{ablation_metrics['condition_none_forecast_loss']:.6f} / "
                    f"{ablation_metrics['condition_shuffled_forecast_loss']:.6f}; "
                    "Delta Full-None/Full-Shuffled: "
                    f"{ablation_metrics['condition_full_minus_none_forecast_loss']:.6f} / "
                    f"{ablation_metrics['condition_full_minus_shuffled_forecast_loss']:.6f}"
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
                    f"({selection_metric}: {best_val_loss:.4f})"
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
                    samples=len(valid_dataset),
                    batches=large_metrics['batches'],
                    manifest_sha256=getattr(
                        valid_dataset, 'fixed_validation_manifest_sha256', None
                    ),
                    period_metrics=large_metrics.get('periods', {}),
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

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
            best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
            device=str(device),
        )
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    signal.signal(signal.SIGINT, request_safe_stop)
    signal.signal(signal.SIGTERM, request_safe_stop)
    rank, world_size, local_rank = setup_ddp()
    if torch.cuda.is_available():
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

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

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
        'use_size_path': bool(config.get('use_size_path', False)),
        'size_path_input_dim': int(config.get('size_path_input_dim', 4)),
    })
    model = Kronos.from_pretrained(config['pretrained_predictor_path'], **model_kwargs)
    reset_conditioning(model, config)
    configure_trainable_parameters(model, config)
    model.to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

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
