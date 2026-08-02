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
from torch.utils.data import DataLoader
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


STOP_REQUESTED = False


def request_safe_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested; training will checkpoint after the current batch.")


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
    temporary = f'{path}.tmp'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': capture_rng_state(),
        **metadata,
    }, temporary)
    os.replace(temporary, path)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        shuffle=False, num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=False
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        shuffle=False, num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=False
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def configure_trainable_parameters(model, config):
    """Freeze the pretrained trunk and train only the adaptation layers."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in [
        model.sector_emb, model.size_emb, model.size_mlp,
        model.norm, model.dep_layer, model.head,
    ]:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    layer_count = int(config.get('trainable_transformer_layers', 0))
    if layer_count > 0:
        for layer in model.transformer[-layer_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable predictor parameters: {trainable:,}/{total:,} ({trainable / total:.1%})")


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


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

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
    if rank == 0:
        print(
            f"Coverage plan: {train_dataset.total_samples:,} windows, "
            f"{train_dataset.n_samples:,}/segment, {segments_per_coverage} segments/pass, "
            f"{coverage_passes} pass(es), up to {effective_epochs} segments."
        )

    core_model = model.module if isinstance(model, DDP) else model
    condition_params = [
        parameter for name, parameter in core_model.named_parameters()
        if parameter.requires_grad and (
            name.startswith('sector_emb.')
            or name.startswith('size_emb.')
            or name.startswith('size_mlp.')
        )
    ]
    adaptation_params = [
        parameter for name, parameter in core_model.named_parameters()
        if parameter.requires_grad and not (
            name.startswith('sector_emb.')
            or name.startswith('size_emb.')
            or name.startswith('size_mlp.')
        )
    ]
    optimizer_groups = []
    max_lrs = []
    if adaptation_params:
        optimizer_groups.append({'params': adaptation_params, 'lr': config['predictor_learning_rate']})
        max_lrs.append(config['predictor_learning_rate'])
    if condition_params:
        optimizer_groups.append({'params': condition_params, 'lr': config.get('condition_learning_rate', 1e-3)})
        max_lrs.append(config.get('condition_learning_rate', 1e-3))
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler_steps = sum(
        math.ceil(
            math.ceil(segment_sample_count(train_dataset, segment) / world_size)
            / config['batch_size']
        )
        for segment in range(effective_epochs)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=scheduler_steps,
        pct_start=0.03, div_factor=10
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    post_coverage_without_improvement = 0
    dt_result = {}
    batch_idx_global = 0
    start_epoch = 0
    start_step = 0
    resume_path = os.path.join(save_dir, 'checkpoints', 'last_state.pt')

    if config.get('resume_training', False) and os.path.exists(resume_path):
        resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
        saved_effective_epochs = int(resume_state.get('effective_epochs', effective_epochs))
        if saved_effective_epochs != effective_epochs:
            raise ValueError(
                f'Resume plan has {saved_effective_epochs} segments but current plan has {effective_epochs}'
            )
        core_model.load_state_dict(resume_state['model'])
        optimizer.load_state_dict(resume_state['optimizer'])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(resume_state['scheduler'])
        restore_rng_state(resume_state.get('rng_state'))
        start_epoch = int(resume_state['next_epoch'])
        start_step = int(resume_state.get('resume_step', 0))
        best_val_loss = float(resume_state.get('best_val_loss', best_val_loss))
        epochs_without_improvement = int(
            resume_state.get('epochs_without_improvement', 0)
        )
        post_coverage_without_improvement = int(
            resume_state.get('post_coverage_without_improvement', 0)
        )
        batch_idx_global = int(resume_state.get('batch_idx_global', 0))
        if rank == 0:
            print(
                f'Resumed training at coverage segment {start_epoch + 1}, '
                f'step {start_step + 1}.'
            )

    if rank == 0:
        write_progress(
            save_dir,
            status='running',
            phase='initializing',
            current_segment=start_epoch + (1 if start_step else 0),
            current_step=start_step,
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
        epoch_batches = 0
        interrupted_step = 0
        for i, batch in enumerate(train_loader):
            if epoch_idx == start_epoch and i < start_step:
                continue
            batch_x, batch_x_stamp = batch[0], batch[1]
            batch_sector = batch[2] if len(batch) > 2 and config.get('use_sector_features', True) else None
            batch_size_bucket = batch[3] if len(batch) > 3 and config.get('use_size_features', True) else None
            batch_size_percentile = batch[4] if len(batch) > 4 and config.get('use_size_percentile', False) else None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            if batch_sector is not None:
                batch_sector = batch_sector.to(device, non_blocking=True)
            if batch_size_bucket is not None:
                batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
            if batch_size_percentile is not None:
                batch_size_percentile = batch_size_percentile.to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(
                token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                sector_id=batch_sector, size_bucket=batch_size_bucket,
                size_percentile=batch_size_percentile,
            )
            core_model = model.module if isinstance(model, DDP) else model
            loss, s1_loss, s2_loss = core_model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Segment {epoch_idx + 1}/{effective_epochs}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
                write_progress(
                    save_dir,
                    status='stopping' if STOP_REQUESTED else 'running',
                    phase='training',
                    current_segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    current_step=i + 1,
                    total_steps=len(train_loader),
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                    total_train_windows=train_dataset.total_samples,
                    samples_per_segment=len(train_dataset),
                    unique_windows_covered=min(
                        completed_coverage_windows(train_dataset, epoch_idx, coverage_passes)
                        + min((i + 1) * config['batch_size'] * world_size, len(train_dataset)),
                        train_dataset.total_samples * coverage_passes,
                    ),
                    train_loss=epoch_loss_sum / epoch_batches,
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
                    learning_rate=float(lr),
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

            if STOP_REQUESTED:
                interrupted_step = i + 1
                break

        if interrupted_step:
            if rank == 0:
                print(
                    f"Stopping after segment {epoch_idx + 1}, step "
                    f"{interrupted_step}/{len(train_loader)}; saving resume checkpoint."
                )
                write_progress(
                    save_dir,
                    status='stopping',
                    phase='checkpointing',
                    current_segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    current_step=interrupted_step,
                    total_steps=len(train_loader),
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                    total_train_windows=train_dataset.total_samples,
                    samples_per_segment=len(train_dataset),
                    unique_windows_covered=min(
                        completed_coverage_windows(train_dataset, epoch_idx, coverage_passes)
                        + interrupted_step * config['batch_size'] * world_size,
                        train_dataset.total_samples * coverage_passes,
                    ),
                    train_loss=epoch_loss_sum / max(epoch_batches, 1),
                    best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
                    device=str(device),
                )
                save_resume_state(
                    resume_path,
                    core_model,
                    optimizer,
                    scheduler,
                    next_epoch=epoch_idx,
                    resume_step=interrupted_step,
                    best_val_loss=best_val_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    post_coverage_without_improvement=post_coverage_without_improvement,
                    batch_idx_global=batch_idx_global,
                    effective_epochs=effective_epochs,
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                )
                write_progress(
                    save_dir,
                    status='stopped',
                    phase='complete',
                    current_segment=epoch_idx + 1,
                    total_segments=effective_epochs,
                    current_step=interrupted_step,
                    total_steps=len(train_loader),
                    segments_per_coverage=segments_per_coverage,
                    coverage_passes=coverage_passes,
                    total_train_windows=train_dataset.total_samples,
                    samples_per_segment=len(train_dataset),
                    unique_windows_covered=min(
                        completed_coverage_windows(train_dataset, epoch_idx, coverage_passes)
                        + interrupted_step * config['batch_size'] * world_size,
                        train_dataset.total_samples * coverage_passes,
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

        start_step = 0

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
                total_steps=len(val_loader),
                segments_per_coverage=segments_per_coverage,
                coverage_passes=coverage_passes,
                total_train_windows=train_dataset.total_samples,
                samples_per_segment=len(train_dataset),
                unique_windows_covered=completed_coverage_windows(
                    train_dataset, epoch_idx + 1, coverage_passes
                ),
                train_loss=epoch_loss_sum / max(epoch_batches, 1),
                best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
                device=str(device),
            )
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_x_stamp = batch[0], batch[1]
                batch_sector = batch[2] if len(batch) > 2 and config.get('use_sector_features', True) else None
                batch_size_bucket = batch[3] if len(batch) > 3 and config.get('use_size_features', True) else None
                batch_size_percentile = batch[4] if len(batch) > 4 and config.get('use_size_percentile', False) else None
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                if batch_sector is not None:
                    batch_sector = batch_sector.to(device, non_blocking=True)
                if batch_size_bucket is not None:
                    batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
                if batch_size_percentile is not None:
                    batch_size_percentile = batch_size_percentile.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    sector_id=batch_sector, size_bucket=batch_size_bucket,
                    size_percentile=batch_size_percentile,
                    use_teacher_forcing=True,
                    s1_targets=token_out[0],
                )
                core_model = model.module if isinstance(model, DDP) else model
                val_loss, _, _ = core_model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        # Reduce validation metrics
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0

        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if epoch_idx + 1 > minimum_coverage_segments:
                post_coverage_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch_idx + 1 > minimum_coverage_segments:
                post_coverage_without_improvement += 1

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Coverage Segment {epoch_idx + 1}/{effective_epochs} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            if improved:
                save_path = f"{save_dir}/checkpoints/best_model"
                core_model = model.module if isinstance(model, DDP) else model
                model_config = dict(getattr(core_model, '_hub_mixin_config', {}) or {})
                model_config.update({
                    'num_sectors': int(config.get('num_sectors', 0)),
                    'num_size_buckets': int(config.get('num_size_buckets', 0)),
                    'context_layer': int(config.get('context_layer', 0)),
                    'use_size_percentile': bool(config.get('use_size_percentile', False)),
                    'size_mlp_hidden_dim': int(config.get('size_mlp_hidden_dim', 64)),
                })
                core_model.save_pretrained(save_path, config=model_config)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
            save_resume_state(
                resume_path,
                core_model,
                optimizer,
                scheduler,
                next_epoch=epoch_idx + 1,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                post_coverage_without_improvement=post_coverage_without_improvement,
                batch_idx_global=batch_idx_global,
                resume_step=0,
                effective_epochs=effective_epochs,
                segments_per_coverage=segments_per_coverage,
                coverage_passes=coverage_passes,
            )
            completed_coverage_segments = epoch_idx + 1
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
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                device=str(device),
            )
            append_metric(
                save_dir,
                type='validation',
                segment=completed_coverage_segments,
                total_segments=effective_epochs,
                step=0,
                loss=float(avg_val_loss),
                best_loss=float(best_val_loss),
                train_average=epoch_loss_sum / max(epoch_batches, 1),
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

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
    if rank == 0:
        display_segment = dt_result.get(
            'partial_segment', dt_result['completed_segments']
        )
        display_step = dt_result.get('partial_step', 0)
        unique_windows_covered = completed_coverage_windows(
            train_dataset, dt_result['completed_segments'], coverage_passes
        )
        if display_step:
            unique_windows_covered = min(
                unique_windows_covered
                + display_step * config['batch_size'] * world_size,
                train_dataset.total_samples * coverage_passes,
            )
        write_progress(
            save_dir,
            status=dt_result['status'],
            phase='complete',
            current_segment=display_segment,
            total_segments=effective_epochs,
            current_step=display_step,
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
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    model = Kronos.from_pretrained(
        config['pretrained_predictor_path'],
        num_sectors=int(config.get('num_sectors', 0)),
        num_size_buckets=int(config.get('num_size_buckets', 0)),
        context_layer=int(config.get('context_layer', 0)),
        use_size_percentile=bool(config.get('use_size_percentile', False)),
        size_mlp_hidden_dim=int(config.get('size_mlp_hidden_dim', 64)),
    )
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
    main(config_instance.__dict__)
