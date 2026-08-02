import os
import sys
import json
import time
from time import gmtime, strftime
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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=True
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


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs,
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        if train_loader.sampler is not None and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, batch in enumerate(train_loader):
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

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
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
        else:
            epochs_without_improvement += 1

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
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
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        patience = int(config.get('early_stopping_patience', 0))
        if patience > 0 and epochs_without_improvement >= patience:
            if rank == 0:
                print(f"Early stopping after {epoch_idx + 1} epochs.")
            break

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
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
