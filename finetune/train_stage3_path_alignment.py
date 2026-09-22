import argparse, json, os, time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from model.kronos import Kronos, KronosTokenizer
from finetune.dataset import QlibDataset
from finetune.stage3_ce_rank import DEFAULT_FORECAST_HORIZON_WEIGHTS, Stage3CERankConfig
from finetune.stage3_path_alignment import PathAlignmentConfig
from finetune.stage3_ar_vol import AR_METRIC_KEYS, ARVolConfig
from finetune.stage3_vol_alignment import VOL_METRIC_KEYS, VolAlignmentConfig
from finetune.stage3_training_model import Stage3TrainingModel, gradient_metrics
from finetune.stage3_trainable_mask import (
    KNOWN_MASKS, apply_trainable_mask, assert_frozen_parameters_unchanged,
    snapshot_frozen_parameters,
)


def parse_horizon_weights(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(',') if item.strip())
    if not values:
        raise ValueError('forecast horizon weights must not be empty')
    return values

def setup_dist():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0

def is_main(rank):
    return rank == 0

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def persist_state(core, opt, out, step, segment, ds, experiment, world, rank):
    rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
    rank_rng = [None] * world
    if world > 1: dist.all_gather_object(rank_rng, rng)
    else: rank_rng[0] = rng
    if rank == 0:
        state = core.checkpoint_state(opt, step, segment)
        state.update(next_epoch=segment, resume_step=0, rank_rng_states=rank_rng,
                     scheduler={'type': 'fixed', 'lr': opt.param_groups[0]['lr']},
                     scaler=None, experiment_manifest=experiment,
                     coverage=ds.coverage_state(segment-1) if segment else {'consumed_samples': 0})
        temp = out / 'checkpoints/last_state.pt.tmp'
        torch.save(state, temp); os.replace(temp, out / 'checkpoints/last_state.pt')
        core.predictor.save_pretrained(out / 'checkpoints/last_model')

def record_metrics(out, values):
    with (out / 'metrics.jsonl').open('a', buffering=1) as handle:
        handle.write(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), **values}) + '\n')

def validate_resume(state, experiment, summary, world, lr, seed):
    """Fail closed on a different objective/data/run or a partial segment."""
    prior = state['experiment_manifest']
    for key in ('sha256', 'run_id', 'loss', 'lambda_path', 'lambda_vol', 'lambda_rank', 'history_weight',
                'forecast_horizon_weights', 'ce_rank', 'trainable_mask', 'huber_delta', 'top_k',
                'candidates', 'vol_temperature', 'vol_top_p', 'vol_samples', 'ema_decay',
                'dependency_causal', 'amp', 'global_batch',
                'batch_per_gpu', 'lookback', 'horizon', 'validation_samples'):
        default = 'all' if key == 'trainable_mask' else 0.0 if key == 'lambda_vol' else None
        if prior.get(key, default) != experiment.get(key, default):
            raise ValueError(f'Resume manifest mismatch: {key}')
    if prior['seed'] != seed or experiment['seed'] != seed:
        raise ValueError('Resume coverage seed changed')
    if prior['lr'] != lr or state['scheduler'] != {'type': 'fixed', 'lr': lr}:
        raise ValueError('Resume scheduler/LR changed')
    if any(group['lr'] != lr for group in state['optimizer']['param_groups']):
        raise ValueError('Resume optimizer LR mismatch')
    segment = state['segment']
    if state['next_epoch'] != segment or state['resume_step'] != 0:
        raise ValueError('Not a complete segment boundary')
    if len(state['rank_rng_states']) != world:
        raise ValueError('Resume world size changed')
    rows = summary['segments']
    if [r['segment'] for r in rows] != list(range(1, segment + 1)):
        raise ValueError('Incomplete historical segment metrics')
    if not rows or rows[-1]['step'] != state['step']:
        raise ValueError('Resume global step mismatch')
    if state['coverage']['unique_samples_covered'] != segment * 20000:
        raise ValueError('Resume coverage cursor mismatch')
    return min(r['validation_objective'] for r in rows)

def _vol_enabled(core):
    return getattr(core, 'vol_config', None) is not None and core.vol_config.weight != 0


def _grad_norm(grads):
    terms = [g.detach().float().square().sum() for g in grads if g is not None]
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).sum().sqrt()


def vol_vs_ce_grad_norm_ratio(core, token_loss, vol_loss):
    if not token_loss.requires_grad:
        raise RuntimeError('token loss has no grad_fn')
    if not vol_loss.requires_grad:
        raise RuntimeError('vol loss has no grad_fn; mixture weights were detached')
    params = [p for p in core.predictor.parameters() if p.requires_grad]
    ce_grads = torch.autograd.grad(token_loss, params, retain_graph=True, allow_unused=True)
    vol_grads = torch.autograd.grad(vol_loss, params, retain_graph=True, allow_unused=True)
    ce_n = _grad_norm(ce_grads).clamp_min(1e-12)
    return _grad_norm(vol_grads) / ce_n


def evaluate(model, loader, device, world, rank=0, log_interval=50):
    was_training = model.training
    model.eval()
    core = model.module if isinstance(model, DDP) else model
    n = torch.zeros((), device=device, dtype=torch.float64)
    keys = ('token_loss', 'raw_path_loss', 'total_loss', 'top16_joint_mass',
            's1_entropy', 's2_conditional_entropy_topk_s1', 'horizon_mae',
            'prediction_mean_hf', 'prediction_second_moment_hf',
            'target_mean_hf', 'target_second_moment_hf')
    if _vol_enabled(core):
        keys = keys + VOL_METRIC_KEYS
    sums = {k: torch.zeros((core.horizon, 6) if k.endswith('_hf') else
                          (core.horizon,) if k == 'horizon_mae' else (),
                          device=device, dtype=torch.float64) for k in keys}
    max_residual = torch.zeros((), device=device)
    with torch.no_grad():
        for batch_index, vb in enumerate(loader, 1):
            vx, vs = vb[0].to(device), vb[1].to(device)
            vsec = vb[2].to(device) if len(vb) > 2 else None
            vpct = vb[4].to(device) if len(vb) > 4 else None
            means = stds = None
            if _vol_enabled(core):
                means = vb[-2].to(device)
                stds = vb[-1].to(device)
            _, metrics = core(vx, vs, sector_id=vsec, size_percentile=vpct,
                              feature_means=means, feature_stds=stds)
            for key in keys:
                sums[key] += metrics[key].double() * len(vx)
            max_residual = torch.maximum(max_residual, metrics['max_residual'])
            n += len(vx)
            if rank == 0 and (batch_index == 1 or batch_index % log_interval == 0):
                print(json.dumps({'phase': 'validation', 'batches_rank0': batch_index,
                                  'samples_rank0': int(n)}), flush=True)
    if world > 1:
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        for value in sums.values(): dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_residual, op=dist.ReduceOp.MAX)
    if not n.item(): raise RuntimeError('Empty validation set')
    result = {key: (value / n).tolist() for key, value in sums.items()}
    for prefix in ('prediction', 'target'):
        variance = (sums[prefix + '_second_moment_hf'] / n -
                    (sums[prefix + '_mean_hf'] / n).square()).clamp_min(0)
        result[prefix + '_variance_horizon'] = variance.mean(-1).tolist()
    result.update(samples=int(n), max_residual=float(max_residual))
    model.train(was_training)
    return result

def main(a):
    import swanlab
    rank, world, local = setup_dist()
    torch.manual_seed(a.seed + rank)
    device = torch.device(f'cuda:{local}')
    milestones = {int(item) for item in str(a.milestone_segments).split(',') if item.strip()}
    if is_main(rank):
        print('initialization=stage3_from_c2_best_model_only', flush=True)
        print('parent_model=' + str(a.model_dir), flush=True)
        print('optimizer_state=' + ('resume' if a.resume_state else 'reset'), flush=True)
        print(f'lambda_path={a.lambda_path}', flush=True)
        print(f'lambda_vol={a.lambda_vol}', flush=True)
        print(f'ar_vol={a.ar_vol}', flush=True)
        if a.lambda_vol or a.ar_vol:
            print(f'vol_temperature={a.vol_temperature}', flush=True)
            print(f'vol_top_p={a.vol_top_p}', flush=True)
            print(f'vol_samples={a.vol_samples}', flush=True)
        print(f'ce_rank={a.ce_rank}', flush=True)
        if a.ce_rank:
            print(f'lambda_rank={a.lambda_rank}', flush=True)
            print(f'history_weight={a.history_weight}', flush=True)
            print(f'forecast_horizon_weights={a.forecast_horizon_weights}', flush=True)
        print(f'milestone_segments={sorted(milestones)}', flush=True)
        print(f'trainable_mask={a.trainable_mask}', flush=True)
        print(f'world_size={world}', flush=True)
        print('validation_mode=full', flush=True)
        print('target_slice=x[:,120:130]', flush=True)
        print(f'chunk1_max_segments={a.segments}', flush=True)
        print(f'chunk1_max_runtime_seconds={a.max_runtime_seconds}', flush=True)
    predictor = Kronos.from_pretrained(a.model_dir).to(device)
    tok = KronosTokenizer.from_pretrained(a.tokenizer_dir).to(device).eval()
    ce_rank_config = Stage3CERankConfig(
        enabled=a.ce_rank,
        lambda_rank=a.lambda_rank,
        history_weight=a.history_weight,
        forecast_horizon_weights=parse_horizon_weights(a.forecast_horizon_weights),
    )
    if a.ce_rank and a.lambda_path != 0:
        raise ValueError('P1 CE+rank requires lambda_path=0')
    if a.ar_vol and (a.lambda_vol <= 0 or a.lambda_path != 0 or a.ce_rank):
        raise ValueError('autoregressive vol requires lambda_vol>0, lambda_path=0, and ce_rank disabled')
    if a.lambda_vol and not a.ar_vol and (a.lambda_path != 0 or a.ce_rank):
        raise ValueError('vol alignment requires lambda_path=0 and ce_rank disabled')
    model = Stage3TrainingModel(
        predictor, tok,
        config=PathAlignmentConfig(weight=0.0 if a.ar_vol else a.lambda_path),
        ce_rank_config=ce_rank_config,
        vol_config=VolAlignmentConfig(
            weight=0.0 if a.ar_vol else a.lambda_vol,
            temperature=a.vol_temperature,
            top_p=a.vol_top_p,
            candidates=a.vol_samples,
        ),
        ar_vol=a.ar_vol,
        ar_config=ARVolConfig(
            weight=a.lambda_vol,
            temperature=a.vol_temperature,
            top_p=a.vol_top_p,
            samples=a.vol_samples,
        ),
    ).to(device)
    freeze_audit = apply_trainable_mask(model.predictor, a.trainable_mask)
    frozen_snapshot = snapshot_frozen_parameters(model.predictor)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local,
                    find_unused_parameters=False, broadcast_buffers=False)
    core = model.module if isinstance(model, DDP) else model
    raw = core.predictor
    trainable_params = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError(f'Trainable mask {a.trainable_mask} selected no parameters')
    run = None
    if is_main(rank):
        api_key = os.environ.get('SWANLAB_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('SWANLAB_API_KEY missing; refusing to start training without dashboard')
        swanlab.login(api_key=api_key)
        run_id = os.environ.get('SWANLAB_RUN_ID', 'small_0.1_stage3_joint_path_alignment_from_c2_best_v2').strip()
        experiment_name = os.environ.get('SWANLAB_EXPERIMENT_NAME', run_id).strip()
        if run_id == 'small_0.1_stage3_path_alignment_from_c2_best' or not run_id:
            raise RuntimeError('Refusing to reuse aborted Stage3 C1 dashboard')
        if a.lambda_vol and 'joint_path_alignment_from_c2_best' in run_id:
            raise RuntimeError('Refusing to reuse Stage3 C3 path-alignment dashboard for vol calibration')
        if a.lambda_vol and run_id == 'small_0.1_stage3_vol_cal_from_c2_best_v1':
            raise RuntimeError('Refusing to reuse the T=1 top-16 vol-cal dashboard')
        if a.ar_vol and 'ar_vol_from_c2' not in run_id:
            raise RuntimeError('Autoregressive vol requires its own SwanLab run id')
        if a.ar_vol and (
            'joint_path_alignment_from_c2_best' in run_id
            or run_id == 'small_0.1_stage3_vol_cal_from_c2_best_v1'
            or run_id == 'small_0.1_stage3_vol_prod_decode_from_c2_v1'
        ):
            raise RuntimeError('Refusing to reuse a teacher-forced vol or C3 dashboard for autoregressive vol')
        if (not a.ar_vol) and 'ar_vol_from_c2' in run_id:
            raise RuntimeError('Refusing to reuse the autoregressive vol dashboard')
        # Match C2: fixed run id + resume=allow so multi-chunk handoffs stay on one dashboard.
        run = swanlab.init(
            id=run_id,
            resume='allow',
            project='finance',
            workspace='roc_fu',
            experiment_name=experiment_name,
            config={'lr': a.lr, 'batch_per_gpu': a.batch, 'global_batch': a.batch * world,
                    'segments': a.segments, 'max_runtime_seconds': a.max_runtime_seconds,
                    'top_k': 16, 'candidates': a.vol_samples if a.lambda_vol else 16, 'parent': str(a.model_dir),
                    'lambda_path': a.lambda_path, 'lambda_vol': a.lambda_vol, 'ar_vol': a.ar_vol,
                    'vol_temperature': a.vol_temperature, 'vol_top_p': a.vol_top_p,
                    'vol_samples': a.vol_samples,
                    'milestone_segments': sorted(milestones),
                    'validation': 'full_causal',
                    'validation_objective': (
                        f'weighted_ce+{a.history_weight:g}*history+{a.lambda_rank:g}*rank'
                        if a.ce_rank else
                        f'token_ce+{a.lambda_vol:g}*ar_log_vol_reinforce'
                        if a.ar_vol else
                        f'token_ce+{a.lambda_vol:g}*log_vol_huber'
                        if a.lambda_vol else
                        f'token_ce+{a.lambda_path:g}*raw_path_huber'
                    ),
                    'ce_rank': a.ce_rank,
                    'lambda_rank': a.lambda_rank,
                    'history_weight': a.history_weight,
                    'forecast_horizon_weights': a.forecast_horizon_weights,
                    'trainable_mask': a.trainable_mask,
                    'optimizer': 'resume' if a.resume_state else 'fresh', 'dependency_causal': True,
                    'ema': 'global_batch_detached', 'coverage_seed': a.seed,
                    'nproc': world, 'chunk': a.chunk},
            mode='cloud',
        )
        run_url = getattr(run, 'url', getattr(run, 'web_url', ''))
        print('SWANLAB_RUN_URL=' + str(run_url), flush=True)
        if not run_url:
            raise RuntimeError('SwanLab init returned no run URL; refusing to start training')
    model.train()
    os.environ['KRONOS_COVERAGE_SEED'] = str(a.seed)
    os.environ['KRONOS_TRAIN_SAMPLES_PER_SEGMENT'] = '20000'
    os.environ['KRONOS_STAGE3_RANK_LOSS'] = '1' if a.ce_rank else '0'
    os.environ['KRONOS_STAGE3_VOL_LOSS'] = '1' if (a.lambda_vol or a.ar_vol) else '0'
    print(json.dumps({'phase': 'load_training_dataset', 'rank': rank}), flush=True)
    ds = QlibDataset('train')
    train_sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, sampler=train_sampler, num_workers=0)
    os.environ['KRONOS_VALIDATION_SAMPLES'] = '0'
    print(json.dumps({'phase': 'load_full_validation_dataset', 'rank': rank}), flush=True)
    val_ds = QlibDataset('val')
    if len(val_ds) != val_ds.total_samples or val_ds.total_samples < 100000:
        raise RuntimeError(f'Validation set unexpectedly small: {len(val_ds)}; full validation is required')
    if is_main(rank):
        print('validation_samples=' + str(val_ds.total_samples), flush=True)
        print('dependency_causal=True; validation_dropout=False; ddp_complete_objective=True; global_step=0', flush=True)
    if world > 1: dist.barrier()
    # Exact coverage, no DistributedSampler padding/duplicate validation rows.
    val_shard = Subset(val_ds, range(rank, len(val_ds), world))
    val_loader = DataLoader(val_shard, batch_size=a.batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(trainable_params, lr=a.lr, weight_decay=0.01)
    out = Path(a.output_dir)
    if is_main(rank):
        (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'freeze_audit.json', freeze_audit)
        print(json.dumps({
            'phase': 'trainable_mask',
            **{key: value for key, value in freeze_audit.items() if key != 'frozen_names'},
        }), flush=True)
    if world > 1: dist.barrier()
    step = 0
    summary = {'segments': [], 'chunk': a.chunk, 'max_runtime_seconds': a.max_runtime_seconds}
    experiment = json.loads((out / 'experiment_manifest.json').read_text())
    if experiment.get('trainable_mask', 'all') != a.trainable_mask:
        raise ValueError('Trainable mask does not match experiment_manifest')
    start_segment = 0
    best_val = float('inf')
    if a.baseline_before_resume:
        # Still the untouched C2 model. Evaluation must not update optimizer/EMA.
        baseline = evaluate(model, val_loader, device, world, rank)
        if baseline['samples'] != len(val_ds): raise RuntimeError('Baseline validation count mismatch')
        if is_main(rank):
            atomic_json(out / 'c2_causal_baseline.json', {'parent': str(a.model_dir), 'validation': baseline})
            print(json.dumps({'phase': 'c2_causal_baseline', 'validation': baseline}), flush=True)
    if a.resume_state:
        state = torch.load(a.resume_state, map_location='cpu', weights_only=True)
        summary = json.loads((out / 'summary.json').read_text())
        best_val = validate_resume(state, experiment, summary, world, a.lr, a.seed)
        step, start_segment = core.load_checkpoint_state(state, opt)
        if start_segment >= a.segments: raise ValueError('No new segments requested')
        if state['coverage']['total_samples'] != ds.total_samples:
            raise ValueError('Training window pool changed')
        if a.baseline_before_resume:
            resumed_baseline = evaluate(model, val_loader, device, world, rank)
            if is_main(rank):
                atomic_json(out / 'resume_causal_baseline.json', {'segment': start_segment, 'validation': resumed_baseline})
        rng = state['rank_rng_states'][rank]
        torch.set_rng_state(rng['cpu'].cpu()); torch.cuda.set_rng_state(rng['cuda'].cpu(), device)
        summary.update(chunk=a.chunk, max_runtime_seconds=a.max_runtime_seconds)
        summary.pop('final_result', None)
        if is_main(rank):
            print(json.dumps({'phase': 'resume_ready', 'next_segment': start_segment + 1,
                              'global_step': step, 'best_validation_objective': best_val,
                              'coverage_seed': a.seed, 'resume_state': a.resume_state}), flush=True)
    elif is_main(rank):
        raw.save_pretrained(out / 'checkpoints/best_model')
        atomic_json(out / 'checkpoints/best_model/best_metric.json',
                    {'segment': 0, 'status': 'initial_unvalidated', 'parent': str(a.model_dir)})
        atomic_json(out / 'progress.json', {'completed_segments': 0, 'next_epoch': 0, 'status': 'running'})
        atomic_json(out / 'summary.json', summary)
    if not a.resume_state:
        persist_state(core, opt, out, 0, 0, ds, experiment, world, rank)
    per_rank_target = (20000 + world - 1) // world
    run_started = time.time()
    for seg in range(start_segment + 1, a.segments + 1):
        if train_sampler is not None: train_sampler.set_epoch(seg - 1)
        ds.set_epoch_seed(seg - 1); seen = 0; started = time.time()
        for batch in loader:
            x, stamp = batch[0].to(device), batch[1].to(device)
            sec = batch[2].to(device) if len(batch) > 2 else None
            pct = batch[4].to(device) if len(batch) > 4 else None
            date_ids = means = stds = None
            if a.ce_rank:
                date_ids = batch[-3].to(device)
                means = batch[-2].to(device)
                stds = batch[-1].to(device)
            elif a.lambda_vol or a.ar_vol:
                means = batch[-2].to(device)
                stds = batch[-1].to(device)
            opt.zero_grad(set_to_none=True)
            # AR sampling writes .grad inside the forward. Keep DDP from
            # reducing that partial graph; all-reduce the summed grads after.
            sync_off = model.no_sync() if a.ar_vol and isinstance(model, DDP) else nullcontext()
            with sync_off:
                loss, metrics = model(
                    x, stamp, sector_id=sec, size_percentile=pct,
                    date_ids=date_ids, feature_means=means, feature_stds=stds,
                )
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite Stage3 loss')
                will_log = step + 1 == 1 or (step + 1) % a.log_interval == 0
                vol_ce_ratio = None
                if a.lambda_vol and not a.ar_vol and will_log:
                    vol_ce_ratio = vol_vs_ce_grad_norm_ratio(
                        core, core._live_token_loss, core._live_vol_loss,
                    )
                loss.backward()
            if a.ar_vol and world > 1:
                for parameter in trainable_params:
                    if parameter.grad is None:
                        raise RuntimeError('AR vol left a trainable parameter without a gradient')
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world)
            group_grads = gradient_metrics(raw)
            grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); step += 1; seen += len(x)
            if will_log:
                scalar_keys = ('token_loss', 'raw_path_loss', 'normalized_path_loss', 'total_loss',
                               'top16_joint_mass', 's1_entropy', 's2_conditional_entropy_topk_s1', 'max_residual')
                if a.ar_vol:
                    scalar_keys = scalar_keys + AR_METRIC_KEYS
                elif a.lambda_vol:
                    scalar_keys = scalar_keys + VOL_METRIC_KEYS
                if a.ce_rank:
                    scalar_keys = ('ce_objective', 'history_loss', 'weighted_forecast_loss', 'rank_loss',
                                   'total_loss', 's1_entropy')
                    tensor_keys = [key for key in scalar_keys if key in metrics]
                    logged = torch.stack([metrics[key] for key in tensor_keys])
                    if world > 1:
                        dist.all_reduce(logged)
                        logged /= world
                    log = dict(zip(tensor_keys, logged.tolist()))
                    for key in ('unique_signal_dates', 'stocks_per_date_mean', 'pair_count', 'return_dispersion'):
                        if key in metrics:
                            log[key] = float(metrics[key])
                else:
                    logged = torch.stack([metrics[key] for key in scalar_keys])
                    if world > 1:
                        dist.all_reduce(logged)
                        logged /= world
                    log = dict(zip(scalar_keys, logged.tolist()))
                if is_main(rank):
                    log.update({key: float(value) for key, value in group_grads.items()})
                    log.update(phase='train', segment=seg, step=step, grad_norm=float(grad))
                    if vol_ce_ratio is not None:
                        log['vol_vs_ce_grad_norm_ratio'] = float(vol_ce_ratio)
                    print(json.dumps(log), flush=True)
                    record_metrics(out, log)
                    if run is not None:
                        run.log({'train/' + k: v for k, v in log.items() if isinstance(v, (int, float))}, step=step)
            if seen >= per_rank_target: break
        validation = evaluate(model, val_loader, device, world, rank)
        if validation['samples'] != len(val_ds): raise RuntimeError('Full validation count mismatch')
        val_loss = validation['total_loss']
        row = {'segment': seg, 'step': step, 'samples_per_rank': seen,
               'global_samples_approx': seen * world,
               'last_batch_token_loss_rank0': float(metrics['token_loss']),
               'validation_objective': val_loss, 'validation': validation,
               'grad_norm': float(grad), 'elapsed_sec': time.time() - started}
        if is_main(rank):
            if val_loss < best_val:
                best_val = val_loss; raw.save_pretrained(out / 'checkpoints/best_model')
                atomic_json(out / 'checkpoints/best_model/best_metric.json',
                            {'segment': seg, 'step': step, 'validation_objective': val_loss,
                             'definition': (
                                 f'weighted_ce+{a.history_weight:g}*history+{a.lambda_rank:g}*rank'
                                 if a.ce_rank else
                                 f'causal_token_ce+{a.lambda_vol:g}*ar_log_vol_reinforce'
                                 if a.ar_vol else
                                 f'causal_token_ce+{a.lambda_vol:g}*log_vol_huber'
                                 if a.lambda_vol else
                                 f'causal_token_ce+{a.lambda_path:g}*raw_path_huber'
                             )})
            if seg in milestones:
                milestone = out / f'checkpoints/milestone_seg{seg:02d}'
                raw.save_pretrained(milestone)
                atomic_json(milestone / 'milestone.json',
                            {'segment': seg, 'step': step, 'validation_objective': val_loss,
                             'lambda_path': a.lambda_path, 'lambda_vol': a.lambda_vol})
                print(json.dumps({'phase': 'milestone_saved', 'segment': seg, 'step': step,
                                  'path': str(milestone)}), flush=True)
            summary['segments'].append(row)
            record_metrics(out, {'phase': 'segment_complete', **row})
            if run is not None:
                log = {'validation/' + key: value for key, value in validation.items() if isinstance(value, (int, float))}
                log.update({f'validation/mae_h{i+1}': value for i, value in enumerate(validation['horizon_mae'])})
                for prefix in ('prediction', 'target'):
                    log.update({f'validation/{prefix}_variance_h{i+1}': value for i, value in enumerate(validation[prefix + '_variance_horizon'])})
                run.log(log, step=step)
            atomic_json(out / 'progress.json', {'segment': seg, 'step': step, 'samples_per_rank': seen, 'status': 'running'})
            atomic_json(out / 'summary.json', summary)
            print(json.dumps(row), flush=True)
        persist_state(core, opt, out, step, seg, ds, experiment, world, rank)
        if is_main(rank):
            assert_frozen_parameters_unchanged(raw, frozen_snapshot)
        stop_for_time = False
        if is_main(rank) and (time.time() - run_started) >= a.max_runtime_seconds:
            stop_for_time = True
            print(json.dumps({'phase': 'stop_max_runtime', 'completed_segment': seg, 'elapsed_sec': time.time() - run_started, 'max_runtime_seconds': a.max_runtime_seconds}), flush=True)
        flag = torch.tensor([1 if stop_for_time else 0], device=device)
        if world > 1:
            dist.broadcast(flag, src=0)
            dist.barrier()
        if int(flag.item()) == 1:
            break
    if is_main(rank):
        summary['final_result'] = {'completed_segments': len(summary['segments']), 'best_validation_objective': best_val, 'status': 'completed'}
        atomic_json(out / 'progress.json', {'completed_segments': len(summary['segments']), 'next_epoch': seg, 'step': step, 'status': 'completed'})
        atomic_json(out / 'summary.json', summary)
        if run is not None: run.finish()
    if world > 1: dist.destroy_process_group()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--segments', type=int, default=1)
    p.add_argument('--max-runtime-seconds', type=int, default=36000)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--lambda-path', type=float, default=0.05)
    p.add_argument('--lambda-vol', type=float, default=0.0)
    p.add_argument('--vol-temperature', type=float, default=0.65)
    p.add_argument('--vol-top-p', type=float, default=0.8)
    p.add_argument('--vol-samples', type=int, default=5)
    p.add_argument('--ar-vol', action='store_true')
    p.add_argument('--ce-rank', action='store_true')
    p.add_argument('--lambda-rank', type=float, default=0.05)
    p.add_argument('--history-weight', type=float, default=0.02)
    p.add_argument('--forecast-horizon-weights', default=','.join(str(v) for v in DEFAULT_FORECAST_HORIZON_WEIGHTS))
    p.add_argument('--milestone-segments', default='')
    p.add_argument('--trainable-mask', default='all', choices=KNOWN_MASKS)
    p.add_argument('--resume-state', default='')
    p.add_argument('--chunk', default='c1')
    p.add_argument('--baseline-before-resume', action='store_true')
    main(p.parse_args())
