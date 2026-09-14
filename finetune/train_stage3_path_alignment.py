import argparse, json, os, time, subprocess, sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--progress-bar', 'off', 'swanlab'], check=True)
import swanlab
from model.kronos import Kronos, KronosTokenizer
from finetune.dataset import QlibDataset
from finetune.stage3_path_alignment import PathAlignmentConfig, DetachedLossEMA, compute_path_alignment_loss

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

def evaluate(model, tok, loader, device, cfg, world):
    model.eval(); total = torch.zeros((), device=device); n = torch.zeros((), device=device)
    raw = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        for vb in loader:
            vx, vs = vb[0].to(device), vb[1].to(device)
            vsec = vb[2].to(device) if len(vb) > 2 else None
            vpct = vb[4].to(device) if len(vb) > 4 else None
            vs1, vs2 = tok.encode(vx, half=True)
            vlogits = raw(vs1[:, :-1], vs2[:, :-1], vs[:, :-1], use_teacher_forcing=True, s1_targets=vs1[:, 1:], sector_id=vsec, size_percentile=vpct)
            vce = raw.head.compute_loss(vlogits[0][:, -10:], vlogits[1][:, -10:], vs1[:, 1:][:, -10:], vs2[:, 1:][:, -10:])[0]
            vpa, _ = compute_path_alignment_loss(tok, vlogits[0][:, -10:], vlogits[1][:, -10:], vx[:, 120:130], cfg, None)
            total += (vce + vpa).detach() * len(vx); n += len(vx)
    if world > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM); dist.all_reduce(n, op=dist.ReduceOp.SUM)
    model.train()
    return float(total / n.clamp_min(1))

def main(a):
    rank, world, local = setup_dist()
    torch.manual_seed(a.seed + rank)
    device = torch.device(f'cuda:{local}')
    if is_main(rank):
        print('initialization=stage3_from_c2_best_model_only', flush=True)
        print('parent_model=' + str(a.model_dir), flush=True)
        print('optimizer_state=reset', flush=True)
        print(f'world_size={world}', flush=True)
        print('validation_mode=full', flush=True)
        print('target_slice=x[:,120:130]', flush=True)
        print(f'chunk1_max_segments={a.segments}', flush=True)
        print(f'chunk1_max_runtime_seconds={a.max_runtime_seconds}', flush=True)
    model = Kronos.from_pretrained(a.model_dir).to(device)
    tok = KronosTokenizer.from_pretrained(a.tokenizer_dir).to(device).eval()
    for p in tok.parameters(): p.requires_grad_(False)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=False)
    raw = model.module if isinstance(model, DDP) else model
    run = None
    if is_main(rank):
        api_key = os.environ.get('SWANLAB_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('SWANLAB_API_KEY missing; refusing to start training without dashboard')
        swanlab.login(api_key=api_key)
        run = swanlab.init(project='finance', workspace='roc_fu', experiment_name='small_0.1_stage3_path_alignment_from_c2_best', config={'lr': a.lr, 'batch_per_gpu': a.batch, 'global_batch': a.batch * world, 'segments': a.segments, 'max_runtime_seconds': a.max_runtime_seconds, 'top_k': 16, 'candidates': 16, 'parent': 'c2_best', 'validation': 'full', 'nproc': world, 'chunk': 'c1'}, mode='cloud')
        run_url = getattr(run, 'url', getattr(run, 'web_url', ''))
        print('SWANLAB_RUN_URL=' + str(run_url), flush=True)
        if not run_url:
            raise RuntimeError('SwanLab init returned no run URL; refusing to start training')
    model.train()
    ds = QlibDataset('train')
    train_sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, sampler=train_sampler, num_workers=0)
    os.environ['KRONOS_VALIDATION_SAMPLES'] = '0'
    val_ds = QlibDataset('val')
    if is_main(rank):
        print('validation_samples=' + str(val_ds.total_samples), flush=True)
        if len(val_ds) != val_ds.total_samples or val_ds.total_samples < 100000:
            raise RuntimeError(f'Validation set unexpectedly small: {len(val_ds)}; full validation is required')
    if world > 1: dist.barrier()
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    val_loader = DataLoader(val_ds, batch_size=a.batch, shuffle=False, sampler=val_sampler, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    ema = DetachedLossEMA(0.99); cfg = PathAlignmentConfig(16, 16, 0.05, 0.02, 0.99)
    out = Path(a.output_dir)
    if is_main(rank): out.mkdir(parents=True, exist_ok=True)
    if world > 1: dist.barrier()
    step = 0
    summary = {'segments': [], 'chunk': 'c1', 'max_runtime_seconds': a.max_runtime_seconds}
    best_val = float('inf')
    per_rank_target = (20000 + world - 1) // world
    run_started = time.time()
    for seg in range(1, a.segments + 1):
        if train_sampler is not None: train_sampler.set_epoch(seg - 1)
        ds.set_epoch_seed(seg - 1); seen = 0; started = time.time()
        for batch in loader:
            x, stamp = batch[0].to(device), batch[1].to(device)
            sec = batch[2].to(device) if len(batch) > 2 else None
            pct = batch[4].to(device) if len(batch) > 4 else None
            with torch.no_grad(): s1, s2 = tok.encode(x, half=True)
            logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1], use_teacher_forcing=True, s1_targets=s1[:, 1:], sector_id=sec, size_percentile=pct)
            ce = raw.head.compute_loss(logits[0][:, -10:], logits[1][:, -10:], s1[:, 1:][:, -10:], s2[:, 1:][:, -10:])[0]
            pa, metrics = compute_path_alignment_loss(tok, logits[0][:, -10:], logits[1][:, -10:], x[:, 120:130], cfg, ema)
            loss = ce + pa
            opt.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1; seen += len(x)
            if seen >= per_rank_target: break
        val_loss = evaluate(model, tok, val_loader, device, cfg, world)
        row = {'segment': seg, 'step': step, 'samples_per_rank': seen, 'global_samples_approx': seen * world, 'token_forecast_loss': float(ce.detach()), 'validation_objective': val_loss, 'path_align_loss': float(metrics['path_align_loss']), 'horizon_mae': metrics['path_align_horizon_mae'].tolist(), 'max_residual': float(metrics['path_align_max_residual']), 'grad_norm': float(grad), 'elapsed_sec': time.time() - started}
        if is_main(rank):
            if val_loss < best_val:
                best_val = val_loss; raw.save_pretrained(out / 'best_model')
            summary['segments'].append(row)
            torch.save({'model': raw.state_dict(), 'optimizer': opt.state_dict(), 'step': step, 'segment': seg, 'path_ema': ema.state_dict()}, out / 'last_state.pt')
            if run is not None:
                run.log({'segment': seg, 'step': step, 'token_forecast_loss': row['token_forecast_loss'], 'validation_objective': val_loss, 'path_align_loss': row['path_align_loss'], 'grad_norm': row['grad_norm'], 'max_residual': row['max_residual']}, step=step)
            (out / 'progress.json').write_text(json.dumps({'segment': seg, 'step': step, 'samples_per_rank': seen, 'status': 'running'}, indent=2))
            (out / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(row), flush=True)
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
        raw.save_pretrained(out / 'last_model')
        summary['final_result'] = {'completed_segments': len(summary['segments']), 'best_validation_objective': best_val, 'status': 'stopped'}
        (out / 'progress.json').write_text(json.dumps({'completed_segments': len(summary['segments']), 'status': 'stopped'}, indent=2))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        if run is not None: run.finish()
    if world > 1: dist.destroy_process_group()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--segments', type=int, default=40)
    p.add_argument('--max-runtime-seconds', type=int, default=36000)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--seed', type=int, default=20260914)
    main(p.parse_args())
