import json, os, subprocess, sys
from pathlib import Path

TRAINER_SRC = "import argparse, json, os, time, subprocess, sys\nfrom pathlib import Path\nimport torch\nimport torch.distributed as dist\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nfrom torch.utils.data import DataLoader\nfrom torch.utils.data.distributed import DistributedSampler\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--progress-bar', 'off', 'swanlab'], check=True)\nimport swanlab\nfrom model.kronos import Kronos, KronosTokenizer\nfrom finetune.dataset import QlibDataset\nfrom finetune.stage3_path_alignment import PathAlignmentConfig, DetachedLossEMA, compute_path_alignment_loss\n\ndef setup_dist():\n    if 'RANK' in os.environ:\n        dist.init_process_group(backend='nccl')\n        rank = dist.get_rank(); world = dist.get_world_size()\n        local = int(os.environ.get('LOCAL_RANK', 0))\n        torch.cuda.set_device(local)\n        return rank, world, local\n    return 0, 1, 0\n\ndef is_main(rank):\n    return rank == 0\n\ndef evaluate(model, tok, loader, device, cfg, world):\n    model.eval(); total = torch.zeros((), device=device); n = torch.zeros((), device=device)\n    raw = model.module if isinstance(model, DDP) else model\n    with torch.no_grad():\n        for vb in loader:\n            vx, vs = vb[0].to(device), vb[1].to(device)\n            vsec = vb[2].to(device) if len(vb) > 2 else None\n            vpct = vb[4].to(device) if len(vb) > 4 else None\n            vs1, vs2 = tok.encode(vx, half=True)\n            vlogits = raw(vs1[:, :-1], vs2[:, :-1], vs[:, :-1], use_teacher_forcing=True, s1_targets=vs1[:, 1:], sector_id=vsec, size_percentile=vpct)\n            vce = raw.head.compute_loss(vlogits[0][:, -10:], vlogits[1][:, -10:], vs1[:, 1:][:, -10:], vs2[:, 1:][:, -10:])[0]\n            vpa, _ = compute_path_alignment_loss(tok, vlogits[0][:, -10:], vlogits[1][:, -10:], vx[:, 120:130], cfg, None)\n            total += (vce + vpa).detach() * len(vx); n += len(vx)\n    if world > 1:\n        dist.all_reduce(total, op=dist.ReduceOp.SUM); dist.all_reduce(n, op=dist.ReduceOp.SUM)\n    model.train()\n    return float(total / n.clamp_min(1))\n\ndef main(a):\n    rank, world, local = setup_dist()\n    torch.manual_seed(a.seed + rank)\n    device = torch.device(f'cuda:{local}')\n    if is_main(rank):\n        print('initialization=stage3_from_c2_best_model_only', flush=True)\n        print('parent_model=' + str(a.model_dir), flush=True)\n        print('optimizer_state=reset', flush=True)\n        print(f'world_size={world}', flush=True)\n        print('validation_mode=full', flush=True)\n        print('target_slice=x[:,120:130]', flush=True)\n        print(f'chunk1_max_segments={a.segments}', flush=True)\n        print(f'chunk1_max_runtime_seconds={a.max_runtime_seconds}', flush=True)\n    model = Kronos.from_pretrained(a.model_dir).to(device)\n    tok = KronosTokenizer.from_pretrained(a.tokenizer_dir).to(device).eval()\n    for p in tok.parameters(): p.requires_grad_(False)\n    if world > 1:\n        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=False)\n    raw = model.module if isinstance(model, DDP) else model\n    run = None\n    if is_main(rank):\n        api_key = os.environ.get('SWANLAB_API_KEY', '').strip()\n        if not api_key:\n            raise RuntimeError('SWANLAB_API_KEY missing; refusing to start training without dashboard')\n        swanlab.login(api_key=api_key)\n        run_id = os.environ.get('SWANLAB_RUN_ID', 'small_0.1_stage3_path_alignment_from_c2_best').strip() or 'small_0.1_stage3_path_alignment_from_c2_best'\n        experiment_name = os.environ.get('SWANLAB_EXPERIMENT_NAME', 'small_0.1_stage3_path_alignment_from_c2_best').strip() or 'small_0.1_stage3_path_alignment_from_c2_best'\n        # Match C2: fixed run id + resume=allow so multi-chunk handoffs stay on one dashboard.\n        run = swanlab.init(\n            id=run_id,\n            resume='allow',\n            project='finance',\n            workspace='roc_fu',\n            experiment_name=experiment_name,\n            config={'lr': a.lr, 'batch_per_gpu': a.batch, 'global_batch': a.batch * world, 'segments': a.segments, 'max_runtime_seconds': a.max_runtime_seconds, 'top_k': 16, 'candidates': 16, 'parent': 'c2_best', 'validation': 'full', 'nproc': world, 'chunk': 'c1'},\n            mode='cloud',\n        )\n        run_url = getattr(run, 'url', getattr(run, 'web_url', ''))\n        print('SWANLAB_RUN_URL=' + str(run_url), flush=True)\n        if not run_url:\n            raise RuntimeError('SwanLab init returned no run URL; refusing to start training')\n    model.train()\n    ds = QlibDataset('train')\n    train_sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None\n    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, sampler=train_sampler, num_workers=0)\n    os.environ['KRONOS_VALIDATION_SAMPLES'] = '0'\n    val_ds = QlibDataset('val')\n    if is_main(rank):\n        print('validation_samples=' + str(val_ds.total_samples), flush=True)\n        if len(val_ds) != val_ds.total_samples or val_ds.total_samples < 100000:\n            raise RuntimeError(f'Validation set unexpectedly small: {len(val_ds)}; full validation is required')\n    if world > 1: dist.barrier()\n    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None\n    val_loader = DataLoader(val_ds, batch_size=a.batch, shuffle=False, sampler=val_sampler, num_workers=0)\n    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)\n    ema = DetachedLossEMA(0.99); cfg = PathAlignmentConfig(16, 16, 0.05, 0.02, 0.99)\n    out = Path(a.output_dir)\n    if is_main(rank): out.mkdir(parents=True, exist_ok=True)\n    if world > 1: dist.barrier()\n    step = 0\n    summary = {'segments': [], 'chunk': 'c1', 'max_runtime_seconds': a.max_runtime_seconds}\n    best_val = float('inf')\n    per_rank_target = (20000 + world - 1) // world\n    run_started = time.time()\n    for seg in range(1, a.segments + 1):\n        if train_sampler is not None: train_sampler.set_epoch(seg - 1)\n        ds.set_epoch_seed(seg - 1); seen = 0; started = time.time()\n        for batch in loader:\n            x, stamp = batch[0].to(device), batch[1].to(device)\n            sec = batch[2].to(device) if len(batch) > 2 else None\n            pct = batch[4].to(device) if len(batch) > 4 else None\n            with torch.no_grad(): s1, s2 = tok.encode(x, half=True)\n            logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1], use_teacher_forcing=True, s1_targets=s1[:, 1:], sector_id=sec, size_percentile=pct)\n            ce = raw.head.compute_loss(logits[0][:, -10:], logits[1][:, -10:], s1[:, 1:][:, -10:], s2[:, 1:][:, -10:])[0]\n            pa, metrics = compute_path_alignment_loss(tok, logits[0][:, -10:], logits[1][:, -10:], x[:, 120:130], cfg, ema)\n            loss = ce + pa\n            opt.zero_grad(set_to_none=True); loss.backward()\n            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n            opt.step(); step += 1; seen += len(x)\n            if seen >= per_rank_target: break\n        val_loss = evaluate(model, tok, val_loader, device, cfg, world)\n        row = {'segment': seg, 'step': step, 'samples_per_rank': seen, 'global_samples_approx': seen * world, 'token_forecast_loss': float(ce.detach()), 'validation_objective': val_loss, 'path_align_loss': float(metrics['path_align_loss']), 'horizon_mae': metrics['path_align_horizon_mae'].tolist(), 'max_residual': float(metrics['path_align_max_residual']), 'grad_norm': float(grad), 'elapsed_sec': time.time() - started}\n        if is_main(rank):\n            if val_loss < best_val:\n                best_val = val_loss; raw.save_pretrained(out / 'best_model')\n            summary['segments'].append(row)\n            torch.save({'model': raw.state_dict(), 'optimizer': opt.state_dict(), 'step': step, 'segment': seg, 'path_ema': ema.state_dict()}, out / 'last_state.pt')\n            if run is not None:\n                run.log({'segment': seg, 'step': step, 'token_forecast_loss': row['token_forecast_loss'], 'validation_objective': val_loss, 'path_align_loss': row['path_align_loss'], 'grad_norm': row['grad_norm'], 'max_residual': row['max_residual']}, step=step)\n            (out / 'progress.json').write_text(json.dumps({'segment': seg, 'step': step, 'samples_per_rank': seen, 'status': 'running'}, indent=2))\n            (out / 'summary.json').write_text(json.dumps(summary, indent=2))\n            print(json.dumps(row), flush=True)\n        stop_for_time = False\n        if is_main(rank) and (time.time() - run_started) >= a.max_runtime_seconds:\n            stop_for_time = True\n            print(json.dumps({'phase': 'stop_max_runtime', 'completed_segment': seg, 'elapsed_sec': time.time() - run_started, 'max_runtime_seconds': a.max_runtime_seconds}), flush=True)\n        flag = torch.tensor([1 if stop_for_time else 0], device=device)\n        if world > 1:\n            dist.broadcast(flag, src=0)\n            dist.barrier()\n        if int(flag.item()) == 1:\n            break\n    if is_main(rank):\n        raw.save_pretrained(out / 'last_model')\n        summary['final_result'] = {'completed_segments': len(summary['segments']), 'best_validation_objective': best_val, 'status': 'stopped'}\n        (out / 'progress.json').write_text(json.dumps({'completed_segments': len(summary['segments']), 'status': 'stopped'}, indent=2))\n        (out / 'summary.json').write_text(json.dumps(summary, indent=2))\n        if run is not None: run.finish()\n    if world > 1: dist.destroy_process_group()\n\nif __name__ == '__main__':\n    p = argparse.ArgumentParser()\n    p.add_argument('--model-dir', required=True)\n    p.add_argument('--tokenizer-dir', required=True)\n    p.add_argument('--output-dir', required=True)\n    p.add_argument('--batch', type=int, default=32)\n    p.add_argument('--segments', type=int, default=40)\n    p.add_argument('--max-runtime-seconds', type=int, default=36000)\n    p.add_argument('--lr', type=float, default=2e-6)\n    p.add_argument('--seed', type=int, default=20260914)\n    main(p.parse_args())\n"

ALIGNMENT_SRC = "from dataclasses import dataclass\nimport torch\nimport torch.nn.functional as F\n@dataclass\nclass PathAlignmentConfig:\n top_k:int=16; candidates:int=16; weight:float=.05; huber_delta:float=.02; ema_decay:float=.99\nclass DetachedLossEMA:\n def __init__(self,decay=.99): self.decay=decay; self.value=None\n def normalize(self,loss):\n  cur=loss.detach().float(); self.value=cur if self.value is None else self.decay*self.value+(1-self.decay)*cur\n  return loss/self.value.clamp_min(1e-6).to(loss.dtype)\n def state_dict(self): return {'decay':self.decay,'value':None if self.value is None else self.value.detach().cpu()}\ndef candidate_mixture_decode(tok,a,b,top_k=16,candidates=16):\n k1=min(top_k,a.shape[-1]); k2=min(top_k,b.shape[-1]); p1=a.float().softmax(-1); p2=b.float().softmax(-1); v1,i1=torch.topk(p1,k1,-1); v2,i2=torch.topk(p2,k2,-1); outs=[]; ws=[]\n for n in range(min(candidates,k1*k2)): outs.append(tok.decode((i1[...,n%k1],i2[...,(n//k1)%k2]),half=True).float()); ws.append(v1[...,n%k1]*v2[...,(n//k1)%k2])\n w=torch.stack(ws); w=w/w.sum(0,keepdim=True).clamp_min(1e-8); return (torch.stack(outs)*w[...,None]).sum(0),w.detach()\ndef compute_path_alignment_loss(tok,a,b,target,cfg,normalizer=None):\n pred,w=candidate_mixture_decode(tok,a,b,cfg.top_k,cfg.candidates); target=target.to(pred.dtype); err=pred-target; per=F.huber_loss(pred,target,delta=cfg.huber_delta,reduction='none').mean((0,2)); loss=per.mean(); norm=normalizer.normalize(loss) if normalizer else loss\n return cfg.weight*norm, {'path_align_loss':loss.detach(),'path_align_horizon_mae':err.detach().abs().mean((0,2)),'path_align_max_residual':err.detach().abs().max()}\n"

ROOT = Path('/kaggle/working/Kronos')
SWANLAB_KEY = os.environ.get('SWANLAB_API_KEY', 'fmEPDGk4IItxgqSZKGLi8')
os.environ['SWANLAB_API_KEY'] = SWANLAB_KEY
os.environ.setdefault('SWANLAB_MODE', 'cloud')
os.environ['SWANLAB_EXPERIMENT_NAME'] = 'small_0.1_stage3_path_alignment_from_c2_best'
os.environ['SWANLAB_RUN_ID'] = 'small_0.1_stage3_path_alignment_from_c2_best'
print('install_swanlab_started', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--progress-bar', 'off', 'swanlab'], check=True)
print('install_swanlab_finished', flush=True)
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/luckfu/Kronos.git', str(ROOT)], check=True)
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')

import torch
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(gpu_names) != 2 or any('T4' not in name for name in gpu_names):
    raise RuntimeError(f'Expected exactly two Tesla T4 GPUs, found {gpu_names}')
print(json.dumps({'phase': 'device_ready', 'gpu_count': len(gpu_names), 'devices': gpu_names}), flush=True)

# The trainer and alignment module come from the cloned GitHub HEAD. Keeping
# one source of truth prevents the Kaggle wrapper from silently running stale
# embedded Stage-3 code.
trainer = ROOT / 'finetune' / 'train_stage3_path_alignment.py'
alignment = ROOT / 'finetune' / 'stage3_path_alignment.py'
if not trainer.is_file() or not alignment.is_file():
    raise FileNotFoundError('Stage3 trainer/module missing from cloned GitHub repository')

def find_one(base, names):
    base = Path(base)
    for name in names:
        hits = list(base.rglob(name))
        if hits:
            return hits[0]
    return None

c2 = Path('/kaggle/input')
best_candidates = [p for p in c2.rglob('model.safetensors') if 'best_model' in str(p)]
model = best_candidates[0] if best_candidates else None
if model is None:
    raise FileNotFoundError('C2 checkpoints/best_model/model.safetensors not found')
model_dir = model.parent
tok = find_one(c2, ['Kronos-Tokenizer-base'])
if tok is None:
    tok = next((p for p in c2.rglob('*') if p.is_dir() and 'tokenizer' in p.name.lower()), None)
if tok is None:
    raise FileNotFoundError('C2 source tokenizer directory not found')
train = find_one('/kaggle/input', ['train_data.pkl'])
val = find_one('/kaggle/input', ['val_data.pkl'])
meta = find_one('/kaggle/input', ['asset_metadata.csv'])
if train is None or val is None or meta is None:
    raise FileNotFoundError('dataset train_data.pkl, val_data.pkl or asset_metadata.csv not found')
dataset_root = train.parent
if val.parent != dataset_root:
    raise RuntimeError(f'train/val are not in one C2-compatible dataset root: {train} {val}')
os.environ.update({
    'KRONOS_DATASET_PATH': str(dataset_root),
    'KRONOS_METADATA_PATH': str(meta),
    'KRONOS_LOOKBACK_WINDOW': '120',
    'KRONOS_PREDICT_WINDOW': '10',
    'KRONOS_USE_SIZE_PERCENTILE': '1',
    'KRONOS_NUM_SIZE_BUCKETS': '0',
    'KRONOS_VALIDATION_SAMPLES': '0',
    'SWANLAB_API_KEY': SWANLAB_KEY,
    'SWANLAB_EXPERIMENT_NAME': 'small_0.1_stage3_path_alignment_from_c2_best',
    'SWANLAB_RUN_ID': 'small_0.1_stage3_path_alignment_from_c2_best',
})
out = Path('/kaggle/working/stage3_path_alignment_from_c2_best_smoke')
cmd = [
    sys.executable, '-m', 'torch.distributed.run',
    '--standalone', '--nproc_per_node=2',
    str(ROOT / 'finetune/train_stage3_path_alignment.py'),
    '--model-dir', str(model_dir),
    '--tokenizer-dir', str(tok),
    '--output-dir', str(out),
    '--segments', '40',
    '--batch', '32',
    '--lr', '2e-6',
    '--max-runtime-seconds', '36000',
]
print(json.dumps({
    'model_dir': str(model_dir),
    'tokenizer_dir': str(tok),
    'train_data': str(train),
    'metadata': str(meta),
    'cmd': cmd,
    'global_batch': 64,
    'chunk': 'c1',
    'max_segments_per_run': 40,
    'max_runtime_seconds': 36000,
    'swanlab_run_id': 'small_0.1_stage3_path_alignment_from_c2_best',
    'swanlab_resume': 'allow',
    'timing_basis_sec_per_segment_single_gpu': 772,
}), flush=True)
subprocess.run(cmd, cwd=ROOT, check=True)
print(json.dumps({'output_files': sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())}), flush=True)
