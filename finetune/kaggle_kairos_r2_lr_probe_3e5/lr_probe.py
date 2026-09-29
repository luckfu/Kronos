"""Full one-epoch ModernBERT decision training with checkpointing and dashboard."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
SEED = 20260927
BATCH_SIZE = 16
LR_PROBE = True
LEARNING_RATE_OVERRIDE = 3e-05
EARLY_STOP_PATIENCE = 2
EARLY_STOP_MIN_DELTA = 0.001
MAX_VALIDATION_DEGRADATION = 0.2
CHUNK_INDEX = 1
SEGMENT_SAMPLES = 20000
GPU_BUDGET_SECONDS = 5100
RUNTIME_RESERVE_SECONDS = 900
SEGMENT_ESTIMATE_SECONDS = 546.412109773
SEGMENT_TIME_MARGIN = 1.1
MAX_SEGMENTS_THIS_RUN = 4
TOTAL_SEGMENTS = None
SHUFFLE_SEED = 20260927
SWANLAB_API_KEY_FALLBACK = ''
SWANLAB_RUN_ID = 'kairos-lr-probe-3e5-20260930'
OUTPUT = Path('/kaggle/working/kairos_r2')
RUN_PURPOSE = 'bounded-lr-probe-from-chunk1-segment1'
FEATURES = ('open', 'high', 'low', 'close', 'volume', 'amount')
TARGET_COLUMNS = ('up_003', 'up_005', 'up_008', 'up_012', 'down_003', 'down_005', 'down_008', 'down_012')

def log(phase: str, **fields: object) -> None:
    if os.environ.get('RANK', '0') != '0':
        return
    row = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'phase': phase, **fields}
    line = json.dumps(row, ensure_ascii=False, default=str)
    print(line, flush=True)
    with (OUTPUT / 'run.log').open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def find_one(pattern: str) -> Path:
    matches = sorted(Path('/kaggle/input').glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f'expected one {pattern}, found {len(matches)}: {matches}')
    return matches[0]

def dashboard(state: dict[str, Any]) -> None:
    if not is_main_process():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'progress.json').write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    processed = int(state.get('processed_samples', 0))
    total = int(state.get('total_samples', 0))
    percent = 100 * processed / total if total else 0
    html = f"""<!doctype html><meta charset="utf-8"><title>Kairos R2 training</title>\n<style>body{{font:16px system-ui;margin:32px;background:#f6f7f9;color:#17202a}}\nmain{{max-width:900px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}\n.card,pre{{background:white;border:1px solid #ddd;border-radius:8px;padding:16px}}\n.label{{color:#667085;font-size:13px}}.value{{font-size:24px;font-weight:650;margin-top:6px}}</style>\n<main><h1>Kairos R2 training</h1><div class="grid">\n<div class="card"><div class="label">Phase</div><div class="value">{state.get('phase', 'starting')}</div></div>\n<div class="card"><div class="label">Progress</div><div class="value">{percent:.2f}%</div></div>\n<div class="card"><div class="label">Samples</div><div class="value">{processed:,}/{total:,}</div></div>\n</div><pre>{json.dumps(state, ensure_ascii=False, indent=2, default=str)}</pre></main>"""
    (OUTPUT / 'dashboard.html').write_text(html, encoding='utf-8')

def shuffled_group_order(num_groups: int) -> list[int]:
    return np.random.default_rng(SHUFFLE_SEED).permutation(num_groups).tolist()

def shuffle_group_rows(rows: list[dict[str, Any]], group_id: int) -> list[dict[str, Any]]:
    order = np.random.default_rng(SHUFFLE_SEED + group_id + 1).permutation(len(rows))
    return [rows[int(index)] for index in order]

def group_order_hash(order: list[int]) -> str:
    return hashlib.sha256(','.join(map(str, order)).encode('ascii')).hexdigest()

def row_identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row['symbol']), int(row['start_index']), str(row['asof_date']))

def verify_segment_coverage(rows: list[dict[str, Any]], expected: list[tuple[str, int, str]], offset: int, seen: set[tuple[str, int, str]]) -> str:
    identities = [row_identity(row) for row in rows]
    if identities != expected[offset:offset + len(identities)]:
        raise RuntimeError('segment sample identities differ from independent prefix')
    if len(set(identities)) != len(identities) or seen.intersection(identities):
        raise RuntimeError('duplicate sample identity in training prefix')
    seen.update(identities)
    return hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()

def probe_stop_reason(score: float, best: float, stale: int, initial: float) -> tuple[str, float, int]:
    if not np.isfinite(score):
        return ('nonfinite_validation', best, stale)
    stale = 0 if score < best - EARLY_STOP_MIN_DELTA else stale + 1
    best = min(best, score)
    if score > initial + MAX_VALIDATION_DEGRADATION:
        return ('validation_degradation', best, stale)
    if stale >= EARLY_STOP_PATIENCE:
        return ('validation_patience', best, stale)
    return ('', best, stale)

def assert_restored_state(expected: Any, actual: Any, torch: Any) -> None:
    if isinstance(expected, torch.Tensor):
        if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
            raise RuntimeError('checkpoint tensor restoration mismatch')
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError('checkpoint state keys mismatch')
        for key in expected:
            assert_restored_state(expected[key], actual[key], torch)
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError('checkpoint state length mismatch')
        for left, right in zip(expected, actual):
            assert_restored_state(left, right, torch)
    elif expected != actual:
        raise RuntimeError('checkpoint scalar restoration mismatch')

def is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0

def calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int=10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        return float('nan')
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1] if index + 1 < bins else probabilities <= edges[index + 1])
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return error

def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int=10) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities <= right if right == 1.0 else probabilities < right)
        result.append({'lower': float(left), 'upper': float(right), 'count': int(mask.sum()), 'mean_probability': float(probabilities[mask].mean()) if mask.any() else None, 'positive_rate': float(labels[mask].mean()) if mask.any() else None})
    return result

def start_swanlab() -> tuple[Any, Any]:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', 'swanlab'], check=True)
    import swanlab
    api_key = os.environ.get('SWANLAB_API_KEY', '').strip() or SWANLAB_API_KEY_FALLBACK
    if not api_key and (not LR_PROBE):
        raise RuntimeError('SWANLAB_API_KEY is empty')
    if api_key:
        swanlab.login(api_key=api_key)
    run = swanlab.init(id=SWANLAB_RUN_ID, resume='allow', project='finance', workspace='roc_fu', experiment_name=SWANLAB_RUN_ID, config={'model': 'ModernBERT-base-style', 'hidden_size': 768, 'layers': 22, 'heads': 12, 'batch_size': BATCH_SIZE, 'chunk_index': CHUNK_INDEX, 'segment_samples': SEGMENT_SAMPLES, 'max_segments_this_run': MAX_SEGMENTS_THIS_RUN, 'segment_total': 'computed_after_data_load', 'shuffle_seed': SHUFFLE_SEED, 'variant': 'kairos-r2', 'learning_rate_override': LEARNING_RATE_OVERRIDE, 'lr_probe': LR_PROBE, 'validation_samples_per_segment': 123836, 'validation_scope': 'full_validation_after_every_segment'}, mode='cloud' if api_key else 'offline')
    url = getattr(run, 'url', '') or getattr(run, 'web_url', '') if api_key else ''
    if not url and api_key:
        raise RuntimeError('SwanLab did not return a run URL')
    print(json.dumps({'phase': 'swanlab_ready', 'url': url}), flush=True)
    return (swanlab, run)

def load_arrays(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    import pandas as pd
    with path.open('rb') as handle:
        panel = pickle.load(handle)
    arrays: dict[str, dict[str, Any]] = {}
    sectors: set[str] = set()
    for symbol in sorted(panel):
        frame = panel[symbol].sort_index()
        sector = frame.get('sector', pd.Series('unknown', index=frame.index)).astype(str).to_numpy()
        sectors.update(sector.tolist())
        arrays[str(symbol)] = {'values': frame.loc[:, FEATURES].to_numpy(dtype=np.float32, copy=True), 'sector': sector, 'size': pd.to_numeric(frame.get('size_percentile', pd.Series(0.5, index=frame.index)), errors='coerce').fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=np.float32), 'dates': frame.index.to_numpy(dtype='datetime64[D]')}
    return (arrays, sectors)

def make_batch(rows: list[dict[str, Any]], arrays: dict[str, dict[str, Any]], sector_ids: dict[str, int], check_labels: bool=False) -> tuple[np.ndarray, ...]:
    histories = np.empty((len(rows), 120, 6), dtype=np.float32)
    sectors = np.empty(len(rows), dtype=np.int64)
    sizes = np.empty(len(rows), dtype=np.float32)
    labels = np.empty((len(rows), 8), dtype=np.float32)
    dates = np.empty(len(rows), dtype='datetime64[D]')
    for i, row in enumerate(rows):
        symbol, start = (str(row['symbol']), int(row['start_index']))
        data = arrays[symbol]
        asof = start + 119
        history = data['values'][start:start + 120].copy()
        history = (history - history.mean(axis=0)) / (history.std(axis=0) + 1e-05)
        histories[i] = np.clip(history, -5.0, 5.0)
        sectors[i] = sector_ids.get(str(data['sector'][asof]), len(sector_ids))
        sizes[i] = data['size'][asof]
        labels[i] = [row[name] for name in TARGET_COLUMNS]
        dates[i] = data['dates'][asof]
        if check_labels:
            future = data['values'][start + 120:start + 130]
            close = float(data['values'][asof, 3])
            if not np.isclose(future[:, 1].max() / close - 1, row['mfe10'], atol=2e-06):
                raise ValueError(f'MFE mismatch: {symbol} {start}')
            if not np.isclose(future[:, 2].min() / close - 1, row['mae10'], atol=2e-06):
                raise ValueError(f'MAE mismatch: {symbol} {start}')
    return (histories, sectors, sizes, labels, dates)

def main() -> None:
    runtime_started = float(os.environ.setdefault('KAIROS_RUNTIME_STARTED', str(time.time())))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    if not torch.cuda.is_available():
        raise RuntimeError('full training requires Kaggle GPU')
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2 and 'WORLD_SIZE' not in os.environ:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(Path(__file__).resolve())]
        log('ddp_relaunch', command=command, gpu_count=gpu_count)
        completed = subprocess.run(command, check=False)
        raise SystemExit(completed.returncode)
    if gpu_count >= 2 and int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise RuntimeError(f"Kairos R2 expects exactly two DDP workers on a two-GPU Kaggle runtime; got WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    swanlab, swanlab_run = start_swanlab() if is_main_process() else (None, None)
    log('run_started', purpose=RUN_PURPOSE, chunk_index=CHUNK_INDEX, max_segments_this_run=MAX_SEGMENTS_THIS_RUN, world_size=world_size, global_batch_size=BATCH_SIZE)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda')
    train_panel_path = find_one('**/processed_datasets/train_data.pkl')
    validation_panel_path = find_one('**/processed_datasets/val_data.pkl')
    train_targets_path = find_one('**/train_targets.parquet')
    validation_targets_path = find_one('**/validation_targets.parquet')
    target_manifest_path = train_targets_path.parent / 'decision_targets_manifest.json'
    source_manifest_path = train_panel_path.parent.parent / 'data_manifest.json'
    target_manifest = json.loads(target_manifest_path.read_text(encoding='utf-8'))
    if int(target_manifest['window']['source_window']) != 131:
        raise ValueError('expected 120+10+1 window contract')
    for path, key in ((train_panel_path, 'train_panel_sha256'), (validation_panel_path, 'validation_panel_sha256'), (source_manifest_path, 'data_manifest_sha256')):
        if sha256_file(path) != target_manifest['source_dataset'][key]:
            raise ValueError(f'source hash mismatch: {path}')
    train_pq = pq.ParquetFile(train_targets_path)
    validation_pq = pq.ParquetFile(validation_targets_path)
    train_total, validation_total = (train_pq.metadata.num_rows, validation_pq.metadata.num_rows)
    if is_main_process():
        dashboard({'phase': 'loading panels', 'processed_samples': 0, 'total_samples': train_total})
    log('runtime', gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], gpu_count=torch.cuda.device_count(), torch=torch.__version__, transformers=importlib.metadata.version('transformers'), train_samples=train_total, validation_samples=validation_total, rank=rank, world_size=world_size, local_rank=local_rank)
    train_arrays, train_sectors = load_arrays(train_panel_path)
    validation_arrays, validation_sectors = load_arrays(validation_panel_path)
    sector_ids = {name: i for i, name in enumerate(sorted(train_sectors | validation_sectors))}
    repo_path = Path('/kaggle/working/Kronos')
    if is_main_process() and (not (repo_path / 'model' / 'kronos.py').is_file()):
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'master', 'https://github.com/luckfu/Kronos.git', str(repo_path)], check=True)
    if distributed:
        dist.barrier()
    sys.path.insert(0, str(repo_path))
    from model import KronosTokenizer
    from huggingface_hub import snapshot_download
    from transformers import ModernBertConfig, ModernBertModel
    tokenizer_path_holder = [snapshot_download(repo_id='NeoQuasar/Kronos-Tokenizer-base') if is_main_process() else None]
    if distributed:
        dist.broadcast_object_list(tokenizer_path_holder, src=0)
    tokenizer_path = Path(tokenizer_path_holder[0])
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    tokenizer_sha256 = sha256_file(tokenizer_path / 'model.safetensors')

    class FullModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            hidden = 768
            self.s1 = nn.Embedding(1024, hidden // 2)
            self.s2 = nn.Embedding(1024, hidden // 2)
            self.fusion = nn.Linear(hidden, hidden)
            self.gate = nn.Parameter(torch.zeros(1))
            self.sector = nn.Embedding(len(sector_ids) + 1, hidden)
            self.size = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, hidden))
            self.cond = nn.Parameter(torch.zeros(1, 1, hidden))
            config = ModernBertConfig(vocab_size=1024, hidden_size=hidden, intermediate_size=1152, num_hidden_layers=22, num_attention_heads=12, max_position_embeddings=128, pad_token_id=0, attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0, reference_compile=False)
            self.backbone = ModernBertModel(config)
            native_embeddings = getattr(self.backbone.embeddings, 'tok_embeddings', None)
            if native_embeddings is None:
                native_embeddings = getattr(self.backbone.embeddings, 'word_embeddings', None)
            if native_embeddings is None:
                raise RuntimeError(f'Cannot locate ModernBERT native token embeddings; available={list(self.backbone.embeddings._modules)}')
            for parameter in native_embeddings.parameters():
                parameter.requires_grad_(False)
            self.up0, self.upd = (nn.Linear(hidden, 1), nn.Linear(hidden, 3))
            self.dn0, self.dnd = (nn.Linear(hidden, 1), nn.Linear(hidden, 3))

        def forward(self, s1: Any, s2: Any, size: Any, sector: Any) -> tuple[Any, Any]:
            condition = self.cond + self.sector(sector).unsqueeze(1) + self.size(size).unsqueeze(1)
            market = self.fusion(torch.cat([self.s1(s1), self.s2(s2)], -1)) * self.gate
            sequence = torch.cat([condition, market], 1)
            mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device)
            pooled = self.backbone(inputs_embeds=sequence, attention_mask=mask).last_hidden_state[:, 0]

            def ordinal(base: Any, delta: Any) -> Any:
                return torch.cat([base, base - torch.cumsum(F.softplus(delta), -1)], -1)
            return (ordinal(self.up0(pooled), self.upd(pooled)), ordinal(self.dn0(pooled), self.dnd(pooled)))
    model = FullModel().to(device)
    raw_model = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    log('multi_gpu_detected', devices=torch.cuda.device_count(), mode='ddp' if distributed else 'single_gpu', active_device=str(device), rank=rank, world_size=world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler('cuda')
    checkpoint = OUTPUT / 'last_checkpoint.pt'
    best_checkpoint = OUTPUT / 'best_model.pt'
    if is_main_process():
        if CHUNK_INDEX > 0 and (not checkpoint.exists()):
            previous = sorted(Path('/kaggle/input').glob('**/last_checkpoint.pt'))
            if len(previous) > 1:
                raise RuntimeError(f'expected at most one previous checkpoint, found {previous}')
            if previous:
                shutil.copy2(previous[0], checkpoint)
        if not best_checkpoint.exists():
            previous_best = sorted(Path('/kaggle/input').glob('**/best_model.pt'))
            if len(previous_best) > 1:
                raise RuntimeError(f'expected at most one previous best model, found {previous_best}')
            if previous_best:
                shutil.copy2(previous_best[0], best_checkpoint)
    best_meta = OUTPUT / 'best_metric.json'
    if is_main_process() and (not best_meta.exists()):
        previous_meta = sorted(Path('/kaggle/input').glob('**/best_metric.json'))
        if len(previous_meta) > 1:
            raise RuntimeError(f'expected at most one previous best metric, found {previous_meta}')
        if previous_meta:
            shutil.copy2(previous_meta[0], best_meta)
    if distributed:
        dist.barrier()
    group_order = shuffled_group_order(train_pq.num_row_groups)
    total_segments = (train_total + SEGMENT_SAMPLES - 1) // SEGMENT_SAMPLES
    if TOTAL_SEGMENTS is not None and TOTAL_SEGMENTS != total_segments:
        raise RuntimeError(f'configured total segments {TOTAL_SEGMENTS} != computed {total_segments}')
    run_segment_limit = min(MAX_SEGMENTS_THIS_RUN, total_segments)
    start_segment_index = 0
    start_group_order_pos = 0
    start_row_offset = 0
    processed = 0
    last_validation = None
    last_best_updated = False
    best_score = float('inf')
    if best_meta.exists():
        best_score = float(json.loads(best_meta.read_text(encoding='utf-8'))['macro_log_loss'])
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        if saved.get('shuffle_seed') != SHUFFLE_SEED:
            raise RuntimeError('checkpoint shuffle seed mismatch')
        if saved.get('group_order_hash') != group_order_hash(group_order):
            raise RuntimeError('checkpoint row-group order mismatch')
        raw_model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        if LR_PROBE:
            if (saved.get('chunk_index'), saved.get('completed_segments'), saved.get('processed_samples'), saved.get('group_order_pos'), saved.get('row_offset')) != (0, 1, 20000, 0, 20000):
                raise RuntimeError('LR probe requires the unmodified Chunk 1 segment-1 checkpoint')
            if abs(saved['last_validation']['all']['macro_log_loss'] - 0.6702645644545555) > 1e-08:
                raise RuntimeError('unexpected source checkpoint metric')
            assert_restored_state(saved['model'], raw_model.state_dict(), torch)
            assert_restored_state(saved['optimizer'], optimizer.state_dict(), torch)
            assert_restored_state(saved['scaler'], scaler.state_dict(), torch)
            log('probe_restore_verified', checkpoint_sha256=sha256_file(checkpoint), model_optimizer_scaler_exact=True, source_learning_rates=[group['lr'] for group in optimizer.param_groups])
        if LEARNING_RATE_OVERRIDE is not None:
            for group in optimizer.param_groups:
                group['lr'] = LEARNING_RATE_OVERRIDE
            log('learning_rate_override', learning_rate=LEARNING_RATE_OVERRIDE, optimizer_moments_preserved=True)
        saved_chunk = int(saved['chunk_index'] if 'chunk_index' in saved else CHUNK_INDEX - 1)
        if saved_chunk in {CHUNK_INDEX, CHUNK_INDEX - 1}:
            start_segment_index = int(saved.get('completed_segments', 0))
            start_group_order_pos = int(saved.get('group_order_pos', 0))
            start_row_offset = int(saved.get('row_offset', 0))
        else:
            raise RuntimeError(f'checkpoint chunk mismatch: saved={saved_chunk}, current={CHUNK_INDEX}')
        processed = int(saved['processed_samples'])
        last_validation = saved.get('last_validation')
        last_best_updated = bool(saved.get('last_best_updated', False))
        log('checkpoint_resumed', chunk_index=saved_chunk, completed_segments=start_segment_index, group_order_pos=start_group_order_pos, row_offset=start_row_offset, processed_samples=processed)
    elif CHUNK_INDEX == 0:
        initial_models = sorted(Path('/kaggle/input').glob('**/final_model.pt'))
        if len(initial_models) != 1:
            raise RuntimeError(f'R2 Chunk 1 requires exactly one R1 final_model.pt, found {initial_models}')
        initial = torch.load(initial_models[0], map_location=device)
        if 'model' not in initial:
            raise RuntimeError(f'R1 artifact has no model state: {initial_models[0]}')
        raw_model.load_state_dict(initial['model'], strict=True)
        log('r1_warm_start', source=str(initial_models[0]), optimizer_reset=True, shuffle_seed=SHUFFLE_SEED)
    else:
        raise RuntimeError("Relay requires the previous chunk's last_checkpoint.pt")
    if checkpoint.exists():
        del saved
        torch.cuda.empty_cache()
    columns = ['symbol', 'start_index', 'asof_date', 'mfe10', 'mae10', *TARGET_COLUMNS]
    expected_prefix: list[tuple[str, int, str]] = []
    seen_identities: set[tuple[str, int, str]] = set()
    if LR_PROBE:
        prefix_limit = processed + run_segment_limit * SEGMENT_SAMPLES
        for expected_group_id in group_order:
            expected_rows = shuffle_group_rows(train_pq.read_row_group(expected_group_id, columns=columns).to_pylist(), expected_group_id)
            expected_prefix.extend((row_identity(row) for row in expected_rows))
            if len(expected_prefix) >= prefix_limit:
                break
        expected_prefix = expected_prefix[:prefix_limit]
        if len(set(expected_prefix)) != len(expected_prefix):
            raise RuntimeError('independent prefix itself contains duplicate identities')
        seen_identities.update(expected_prefix[:processed])
        log('probe_prefix_verified', samples=len(expected_prefix), sha256=hashlib.sha256(json.dumps(expected_prefix, separators=(',', ':')).encode()).hexdigest(), source_processed_samples=processed)
    local_batch_size = max(1, BATCH_SIZE // world_size)
    validation_token_cache: tuple[np.ndarray, ...] | None = None
    validation_cache_path = OUTPUT / f'validation_token_cache_rank{rank}.npz'
    validation_cache_identity = {'schema_version': 1, 'validation_panel_sha256': target_manifest['source_dataset']['validation_panel_sha256'], 'validation_targets_sha256': sha256_file(validation_targets_path), 'tokenizer_sha256': tokenizer_sha256, 'rank': rank, 'world_size': world_size, 'lookback': 120, 'features': list(FEATURES), 'normalization': 'per_window_zscore_std_plus_1e-5_clip_5'}

    def encode_token_cache(rows: list[dict[str, Any]], arrays: dict[str, dict[str, Any]], check_labels: bool=False) -> tuple[np.ndarray, ...]:
        """Encode rows once and retain compact CPU token arrays for reuse."""
        token_s1: list[np.ndarray] = []
        token_s2: list[np.ndarray] = []
        cached_sectors: list[np.ndarray] = []
        cached_sizes: list[np.ndarray] = []
        cached_labels: list[np.ndarray] = []
        cached_dates: list[np.ndarray] = []
        for offset in range(0, len(rows), local_batch_size):
            batch = rows[offset:offset + local_batch_size]
            history, sectors, sizes, labels, dates = make_batch(batch, arrays, sector_ids, check_labels=check_labels and offset == 0)
            with torch.no_grad():
                encoded_s1, encoded_s2 = tokenizer.encode(torch.from_numpy(history).to(device), half=True)
            token_s1.append(encoded_s1.cpu().numpy().astype(np.uint16, copy=False))
            token_s2.append(encoded_s2.cpu().numpy().astype(np.uint16, copy=False))
            cached_sectors.append(sectors)
            cached_sizes.append(sizes)
            cached_labels.append(labels)
            cached_dates.append(dates)
        return (np.concatenate(token_s1, axis=0), np.concatenate(token_s2, axis=0), np.concatenate(cached_sectors, axis=0), np.concatenate(cached_sizes, axis=0), np.concatenate(cached_labels, axis=0), np.concatenate(cached_dates, axis=0))

    def full_validation() -> dict[str, Any]:
        import sklearn.metrics
        nonlocal validation_token_cache
        if validation_token_cache is None:
            if not validation_cache_path.exists():
                previous = sorted(Path('/kaggle/input').glob(f'**/validation_token_cache_rank{rank}.npz'))
                if len(previous) > 1:
                    raise RuntimeError(f'expected at most one prior validation token cache, found {previous}')
                if previous:
                    shutil.copy2(previous[0], validation_cache_path)
            if validation_cache_path.exists():
                with np.load(validation_cache_path, allow_pickle=False) as cached:
                    identity = json.loads(str(cached['identity'].item()))
                    if identity != validation_cache_identity:
                        raise RuntimeError(f'validation token cache identity mismatch: expected={validation_cache_identity}, found={identity}')
                    validation_token_cache = tuple((cached[name].copy() for name in ('s1', 's2', 'sectors', 'sizes', 'labels', 'dates')))
                log('validation_token_cache_loaded', path=str(validation_cache_path), rank=rank, samples=len(validation_token_cache[0]), tokenizer_sha256=tokenizer_sha256)
            else:
                rows: list[dict[str, Any]] = []
                for group_id in range(validation_pq.num_row_groups):
                    rows.extend(validation_pq.read_row_group(group_id, columns=columns).to_pylist())
                local_rows = rows[rank::world_size]
                validation_token_cache = encode_token_cache(local_rows, validation_arrays)
                temporary_path = validation_cache_path.with_suffix('.npz.tmp')
                with temporary_path.open('wb') as handle:
                    np.savez(handle, identity=np.asarray(json.dumps(validation_cache_identity, sort_keys=True)), s1=validation_token_cache[0], s2=validation_token_cache[1], sectors=validation_token_cache[2], sizes=validation_token_cache[3], labels=validation_token_cache[4], dates=validation_token_cache[5])
                temporary_path.replace(validation_cache_path)
                log('validation_token_cache_created', path=str(validation_cache_path), rank=rank, samples=len(validation_token_cache[0]), tokenizer_sha256=tokenizer_sha256)
        local_cache = validation_token_cache
        local_rows = local_cache[0]
        model.eval()
        predictions, truth, dates = ([], [], [])
        for offset in range(0, len(local_rows), local_batch_size):
            s1 = torch.from_numpy(local_cache[0][offset:offset + local_batch_size]).to(device)
            s2 = torch.from_numpy(local_cache[1][offset:offset + local_batch_size]).to(device)
            sizes = torch.from_numpy(local_cache[3][offset:offset + local_batch_size, None]).to(device)
            sectors = torch.from_numpy(local_cache[2][offset:offset + local_batch_size]).to(device)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                up, down = model(s1.long(), s2.long(), sizes, sectors)
            predictions.append(torch.cat([up.sigmoid(), down.sigmoid()], 1).cpu().numpy())
            truth.append(local_cache[4][offset:offset + local_batch_size])
            dates.append(local_cache[5][offset:offset + local_batch_size])
        local_result = (np.concatenate(predictions), np.concatenate(truth), np.concatenate(dates))
        if distributed:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_result)
        else:
            gathered = [local_result]
        if not is_main_process():
            if distributed:
                dist.barrier()
            result_holder = [None]
            if distributed:
                dist.broadcast_object_list(result_holder, src=0)
            model.train()
            return result_holder[0]
        probabilities = np.concatenate([item[0] for item in gathered]).astype(np.float32)
        labels = np.concatenate([item[1] for item in gathered])
        dates = np.concatenate([item[2] for item in gathered])
        result: dict[str, Any] = {}
        for name, mask in (('all', np.ones(len(labels), dtype=bool)), ('2025H2', dates < np.datetime64('2026-01-01')), ('2026H1', dates >= np.datetime64('2026-01-01'))):
            values = []
            for index in range(8):
                p = np.clip(probabilities[mask, index], 1e-07, 1 - 1e-07)
                y = labels[mask, index]
                values.append({'log_loss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 'brier': float(np.square(p - y).mean()), 'ece_10bin': calibration_error(p, y), 'roc_auc': float(sklearn.metrics.roc_auc_score(y, p)) if np.unique(y).size == 2 else None, 'pr_auc': float(sklearn.metrics.average_precision_score(y, p)) if np.any(y == 1) else None, 'reliability': reliability_bins(p, y)})
            result[name] = {'samples': int(mask.sum()), 'macro_log_loss': float(np.mean([value['log_loss'] for value in values])), 'macro_brier': float(np.mean([value['brier'] for value in values])), 'macro_ece_10bin': float(np.mean([value['ece_10bin'] for value in values])), 'per_threshold': dict(zip(TARGET_COLUMNS, values))}
        result_holder = [result]
        if distributed:
            dist.barrier()
            dist.broadcast_object_list(result_holder, src=0)
        model.train()
        return result
    started = time.monotonic()
    model.train()
    completed_segments = start_segment_index
    group_order_pos = start_group_order_pos
    row_offset = start_row_offset
    segments_this_run = 0
    segment_rows: list[dict[str, Any]] = []
    segment_processed = 0
    segment_global_start = processed
    segment_started = time.monotonic()
    longest_segment_seconds = SEGMENT_ESTIMATE_SECONDS
    stop_reason = 'segment_limit'
    validation_history = []
    probe_initial_score = best_score
    probe_best_score = best_score
    probe_stale_segments = 0
    if LR_PROBE:
        initial_validation = full_validation()
        probe_initial_score = initial_validation['all']['macro_log_loss']
        if not np.isfinite(probe_initial_score) or abs(probe_initial_score - best_score) > 0.002:
            raise RuntimeError('initial validation does not reproduce Chunk 1 baseline')
        probe_best_score = probe_initial_score
        log('probe_initial_validation', learning_rate=LEARNING_RATE_OVERRIDE, validation=initial_validation)
        if time.time() - runtime_started > GPU_BUDGET_SECONDS - RUNTIME_RESERVE_SECONDS - longest_segment_seconds:
            raise RuntimeError('initialization consumed the probe training budget')
    while group_order_pos < len(group_order) and segments_this_run < run_segment_limit:
        loaded_group_order_pos = group_order_pos
        group_id = group_order[group_order_pos]
        rows = shuffle_group_rows(train_pq.read_row_group(group_id, columns=columns).to_pylist(), group_id)
        while group_order_pos == loaded_group_order_pos and row_offset < len(rows) and (segments_this_run < run_segment_limit):
            remaining = SEGMENT_SAMPLES - len(segment_rows)
            take = min(remaining, len(rows) - row_offset)
            segment_rows.extend(rows[row_offset:row_offset + take])
            row_offset += take
            if len(segment_rows) < SEGMENT_SAMPLES:
                if row_offset == len(rows) and group_order_pos + 1 < len(group_order):
                    group_order_pos += 1
                    row_offset = 0
                    break
            global_segment_index = completed_segments + 1
            segment_samples = len(segment_rows)
            if LR_PROBE:
                identity_hash = verify_segment_coverage(segment_rows, expected_prefix, processed, seen_identities)
                log('segment_coverage_verified', segment_index=global_segment_index, samples=segment_samples, identity_sha256=identity_hash, group_order_pos=group_order_pos, row_offset=row_offset)
            segment_global_start = processed
            segment_processed = 0
            segment_started = time.monotonic()
            padded_count = (segment_samples + world_size - 1) // world_size * world_size
            padded_rows = segment_rows + [segment_rows[-1]] * (padded_count - segment_samples)
            local_rows = padded_rows[rank::world_size]
            local_valid = np.arange(rank, padded_count, world_size) < segment_samples
            local_cache = encode_token_cache(local_rows, train_arrays, check_labels=rank == 0 and completed_segments == 0)
            for offset in range(0, len(local_rows), local_batch_size):
                batch_slice = slice(offset, offset + local_batch_size)
                valid = torch.from_numpy(local_valid[batch_slice]).to(device)
                s1 = torch.from_numpy(local_cache[0][batch_slice]).to(device)
                s2 = torch.from_numpy(local_cache[1][batch_slice]).to(device)
                sector = torch.from_numpy(local_cache[2][batch_slice]).to(device)
                size = torch.from_numpy(local_cache[3][batch_slice, None]).to(device)
                target = torch.from_numpy(local_cache[4][batch_slice]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    up, down = model(s1.long(), s2.long(), size, sector)
                    per_row = F.binary_cross_entropy_with_logits(up, target[:, :4], reduction='none').mean(dim=1)
                    per_row += F.binary_cross_entropy_with_logits(down, target[:, 4:], reduction='none').mean(dim=1)
                    local_loss_sum = per_row[valid].sum()
                    global_valid_count = valid.sum().to(dtype=torch.float32)
                    if distributed:
                        dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
                    loss = local_loss_sum * world_size / global_valid_count.clamp_min(1)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                local_done = int(local_valid[:offset + local_batch_size].sum())
                estimated_segment_processed = min(segment_samples, local_done * world_size)
                estimated_processed = segment_global_start + estimated_segment_processed
                if is_main_process() and (estimated_processed % 25000 < BATCH_SIZE or offset + local_batch_size >= len(local_rows)):
                    rate = estimated_processed / max(time.monotonic() - started, 1e-06)
                    segment_processed = estimated_segment_processed
                    state = {'phase': 'training', 'processed_samples': estimated_processed, 'total_samples': train_total, 'row_group': group_id, 'order_pos': group_order_pos, 'segment_total': total_segments, 'segment_index': global_segment_index, 'segment_samples': segment_samples, 'segment_processed_samples': segment_processed, 'segment_progress': min(1.0, segment_processed / max(segment_samples, 1)), 'loss': float(loss.detach().cpu()), 'samples_per_second': rate, 'eta_seconds': (segment_samples - segment_processed) / max(rate, 1e-06)}
                    log('training_progress', **{key: value for key, value in state.items() if key != 'phase'})
                    dashboard(state)
                    swanlab_run.log({'train/loss': float(loss.detach().cpu()), 'train/processed_samples': estimated_processed, 'train/global_processed_samples': estimated_processed, 'train/global_total_samples': train_total, 'train/segment_total': total_segments, 'train/segment_index': global_segment_index, 'train/segment_processed_samples': segment_processed, 'train/segment_samples': segment_samples, 'train/segment_progress': state['segment_progress'], 'train/samples_per_second': rate, 'train/eta_seconds': state['eta_seconds']}, step=estimated_processed)
            if distributed:
                dist.barrier()
            if is_main_process():
                processed = segment_global_start + segment_samples
            if distributed:
                processed_holder = [processed if is_main_process() else None]
                dist.broadcast_object_list(processed_holder, src=0)
                processed = int(processed_holder[0])
            segment_rows = []
            completed_segments += 1
            segments_this_run += 1
            if row_offset == len(rows):
                group_order_pos += 1
                row_offset = 0
            last_validation = full_validation()
            validation_score = last_validation['all']['macro_log_loss']
            validation_history.append({'segment_index': global_segment_index, 'processed_samples': processed, 'learning_rate': optimizer.param_groups[0]['lr'], 'validation': last_validation})
            if is_main_process():
                history_temp = OUTPUT / 'validation_history.json.tmp'
                history_temp.write_text(json.dumps(validation_history, indent=2) + '\n')
                history_temp.replace(OUTPUT / 'validation_history.json')
            log('segment_validation', segment_index=global_segment_index, learning_rate=optimizer.param_groups[0]['lr'], validation=last_validation)
            last_best_updated = validation_score < best_score
            if last_best_updated:
                best_score = validation_score
                if is_main_process():
                    torch.save({'model': raw_model.state_dict(), 'metrics': last_validation, 'chunk_index': CHUNK_INDEX, 'segment_index': global_segment_index, 'processed_samples': processed}, best_checkpoint.with_suffix('.pt.tmp'))
                    best_checkpoint.with_suffix('.pt.tmp').replace(best_checkpoint)
                    best_meta.write_text(json.dumps({'macro_log_loss': best_score, 'chunk_index': CHUNK_INDEX, 'segment_index': global_segment_index, 'processed_samples': processed}, indent=2) + '\n', encoding='utf-8')
                    log('best_model_updated', macro_log_loss=best_score, segment_index=global_segment_index, processed_samples=processed)
            if is_main_process():
                swanlab_run.log({'validation/macro_log_loss': last_validation['all']['macro_log_loss'], 'validation/macro_brier': last_validation['all']['macro_brier'], 'validation/macro_ece': last_validation['all']['macro_ece_10bin'], 'validation/2025H2_log_loss': last_validation['2025H2']['macro_log_loss'], 'validation/2026H1_log_loss': last_validation['2026H1']['macro_log_loss'], 'validation/samples': last_validation['all']['samples'], 'validation/segment_complete': 1, 'validation/best_updated': int(last_best_updated), 'train/segment_total': total_segments, 'train/segment_index': global_segment_index}, step=processed)
                torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'chunk_index': CHUNK_INDEX, 'completed_segments': completed_segments, 'group_order_pos': group_order_pos, 'row_offset': row_offset, 'shuffle_seed': SHUFFLE_SEED, 'group_order_hash': group_order_hash(group_order), 'processed_samples': processed, 'last_validation': last_validation, 'last_best_updated': last_best_updated}, checkpoint.with_suffix('.pt.tmp'))
                checkpoint.with_suffix('.pt.tmp').replace(checkpoint)
            if distributed:
                dist.barrier()
            segment_seconds = time.monotonic() - segment_started
            longest_segment_seconds = max(longest_segment_seconds, segment_seconds)
            log('segment_complete', segment_index=global_segment_index, segment_samples=segment_samples, processed_samples=processed, segment_seconds=segment_seconds)
            budget_decision = [None]
            if is_main_process():
                remaining_seconds = GPU_BUDGET_SECONDS - RUNTIME_RESERVE_SECONDS - (time.time() - runtime_started)
                budget_decision[0] = remaining_seconds < longest_segment_seconds * SEGMENT_TIME_MARGIN
                log('runtime_budget', remaining_seconds=remaining_seconds, estimated_next_segment_seconds=longest_segment_seconds * SEGMENT_TIME_MARGIN, stop_before_next_segment=budget_decision[0])
            if distributed:
                dist.broadcast_object_list(budget_decision, src=0)
            if budget_decision[0]:
                run_segment_limit = segments_this_run
                stop_reason = 'runtime_budget'
            if LR_PROBE:
                reason, probe_best_score, probe_stale_segments = probe_stop_reason(validation_score, probe_best_score, probe_stale_segments, probe_initial_score)
                if reason:
                    run_segment_limit = segments_this_run
                    stop_reason = reason
                    log('probe_early_stop', reason=reason, stale_segments=probe_stale_segments)
    if completed_segments < total_segments:
        report = {'status': 'CHUNK_COMPLETE', 'purpose': RUN_PURPOSE, 'learning_rate': optimizer.param_groups[0]['lr'], 'initial_macro_log_loss': probe_initial_score if LR_PROBE else None, 'validation_history': validation_history, 'stop_reason': stop_reason, 'gpu_budget_seconds': GPU_BUDGET_SECONDS, 'runtime_elapsed_seconds': time.time() - runtime_started, 'chunk_index': CHUNK_INDEX, 'processed_samples': processed, 'completed_segments': completed_segments, 'segments_this_run': segments_this_run, 'shuffle_seed': SHUFFLE_SEED, 'group_order_hash': group_order_hash(group_order), 'segment_total': total_segments, 'segment_index': completed_segments, 'segment_samples': segment_samples, 'segment_processed_samples': max(0, processed - segment_global_start), 'checkpoint': str(checkpoint), 'best_model': str(best_checkpoint), 'best_macro_log_loss': best_score, 'validation_samples': last_validation['all']['samples'], 'best_updated': int(last_best_updated), 'validation': last_validation, 'elapsed_seconds': time.monotonic() - started}
        if is_main_process():
            (OUTPUT / 'chunk_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            dashboard({'phase': 'chunk complete', 'processed_samples': processed, 'total_samples': train_total, 'chunk_index': CHUNK_INDEX, 'completed_segments': completed_segments, 'segment_total': total_segments})
            log('chunk_complete', **report)
            swanlab_run.log({'chunk/completed': 1, 'chunk/processed_samples': processed, 'segment_complete': segments_this_run, 'validation_samples': last_validation['all']['samples'], 'validation_macro_log_loss': last_validation['all']['macro_log_loss'], 'validation_macro_brier': last_validation['all']['macro_brier'], 'validation_ece': last_validation['all']['macro_ece_10bin'], 'best_updated': int(last_best_updated)}, step=processed)
            swanlab.finish()
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    if last_validation is None:
        raise RuntimeError('no completed segment validation was produced')
    metrics = last_validation
    if is_main_process():
        torch.save({'model': raw_model.state_dict(), 'metrics': metrics}, OUTPUT / 'final_model.pt')
    report = {'status': 'COMPLETE', 'purpose': 'full-temporal-training', 'train_samples': train_total, 'validation_samples': validation_total, 'epochs': 1, 'batch_size': BATCH_SIZE, 'metrics': metrics, 'checkpoint': str(OUTPUT / 'final_model.pt'), 'best_model': str(best_checkpoint), 'best_macro_log_loss': best_score, 'segment_total': total_segments, 'validation_runs': completed_segments, 'last_segment_validation': last_validation, 'model_selection': 'final_checkpoint; best_model_selected_on_full_segment_validation', 'elapsed_seconds': time.monotonic() - started}
    if is_main_process():
        (OUTPUT / 'full_training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        dashboard({'phase': 'complete', 'processed_samples': processed, 'total_samples': train_total, 'metrics': metrics})
        swanlab_run.log({'validation/macro_log_loss': metrics['all']['macro_log_loss'], 'validation/macro_brier': metrics['all']['macro_brier'], 'validation/2025H2_log_loss': metrics['2025H2']['macro_log_loss'], 'validation/2026H1_log_loss': metrics['2026H1']['macro_log_loss'], 'validation/completed': 1}, step=processed)
        swanlab.finish()
        log('full_training_complete', **report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
