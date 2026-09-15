"""NCCL correctness/memory gate. Disposable updates; never a training parent."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from model import Kronos, KronosTokenizer
from finetune.dataset import QlibDataset
from finetune.stage3_training_model import Stage3TrainingModel, gradient_metrics


def main(a):
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260915)
    device = torch.device('cuda', local)
    print(json.dumps({'phase': 'gpu_probe_load', 'rank': rank, 'device': torch.cuda.get_device_name(local)}), flush=True)
    ds = QlibDataset('val')
    assert len(ds) == ds.total_samples == 123836
    def batch(indices):
        fields = default_collate([ds[i] for i in indices])
        return tuple(fields[i].to(device) for i in (0, 1, 2, 4))
    core = Stage3TrainingModel(Kronos.from_pretrained(a.model_dir),
                              KronosTokenizer.from_pretrained(a.tokenizer_dir)).to(device).train()
    # Deterministic microbatch reference only; real training restores C2 dropout
    # in a new process and loads C2 weights again, not these probe updates.
    for module in core.predictor.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
        if hasattr(module, 'attn_dropout_p'): module.attn_dropout_p = 0.
    ref = copy.deepcopy(core); ref.synchronize_ema = False
    ddp = DDP(core, device_ids=[local], broadcast_buffers=False)
    x, stamp, sec, pct = batch([0, len(ds) // 2])
    loss, _ = ddp(x[rank:rank+1], stamp[rank:rank+1], sector_id=sec[rank:rank+1], size_percentile=pct[rank:rank+1])
    loss.backward()
    ref(x, stamp, sector_id=sec, size_percentile=pct)[0].backward()
    grad_diff = torch.zeros((), device=device)
    for p, q in zip(core.predictor.parameters(), ref.predictor.parameters()):
        if not p.requires_grad: continue
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=2e-3)
        grad_diff = torch.maximum(grad_diff, (p.grad-q.grad).abs().max())
    dist.all_reduce(grad_diff, op=dist.ReduceOp.MAX)
    del ref
    ddp.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    # Actual requested per-GPU batch, including optimizer states and decoder.
    ids = torch.linspace(0, len(ds)-1, a.batch * world).long().tolist()
    x, stamp, sec, pct = batch(ids[rank*a.batch:(rank+1)*a.batch])
    opt = torch.optim.AdamW(core.predictor.parameters(), lr=2e-6, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
    loss, metrics = ddp(x, stamp, sector_id=sec, size_percentile=pct)
    assert torch.isfinite(loss)
    loss.backward()
    grads = gradient_metrics(core.predictor)
    assert all(torch.isfinite(v) and v > 0 for v in grads.values())
    torch.nn.utils.clip_grad_norm_(core.predictor.parameters(), 1., error_if_nonfinite=True)
    opt.step(); torch.cuda.synchronize()
    train_seconds = time.monotonic() - started
    flat = torch.cat([p.detach().flatten() for p in core.predictor.parameters()])
    other = flat.clone(); dist.broadcast(other, src=0)
    param_diff = (flat-other).abs().max(); dist.all_reduce(param_diff, op=dist.ReduceOp.MAX)
    assert float(param_diff) == 0
    with tempfile.TemporaryDirectory(prefix='stage3-gpu-checkpoint-') as temp:
        path = Path(temp) / 'state.pt'
        torch.save(core.checkpoint_state(opt, 1, 0), path)
        state = torch.load(path, map_location=device, weights_only=True)
        core.load_checkpoint_state(state, opt)
        restored = torch.cat([p.detach().flatten() for p in core.predictor.parameters()])
        torch.testing.assert_close(flat, restored, atol=0, rtol=0)
        del state, restored
    del flat, other
    ddp.zero_grad(set_to_none=True); core.eval()
    torch.cuda.synchronize(); started = time.monotonic()
    with torch.no_grad(): core(x, stamp, sector_id=sec, size_percentile=pct)
    torch.cuda.synchronize(); val_seconds = time.monotonic() - started
    stats = torch.tensor([train_seconds, val_seconds, torch.cuda.max_memory_allocated()/2**30], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    result = {'status': 'passed', 'devices': [torch.cuda.get_device_name(i) for i in range(2)],
              'batch_per_gpu': a.batch, 'global_batch': a.batch * world, 'amp': False,
              'gradient_max_diff_vs_global_reference': float(grad_diff),
              'parameter_max_diff_between_ranks': float(param_diff), 'checkpoint_roundtrip': True,
              'train_step_seconds': float(stats[0]), 'validation_batch_seconds': float(stats[1]),
              'peak_allocated_gib': float(stats[2]), 'tokenizer_nonnull_grads': sum(p.grad is not None for p in core.tokenizer.parameters()),
              'estimated_segment_seconds': float(stats[0])*313 + float(stats[1])*1935,
              'probe_updates_are_discarded': True}
    if result['peak_allocated_gib'] > 12.8:
        raise RuntimeError('Probe exceeded 80% T4 VRAM safety gate; no segment launched')
    if result['estimated_segment_seconds'] > 10800:
        raise RuntimeError('Probe exceeds three-hour segment budget; no segment launched')
    if rank == 0:
        Path(a.output).write_text(json.dumps(result, indent=2))
        print(json.dumps({'phase': 'gpu_probe_passed', **result}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True); p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--batch', type=int, default=32)
    main(p.parse_args())
