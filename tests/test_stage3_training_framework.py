"""CPU/Gloo acceptance gates; synthetic tensors only, never production updates."""
import copy
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, Subset

from model import Kronos, KronosTokenizer
from finetune.stage3_training_model import Stage3TrainingModel
from finetune.train_stage3_path_alignment import evaluate


def make_model(dropout=0., synchronize_ema=True):
    torch.manual_seed(715)
    predictor = Kronos(4, 4, 2, 32, 4, 64, dropout, dropout, dropout, dropout, False)
    tokenizer = KronosTokenizer(6, 32, 4, 64, 2, 2, dropout, dropout, dropout,
                                4, 4, 0.25, 1., 1., 1., 4)
    return Stage3TrainingModel(predictor, tokenizer, lookback=7, horizon=10,
                               synchronize_ema=synchronize_ema)


def batch(n=4):
    generator = torch.Generator().manual_seed(617)
    x = torch.randn(n, 18, 6, generator=generator)
    # Valid minute/hour/weekday/day/month indices for all temporal embeddings.
    stamps = torch.zeros(n, 18, 5)
    return x, stamps


def test_explicit_causal_eval_and_legacy_unchanged():
    core = make_model(dropout=0.25).eval()
    m = core.predictor
    context = torch.randn(2, 17, 32)
    future_changed = context.clone(); future_changed[:, -1] += 10 * torch.randn(2, 32)
    ids = torch.randint(0, 16, (2, 10, 4))
    causal = m.predict_s2_candidates(context, ids, position_start=6, is_causal=True)
    causal_changed = m.predict_s2_candidates(future_changed, ids, position_start=6, is_causal=True)
    torch.testing.assert_close(causal, causal_changed, atol=0, rtol=0)
    # Repeated eval calls must be identical even with configured dropout > 0.
    torch.testing.assert_close(causal, m.predict_s2_candidates(context, ids, position_start=6, is_causal=True), atol=0, rtol=0)
    legacy = m.predict_s2_candidates(context, ids, position_start=6)
    explicit_noncausal = m.predict_s2_candidates(context, ids, position_start=6, is_causal=False)
    torch.testing.assert_close(legacy, explicit_noncausal, atol=0, rtol=0)
    assert (legacy - m.predict_s2_candidates(future_changed, ids, position_start=6)).abs().max() > 0
    # Both CE and joint candidates receive the same explicit causal override.
    full_ids = torch.randint(0, 16, (2, 17))
    for k in range(4):
        full_ids[:, 6:16] = ids[..., k]
        torch.testing.assert_close(causal[:, :, k], m.predict_s2(context, full_ids, is_causal=True)[:, 6:16])


def test_full_objective_eval_causality_no_dropout_no_ema_update():
    core = make_model(dropout=0.25).train()
    assert not core.tokenizer.training
    x, stamps = batch()
    core(x, stamps)  # Initializes training EMA but performs no optimizer step.
    ema_before = core.path_ema.value.clone()
    core.eval()
    assert all(not module.training for module in core.modules())
    _, metrics = core(x, stamps)
    _, repeated = core(x, stamps)
    for name in metrics:
        torch.testing.assert_close(metrics[name], repeated[name], atol=0, rtol=0)
    changed = x.clone(); changed[:, 16:] += 3.
    _, perturbed = core(changed, stamps)
    # Target rows 7..15 must not be affected by later rows 16 and 17.
    torch.testing.assert_close(metrics['horizon_mae'][:9], perturbed['horizon_mae'][:9], atol=0, rtol=0)
    torch.testing.assert_close(core.path_ema.value, ema_before, atol=0, rtol=0)
    assert not any(value.requires_grad for value in metrics.values())


def test_checkpoint_roundtrip_and_following_update(tmp_path):
    core = make_model(synchronize_ema=False).train()
    x, stamps = batch()
    optimizer = torch.optim.AdamW(core.predictor.parameters(), lr=2e-6)
    loss, metrics = core(x, stamps)
    torch.testing.assert_close(loss, metrics['token_loss'] + metrics['weighted_path_loss'])
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    path = tmp_path / 'last_state.pt'
    torch.save(core.checkpoint_state(optimizer, 1, 0), path)
    restored = make_model(synchronize_ema=False).train()
    restored_opt = torch.optim.AdamW(restored.predictor.parameters(), lr=2e-6)
    assert restored.load_checkpoint_state(torch.load(path, weights_only=True), restored_opt) == (1, 0)
    for obj, opt in ((core, optimizer), (restored, restored_opt)):
        obj(x, stamps)[0].backward(); opt.step()
    for a, b in zip(core.predictor.parameters(), restored.predictor.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(core.path_ema.value, restored.path_ema.value, atol=0, rtol=0)
    restored.predictor.save_pretrained(tmp_path / 'last_model')
    production = Kronos.from_pretrained(tmp_path / 'last_model')
    assert production.state_dict().keys() == core.predictor.state_dict().keys()
    for name, value in restored.predictor.state_dict().items():
        torch.testing.assert_close(production.state_dict()[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match='Not a compatible'):
        restored.load_checkpoint_state({'model': core.predictor.state_dict()}, restored_opt)


def _ddp_worker(rank, rendezvous, report_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        core = make_model().train()
        reference = copy.deepcopy(core)
        reference.synchronize_ema = False
        ddp = DDP(core, find_unused_parameters=False, broadcast_buffers=False)
        opt = torch.optim.AdamW(core.predictor.parameters(), lr=2e-6)
        ref_opt = torch.optim.AdamW(reference.predictor.parameters(), lr=2e-6)
        x, stamps = batch()
        rows = []
        for step in range(2):
            opt.zero_grad(set_to_none=True); ref_opt.zero_grad(set_to_none=True)
            this_x = x + step * 0.13
            loss, _ = ddp(this_x[rank * 2:(rank + 1) * 2], stamps[rank * 2:(rank + 1) * 2])
            loss.backward()
            ref_loss, _ = reference(this_x, stamps)
            ref_loss.backward()
            average_loss = loss.detach().clone(); dist.all_reduce(average_loss); average_loss /= 2
            torch.testing.assert_close(average_loss, ref_loss.detach(), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(core.path_ema.value, reference.path_ema.value, atol=1e-8, rtol=2e-6)
            max_grad_diff = 0.
            for (name, p), (_, q) in zip(core.predictor.named_parameters(), reference.predictor.named_parameters()):
                if not p.requires_grad: continue
                assert p.grad is not None and q.grad is not None, name
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)
                max_grad_diff = max(max_grad_diff, float((p.grad - q.grad).abs().max()))
            assert all(p.grad is None for p in core.tokenizer.parameters())
            opt.step(); ref_opt.step()
            local = torch.cat([p.detach().flatten() for p in core.predictor.parameters()])
            all_params = [torch.zeros_like(local) for _ in range(2)]
            dist.all_gather(all_params, local)
            torch.testing.assert_close(all_params[0], all_params[1], atol=0, rtol=0)
            reference_params = torch.cat([p.detach().flatten() for p in reference.predictor.parameters()])
            torch.testing.assert_close(local, reference_params, atol=2e-6, rtol=2e-6)
            rows.append({'step': step + 1, 'global_loss': float(average_loss),
                         'gradient_max_diff_vs_global_batch': max_grad_diff,
                         'parameter_max_diff_between_ranks': float((all_params[0] - all_params[1]).abs().max()),
                         'parameter_max_diff_vs_global_batch': float((local - reference_params).abs().max()),
                         'ema': float(core.path_ema.value)})
        # Five samples split 3/2: validate exactly once, no padded duplicate.
        vx, vs = batch(5)
        dataset = TensorDataset(vx, vs)
        loader = DataLoader(Subset(dataset, range(rank, len(dataset), 2)), batch_size=2)
        result = evaluate(ddp, loader, torch.device('cpu'), 2, rank)
        ref_result = evaluate(reference, DataLoader(dataset, batch_size=2), torch.device('cpu'), 1, rank)
        assert result['samples'] == ref_result['samples'] == 5
        for key in ('token_loss', 'raw_path_loss', 'total_loss', 'horizon_mae'):
            torch.testing.assert_close(torch.tensor(result[key]), torch.tensor(ref_result[key]), atol=2e-6, rtol=2e-5)
        if rank == 0:
            Path(report_path).write_text(json.dumps({'steps': rows, 'validation_samples': result['samples']}, indent=2))
    finally:
        dist.destroy_process_group()


def test_ddp_matches_global_batch_and_exact_validation(tmp_path):
    rendezvous = str(tmp_path / 'rendezvous')
    report = str(tmp_path / 'ddp_report.json')
    mp.spawn(_ddp_worker, args=(rendezvous, report), nprocs=2, join=True)
    print(Path(report).read_text())
