"""Offline, no-update Stage3 verification on C2 best and Qlib validation.

This checks a teacher-forced, truncated candidate-mixture surrogate, not an
unbiased expectation over autoregressive paths or out-of-sample accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate
from model import Kronos, KronosTokenizer
from finetune.dataset import QlibDataset
from finetune.stage3_path_alignment import PathAlignmentConfig, compute_path_alignment_loss
from finetune.stage3_training_model import Stage3TrainingModel, gradient_metrics


def norm(module):
    grads = [p.grad.detach().double() for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    result = float(sum(g.square().sum() for g in grads).sqrt())
    assert result > 0
    return result


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
    root = Path(args.dataset_root).resolve()
    os.environ.update({
        'KRONOS_DATASET_PATH': str(root / 'processed_datasets'),
        'KRONOS_METADATA_PATH': str(root / 'asset_metadata.csv'),
        'KRONOS_LOOKBACK_WINDOW': '120', 'KRONOS_PREDICT_WINDOW': '10',
        'KRONOS_USE_SIZE_PERCENTILE': '1', 'KRONOS_NUM_SIZE_BUCKETS': '0',
        'KRONOS_VALIDATION_SAMPLES': '0', 'KRONOS_VAL_SIGNAL_START': '2025-07-01',
        'KRONOS_VAL_SIGNAL_END': '2026-07-02',
    })
    ds = QlibDataset('val')
    assert len(ds) == ds.total_samples
    model = Kronos.from_pretrained(args.model_dir).float().eval()
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer_dir).float().eval()
    tokenizer.requires_grad_(False)
    indices = torch.linspace(0, len(ds) - 1, args.samples).long().tolist()
    rows = []
    for offset in range(0, len(indices), args.batch):
        chosen = indices[offset:offset + args.batch]
        x, stamp, sec, _, pct = default_collate([ds[i] for i in chosen])
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            s1, s2 = tokenizer.encode(x.float(), half=True)
        context = model.encode_context(s1[:, :-1], s2[:, :-1], stamp[:, :-1],
                                       sector_id=sec, size_percentile=pct)
        logits = model.predict_s1(context[:, 119:129])
        logits.retain_grad()
        loss, metrics = compute_path_alignment_loss(
            model, tokenizer, context, logits, x[:, 120:130],
            PathAlignmentConfig(), position_start=119, is_causal=True,
        )
        metrics['s2_conditional_logits'].retain_grad()
        loss.backward()
        grad_norms = {
            's1_head': norm(model.head.proj_s1),
            's2_head': norm(model.head.proj_s2),
            'dependency_layer': norm(model.dep_layer),
            'shared_transformer': norm(model.transformer),
        }
        assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
        s2grad = metrics['s2_conditional_logits'].grad
        assert torch.isfinite(s2grad).all() and s2grad.abs().sum() > 0
        assert all(p.grad is None for p in tokenizer.parameters())
        with torch.no_grad():
            _, top1 = logits.topk(16, dim=-1)
            conditional = model.predict_s2_candidates(context, top1, position_start=119, is_causal=True)
            reference_ids = s1[:, 1:].clone()
            reference_ids[:, 119:129] = top1[..., 0]
            reference = model.predict_s2(context, reference_ids, is_causal=True)[:, 119:129]
            max_diff = float((conditional[:, :, 0] - reference).abs().max())
            torch.testing.assert_close(conditional[:, :, 0], reference, atol=2e-5, rtol=2e-5)
            sensitivity = float((conditional[:, :, 0] - conditional[:, :, 1]).abs().max())
            assert sensitivity > 0
            changed_context = context.clone()
            changed_context[:, 129] += torch.randn_like(changed_context[:, 129]) * 10
            changed_conditional = model.predict_s2_candidates(
                changed_context, top1, position_start=119, is_causal=True)
            future_diff = float((conditional - changed_conditional).abs().max())
            assert future_diff == 0
            logp = metrics['selected_joint_logp'].detach().double()
            paths = metrics['candidate_paths'].double()
            target = x[:, 120:130].double()
        # Finite difference through the actual selected log-probabilities,
        # keeping discrete support and decoded anchors fixed as in training.
        logp.requires_grad_(True)
        def objective(z):
            pred = (paths * z.softmax(-1).unsqueeze(-2)).sum(-1)
            return torch.nn.functional.huber_loss(pred, target, delta=0.02)
        grad, = torch.autograd.grad(objective(logp), logp)
        index = int(grad.abs().argmax())
        shift = torch.zeros_like(logp); shift.view(-1)[index] = 1e-4
        fd = (objective(logp + shift) - objective(logp - shift)) / 2e-4
        expected = grad.flatten()[index]
        torch.testing.assert_close(fd, expected, atol=1e-8, rtol=1e-4)
        row = {
            'validation_indices': chosen, 'prediction_shape': list(metrics['path_align_prediction'].shape),
            'conditional_forward_max_abs_diff': max_diff,
            's1_candidate_change_s2_max_abs_diff': sensitivity,
            'causal_eval_future_context_max_abs_diff': future_diff,
            'path_only_grad_norms': grad_norms, 'tokenizer_nonnull_grads': 0,
            'path_huber_raw': float(metrics['path_align_loss']),
            'horizon_mae_normalized_six_features': metrics['path_align_horizon_mae'].tolist(),
            'max_residual_normalized': float(metrics['path_align_max_residual']),
            'finite_difference': float(fd.detach()), 'autograd': float(expected.detach()),
            'selected_joint_probability_mass_mean': float(logp.detach().exp().sum(-1).mean()),
            'seconds': time.perf_counter() - started,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Real C2 weights through the complete training wrapper (no optimizer step).
    core = Stage3TrainingModel(model, tokenizer, synchronize_ema=False).train()
    core.zero_grad(set_to_none=True)
    total, metrics = core(x, stamp, sector_id=sec, size_percentile=pct)
    assert torch.isfinite(total)
    total.backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in tokenizer.parameters())
    wrapper_result = {key: value.tolist() for key, value in metrics.items()}
    wrapper_result['gradients'] = {key: float(value) for key, value in gradient_metrics(model).items()}
    wrapper_result['missing_predictor_gradients'] = missing
    wrapper_result['tokenizer_training'] = tokenizer.training
    print('TRAINING_WRAPPER=' + json.dumps(wrapper_result), flush=True)
    model_path = Path(args.model_dir) / 'model.safetensors'
    with model_path.open('rb') as handle:
        model_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    report = {
        'model_dir': str(Path(args.model_dir).resolve()),
        'model_sha256': model_sha256,
        'validation_total': len(ds), 'tested_samples': len(indices),
        'optimizer_steps': 0, 'oos_used': False, 'device': 'cpu', 'dtype': 'float32',
        'mode': 'explicit causal eval plus complete training-wrapper backward',
        'limitations': ['Not a GPU memory or DDP test', 'Not a full validation evaluation',
                        'Teacher-forced horizon-wise truncated mixture, not full-path expectation',
                        'Loss is normalized six-feature Huber, not raw-return Huber'],
        'batches': rows, 'training_wrapper': wrapper_result, 'status': 'passed',
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT=' + args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
