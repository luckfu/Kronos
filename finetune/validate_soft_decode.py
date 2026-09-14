"""Read-only differentiable soft-decode acceptance test for Stage 3.

This script does not train or save weights.  It compares the existing hard
tokenizer decode with a differentiable expectation over the binary token bits,
then verifies that a numeric path loss reaches Predictor logits/backbone while
tokenizer parameters remain frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.kronos import Kronos, KronosTokenizer


def _bits_expectation(logits: torch.Tensor) -> torch.Tensor:
    """Expected binary code bits under a categorical token distribution."""
    probs = logits.float().softmax(dim=-1)
    vocab = probs.shape[-1]
    ids = torch.arange(vocab, device=logits.device, dtype=torch.long)
    nbits = max(1, (vocab - 1).bit_length())
    masks = (1 << torch.arange(nbits, device=logits.device)).long()
    bits = (ids[:, None] & masks[None, :]).to(probs.dtype)
    return probs @ bits


def soft_decode(tokenizer: KronosTokenizer, s1_logits: torch.Tensor,
                s2_logits: torch.Tensor) -> torch.Tensor:
    """Decode expected bits through the frozen tokenizer decoder."""
    s1_bits = _bits_expectation(s1_logits)[..., : tokenizer.s1_bits]
    s2_bits = _bits_expectation(s2_logits)[..., : tokenizer.s2_bits]
    # KronosTokenizer.indices_to_bits maps {0,1} to {-1,+1} and applies
    # the BSQ spherical scale.  The soft path must preserve that exact map.
    quantized = torch.cat([s1_bits, s2_bits], dim=-1)
    quantized = (quantized * 2.0 - 1.0) / (tokenizer.codebook_dim ** 0.5)
    z = tokenizer.post_quant_embed(quantized)
    for layer in tokenizer.decoder:
        z = layer(z)
    return tokenizer.head(z)


def sampled_mixture_decode(tokenizer: KronosTokenizer, s1_logits: torch.Tensor,
                           s2_logits: torch.Tensor, top_k: int = 8,
                           samples: int = 16) -> torch.Tensor:
    """Decode fixed top-k candidate paths and combine by differentiable weights.

    Candidate token ids are selected without replacement from the top-k support;
    tokenizer decoding is hard per candidate, while the probability-weighted
    mixture remains differentiable through candidate weights. This is a
    feasibility probe, not yet the Stage 3 training implementation.
    """
    b, t, _ = s1_logits.shape
    k1 = min(top_k, s1_logits.shape[-1])
    k2 = min(top_k, s2_logits.shape[-1])
    p1 = s1_logits.float().softmax(-1)
    p2 = s2_logits.float().softmax(-1)
    v1, i1 = torch.topk(p1, k1, dim=-1)
    v2, i2 = torch.topk(p2, k2, dim=-1)
    # Deterministic rank-based candidates avoid random nondeterminism in the
    # acceptance test while still covering distinct support points.
    candidate_outputs, candidate_weights = [], []
    for n in range(max(1, samples)):
        # Enumerate the Cartesian top-k support without repeating the same
        # pair when samples > k.  The full support has k1*k2 combinations.
        r1 = n % k1
        r2 = (n // k1) % k2
        ids1, ids2 = i1[..., r1], i2[..., r2]
        weight = v1[..., r1] * v2[..., r2]
        candidate_outputs.append(tokenizer.decode((ids1, ids2), half=True).float())
        candidate_weights.append(weight)
    weights = torch.stack(candidate_weights, dim=0)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (torch.stack(candidate_outputs, dim=0) * weights[..., None]).sum(dim=0)


def load_from_dirs(model_dir: Path, tokenizer_dir: Path, device: torch.device):
    model = Kronos.from_pretrained(str(model_dir)).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir)).to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    model, tokenizer = load_from_dirs(Path(args.model_dir), Path(args.tokenizer_dir), device)
    torch.manual_seed(args.seed)

    sector = size_percentile = target = None
    if args.val_data:
        os.environ.update({
            "KRONOS_VAL_DATA_PATHS": str(Path(args.val_data).resolve()),
            "KRONOS_LOOKBACK_WINDOW": "120",
            "KRONOS_PREDICT_WINDOW": "10",
            "KRONOS_USE_SIZE_PERCENTILE": "1",
            "KRONOS_NUM_SIZE_BUCKETS": "0",
            "KRONOS_METADATA_PATH": str(Path(args.metadata).resolve()),
            "KRONOS_VAL_SIGNAL_START": args.signal_start,
            "KRONOS_VAL_SIGNAL_END": args.signal_end,
        })
        from finetune.dataset import QlibDataset
        dataset = QlibDataset("val")
        positions = torch.linspace(0, len(dataset) - 1, args.sample_count).long().tolist()
        records = [dataset[position] for position in positions]
        batch_x = torch.stack([record[0] for record in records]).to(device)
        stamp = torch.stack([record[1] for record in records]).to(device)
        sector = torch.stack([record[2] for record in records]).to(device)
        size_percentile = torch.stack([record[4] for record in records]).to(device)
        target = batch_x[:, -10:].float()
        with torch.no_grad():
            s1, s2 = tokenizer.encode(batch_x, half=True)
        source = "validation"
    else:
        # Synthetic normalized window probes only graph connectivity.
        batch, seq = 2, 130
        s1 = torch.randint(0, 2 ** model.s1_bits, (batch, seq), device=device)
        s2 = torch.randint(0, 2 ** model.s2_bits, (batch, seq), device=device)
        stamp = torch.zeros(batch, seq, 5, device=device)
        source = "synthetic"
    with torch.enable_grad():
        output = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1],
                       use_teacher_forcing=True,
                       s1_targets=s1[:, 1:], sector_id=sector,
                       size_percentile=size_percentile)
        s1_logits, s2_logits = output
        # Probe the final ten teacher-forced positions.
        s1_probe, s2_probe = s1_logits[:, -10:], s2_logits[:, -10:]
        hard = tokenizer.decode((s1_probe.argmax(-1), s2_probe.argmax(-1)), half=True)
        soft = soft_decode(tokenizer, s1_probe, s2_probe)
        mixture = sampled_mixture_decode(tokenizer, s1_probe, s2_probe,
                                         top_k=args.top_k,
                                         samples=args.mixture_samples)
        diff = (soft.float() - hard.float()).abs()
        mixture_diff = (mixture.float() - hard.float()).abs()
        confidence = torch.cat((s1_probe.float().log_softmax(-1).exp().amax(-1),
                                s2_probe.float().log_softmax(-1).exp().amax(-1)), dim=-1)
        print(json.dumps({
            "device": str(device),
            "source": source,
            "samples": int(s1.shape[0]),
            "hard_shape": list(hard.shape),
            "soft_shape": list(soft.shape),
            "mean_abs_diff": float(diff.mean().detach()),
            "median_abs_diff": float(diff.median().detach()),
            "p95_abs_diff": float(torch.quantile(diff.flatten(), 0.95).detach()),
            "max_abs_diff": float(diff.max().detach()),
            "hard_abs_max": float(hard.float().abs().max().detach()),
            "soft_abs_max": float(soft.float().abs().max().detach()),
            "mixture_mean_abs_diff": float(mixture_diff.mean().detach()),
            "mixture_p95_abs_diff": float(torch.quantile(mixture_diff.flatten(), 0.95).detach()),
            "mixture_abs_max": float(mixture.float().abs().max().detach()),
            "soft_finite": bool(torch.isfinite(soft).all()),
            "hard_finite": bool(torch.isfinite(hard).all()),
            "mean_token_confidence": float(confidence.mean().detach()),
        }, ensure_ascii=False, indent=2))

        # Realistic differentiable path probe: every horizon and every output
        # feature participates; the target is deliberately finite and local.
        if target is None:
            target = torch.zeros_like(soft)
        loss = F.huber_loss(mixture, target, delta=0.02)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_values = [p.grad.detach().float().norm() for p in model.parameters()
                       if p.grad is not None]
        tokenizer_grads = [p.grad for p in tokenizer.parameters() if p.grad is not None]
        grad_norm = torch.linalg.vector_norm(torch.stack(grad_values)) if grad_values else torch.tensor(float("nan"))
        print(json.dumps({
            "path_huber_loss": float(loss.detach()),
            "predictor_grad_norm": float(grad_norm),
            "predictor_grad_finite": bool(torch.isfinite(grad_norm)),
            "tokenizer_nonnull_grads": len(tokenizer_grads),
            "soft_prediction_collapsed_to_zero": bool(soft.detach().abs().max() < 1e-8),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--val-data")
    parser.add_argument("--metadata")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--signal-start", default="2025-07-03")
    parser.add_argument("--signal-end", default="2026-07-02")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mixture-samples", type=int, default=16)
    raise SystemExit(run(parser.parse_args()))
