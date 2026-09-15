"""Structure-preserving multi-horizon path-alignment objective.

This module is intentionally separate from the legacy/Beta-v2.1 objective.
It adds no model parameters: logits are decoded through fixed candidate paths
and compared with the ten teacher-forced normalized target rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class PathAlignmentConfig:
    top_k: int = 16
    candidates: int = 16
    weight: float = 0.05
    huber_delta: float = 0.02
    ema_decay: float = 0.99


class DetachedLossEMA:
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value = None

    def normalize(self, loss: torch.Tensor, sample_count=1, synchronize=False) -> torch.Tensor:
        current = loss.detach().float()
        if synchronize and dist.is_available() and dist.is_initialized():
            # A shared denominator is necessary for equivalence to a single
            # global-batch update. Rank-local EMA would reweight each shard.
            stats = torch.stack((current * sample_count, current.new_tensor(float(sample_count))))
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            current = stats[0] / stats[1].clamp_min(1)
        previous = None if self.value is None else self.value.to(current.device)
        self.value = current if previous is None else self.decay * previous + (1 - self.decay) * current
        return loss / self.value.clamp_min(1e-6).to(loss.dtype)

    def state_dict(self):
        return {'decay': self.decay, 'value': None if self.value is None else self.value.detach().cpu()}

    def load_state_dict(self, state):
        self.decay = float(state.get('decay', self.decay))
        value = state.get('value')
        self.value = None if value is None else torch.as_tensor(value).float()


def candidate_mixture_decode(model, tokenizer, context, s1_logits, top_k=16, candidates=16, position_start=None, is_causal=None):
    """Decode a horizon-wise truncated mixture using P(s1)P(s2|context,s1).

    Candidate rank n across horizons forms a deterministic decoder anchor.
    Weights are per horizon, not probabilities of full autoregressive paths;
    the temporal decoder makes this a surrogate, not an exact expectation.
    """
    if int(top_k) < 1 or int(candidates) < 1:
        raise ValueError('top_k and candidates must be positive')
    k1 = min(int(top_k), s1_logits.shape[-1])
    p1 = s1_logits.float().softmax(-1)
    v1, i1 = torch.topk(p1, k1, dim=-1)
    s2_cond = model.predict_s2_candidates(context, i1, position_start=position_start, is_causal=is_causal)
    p2 = s2_cond.float().softmax(-1)
    k2 = min(int(top_k), p2.shape[-1])
    v2, i2 = torch.topk(p2, k2, dim=-1)
    joint_logp = v1.clamp_min(1e-12).log()[..., :, None] + v2.clamp_min(1e-12).log()
    flat = joint_logp.reshape(*joint_logp.shape[:-2], -1)
    selected = min(max(1, int(candidates)), flat.shape[-1])
    top_logp, top_idx = torch.topk(flat, selected, dim=-1)
    r1 = top_idx // k2
    s1_ids = i1.gather(-1, r1)
    s2_ids = i2.flatten(-2).gather(-1, top_idx)
    outputs = []
    with torch.no_grad():
        for n in range(selected):
            outputs.append(tokenizer.decode((s1_ids[..., n], s2_ids[..., n]), half=True).float())
    weights = top_logp.softmax(-1)
    prediction = (torch.stack(outputs, dim=-1) * weights.unsqueeze(-2)).sum(-1)
    return prediction, weights.detach(), {
        'joint_candidate_count': selected, 's2_conditional_logits': s2_cond,
        'selected_s1_ids': s1_ids, 'selected_s2_ids': s2_ids,
        'selected_joint_logp': top_logp,
        'candidate_paths': torch.stack(outputs, dim=-1),
    }


def compute_path_alignment_loss(model, tokenizer, context, s1_logits, target,
                               config: PathAlignmentConfig, normalizer: DetachedLossEMA | None = None,
                               position_start=None, is_causal=None, synchronize_ema=False):
    """Return normalized loss and per-horizon MAE for target [B,10,6]."""
    prediction, weights, extra = candidate_mixture_decode(
        model, tokenizer, context, s1_logits, config.top_k, config.candidates, position_start, is_causal
    )
    target = target.to(prediction.dtype)
    errors = prediction - target
    per_horizon = F.huber_loss(prediction, target, delta=config.huber_delta, reduction='none').mean(dim=(0, 2))
    loss = per_horizon.mean()
    normalized = normalizer.normalize(loss, sample_count=target.shape[0], synchronize=synchronize_ema) if normalizer is not None else loss
    return config.weight * normalized, {
        'path_align_loss': loss.detach(),
        'path_align_normalized': normalized.detach(),
        'path_align_horizon_mae': errors.detach().abs().mean(dim=(0, 2)),
        'path_align_max_residual': errors.detach().abs().max(),
        'path_align_prediction': prediction,
        'path_align_candidate_weights': weights,
        **extra,
    }
