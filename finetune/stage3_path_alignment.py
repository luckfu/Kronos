"""Structure-preserving multi-horizon path-alignment objective.

This module is intentionally separate from the legacy/Beta-v2.1 objective.
It adds no model parameters: logits are decoded through fixed candidate paths
and compared with the ten teacher-forced normalized target rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
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

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        current = loss.detach().float()
        self.value = current if self.value is None else self.decay * self.value + (1 - self.decay) * current
        return loss / self.value.clamp_min(1e-6).to(loss.dtype)

    def state_dict(self):
        return {'decay': self.decay, 'value': None if self.value is None else self.value.detach().cpu()}

    def load_state_dict(self, state):
        self.decay = float(state.get('decay', self.decay))
        value = state.get('value')
        self.value = None if value is None else torch.as_tensor(value).float()


def candidate_mixture_decode(tokenizer, s1_logits, s2_logits, top_k=16, candidates=16):
    """Decode distinct top-k token pairs and mix by their probability mass.

    Token ids are selected from a fixed top-k support (non-differentiable);
    gradients flow through the candidate probability weights. This avoids
    feeding off-manifold fractional bits into the nonlinear tokenizer decoder.
    """
    k1, k2 = min(int(top_k), s1_logits.shape[-1]), min(int(top_k), s2_logits.shape[-1])
    p1, p2 = s1_logits.float().softmax(-1), s2_logits.float().softmax(-1)
    v1, i1 = torch.topk(p1, k1, dim=-1); v2, i2 = torch.topk(p2, k2, dim=-1)
    outputs, weights = [], []
    for n in range(min(max(1, int(candidates)), k1 * k2)):
        r1, r2 = n % k1, (n // k1) % k2
        outputs.append(tokenizer.decode((i1[..., r1], i2[..., r2]), half=True).float())
        weights.append(v1[..., r1] * v2[..., r2])
    w = torch.stack(weights); w = w / w.sum(0, keepdim=True).clamp_min(1e-8)
    return (torch.stack(outputs) * w[..., None]).sum(0), w.detach()


def compute_path_alignment_loss(tokenizer, s1_logits, s2_logits, target,
                               config: PathAlignmentConfig, normalizer: DetachedLossEMA | None = None):
    """Return normalized loss and per-horizon MAE for target [B,10,6]."""
    prediction, weights = candidate_mixture_decode(
        tokenizer, s1_logits, s2_logits, config.top_k, config.candidates
    )
    target = target.to(prediction.dtype)
    errors = prediction - target
    per_horizon = F.huber_loss(prediction, target, delta=config.huber_delta, reduction='none').mean(dim=(0, 2))
    loss = per_horizon.mean()
    normalized = normalizer.normalize(loss) if normalizer is not None else loss
    return config.weight * normalized, {
        'path_align_loss': loss.detach(),
        'path_align_normalized': normalized.detach(),
        'path_align_horizon_mae': errors.detach().abs().mean(dim=(0, 2)),
        'path_align_max_residual': errors.detach().abs().max(),
        'path_align_prediction': prediction,
        'path_align_candidate_weights': weights,
    }
