"""Losses and validation statistics for the Beta v2.1 decision heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F


RETURN_HORIZONS = (1, 3, 5, 10)
RETURN_HORIZON_WEIGHTS = (0.25, 0.25, 0.30, 0.20)


def same_date_return_bias_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    date_ids: torch.Tensor,
) -> torch.Tensor:
    """Penalize systematic horizon bias within each signal-date cross section."""
    same_date = date_ids[:, None] == date_ids[None, :]
    group_sizes = same_date.sum(dim=1)
    mean_errors = (
        same_date.to(predictions.dtype) @ (predictions - targets)
    ) / group_sizes.clamp_min(1).to(predictions.dtype)[:, None]
    per_sample_group_loss = F.smooth_l1_loss(
        mean_errors, torch.zeros_like(mean_errors), reduction='none'
    ).mean(dim=1)
    group_weights = (group_sizes >= 2).to(predictions.dtype) / group_sizes.clamp_min(1)
    group_count = group_weights.sum()
    grouped_loss = (per_sample_group_loss * group_weights).sum() / group_count.clamp_min(1)
    fallback_error = predictions.mean(dim=0) - targets.mean(dim=0)
    fallback_loss = F.smooth_l1_loss(
        fallback_error, torch.zeros_like(fallback_error)
    )
    return torch.where(group_count > 0, grouped_loss, fallback_loss)


def class_balanced_barrier_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy with inverse-square-root batch class frequencies."""
    valid_mask = valid_mask.bool()
    targets = targets.long()
    valid_values = valid_mask.to(logits.dtype)
    counts = (
        F.one_hot(targets, num_classes=logits.shape[-1]).to(logits.dtype)
        * valid_values[:, None]
    ).sum(dim=0)
    weights = torch.where(counts > 0, counts.rsqrt(), torch.zeros_like(counts))
    present_classes = (counts > 0).to(logits.dtype).sum()
    weights = weights / (
        weights.sum() / present_classes.clamp_min(1)
    ).clamp_min(torch.finfo(logits.dtype).eps)
    sample_weights = weights.gather(0, targets) * valid_values
    losses = F.cross_entropy(logits, targets, reduction='none')
    denominator = sample_weights.sum()
    weighted_loss = (losses * sample_weights).sum() / denominator.clamp_min(
        torch.finfo(logits.dtype).eps
    )
    return torch.where(denominator > 0, weighted_loss, logits.sum() * 0.0)


def expected_utility_score(
    normalized_returns: torch.Tensor,
    barrier_logits: torch.Tensor,
    return_scales: torch.Tensor,
) -> torch.Tensor:
    """Derive one ranking score from return and barrier predictions."""
    raw_log_return_10 = normalized_returns[:, -1] * return_scales[:, -1]
    unresolved_utility = torch.expm1(raw_log_return_10) - 0.004
    probabilities = F.softmax(barrier_logits, dim=-1)
    return (
        probabilities[:, 0] * 0.046
        + probabilities[:, 1] * -0.034
        + probabilities[:, 2] * unresolved_utility
    )


def same_date_pairwise_ranking_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    date_ids: torch.Tensor,
    minimum_gap: float = 0.005,
) -> torch.Tensor:
    """Compare only economically distinct pairs from the same signal date."""
    same_date = date_ids[:, None] == date_ids[None, :]
    group_sizes = same_date.sum(dim=1)
    score_delta = scores[:, None] - scores[None, :]
    utility_delta = utilities[:, None] - utilities[None, :]
    pair_mask = same_date & torch.triu(
        utility_delta.abs() >= minimum_gap, diagonal=1
    )
    pair_values = pair_mask.to(scores.dtype)
    pair_counts_by_row = pair_values.sum(dim=1)
    group_pair_counts = same_date.to(scores.dtype) @ pair_counts_by_row
    pair_losses = F.softplus(-score_delta * utility_delta.sign())
    normalized_pairs = (
        pair_losses * pair_values / group_pair_counts.clamp_min(1)[:, None]
    )
    groups_with_pairs = (group_pair_counts > 0).to(scores.dtype)
    group_count = (groups_with_pairs / group_sizes.clamp_min(1)).sum()
    grouped_loss = normalized_pairs.sum() / group_count.clamp_min(1)
    return torch.where(group_count > 0, grouped_loss, scores.sum() * 0.0)


def compute_auxiliary_losses(
    return_predictions: torch.Tensor,
    barrier_logits: torch.Tensor,
    labels: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute return, bias, barrier, and derived ranking objectives."""
    targets = labels['return_targets']
    horizon_weights = return_predictions.new_tensor(RETURN_HORIZON_WEIGHTS)
    horizon_huber = F.smooth_l1_loss(
        return_predictions, targets, reduction='none'
    ).mean(dim=0)
    return_huber = torch.sum(horizon_huber * horizon_weights)
    return_bias = same_date_return_bias_loss(
        return_predictions, targets, labels['date_id']
    )
    return_total = 0.80 * return_huber + 0.20 * return_bias
    barrier = class_balanced_barrier_loss(
        barrier_logits, labels['barrier_target'], labels['barrier_valid']
    )
    ranking_scores = expected_utility_score(
        return_predictions, barrier_logits, labels['return_scales']
    )
    ranking = same_date_pairwise_ranking_loss(
        ranking_scores, labels['utility'], labels['date_id']
    )
    return {
        'return': return_total,
        'return_huber': return_huber,
        'return_bias': return_bias,
        'barrier': barrier,
        'ranking': ranking,
    }


class DetachedEMANormalizer:
    """Normalize heterogeneous training losses without backpropagating through EMA."""

    def __init__(self, decay: float = 0.99, epsilon: float = 1e-4):
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.values: dict[str, float] = {}

    def normalize(self, name: str, loss: torch.Tensor, update: bool = True) -> torch.Tensor:
        observed = float(loss.detach().float().item())
        if update:
            previous = self.values.get(name, observed)
            self.values[name] = self.decay * previous + (1.0 - self.decay) * observed
        denominator = max(self.values.get(name, observed), self.epsilon)
        return loss / loss.new_tensor(denominator)

    def state_dict(self) -> dict[str, object]:
        return {'decay': self.decay, 'epsilon': self.epsilon, 'values': dict(self.values)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state['decay'])
        self.epsilon = float(state['epsilon'])
        self.values = {str(key): float(value) for key, value in state['values'].items()}


def compose_beta_v21_objective(
    path_loss: torch.Tensor,
    history_loss: torch.Tensor,
    auxiliary_losses: dict[str, torch.Tensor],
    normalizer: DetachedEMANormalizer,
    global_step: int,
    warmup_steps: int = 1000,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Build the warm-started, scale-free Beta v2.1 objective."""
    normalized = {
        'path': normalizer.normalize('path', path_loss),
        'history': normalizer.normalize('history', history_loss),
        'return': normalizer.normalize('return', auxiliary_losses['return']),
        'barrier': normalizer.normalize('barrier', auxiliary_losses['barrier']),
        'ranking': normalizer.normalize('ranking', auxiliary_losses['ranking']),
    }
    ramp = min(1.0, max(0.0, float(global_step) / max(1, int(warmup_steps))))
    base = 0.68 * normalized['path'] + 0.02 * normalized['history']
    auxiliary = (
        0.15 * normalized['return']
        + 0.10 * normalized['barrier']
        + 0.05 * normalized['ranking']
    )
    objective = (base + ramp * auxiliary) / (0.70 + 0.30 * ramp)
    return objective, {**normalized, 'ramp': ramp}


def generated_return_targets(
    generated_path: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    return_scales: torch.Tensor,
) -> torch.Tensor:
    """Convert normalized generated OHLC paths to the auxiliary head's target space."""
    denormalized = generated_path * (feature_stds[:, None, :] + 1e-5) + feature_means[:, None, :]
    entry = denormalized[:, 0, 0].clamp_min(1e-8)
    closes = denormalized[:, :, 3].clamp_min(1e-8)
    horizon_indices = generated_path.new_tensor(
        [horizon - 1 for horizon in RETURN_HORIZONS], dtype=torch.long
    )
    raw_returns = torch.log(closes.index_select(1, horizon_indices) / entry[:, None])
    return (raw_returns / return_scales.clamp_min(1e-8)).clamp(-3.0, 3.0)


def consistency_statistics(
    auxiliary_returns: torch.Tensor,
    generated_returns: torch.Tensor,
    actual_returns: torch.Tensor,
) -> dict[str, list[float] | int]:
    """Summarize generated-path and auxiliary-return agreement by horizon."""
    auxiliary = auxiliary_returns.detach().float()
    generated = generated_returns.detach().float()
    actual = actual_returns.detach().float()
    count = int(auxiliary.shape[0])
    if count == 0:
        return {'samples': 0}
    auxiliary_centered = auxiliary - auxiliary.mean(dim=0)
    generated_centered = generated - generated.mean(dim=0)
    covariance = (auxiliary_centered * generated_centered).mean(dim=0)
    denominator = auxiliary_centered.square().mean(dim=0).sqrt() * generated_centered.square().mean(dim=0).sqrt()
    correlation = covariance / denominator.clamp_min(1e-8)
    return {
        'samples': count,
        'mae': (auxiliary - generated).abs().mean(dim=0).tolist(),
        'sign_agreement': ((auxiliary >= 0) == (generated >= 0)).float().mean(dim=0).tolist(),
        'correlation': correlation.tolist(),
        'auxiliary_mean': auxiliary.mean(dim=0).tolist(),
        'generated_mean': generated.mean(dim=0).tolist(),
        'auxiliary_std': auxiliary.std(dim=0, unbiased=False).tolist(),
        'generated_std': generated.std(dim=0, unbiased=False).tolist(),
        'auxiliary_bias': (auxiliary - actual).mean(dim=0).tolist(),
        'generated_bias': (generated - actual).mean(dim=0).tolist(),
    }
