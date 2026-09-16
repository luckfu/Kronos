"""Same-day pairwise rank loss for Stage3 P1."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def naive_same_date_pairwise_ranking_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    date_ids: torch.Tensor,
) -> torch.Tensor:
    """All ordered same-date pairs; logistic on score ordering vs actual return order."""
    same_date = date_ids[:, None] == date_ids[None, :]
    score_delta = scores[:, None] - scores[None, :]
    utility_delta = utilities[:, None] - utilities[None, :]
    pair_mask = same_date & torch.triu(utility_delta != 0, diagonal=1)
    pair_values = pair_mask.to(scores.dtype)
    pair_counts_by_row = pair_values.sum(dim=1)
    group_pair_counts = same_date.to(scores.dtype) @ pair_counts_by_row
    pair_losses = F.softplus(-score_delta * utility_delta.sign())
    normalized_pairs = (
        pair_losses * pair_values / group_pair_counts.clamp_min(1)[:, None]
    )
    groups_with_pairs = (group_pair_counts > 0).to(scores.dtype)
    group_count = (groups_with_pairs / same_date.sum(dim=1).clamp_min(1)).sum()
    grouped_loss = normalized_pairs.sum() / group_count.clamp_min(1)
    return torch.where(group_count > 0, grouped_loss, scores.sum() * 0.0)


def rank_batch_diagnostics(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    date_ids: torch.Tensor,
) -> dict[str, float]:
    """Loggable rank batch stats required by the P1 handoff."""
    unique_dates = int(torch.unique(date_ids).numel())
    counts = torch.bincount(date_ids - int(date_ids.min()))
    counts = counts[counts > 0]
    stocks_per_date_mean = float(counts.float().mean()) if counts.numel() else 0.0
    same_date = date_ids[:, None] == date_ids[None, :]
    utility_delta = utilities[:, None] - utilities[None, :]
    pair_mask = same_date & torch.triu(utility_delta != 0, diagonal=1)
    pair_count = int(pair_mask.sum().item())
    dispersion = float(utilities.std(unbiased=False).item()) if utilities.numel() > 1 else 0.0
    return {
        'unique_signal_dates': unique_dates,
        'stocks_per_date_mean': stocks_per_date_mean,
        'valid_pairs': pair_count,
        'pair_count': pair_count,
        'return_dispersion': dispersion,
    }
