"""Vol-calibration objective for Stage 3 smoke.

This is not the old path-level Huber. Production OOS scores
`predicted_path_vol` as the mean of per-sample close-path daily-return
stds. Training must match that: E[vol(path)], never vol(E[path]).

Candidate decoder values are stop-grad; gradients flow through the
per-horizon mixture weights and therefore through s1/s2 logits.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from finetune.stage3_path_alignment import DetachedLossEMA, candidate_mixture_decode


CLOSE = 3
EPS = 1e-5


@dataclass
class VolAlignmentConfig:
    top_k: int = 16
    candidates: int = 16
    weight: float = 0.15
    huber_delta: float = 0.15
    ema_decay: float = 0.99
    close_index: int = CLOSE


def denorm_close(close_z, mean, std):
    scale = std.clamp_min(EPS)
    return close_z * scale + mean


def daily_returns_from_close(future_close, last_close):
    """future_close [..., H], last_close broadcastable -> daily returns [..., H]."""
    last = last_close
    while last.ndim < future_close.ndim:
        last = last.unsqueeze(-1)
    last = last.clamp_min(EPS)
    cumulative = future_close / last - 1.0
    previous = torch.cat(
        [torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]],
        dim=-1,
    )
    return (1.0 + cumulative) / (1.0 + previous).clamp_min(EPS) - 1.0


def path_vol(daily):
    """Std of daily returns over the horizon. daily [..., H] or [..., H, N]."""
    return daily.std(dim=-2 if daily.ndim >= 3 else -1, unbiased=False)


def expected_path_vol(candidate_close, weights, last_close):
    """Mean of per-candidate path vols, weights averaged over horizons.

    candidate_close: [B, H, N] denormalized
    weights: [B, H, N]
    last_close: [B]
    """
    daily = daily_returns_from_close(candidate_close.transpose(-1, -2), last_close)
    # daily [B, N, H]
    vol = daily.std(dim=-1, unbiased=False)
    path_weights = weights.mean(dim=1)
    path_weights = path_weights / path_weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return (vol * path_weights).sum(-1), vol, path_weights


def mixture_mean_path_vol(candidate_close, weights, last_close):
    """vol(E[path]) — the compressed object we are NOT training."""
    mixed = (candidate_close * weights).sum(-1)
    daily = daily_returns_from_close(mixed, last_close)
    return daily.std(dim=-1, unbiased=False)


def compute_vol_alignment_loss(
    model, tokenizer, context, s1_logits, target_z, last_close_z,
    feature_mean, feature_std, config: VolAlignmentConfig,
    normalizer: DetachedLossEMA | None = None,
    position_start=None, is_causal=None, synchronize_ema=False,
):
    _, weights, extra = candidate_mixture_decode(
        model, tokenizer, context, s1_logits, config.top_k, config.candidates,
        position_start, is_causal,
    )
    close_mean = feature_mean[:, config.close_index]
    close_std = feature_std[:, config.close_index]
    candidate_z = extra["candidate_paths"][:, :, config.close_index, :]
    candidate_px = denorm_close(candidate_z, close_mean[:, None, None], close_std[:, None, None])
    last_px = denorm_close(last_close_z, close_mean, close_std)
    target_px = denorm_close(
        target_z[:, :, config.close_index], close_mean[:, None], close_std[:, None],
    )
    pred_vol, candidate_vol, path_weights = expected_path_vol(candidate_px, weights, last_px)
    realized_vol = daily_returns_from_close(target_px, last_px).std(dim=-1, unbiased=False)
    mixed_vol = mixture_mean_path_vol(candidate_px, weights.detach(), last_px)
    log_pred = pred_vol.clamp_min(EPS).log()
    log_real = realized_vol.clamp_min(EPS).log()
    raw = F.huber_loss(log_pred, log_real, delta=config.huber_delta, reduction="mean")
    normalized = (
        normalizer.normalize(raw, sample_count=target_z.shape[0], synchronize=synchronize_ema)
        if normalizer is not None else raw
    )
    ratio = (realized_vol / pred_vol.clamp_min(EPS)).mean()
    return config.weight * normalized, {
        "vol_align_loss": raw.detach(),
        "vol_align_normalized": normalized.detach(),
        "pred_path_vol": pred_vol.detach().mean(),
        "realized_path_vol": realized_vol.detach().mean(),
        "vol_calibration_ratio": ratio.detach(),
        "mixture_mean_path_vol": mixed_vol.detach().mean(),
        "candidate_vol_mean": candidate_vol.detach().mean(),
        "vol_align_weights": path_weights.detach(),
        **extra,
    }
