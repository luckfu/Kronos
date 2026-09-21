"""Vol-calibration objective for Stage 3.

Production OOS scores `predicted_path_vol` as the mean of per-sample
close-path daily-return stds at T=0.65 / top_p=0.8 / N=5. Training
must match that concentrated decode: E[vol(path)], never vol(E[path]),
never the T=1 top-16 mixture that already over-vol'd on causal val.

Candidate decoder values are stop-grad; gradients flow through the
per-horizon mixture weights and therefore through s1/s2 logits.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from finetune.stage3_path_alignment import DetachedLossEMA


CLOSE = 3
EPS = 1e-5

# evaluate() and train logging require every key on this tuple. The vol
# loss already computes them; Stage3TrainingModel must copy them into
# the metrics dict or baseline evaluate KeyErrors (uniform_path_vol).
VOL_METRIC_KEYS = (
    'vol_calibration_ratio',
    'pred_path_vol',
    'realized_path_vol',
    'mixture_mean_path_vol',
    'uniform_path_vol',
)


@dataclass
class VolAlignmentConfig:
    support_k: int = 16
    candidates: int = 5
    weight: float = 0.15
    huber_delta: float = 0.15
    ema_decay: float = 0.99
    close_index: int = CLOSE
    temperature: float = 0.65
    top_p: float = 0.8


def apply_temperature_nucleus(logits, temperature, top_p):
    """Scale logits by T, then keep the nucleus mass used in production decode."""
    scaled = logits.float() / max(float(temperature), EPS)
    if float(top_p) >= 1.0 - 1e-12:
        return scaled
    vocab = scaled.shape[-1]
    flat = scaled.reshape(-1, vocab)
    sorted_logits, sorted_indices = torch.sort(flat, descending=True, dim=-1)
    cumulative = sorted_logits.softmax(-1).cumsum(-1)
    sorted_remove = cumulative > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = torch.zeros_like(sorted_remove)
    remove.scatter_(1, sorted_indices, sorted_remove)
    return flat.masked_fill(remove, float('-inf')).view_as(logits)


def production_style_mixture_decode(
    model, tokenizer, context, s1_logits, config: VolAlignmentConfig,
    position_start=None, is_causal=None,
):
    """Teacher-forced joint mixture with production T / top_p / N.

    Same stop-grad tokenizer as the old Stage3 decoder; the distribution
    is the concentrated one used at inference, not T=1 top-16.
    """
    temperature = max(float(config.temperature), EPS)
    top_p = float(config.top_p)
    support = min(int(config.support_k), s1_logits.shape[-1])
    if support < 1 or int(config.candidates) < 1:
        raise ValueError('support_k and candidates must be positive')
    scaled1 = apply_temperature_nucleus(s1_logits, temperature, top_p)
    p1 = scaled1.softmax(-1)
    v1, i1 = torch.topk(p1, support, dim=-1)
    s2_cond = model.predict_s2_candidates(
        context, i1, position_start=position_start, is_causal=is_causal,
    )
    scaled2 = apply_temperature_nucleus(s2_cond, temperature, top_p)
    p2 = scaled2.softmax(-1)
    k2 = min(support, p2.shape[-1])
    v2, i2 = torch.topk(p2, k2, dim=-1)
    joint_logp = v1.clamp_min(EPS).log()[..., :, None] + v2.clamp_min(EPS).log()
    flat = joint_logp.reshape(*joint_logp.shape[:-2], -1)
    joint_logits = apply_temperature_nucleus(flat, 1.0, top_p)
    selected = min(max(1, int(config.candidates)), joint_logits.shape[-1])
    top_logp, top_idx = torch.topk(joint_logits, selected, dim=-1)
    r1 = top_idx // k2
    s1_ids = i1.gather(-1, r1)
    s2_ids = i2.flatten(-2).gather(-1, top_idx)
    outputs = []
    with torch.no_grad():
        for n in range(selected):
            outputs.append(tokenizer.decode((s1_ids[..., n], s2_ids[..., n]), half=True).float())
    weights = top_logp.softmax(-1)
    if torch.is_grad_enabled() and not weights.requires_grad:
        raise RuntimeError('vol alignment mixture weights are detached')
    paths = torch.stack(outputs, dim=-1)
    prediction = (paths * weights.unsqueeze(-2)).sum(-1)
    return prediction, weights, {
        'joint_candidate_count': selected,
        's2_conditional_logits': s2_cond,
        'selected_s1_ids': s1_ids,
        'selected_s2_ids': s2_ids,
        'selected_joint_logp': top_logp,
        'candidate_paths': paths,
        'vol_temperature': temperature,
        'vol_top_p': top_p,
    }


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
    _, weights, extra = production_style_mixture_decode(
        model, tokenizer, context, s1_logits, config, position_start, is_causal,
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
    uniform_vol = candidate_vol.mean(-1)
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
        "uniform_path_vol": uniform_vol.detach().mean(),
        "candidate_vol_mean": candidate_vol.detach().mean(),
        "vol_align_weights": path_weights.detach(),
        **extra,
    }
