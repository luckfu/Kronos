"""Autoregressive vol calibration.

Sample N full paths at the production decode (T, top_p), score each path's
close-vol against the realized path, and push probability toward the paths
whose vol is closer to realized. Prices are stop-grad. The score is the
within-window advantage of per-path log-vol Huber, times the token log-prob
under the same temperature and nucleus used to sample. Both sides call
`temperature_nucleus_logits`, so a drawn token stays in support at the
production (T, top_p), including ties and fp16/bf16 logits.

One horizon step is differentiated at a time so the graph is a single decode.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from finetune.stage3_vol_alignment import (
    EPS, denorm_close, daily_returns_from_close,
)
from model.kronos import auto_regressive_inference, temperature_nucleus_logits

CLOSE = 3
MAX_CONTEXT = 512
BASELINE_AR_RATIO = 1.3533238272873447
CE_INCREASE_LIMIT = 0.03

AR_METRIC_KEYS = (
    'ar_calibration_ratio',
    'ar_pred_path_vol',
    'ar_realized_path_vol',
    'ar_advantage_abs',
    'ar_within_window_vol_std',
    'ar_logp_mean',
    'ar_mean_huber',
)


@dataclass
class ARVolConfig:
    weight: float = 0.15
    temperature: float = 0.65
    top_p: float = 0.8
    samples: int = 5
    huber_delta: float = 0.15


def decide(
    ar_ratio,
    baseline_token_ce,
    token_ce,
    baseline_ar_ratio=BASELINE_AR_RATIO,
    ce_increase_limit=CE_INCREASE_LIMIT,
):
    """Both bars are required: AR ratio moves toward 1, and token CE rises by less than the limit."""
    ce_increase = float(token_ce) - float(baseline_token_ce)
    distance = abs(float(ar_ratio) - 1.0)
    baseline_distance = abs(float(baseline_ar_ratio) - 1.0)
    closer = distance < baseline_distance
    ce_ok = ce_increase < float(ce_increase_limit)
    return {
        'ar_calibration_ratio': float(ar_ratio),
        'baseline_ar_calibration_ratio': float(baseline_ar_ratio),
        'distance_to_one': distance,
        'baseline_distance_to_one': baseline_distance,
        'closer_to_one': bool(closer),
        'baseline_token_ce': float(baseline_token_ce),
        'token_ce': float(token_ce),
        'ce_increase': ce_increase,
        'ce_increase_limit': float(ce_increase_limit),
        'ce_ok': bool(ce_ok),
        'go': bool(closer and ce_ok),
    }


def filtered_log_prob(logits, token, temperature, top_p, include_sampled_token=False):
    """Log-prob of `token` under the same temperature and nucleus used by sample_from_logits.

    `include_sampled_token` puts a drawn index back into the support when the
    replay forward is not bitwise-identical to the sampling forward (attention
    kernels) and the rebuilt nucleus drops a boundary token. The restored value
    is that token's temperature-scaled logit, so the gradient still flows only
    through the path probability. Without this flag a token outside the
    nucleus still raises.
    """
    filtered = temperature_nucleus_logits(logits, temperature, top_p)
    index = token.long().view(-1, 1)
    if include_sampled_token:
        chosen = filtered.gather(-1, index)
        missing = ~torch.isfinite(chosen)
        if bool(missing.any()):
            raw = logits.float() / float(temperature)
            positions = torch.zeros(
                filtered.shape, dtype=torch.bool, device=filtered.device,
            )
            positions.scatter_(1, index, missing)
            filtered = torch.where(positions, raw, filtered)
    gathered = filtered.log_softmax(-1).gather(-1, index).squeeze(-1)
    if not torch.isfinite(gathered).all():
        raise RuntimeError('sampled token fell outside the nucleus used for log-prob')
    return gathered


def within_window_advantage(pred_vol, realized_vol, delta):
    """pred_vol [B, N], realized_vol [B]. Positive advantage means worse than the window mean."""
    log_pred = pred_vol.clamp_min(EPS).log()
    log_real = realized_vol.clamp_min(EPS).log().unsqueeze(-1).expand_as(log_pred)
    error = F.huber_loss(log_pred, log_real, delta=delta, reduction='none')
    advantage = error - error.mean(dim=-1, keepdim=True)
    return advantage, error


def _path_vols(forecast, x, mean, std, lookback, horizon):
    future_z = torch.as_tensor(forecast[:, :, -horizon:, CLOSE], device=x.device, dtype=torch.float32)
    close_mean = mean[:, CLOSE]
    close_std = std[:, CLOSE]
    last_px = denorm_close(x[:, lookback - 1, CLOSE], close_mean, close_std)
    future_px = denorm_close(future_z, close_mean[:, None, None], close_std[:, None, None])
    pred_vol = daily_returns_from_close(future_px, last_px).std(dim=-1, unbiased=False)
    target_px = denorm_close(
        x[:, lookback:lookback + horizon, CLOSE], close_mean[:, None], close_std[:, None],
    )
    realized = daily_returns_from_close(target_px, last_px).std(dim=-1, unbiased=False)
    return pred_vol, realized


def _step_logp(model, s1_prefix, s2_prefix, stamp_prefix, sector, percentile, token_s1, token_s2, temperature, top_p):
    s1_logits, context = model.decode_s1(
        s1_prefix, s2_prefix, stamp_prefix,
        sector_id=sector, size_percentile=percentile,
    )
    logp1 = filtered_log_prob(
        s1_logits[:, -1, :], token_s1, temperature, top_p, include_sampled_token=True,
    )
    s2_logits = model.decode_s2(context, token_s1.view(-1, 1))
    logp2 = filtered_log_prob(
        s2_logits[:, -1, :], token_s2, temperature, top_p, include_sampled_token=True,
    )
    return logp1 + logp2


def reinforce_backward_from_tokens(
    model, s1_hist, s2_hist, stamp, s1_tokens, s2_tokens, advantage, config: ARVolConfig,
    sector=None, percentile=None, lookback=120,
):
    """Backward λ * advantage * log π(path). `advantage` is detached. One horizon step at a time.

    s1_hist/s2_hist: [B, lookback]
    s1_tokens/s2_tokens: [B, N, H]
    stamp: [B, lookback + H, stamp_dim] covering history and the horizon, in that order.
    """
    batch, samples, horizon = s1_tokens.shape
    if advantage.shape != (batch, samples):
        raise ValueError(f'advantage shape {tuple(advantage.shape)} != {(batch, samples)}')
    if lookback + horizon > MAX_CONTEXT:
        raise ValueError('AR replay buffer would roll; production lookback+horizon must stay within 512')
    flat = batch * samples
    s1_hist = s1_hist.long()
    s2_hist = s2_hist.long()
    s1_hist_flat = s1_hist.repeat_interleave(samples, dim=0)
    s2_hist_flat = s2_hist.repeat_interleave(samples, dim=0)
    s1_new = s1_tokens.reshape(flat, horizon).long()
    s2_new = s2_tokens.reshape(flat, horizon).long()
    stamp_flat = stamp.repeat_interleave(samples, dim=0)
    sector_flat = None if sector is None else sector.repeat_interleave(samples, dim=0)
    percentile_flat = None if percentile is None else percentile.repeat_interleave(samples, dim=0)
    adv_flat = advantage.reshape(flat)
    total = advantage.new_zeros(())
    logp_sum = advantage.new_zeros(())
    for step in range(horizon):
        length = lookback + step
        s1_prefix = s1_hist_flat if step == 0 else torch.cat([s1_hist_flat, s1_new[:, :step]], dim=1)
        s2_prefix = s2_hist_flat if step == 0 else torch.cat([s2_hist_flat, s2_new[:, :step]], dim=1)
        logp = _step_logp(
            model, s1_prefix, s2_prefix, stamp_flat[:, :length].contiguous(),
            sector_flat, percentile_flat, s1_new[:, step], s2_new[:, step],
            config.temperature, config.top_p,
        )
        step_loss = (config.weight * adv_flat.detach() * logp).mean()
        if not step_loss.requires_grad:
            raise RuntimeError('AR vol log-prob has no grad')
        step_loss.backward()
        total = total + step_loss.detach()
        logp_sum = logp_sum + logp.detach().mean()
    return total, logp_sum / horizon


def compute_ar_vol_loss(
    model, tokenizer, x, stamp, sector, percentile, feature_mean, feature_std,
    config: ARVolConfig, lookback=120, horizon=10,
):
    """Sample production paths, backward the score function, return a detached loss and metrics."""
    if feature_mean is None or feature_std is None:
        raise ValueError('autoregressive vol requires feature_means and feature_stds')
    if stamp is None:
        raise ValueError('autoregressive vol requires stamps')
    model.eval()
    hist = x[:, :lookback]
    x_stamp = stamp[:, :lookback]
    y_stamp = stamp[:, lookback:lookback + horizon]
    with torch.no_grad():
        hist_clip = torch.clip(hist, -5, 5)
        s1_hist, s2_hist = tokenizer.encode(hist_clip, half=True)
        forecast, tokens = auto_regressive_inference(
            tokenizer, model, hist, x_stamp, y_stamp,
            max_context=MAX_CONTEXT, pred_len=horizon, clip=5,
            T=config.temperature, top_k=0, top_p=config.top_p,
            sample_count=config.samples, verbose=False,
            sector_id=sector, size_percentile=percentile,
            return_samples=True, return_generated_tokens=True,
        )
    pred_vol, realized = _path_vols(forecast, x, feature_mean, feature_std, lookback, horizon)
    advantage, error = within_window_advantage(pred_vol, realized, config.huber_delta)
    s1_tokens = torch.as_tensor(tokens['s1'], device=x.device)
    s2_tokens = torch.as_tensor(tokens['s2'], device=x.device)
    if s1_tokens.shape != (x.shape[0], config.samples, horizon):
        raise RuntimeError(f'unexpected AR token shape {tuple(s1_tokens.shape)}')
    ar_loss, mean_logp = reinforce_backward_from_tokens(
        model, s1_hist, s2_hist, stamp[:, :lookback + horizon],
        s1_tokens, s2_tokens, advantage, config,
        sector=sector, percentile=percentile, lookback=lookback,
    )
    ratio = (realized / pred_vol.mean(-1).clamp_min(EPS)).mean()
    metrics = {
        'ar_calibration_ratio': ratio.detach(),
        'ar_pred_path_vol': pred_vol.mean().detach(),
        'ar_realized_path_vol': realized.mean().detach(),
        'ar_advantage_abs': advantage.detach().abs().mean(),
        'ar_within_window_vol_std': pred_vol.detach().std(dim=-1, unbiased=False).mean(),
        'ar_logp_mean': mean_logp.detach(),
        'ar_mean_huber': error.detach().mean(),
    }
    missing = [key for key in AR_METRIC_KEYS if key not in metrics]
    if missing:
        raise RuntimeError(f'AR vol metrics missing {missing}')
    return ar_loss, metrics
