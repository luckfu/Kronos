"""Stage3 P1: Stage2 weighted CE + history + same-day rank auxiliary."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from finetune.stage3_path_alignment import candidate_mixture_decode
from finetune.stage3_rank_loss import naive_same_date_pairwise_ranking_loss, rank_batch_diagnostics


def objective_token_slices(sequence_length, lookback_window, predict_window):
    """Return next-token positions for history reconstruction and forecasting."""
    history_stop = int(lookback_window) - 1
    forecast_stop = history_stop + int(predict_window)
    if history_stop < 1:
        raise ValueError('lookback_window must provide at least one history target')
    if forecast_stop > int(sequence_length):
        raise ValueError(
            f'Need {forecast_stop} target positions for the forecast objective, '
            f'but only {sequence_length} are available'
        )
    return slice(0, history_stop), slice(history_stop, forecast_stop)

CLOSE_INDEX = 3
DEFAULT_FORECAST_HORIZON_WEIGHTS = (
    1.364, 1.364, 1.364, 1.136, 1.136,
    0.909, 0.909, 0.682, 0.682, 0.455,
)


@dataclass
class Stage3CERankConfig:
    enabled: bool = False
    lambda_rank: float = 0.05
    history_weight: float = 0.02
    forecast_horizon_weights: tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_FORECAST_HORIZON_WEIGHTS
    )
    rank_top_k: int = 16
    rank_candidates: int = 16


def compute_weighted_ce_objective(
    head,
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets1: torch.Tensor,
    targets2: torch.Tensor,
    lookback: int,
    horizon: int,
    config: Stage3CERankConfig,
) -> dict[str, torch.Tensor]:
    """Mirror Stage2 forecast+history objective on Stage3 logit layout."""
    history_slice, forecast_slice = objective_token_slices(
        targets1.shape[1], lookback, horizon,
    )
    history_loss, history_s1, history_s2 = head.compute_loss(
        logits1[:, history_slice], logits2[:, history_slice],
        targets1[:, history_slice], targets2[:, history_slice],
    )
    forecast_loss, forecast_s1, forecast_s2 = head.compute_loss(
        logits1[:, forecast_slice], logits2[:, forecast_slice],
        targets1[:, forecast_slice], targets2[:, forecast_slice],
    )
    forecast_weights = torch.as_tensor(
        config.forecast_horizon_weights,
        device=logits1.device,
        dtype=logits1.dtype,
    )
    if forecast_weights.numel() != horizon:
        raise ValueError('forecast_horizon_weights length must equal horizon')
    forecast_weights = forecast_weights / forecast_weights.sum()
    if torch.all(forecast_weights == forecast_weights[0]):
        weighted_forecast_loss = forecast_loss
        weighted_forecast_s1 = forecast_s1
        weighted_forecast_s2 = forecast_s2
    else:
        weighted_s1 = F.cross_entropy(
            logits1[:, forecast_slice].transpose(1, 2),
            targets1[:, forecast_slice],
            reduction='none',
        ).mean(0)
        weighted_s2 = F.cross_entropy(
            logits2[:, forecast_slice].transpose(1, 2),
            targets2[:, forecast_slice],
            reduction='none',
        ).mean(0)
        weighted_forecast_s1 = torch.sum(weighted_s1 * forecast_weights)
        weighted_forecast_s2 = torch.sum(weighted_s2 * forecast_weights)
        weighted_forecast_loss = (weighted_forecast_s1 + weighted_forecast_s2) / 2
    objective = weighted_forecast_loss + config.history_weight * history_loss
    return {
        'objective': objective,
        'history_loss': history_loss,
        'forecast_loss': forecast_loss,
        'weighted_forecast_loss': weighted_forecast_loss,
        'history_s1': history_s1,
        'history_s2': history_s2,
        'forecast_s1': forecast_s1,
        'forecast_s2': forecast_s2,
    }


def denormalized_close(
    normalized_close: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
) -> torch.Tensor:
    return normalized_close * (feature_stds[:, CLOSE_INDEX] + 1e-5) + feature_means[:, CLOSE_INDEX]


def close_return_from_normalized(
    future_close_norm: torch.Tensor,
    anchor_close_norm: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
) -> torch.Tensor:
    """Match evaluator: predicted_closes[-1] / last_close - 1."""
    future = denormalized_close(future_close_norm, feature_means, feature_stds)
    anchor = denormalized_close(anchor_close_norm, feature_means, feature_stds)
    return future / anchor.clamp_min(1e-8) - 1.0


def compute_rank_terms(
    predictor,
    tokenizer,
    context,
    logits1,
    x,
    feature_means,
    feature_stds,
    date_ids,
    lookback: int,
    horizon: int,
    config: Stage3CERankConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    start = lookback - 1
    prediction, _, _ = candidate_mixture_decode(
        predictor, tokenizer, context, logits1,
        top_k=config.rank_top_k, candidates=config.rank_candidates,
        position_start=start, is_causal=True,
    )
    anchor_norm = x[:, lookback - 1, CLOSE_INDEX]
    actual_norm = x[:, lookback + horizon - 1, CLOSE_INDEX]
    pred_norm = prediction[:, horizon - 1, CLOSE_INDEX]
    actual_returns = close_return_from_normalized(
        actual_norm, anchor_norm, feature_means, feature_stds,
    ).detach()
    predicted_scores = close_return_from_normalized(
        pred_norm, anchor_norm, feature_means, feature_stds,
    )
    rank_loss = naive_same_date_pairwise_ranking_loss(
        predicted_scores, actual_returns, date_ids,
    )
    diagnostics = rank_batch_diagnostics(predicted_scores.detach(), actual_returns, date_ids)
    diagnostics['rank_loss'] = float(rank_loss.detach().item())
    return rank_loss, diagnostics
