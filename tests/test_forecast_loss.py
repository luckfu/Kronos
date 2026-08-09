import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune'))
from train_predictor import compute_predictor_losses, objective_token_slices


class SquaredErrorHead:
    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets):
        value = (
            (s1_logits - s1_targets.float()).square().mean()
            + (s2_logits - s2_targets.float()).square().mean()
        ) / 2
        return value, value, value


def test_forecast_slice_starts_at_last_context_token():
    history, forecast = objective_token_slices(130, 120, 10)

    assert (history.start, history.stop) == (0, 119)
    assert (forecast.start, forecast.stop) == (119, 129)


def test_forecast_only_loss_has_no_history_gradient():
    logits = [
        torch.ones((1, 130), requires_grad=True),
        torch.ones((1, 130), requires_grad=True),
    ]
    targets = [torch.zeros((1, 130), dtype=torch.long)] * 2
    losses = compute_predictor_losses(
        SquaredErrorHead(), logits, targets,
        {
            'lookback_window': 120,
            'predict_window': 10,
            'predictor_loss_mode': 'forecast',
            'history_loss_weight': 0.0,
        },
    )

    losses['objective'].backward()

    assert torch.count_nonzero(logits[0].grad[:, :119]) == 0
    assert torch.count_nonzero(logits[0].grad[:, 119:129]) == 10
    assert torch.count_nonzero(logits[0].grad[:, 129:]) == 0


def test_history_auxiliary_weight_is_explicit_and_bounded_to_history():
    logits = [
        torch.ones((1, 130), requires_grad=True),
        torch.ones((1, 130), requires_grad=True),
    ]
    targets = [torch.zeros((1, 130), dtype=torch.long)] * 2
    losses = compute_predictor_losses(
        SquaredErrorHead(), logits, targets,
        {
            'lookback_window': 120,
            'predict_window': 10,
            'predictor_loss_mode': 'forecast',
            'history_loss_weight': 0.02,
        },
    )

    losses['objective'].backward()

    assert torch.count_nonzero(logits[0].grad[:, :119]) == 119
    assert torch.count_nonzero(logits[0].grad[:, 119:129]) == 10
    assert torch.count_nonzero(logits[0].grad[:, 129:]) == 0
