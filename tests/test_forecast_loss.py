import sys
from pathlib import Path

import pytest
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


def test_aux_off_validation_records_weighted_forecast_loss(monkeypatch):
    """Phase-1 aux-off validation must accumulate weighted_forecast_loss.

    The dual-T4 p1 run KeyError'd in ValidationLossAccumulator.add because
    evaluate_validation always puts weighted_forecast into the batch dict,
    but only registered that key when use_beta_v21_auxiliary was on.
    """
    from torch.utils.data import DataLoader, Dataset

    from train_predictor import best_selection_value, evaluate_validation

    class _Batch(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return (
                torch.zeros(8, 6),
                torch.zeros(8, 5),
                torch.zeros((), dtype=torch.long),
                torch.zeros((), dtype=torch.long),
                torch.zeros(1),
                torch.tensor(index, dtype=torch.long),
            )

    class _Tokenizer:
        def encode(self, x, half=False):
            batch, length, _features = x.shape
            tokens = torch.zeros(batch, length, dtype=torch.long)
            return tokens, tokens.clone()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(1, 1)

        def forward(self, s1, s2, stamp, **kwargs):
            batch, length = s1.shape
            zeros = torch.zeros(batch, length)
            return [zeros, zeros.clone()]

    def fake_losses(head, logits, targets, config):
        return {
            "objective": torch.tensor(1.5),
            "full_sequence": torch.tensor(1.2),
            "history": torch.tensor(0.4),
            "forecast": torch.tensor(2.40),
            "weighted_forecast": torch.tensor(2.31),
        }

    monkeypatch.setattr(
        "train_predictor.compute_predictor_losses", fake_losses
    )
    loader = DataLoader(_Batch(), batch_size=2)
    metrics = evaluate_validation(
        _Model(),
        _Tokenizer(),
        loader,
        torch.device("cpu"),
        {
            "use_beta_v21_auxiliary": False,
            "use_sector_features": False,
            "use_size_features": False,
            "use_size_percentile": False,
            "lookback_window": 4,
            "predict_window": 2,
            "predictor_loss_mode": "forecast",
            "history_loss_weight": 0.02,
            "beta_v21_consistency_samples": 0,
            "collect_validation_auxiliary": False,
        },
        None,
        run_condition_ablation=False,
        period_names={0: "period_a", 1: "period_b"},
        rank=0,
    )
    assert metrics["weighted_forecast_loss"] == pytest.approx(2.31)
    assert metrics["forecast_loss"] == pytest.approx(2.40)
    assert "beta_v21_score" not in metrics
    assert "return_loss" not in metrics
    assert best_selection_value("forecast", metrics) == pytest.approx(2.31)
    for name in ("period_a", "period_b"):
        assert metrics["periods"][name]["weighted_forecast_loss"] == pytest.approx(2.31)


