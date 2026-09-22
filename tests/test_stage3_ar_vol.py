import torch
from torch import nn

from finetune.stage3_ar_vol import (
    AR_METRIC_KEYS, ARVolConfig, BASELINE_AR_RATIO, CE_INCREASE_LIMIT,
    decide, filtered_log_prob, reinforce_backward_from_tokens, within_window_advantage,
)


def test_within_window_advantage_upweights_the_closer_path():
    pred = torch.tensor([[0.02, 0.04]])
    realized = torch.tensor([0.04])
    advantage, error = within_window_advantage(pred, realized, 0.15)
    assert torch.allclose(advantage.sum(-1), torch.zeros(1), atol=1e-6)
    assert advantage[0, 0] > 0
    assert advantage[0, 1] < 0
    assert error[0, 1] < error[0, 0]


def test_filtered_log_prob_matches_the_sampled_nucleus_and_rejects_the_tail():
    logits = torch.tensor([[0.0, 3.0, 0.0]], requires_grad=True)
    token = torch.tensor([1])
    logp = filtered_log_prob(logits, token, temperature=1.0, top_p=1.0)
    logp.sum().backward()
    assert logits.grad[0, 1] > 0
    tail = torch.tensor([[10.0, -10.0]], requires_grad=True)
    try:
        filtered_log_prob(tail, torch.tensor([1]), temperature=1.0, top_p=0.5)
    except RuntimeError as exc:
        assert 'nucleus' in str(exc)
    else:
        raise AssertionError('tail token was treated as in-support')


def test_go_requires_both_a_closer_ar_ratio_and_a_small_ce_increase():
    moved = decide(1.20, 2.30, 2.32)
    assert moved['closer_to_one'] and moved['ce_ok'] and moved['go']
    farther = decide(1.50, 2.30, 2.31)
    assert not farther['closer_to_one'] and not farther['go']
    ce_break = decide(1.20, 2.30, 2.30 + CE_INCREASE_LIMIT)
    assert ce_break['closer_to_one'] and not ce_break['ce_ok'] and not ce_break['go']
    unchanged = decide(BASELINE_AR_RATIO, 2.30, 2.30)
    assert not unchanged['closer_to_one'] and not unchanged['go']


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)

    def decode_s1(self, s1, s2, stamp, sector_id=None, size_percentile=None):
        hidden = self.emb(s1) + self.emb(s2)
        return self.head(hidden), hidden

    def decode_s2(self, context, s1_ids):
        hidden = context[:, -1:, :] + self.emb(s1_ids)
        return self.head(hidden)


def test_reinforce_backward_writes_a_finite_gradient():
    model = _Tiny()
    batch, samples, horizon, lookback = 2, 2, 2, 3
    config = ARVolConfig(weight=0.15, temperature=1.0, top_p=1.0, samples=samples)
    loss, mean_logp = reinforce_backward_from_tokens(
        model,
        torch.randint(0, 8, (batch, lookback)),
        torch.randint(0, 8, (batch, lookback)),
        torch.zeros(batch, lookback + horizon, 5),
        torch.randint(0, 8, (batch, samples, horizon)),
        torch.randint(0, 8, (batch, samples, horizon)),
        torch.tensor([[1.0, -1.0], [0.4, -0.4]]),
        config,
        lookback=lookback,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(mean_logp)
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0


def test_ar_metric_keys_are_the_logged_contract():
    assert 'ar_calibration_ratio' in AR_METRIC_KEYS
    assert 'ar_mean_huber' in AR_METRIC_KEYS
