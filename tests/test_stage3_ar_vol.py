import math
from unittest.mock import patch

import torch
from torch import nn

from finetune.stage3_ar_vol import (
    AR_METRIC_KEYS, ARVolConfig, BASELINE_AR_RATIO, CE_INCREASE_LIMIT,
    decide, filtered_log_prob, reinforce_backward_from_tokens, within_window_advantage,
)
from model.kronos import sample_from_logits, temperature_nucleus_logits


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
    # Replay may see a boundary token fall out of a rebuilt nucleus. Putting
    # that drawn index back keeps a finite score-function term and a gradient
    # through its logit. The call above does not opt in, so the tail raises.
    replay = filtered_log_prob(
        tail, torch.tensor([1]), temperature=1.0, top_p=0.5, include_sampled_token=True,
    )
    assert torch.isfinite(replay).all()
    replay.sum().backward()
    assert tail.grad[0, 1] != 0
    inside = torch.tensor([[0.2, 1.4, -0.3, -2.0]], requires_grad=True)
    token = torch.tensor([1])
    plain = filtered_log_prob(inside, token, temperature=0.65, top_p=0.8)
    kept = filtered_log_prob(
        inside, token, temperature=0.65, top_p=0.8, include_sampled_token=True,
    )
    assert torch.allclose(plain, kept)


def test_sampled_tokens_stay_in_the_temperature_nucleus_including_half_logits():
    """fp16 temperature scaling used to keep a token the float32 log-prob masked."""
    logits = torch.tensor([[
        2.0, 0.329833984375, 0.86279296875, 0.86865234375,
        -1.11328125, -0.1619873046875, 0.75, 0.2685546875,
    ]], dtype=torch.float16)
    temperature, top_p = 0.65, 0.8
    support = torch.isfinite(temperature_nucleus_logits(logits, temperature, top_p)[0])
    torch.manual_seed(0)
    draws = sample_from_logits(
        logits.expand(4000, -1).contiguous(),
        temperature=temperature, top_k=0, top_p=top_p,
    ).view(-1)
    assert support[draws].all()
    logp = filtered_log_prob(
        logits.expand(4000, -1).contiguous(), draws, temperature, top_p,
    )
    assert torch.isfinite(logp).all()
    # Cold temperature is the production contract, not an unscaled nucleus.
    flat = torch.linspace(-2.0, 3.0, 32).view(1, -1)
    hot = torch.isfinite(temperature_nucleus_logits(flat, 1.0, top_p)).sum()
    cold = torch.isfinite(temperature_nucleus_logits(flat, temperature, top_p)).sum()
    assert cold < hot


def test_tied_logits_use_a_stable_nucleus_and_exact_mass_is_inclusive():
    tied = torch.ones(1, 32)
    # 16/32 == 0.5 exactly, so the nucleus is the first 16 indices and not
    # whichever tied tokens an unstable sort happened to emit.
    kept = torch.isfinite(temperature_nucleus_logits(tied, temperature=1.0, top_p=0.5)[0])
    assert kept.nonzero().view(-1).tolist() == list(range(16))
    torch.manual_seed(0)
    draws = sample_from_logits(
        tied.expand(256, -1).contiguous(), temperature=0.65, top_k=0, top_p=0.5,
    ).view(-1)
    assert int(draws.max()) <= 15
    assert torch.isfinite(filtered_log_prob(
        tied.expand(256, -1).contiguous(), draws, temperature=0.65, top_p=0.5,
    )).all()

    # Four equal tokens, top_p=0.5. Mass hits 0.5 on the second token. The
    # shifted `cumsum > top_p` rule kept the third token as well.
    uniform = torch.zeros(1, 4)
    kept = torch.isfinite(temperature_nucleus_logits(uniform, temperature=1.0, top_p=0.5)[0])
    assert kept.tolist() == [True, True, False, False]
    draws = sample_from_logits(
        uniform.expand(128, -1).contiguous(), temperature=1.0, top_k=0, top_p=0.5,
    ).view(-1)
    assert int(draws.max()) <= 1
    assert torch.isfinite(filtered_log_prob(
        uniform.expand(128, -1).contiguous(), draws, temperature=1.0, top_p=0.5,
    )).all()


def test_multinomial_index_outside_the_mask_is_repaired_into_the_nucleus():
    logits = torch.tensor([[5.0, 1.0, -1.0, -4.0]])
    filtered = temperature_nucleus_logits(logits, temperature=0.65, top_p=0.8)
    assert not torch.isfinite(filtered[0, -1])

    def _draw_the_masked_tail(probs, num_samples=1, replacement=False, generator=None, out=None):
        return torch.full((probs.shape[0], num_samples), probs.shape[-1] - 1, dtype=torch.long)

    with patch('torch.multinomial', _draw_the_masked_tail):
        draws = sample_from_logits(logits, temperature=0.65, top_k=0, top_p=0.8)
    assert int(draws[0, 0]) != logits.shape[-1] - 1
    assert torch.isfinite(filtered.gather(-1, draws)).all()
    assert torch.isfinite(filtered_log_prob(
        logits, draws.view(-1), temperature=0.65, top_p=0.8,
    )).all()


def test_untied_nucleus_matches_the_historical_shifted_cumsum_rule():
    """Distinct fp32 logits keep the same production nucleus as the old shift."""
    logits = torch.linspace(4.0, -6.0, 128).view(1, -1)
    temperature, top_p = 0.65, 0.8
    filtered = temperature_nucleus_logits(logits, temperature, top_p)
    scaled = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True, stable=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
    remove = cumulative > top_p
    shifted = remove.clone()
    shifted[:, 1:] = remove[:, :-1]
    shifted[:, 0] = False
    remove = shifted
    original = torch.zeros_like(remove)
    original.scatter_(1, sorted_indices, remove)
    assert torch.equal(torch.isfinite(filtered), ~original)


def test_go_requires_both_a_closer_ar_ratio_and_a_small_ce_increase():
    moved = decide(1.20, 2.30, 2.32)
    assert moved['closer_to_one'] and moved['ce_ok'] and moved['go']
    farther = decide(1.50, 2.30, 2.31)
    assert not farther['closer_to_one'] and not farther['go']
    # Equality at the limit is not a go. The increase has to be the limit
    # float itself: (2.30 + 0.03) - 2.30 is slightly under 0.03, so that sum
    # still has ce_ok and does not exercise the boundary.
    limit = float(CE_INCREASE_LIMIT)
    at_limit = decide(1.20, 0.0, limit)
    assert at_limit['ce_increase'] == limit
    assert at_limit['closer_to_one']
    assert not at_limit['ce_ok']
    assert not at_limit['go']
    above = decide(1.20, 0.0, math.nextafter(limit, math.inf))
    assert above['ce_increase'] > limit
    assert above['closer_to_one']
    assert not above['ce_ok']
    assert not above['go']
    under = decide(1.20, 0.0, math.nextafter(limit, 0.0))
    assert under['ce_increase'] < limit
    assert under['closer_to_one'] and under['ce_ok'] and under['go']
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
    advantage = torch.tensor([[1.0, -1.0], [0.4, -0.4]], requires_grad=True)
    loss, mean_logp = reinforce_backward_from_tokens(
        model,
        torch.randint(0, 8, (batch, lookback)),
        torch.randint(0, 8, (batch, lookback)),
        torch.zeros(batch, lookback + horizon, 5),
        torch.randint(0, 8, (batch, samples, horizon)),
        torch.randint(0, 8, (batch, samples, horizon)),
        advantage,
        config,
        lookback=lookback,
    )
    assert advantage.grad is None
    assert torch.isfinite(loss)
    assert torch.isfinite(mean_logp)
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0


def test_production_nucleus_replay_of_drawn_tokens_stays_finite():
    """top_p=0.8 used to raise when a replayed token sat outside the rebuilt nucleus."""
    model = _Tiny()
    batch, samples, horizon, lookback = 2, 2, 2, 3
    config = ARVolConfig(weight=0.15, temperature=0.65, top_p=0.8, samples=samples)
    loss, mean_logp = reinforce_backward_from_tokens(
        model,
        torch.randint(0, 8, (batch, lookback)),
        torch.randint(0, 8, (batch, lookback)),
        torch.zeros(batch, lookback + horizon, 5),
        torch.randint(0, 8, (batch, samples, horizon)),
        torch.randint(0, 8, (batch, samples, horizon)),
        torch.tensor([[1.0, -1.0], [0.2, -0.2]]),
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
