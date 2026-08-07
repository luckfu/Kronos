from collections import Counter

import pytest
import torch

from finetune.evaluate_size_condition_ablation import (
    balance_period_records,
    distribution_metrics,
    probability_shift_metrics,
)


def test_identical_token_distributions_have_zero_js_divergence():
    counts = Counter({1: 3, 2: 1})

    metrics = distribution_metrics(counts, counts)

    assert metrics['unique_tokens'] == 2
    assert metrics['entropy'] == pytest.approx(0.811278)
    assert metrics['js_divergence_vs_bucket_0'] == pytest.approx(0.0)


def test_disjoint_token_distributions_have_max_js_divergence():
    metrics = distribution_metrics(Counter({1: 4}), Counter({2: 4}))

    assert metrics['js_divergence_vs_bucket_0'] == pytest.approx(1.0)


def test_probability_shift_metrics_detect_changed_distribution():
    reference = torch.tensor([[[8.0, 0.0]]])

    same = probability_shift_metrics(reference, reference)
    changed = probability_shift_metrics(-reference, reference)

    assert same['mean_js_divergence_vs_bucket_0'] == pytest.approx(0.0)
    assert same['top1_change_rate_vs_bucket_0'] == 0.0
    assert changed['mean_js_divergence_vs_bucket_0'] > 0.9
    assert changed['mean_total_variation_vs_bucket_0'] > 0.9
    assert changed['top1_change_rate_vs_bucket_0'] == 1.0


def test_balance_period_records_caps_each_bucket_deterministically():
    period = {
        'period': 0,
        'records': [
            {'symbol': f'{bucket}-{index}', 'size_bucket': bucket}
            for bucket in range(10)
            for index in range(20)
        ],
    }

    first = balance_period_records([period], stocks_per_bucket=10, seed=7)
    second = balance_period_records([period], stocks_per_bucket=10, seed=7)

    assert first == second
    assert len(first[0]['records']) == 100
    counts = Counter(record['size_bucket'] for record in first[0]['records'])
    assert counts == Counter({bucket: 10 for bucket in range(10)})
