"""CPU tests for Stage3 P1 CE+rank objective."""
import torch

from finetune.stage3_rank_loss import naive_same_date_pairwise_ranking_loss, rank_batch_diagnostics


def test_naive_same_date_pairwise_prefers_correct_order():
    utilities = torch.tensor([0.05, -0.03, 0.04, -0.02], dtype=torch.float32)
    date_ids = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    good = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float32, requires_grad=True)
    bad = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)
    good_loss = naive_same_date_pairwise_ranking_loss(good, utilities, date_ids)
    bad_loss = naive_same_date_pairwise_ranking_loss(bad, utilities, date_ids)
    assert float(good_loss.item()) >= 0.0
    assert float(bad_loss.item()) > float(good_loss.item())
    good_loss.backward()
    assert good.grad is not None


def test_rank_batch_diagnostics_counts_pairs():
    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    utilities = torch.tensor([0.01, 0.02, -0.01], dtype=torch.float32)
    date_ids = torch.tensor([7, 7, 7], dtype=torch.long)
    diag = rank_batch_diagnostics(scores, utilities, date_ids)
    assert diag['unique_signal_dates'] == 1
    assert diag['pair_count'] == 3
