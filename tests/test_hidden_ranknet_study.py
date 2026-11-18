"""Unit tests for hidden_ranknet_study cross-sectional training & loss."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"))

import hidden_ranknet_study as hrs  # noqa: E402


def test_cosine_ic_mlp_recovers_nonlinear_cross_sectional_signal() -> None:
    rng = np.random.default_rng(123)
    dates = np.repeat([f"2026-01-{d:02d}" for d in range(1, 21)], 60)
    x = rng.normal(0.0, 1.0, size=(len(dates), 32)).astype(np.float32)
    # Non-linear planted signal: product / ReLU interaction + small noise
    y = (np.maximum(x[:, 0], 0.0) * x[:, 1] + 0.3 * x[:, 2] + rng.normal(0.0, 0.1, size=len(dates))).astype(np.float64)
    xz = hrs.zscore_hidden_by_date(dates, x)
    yz = hrs.date_standardized_targets(dates, y)
    batches = hrs.build_date_batches(dates)

    # Modify model dim to 32 for synthetic test via train_mlp_fold logic
    torch.manual_seed(1)
    model = hrs.CosineICResidualMLP(in_dim=32, hidden_dim=32, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy(xz), torch.from_numpy(yz)
    for _ in range(25):
        for b in batches:
            idx = torch.from_numpy(b)
            opt.zero_grad(set_to_none=True)
            loss = hrs.cosine_ic_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.inference_mode():
        pred = model(xt).numpy()
    ic = hrs.daily_pearson_on_z(dates, pred, yz)
    assert ic > 0.70, f"Expected non-linear ranker to recover strong IC, got {ic}"
