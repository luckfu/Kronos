"""Unit tests for G13/G14 extractors and bundles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from model import Kronos  # noqa: E402
from g14_base_dense_short_entry import WINDOWS, extract_band_batch  # noqa: E402
from push_g13 import source_bundle as g13_bundle  # noqa: E402
from push_g14 import source_bundle as g14_bundle  # noqa: E402


class _FakeTokenizer:
    def encode(self, x: torch.Tensor, half: bool = True):
        b, t, _ = x.shape
        return torch.zeros((b, t), dtype=torch.long, device=x.device), torch.zeros((b, t), dtype=torch.long, device=x.device)


def test_g14_extract_band_batch_and_bundles():
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=12, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False, num_sectors=86, num_size_buckets=0, context_layer=10, use_size_percentile=True,
    ).eval()
    raw_b = np.random.randn(4, 120, 6).astype(np.float32)
    st_b = np.zeros((4, 120, 5), dtype=np.float32)
    sec_b = np.zeros(4, dtype=np.int64)
    pct_b = np.full(4, 0.5, dtype=np.float32)
    feat = extract_band_batch(model, _FakeTokenizer(), raw_b, st_b, sec_b, pct_b, torch.device("cpu"))
    assert feat.shape == (4, 32 * len(WINDOWS))
    assert torch.isfinite(feat).all()
    assert len(g13_bundle()) > 5000
    assert len(g14_bundle()) > 5000
