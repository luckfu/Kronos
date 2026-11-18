"""Unit tests for G11 MultiLayerBaseExtractor forward hooks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from model import Kronos  # noqa: E402
from g11_base_multilayer_entry import MultiLayerBaseExtractor  # noqa: E402
from push_g11 import source_bundle  # noqa: E402


class _FakeTokenizer:
    def encode(self, x: torch.Tensor, half: bool = True):
        b, t, _ = x.shape
        return torch.zeros((b, t), dtype=torch.long, device=x.device), torch.zeros((b, t), dtype=torch.long, device=x.device)


def test_multilayer_base_extractor_hooks_match_dim_and_clean_up():
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=12, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False, num_sectors=86, num_size_buckets=0, context_layer=10, use_size_percentile=True,
    ).eval()
    ext = MultiLayerBaseExtractor(model, _FakeTokenizer(), torch.device("cpu"))
    raw_b = np.random.randn(4, 20, 6).astype(np.float32)
    st_b = np.zeros((4, 20, 5), dtype=np.float32)
    sec_b = np.zeros(4, dtype=np.int64)
    pct_b = np.full(4, 0.5, dtype=np.float32)
    feat = ext.extract_batch(raw_b, st_b, sec_b, pct_b)
    assert feat.shape == (4, 32 * 4)
    assert torch.isfinite(feat).all()
    ext.close()
    assert len(source_bundle()) > 5000
