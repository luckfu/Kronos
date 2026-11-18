"""Unit tests for G12 ShortHorizonsBaseExtractor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from model import Kronos  # noqa: E402
from g12_base_short_horizons_entry import ShortHorizonsBaseExtractor  # noqa: E402
from push_g12 import source_bundle  # noqa: E402


class _FakeTokenizer:
    def encode(self, x: torch.Tensor, half: bool = True):
        b, t, _ = x.shape
        return torch.zeros((b, t), dtype=torch.long, device=x.device), torch.zeros((b, t), dtype=torch.long, device=x.device)


def test_short_horizons_base_extractor_shapes():
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=12, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False, num_sectors=86, num_size_buckets=0, context_layer=10, use_size_percentile=True,
    ).eval()
    ext = ShortHorizonsBaseExtractor(model, _FakeTokenizer(), torch.device("cpu"))
    raw_b = np.random.randn(4, 120, 6).astype(np.float32)
    st_b = np.zeros((4, 120, 5), dtype=np.float32)
    sec_b = np.zeros(4, dtype=np.int64)
    pct_b = np.full(4, 0.5, dtype=np.float32)
    feat = ext.extract_batch(raw_b, st_b, sec_b, pct_b)
    assert feat.shape == (4, 32 * 4)
    assert torch.isfinite(feat).all()
    ext.close()
    assert len(source_bundle()) > 5000
