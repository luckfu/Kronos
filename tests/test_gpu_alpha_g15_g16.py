"""Unit tests for G15 L60 patcher and G16 pretrained small extractor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from model import Kronos  # noqa: E402
from g15_base_l60_ar_entry import patch_evaluate_source_for_l60  # noqa: E402
from g16_pretrained_small_entry import WINDOWS, extract_pt_small_batch  # noqa: E402
from push_g15 import source_bundle as g15_bundle  # noqa: E402
from push_g16 import source_bundle as g16_bundle  # noqa: E402


class _FakeTokenizer:
    def encode(self, x: torch.Tensor, half: bool = True):
        b, t, _ = x.shape
        return torch.zeros((b, t), dtype=torch.long, device=x.device), torch.zeros((b, t), dtype=torch.long, device=x.device)


def test_patch_evaluate_source_for_l60_rewrites_all_slices():
    ev_path = ROOT / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"
    patched = patch_evaluate_source_for_l60(ev_path.read_text())
    assert "LOOKBACK, PREDICT, WINDOW = 60, 10, 70" in patched
    assert "stamp[:, :60], stamp[:, 60:70]" in patched
    assert "dates[120:130]" not in patched
    assert "raw_close[120:130]" not in patched
    assert len(g15_bundle()) > 5000


def test_g16_extract_pt_small_batch_shapes():
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=8, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False, num_sectors=86, num_size_buckets=0, context_layer=6, use_size_percentile=True,
    ).eval()
    raw_b = np.random.randn(4, 120, 6).astype(np.float32)
    st_b = np.zeros((4, 120, 5), dtype=np.float32)
    sec_b = np.zeros(4, dtype=np.int64)
    pct_b = np.full(4, 0.5, dtype=np.float32)
    feat = extract_pt_small_batch(model, _FakeTokenizer(), raw_b, st_b, sec_b, pct_b, torch.device("cpu"))
    assert feat.shape == (4, 32 * len(WINDOWS))
    assert torch.isfinite(feat).all()
    assert len(g16_bundle()) > 5000
