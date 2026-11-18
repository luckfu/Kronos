"""Unit tests for G19-G22 short-probe extractors, RAM-safe chunked val merge, and bundles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))

from model import Kronos  # noqa: E402
from g19_fulldensity_l45_triarch_entry import TriArchL45Extractor  # noqa: E402
from g20_base_ultrashort_ml_entry import BaseUltrashortMLExtractor  # noqa: E402
from g21_ptsmall_fulldensity_entry import PTSmallUltrashortMLExtractor  # noqa: E402
from g22_ptsmall_microband_entry import WINDOWS as G22_WINDOWS, extract_microband_batch  # noqa: E402
from g23_ptsmall_l45_ultrareg_entry import PTSmallL45MultiLayerExtractor  # noqa: E402
from push_g19 import DATASETS as G19_DATASETS, SLUG as G19_SLUG  # noqa: E402
from push_g20 import DATASETS as G20_DATASETS, SLUG as G20_SLUG  # noqa: E402
from push_g21 import DATASETS as G21_DATASETS, SLUG as G21_SLUG  # noqa: E402
from push_g22 import DATASETS as G22_DATASETS, SLUG as G22_SLUG  # noqa: E402
from push_g23 import DATASETS as G23_DATASETS, SLUG as G23_SLUG  # noqa: E402
from short_probe_common import chunked_val_accumulate_and_sel, push_kernel, solve_block_penalties  # noqa: E402


class _FakeTokenizer:
    def encode(self, x: torch.Tensor, half: bool = True):
        b, t, _ = x.shape
        return torch.zeros((b, t), dtype=torch.long, device=x.device), torch.zeros((b, t), dtype=torch.long, device=x.device)


def _toy_model(n_layers: int, d_model: int, context_layer: int) -> Kronos:
    return Kronos(
        s1_bits=4, s2_bits=4, n_layers=n_layers, d_model=d_model, n_heads=4, ff_dim=d_model * 2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False, num_sectors=86, num_size_buckets=0, context_layer=context_layer, use_size_percentile=True,
    ).eval()


def test_g19_to_g22_extractors_shapes_and_cleanup():
    tok = _FakeTokenizer()
    raw_b = np.random.randn(3, 120, 6).astype(np.float32)
    st_b = np.zeros((3, 120, 5), dtype=np.float32)
    sec_b = np.zeros(3, dtype=np.int64)
    pct_b = np.full(3, 0.5, dtype=np.float32)

    c2 = _toy_model(8, 24, 6)
    pts = _toy_model(8, 24, 6)
    base = _toy_model(12, 32, 10)

    ext19 = TriArchL45Extractor(c2, pts, base, tok, torch.device("cpu"))
    f19 = ext19.extract_batch(raw_b, st_b, sec_b, pct_b)
    ext19.close()
    assert f19.shape == (3, 24 * 3 + 32 * 3)
    assert torch.isfinite(f19).all()

    ext20 = BaseUltrashortMLExtractor(base, tok, torch.device("cpu"))
    f20 = ext20.extract_batch(raw_b, st_b, sec_b, pct_b)
    ext20.close()
    assert f20.shape == (3, 32 * 6)
    assert torch.isfinite(f20).all()

    ext21 = PTSmallUltrashortMLExtractor(pts, tok, torch.device("cpu"))
    f21 = ext21.extract_batch(raw_b, st_b, sec_b, pct_b)
    ext21.close()
    assert f21.shape == (3, 24 * 5)
    assert torch.isfinite(f21).all()

    f22 = extract_microband_batch(pts, tok, raw_b, st_b, sec_b, pct_b, torch.device("cpu"))
    assert f22.shape == (3, 24 * len(G22_WINDOWS))
    assert torch.isfinite(f22).all()

    ext23 = PTSmallL45MultiLayerExtractor(pts, tok, torch.device("cpu"))
    f23 = ext23.extract_batch(raw_b, st_b, sec_b, pct_b)
    ext23.close()
    assert f23.shape == (3, 24 * 3)
    assert torch.isfinite(f23).all()


def test_chunked_val_accumulate_matches_exact_and_aniso_penalties(tmp_path: Path):
    rng = np.random.default_rng(7)
    d = 10
    x1 = rng.standard_normal((12, d)).astype(np.float16)
    x2 = rng.standard_normal((15, d)).astype(np.float16)
    recs = [
        {"date": "2026-04-01", "x": x1, "y_raw": rng.standard_normal(12).astype(np.float32), "y_orth": rng.standard_normal(12).astype(np.float32), "y_multi": rng.standard_normal(12).astype(np.float32), "ret10": rng.standard_normal(12).astype(np.float32)},
        {"date": "2026-06-01", "x": x2, "y_raw": rng.standard_normal(15).astype(np.float32), "y_orth": rng.standard_normal(15).astype(np.float32), "y_multi": rng.standard_normal(15).astype(np.float32), "ret10": rng.standard_normal(15).astype(np.float32)},
    ]
    g_tr0 = rng.standard_normal((d, d))
    g_tr = g_tr0.T @ g_tr0
    rhs_tr = {k: rng.standard_normal(d) for k in ("raw", "orth", "multi")}
    pt0 = tmp_path / "r0.pt"
    torch.save({"n_rows": 50, "gram": g_tr.copy(), "b_raw": rhs_tr["raw"].copy(), "b_orth": rhs_tr["orth"].copy(), "b_multi": rhs_tr["multi"].copy(), "val_recs": recs}, pt0)

    gram_in = np.zeros((d, d), dtype=np.float64)
    rhs_in = {k: np.zeros(d, dtype=np.float64) for k in ("raw", "orth", "multi")}
    n_rows, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel = chunked_val_accumulate_and_sel(
        [str(pt0)], d, "2026-05-18", "2026-07-02", gram_in, rhs_in,
    )
    assert n_rows == 50
    x_cat = np.concatenate([x1.astype(np.float64), x2.astype(np.float64)], axis=0)
    np.testing.assert_allclose(gram_all, g_tr + x_cat.T @ x_cat, rtol=1e-6, atol=1e-6)
    assert x_sel.shape == (15, d)
    assert list(set(d_sel)) == ["2026-06-01"]
    assert len(r_sel) == 15 and len(yo_sel) == 15

    w = solve_block_penalties(gram_all, rhs_all["raw"], slice(2, 10), [(slice(2, 5), 10.0), (slice(5, 10), 100.0)])
    expected = np.linalg.solve(gram_all[2:10, 2:10] + np.diag([10.0] * 3 + [100.0] * 5), rhs_all["raw"][2:10])
    np.testing.assert_allclose(w, expected, rtol=1e-10)


def test_g19_to_g22_dry_bundles_compile():
    specs = [
        ("smmt315", G19_SLUG, "Kronos Alpha G19 Fulldensity L45 Triarch", G19_DATASETS, "g19_fulldensity_l45_triarch_entry.py", "g19", ["g4_base_kaggle_entry.py", "g16_pretrained_small_entry.py"]),
        ("user281434", G20_SLUG, "Kronos Alpha G20 Base Ultrashort ML", G20_DATASETS, "g20_base_ultrashort_ml_entry.py", "g20", ["g4_base_kaggle_entry.py"]),
        ("user281434", G21_SLUG, "Kronos Alpha G21 PTSmall Fulldensity", G21_DATASETS, "g21_ptsmall_fulldensity_entry.py", "g21", ["g16_pretrained_small_entry.py"]),
        ("wynstonliu", G22_SLUG, "Kronos Alpha G22 PTSmall Microband", G22_DATASETS, "g22_ptsmall_microband_entry.py", "g22", ["g16_pretrained_small_entry.py"]),
        ("wynstonliu", G23_SLUG, "Kronos Alpha G23 PTSmall L45 Ultrareg", G23_DATASETS, "g23_ptsmall_l45_ultrareg_entry.py", "g23", ["g16_pretrained_small_entry.py"]),
    ]
    for owner, slug, title, datasets, entry, tag, extra in specs:
        assert title.lower().replace(" ", "-") == slug
        meta = push_kernel(owner=owner, slug=slug, title=title, datasets=datasets, entry_filename=entry, work_tag=tag, extra_gpu_files=extra, push=False)
        assert meta["bundle_bytes"] > 5000
