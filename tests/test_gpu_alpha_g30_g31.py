"""Unit tests for Wave 3A (G30 quad-family non-linear/xstock & G31 Mini mastery/C2 pooling)."""
from __future__ import annotations

import ast
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))


def test_g30_nn_architectures_and_rank_weighted_loss() -> None:
    import g30_quadfam_nonlin_xstock_entry as g30

    torch.manual_seed(0)
    x = torch.randn(64, g30.IN_DIM)
    y = torch.randn(64)
    for kind in ("resid", "film", "xstock"):
        m = g30._build_nn_model(kind)
        out = m(x)
        assert out.shape == (64,)
        assert torch.isfinite(out).all()
        loss = g30.rank_weighted_cosic_loss(out, y)
        assert torch.isfinite(loss)
        loss.backward()


def test_g30_xstock_attention_permutation_equivariant() -> None:
    import g30_quadfam_nonlin_xstock_entry as g30

    torch.manual_seed(7)
    m = g30.CrossStockFactorAttentionNet(g30.IN_DIM).eval()
    m.res_scale.data.fill_(0.5)
    x = torch.randn(32, g30.IN_DIM)
    perm = torch.randperm(32)
    with torch.no_grad():
        y1 = m(x)[perm]
        y2 = m(x[perm])
    assert torch.allclose(y1, y2, atol=1e-5)


def test_g31_date_weights_and_dimensions() -> None:
    import g31_mini_mastery_c2pool_ft_entry as g31

    w_bull = g31.compute_date_weights_g31("2025-07-02", 4500, 0.55)
    w_bear = g31.compute_date_weights_g31("2025-07-02", 4500, 0.20)
    assert abs(w_bull["decay365"] - 0.5) < 1e-2
    assert abs(w_bull["decay730"] - 0.5 ** 0.5) < 1e-2
    assert abs(w_bear["regime"] / w_bull["regime"] - 3.0) < 1e-6
    assert g31.JOINT_DIM == 3072 + 8 + 3584


def test_g30_g31_bundles_have_no_missing_local_imports() -> None:
    import short_probe_common as spc

    cases = [
        (
            "g30_quadfam_nonlin_xstock_entry.py",
            [
                "g2_kaggle_entry.py",
                "g4_base_kaggle_entry.py",
                "g16_pretrained_small_entry.py",
                "g24_common.py",
                "g26_mini2k_pts_alllayers_entry.py",
            ],
        ),
        (
            "g31_mini_mastery_c2pool_ft_entry.py",
            [
                "g2_kaggle_entry.py",
                "g16_pretrained_small_entry.py",
                "g24_common.py",
                "g26_mini2k_pts_alllayers_entry.py",
                "g28_l45_gentle_ft_entry.py",
            ],
        ),
    ]
    for entry, extras in cases:
        zf = zipfile.ZipFile(io.BytesIO(spc.source_bundle(entry, extras)))
        names = set(zf.namelist())
        gpu_mods = {
            n.split("/")[-1][:-3]
            for n in names
            if n.startswith("finetune/gpu_alpha/") and n.endswith(".py")
        }
        for n in names:
            if not n.startswith("finetune/gpu_alpha/") or not n.endswith(".py"):
                continue
            tree = ast.parse(zf.read(n).decode())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                    if mod.startswith("g") and "_" in mod:
                        assert mod in gpu_mods, f"{entry}: {n} missing {mod}"
