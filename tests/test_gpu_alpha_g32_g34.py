"""Unit tests for Wave 3B/3C (G32 L=45 AR, G33 Base dep/pool/FT, G34 Quad dep+s1/s2 logits, failfast helper)."""
from __future__ import annotations

import ast
import io
import os
import signal
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))


def _ok_worker(arg: tuple[int, int]) -> int:
    rank, extra = arg
    return rank + extra


def _sigkill_worker(arg: int) -> int:
    if arg == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    return arg


def test_run_gpu_workers_failfast_ok_and_sigkill() -> None:
    import short_probe_common as spc

    res = spc.run_gpu_workers_failfast(_ok_worker, [(0, 10), (1, 20)], tag="test_ok")
    assert res == [10, 21]
    with pytest.raises(RuntimeError, match="exitcode"):
        spc.run_gpu_workers_failfast(_sigkill_worker, [0, 1], tag="test_kill")


def test_g32_summarize_paths_contract() -> None:
    import g32_l45_ar16_triarch_entry as g32

    rng = np.random.default_rng(42)
    paths = rng.normal(loc=0.01, scale=0.03, size=(8, 16, 10)).astype(np.float32)
    paths[0] += 0.10
    paths[1] -= 0.10
    d10_med, d10_mean, d10_sh, d10_up, d5_med = g32._summarize_paths(paths)
    for arr in (d10_med, d10_mean, d10_sh, d10_up, d5_med):
        assert arr.shape == (8,)
        assert np.isfinite(arr).all()
    assert d10_med[0] > d10_med[1]
    assert d10_mean[0] > d10_mean[1]
    assert d10_up[0] >= d10_up[1]
    assert np.all((d10_up >= 0.0) & (d10_up <= 1.0))


def test_g34_feature_slices_exact() -> None:
    import g34_quad_dep_s1s2_logits_entry as g34

    assert g34.SL_DEP_MINI == slice(0, 256)
    assert g34.SL_DEP_PTS == slice(256, 768)
    assert g34.SL_DEP_C2 == slice(768, 1280)
    assert g34.SL_DEP_BASE == slice(1280, 2112)
    assert g34.SL_DEP_QUAD == slice(0, 2112)
    assert g34.SL_ST_QUAD == slice(2112, 2152)
    assert g34.FEAT_DIM == 2152


def test_g32_g33_g34_g30score_bundles_have_no_missing_local_imports() -> None:
    import short_probe_common as spc

    cases = [
        (
            "g32_l45_ar16_triarch_entry.py",
            [
                "g16_pretrained_small_entry.py",
                "g24_common.py",
                "g26_mini2k_pts_alllayers_entry.py",
            ],
        ),
        (
            "g33_base_dep_pool_gentle_ft_entry.py",
            [
                "g2_kaggle_entry.py",
                "g4_base_kaggle_entry.py",
                "g16_pretrained_small_entry.py",
                "g28_l45_gentle_ft_entry.py",
            ],
        ),
        (
            "g34_quad_dep_s1s2_logits_entry.py",
            [
                "g2_kaggle_entry.py",
                "g4_base_kaggle_entry.py",
                "g16_pretrained_small_entry.py",
                "g24_common.py",
                "g26_mini2k_pts_alllayers_entry.py",
                "g28_l45_gentle_ft_entry.py",
                "g31_mini_mastery_c2pool_ft_entry.py",
            ],
        ),
        (
            "g30_score_oos_entry.py",
            [
                "g2_kaggle_entry.py",
                "g4_base_kaggle_entry.py",
                "g16_pretrained_small_entry.py",
                "g24_common.py",
                "g26_mini2k_pts_alllayers_entry.py",
                "g30_quadfam_nonlin_xstock_entry.py",
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
