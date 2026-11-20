"""Unit and bundle verification tests for Wave 2 GPU alpha probes (G24-G26)."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "finetune" / "gpu_alpha"
if str(GPU_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_DIR))


def test_g24_common_weights_and_r5_breadth() -> None:
    import g24_common as gc

    w_recent, w_reg_bull = gc.compute_date_weights("2026-07-02", n_stocks=4500, breadth_ma20=0.60)
    w_old, w_reg_bear = gc.compute_date_weights("2024-07-02", n_stocks=4500, breadth_ma20=0.20)
    assert abs(w_recent - 1.0) < 1e-6
    assert abs(w_old - 0.5) < 0.01
    assert abs(w_reg_bull - 1.0) < 1e-6
    assert abs(w_reg_bear - 3.0) < 1e-6

    # Synthetic 2-stock panel: stock 0 trending up (+10% over 5d, above MA20), stock 1 flat/down
    vals0 = np.ones((140, 6), dtype=np.float32) * 10.0
    vals0[120, 3] = 12.0  # above MA20
    vals0[125, 3] = 13.2  # +10%
    vals1 = np.ones((140, 6), dtype=np.float32) * 10.0
    vals1[120, 3] = 9.0   # below MA20
    vals1[125, 3] = 8.1   # -10%
    sym_arr = [(vals0, np.zeros((140, 5), dtype=np.float32)), (vals1, np.zeros((140, 5), dtype=np.float32))]
    r5, breadth = gc.extract_r5_and_breadth(sym_arr, np.array([0, 1]), np.array([120, 120]))
    assert abs(breadth - 0.5) < 1e-6
    assert abs(r5[0] - 0.10) < 1e-5
    assert abs(r5[1] - (-0.10)) < 1e-5


def test_g24_g26_bundles_compile_and_contain_dependencies() -> None:
    import short_probe_common as spc

    cases = [
        ("g24_small_apex_pool_decay_entry.py", ["g16_pretrained_small_entry.py", "g24_common.py"]),
        ("g25_base_fulldensity_l45_l52_entry.py", ["g4_base_kaggle_entry.py", "g24_common.py"]),
        ("g26_mini2k_pts_alllayers_entry.py", ["g16_pretrained_small_entry.py", "g24_common.py"]),
        ("g27_nonlinear_moe_lgbm_entry.py", ["g4_base_kaggle_entry.py", "g16_pretrained_small_entry.py", "g24_common.py"]),
        ("g28_l45_gentle_ft_entry.py", ["g2_kaggle_entry.py", "g16_pretrained_small_entry.py"]),
    ]
    for entry, extra in cases:
        payload = spc.source_bundle(entry, extra)
        import io
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = set(zf.namelist())
            assert f"finetune/gpu_alpha/{entry}" in names
            assert "finetune/gpu_alpha/short_probe_common.py" in names
            assert "model/kronos.py" in names
        src = (GPU_DIR / entry).read_text()
        compile(src, str(GPU_DIR / entry), "exec")


def test_g24_g26_slice_dimensions_consistent() -> None:
    import g24_small_apex_pool_decay_entry as g24
    import g25_base_fulldensity_l45_l52_entry as g25
    import g26_mini2k_pts_alllayers_entry as g26

    assert g24.FEAT_DIM == 4096
    assert g24.SL_PTS_ALL.stop == g24.FEAT_DIM
    assert g25.FEAT_DIM == 5824
    assert g25.SL_ALL.stop == g25.FEAT_DIM
    assert g26.FEAT_DIM == 5376
    assert len(g26.SL_PTS_LAYERS) == 8
    assert g26.SL_PTS_LAYERS[-1].stop == g26.FEAT_DIM
