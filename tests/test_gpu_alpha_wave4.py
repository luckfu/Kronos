"""Unit tests for Wave 4 (G35-G38) Kaggle bundles, losses, target anti-leakage, and Study E1 helpers."""
from __future__ import annotations

import ast
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))


def test_wave4_path_targets_never_access_past_ti_plus_10() -> None:
    import wave4_common as w4

    # Construct synthetic price series where ti+11 jumps 100x — extract_10d_path_targets MUST ignore it
    T = 150
    vals = np.ones((T, 6), dtype=np.float32) * 10.0
    ti = 120
    vals[ti : ti + 11, 3] = np.array([10, 10.2, 9.8, 10.3, 10.5, 10.8, 10.0, 9.5, 9.4, 9.6, 10.1], dtype=np.float32)
    vals[ti + 11 :, 3] = 1000.0  # future beyond horizon!
    sym_arr = [(vals, np.zeros((T, 5), dtype=np.float32))]

    out = w4.extract_10d_path_targets(sym_arr, np.array([0]), np.array([ti]))
    assert out["r5"][0] == pytest.approx(10.8 / 10.0 - 1.0, rel=1e-5)
    assert out["r10"][0] == pytest.approx(10.1 / 10.0 - 1.0, rel=1e-5)
    assert out["r_fwd5_10"][0] == pytest.approx(10.1 / 10.8 - 1.0, rel=1e-5)
    assert out["floor10"][0] == pytest.approx(9.4 / 10.0 - 1.0, rel=1e-5)
    assert out["maxdd10"][0] == pytest.approx(1.0 - 9.4 / 10.8, rel=1e-5)


def test_g35_dual_reward_mlp_and_tail_loss() -> None:
    import g35_crash_shield_tailrisk_entry as g35

    torch.manual_seed(0)
    x = torch.randn(48, g35.IN_DIM)
    yc = torch.randn(48)
    yf = torch.randn(48)
    m = g35.DualRewardTailShieldMLP(g35.IN_DIM)
    mu, fl = m(x)
    assert mu.shape == (48,) and fl.shape == (48,)
    loss = g35.tail_penalized_cosic_loss(mu, yc, yf)
    assert torch.isfinite(loss)
    loss.backward()


def test_g36_relevance_labels_and_focal_listnet() -> None:
    import g36_lambdarank_topk_liquid_entry as g36

    rets = np.linspace(-0.1, 0.2, 100)
    is_liq = np.arange(100) >= 35
    # Force top stock to be illiquid -> must be capped at <= 3!
    is_liq[-1] = False
    rel = g36.make_relevance_labels(rets, is_liq)
    assert rel[-1] <= 3
    assert rel[-2] >= 12
    assert rel.min() >= 0 and rel.max() <= 15

    pred = torch.randn(100, requires_grad=True)
    loss = g36.focal_listnet_loss(pred, torch.as_tensor(rets, dtype=torch.float32), torch.as_tensor(is_liq, dtype=torch.float32))
    assert torch.isfinite(loss)
    loss.backward()


def test_g37_student_architectures_and_zero_teacher_oos() -> None:
    import inspect
    import g37_quad_to_light_distill_entry as g37

    for kind, in_dim, n_anc in [("resid", g37.IN_DIM_C2, g37.N_ANC_C2), ("xstock", g37.IN_DIM_MC2, g37.N_ANC_MC2)]:
        m = g37._build_stu_nn(kind, in_dim, n_anc)
        out = m(torch.randn(32, in_dim))
        assert out.shape == (32,) and torch.isfinite(out).all()

    src = inspect.getsource(g37._score_oos_worker)
    assert "_load_student_only" in src
    assert "_load_teacher_and_student" not in src


def test_g38_termstruct_net_and_hazard_fit() -> None:
    import g38_termstruct_hazard_net_entry as g38

    m = g38.TermStructContNet(g38.IN_DIM)
    out = m(torch.randn(24, g38.IN_DIM))
    assert out.shape == (24, 3) and torch.isfinite(out).all()

    rng = np.random.default_rng(7)
    macro_tr = [
        {"date": f"2024-01-{i+1:02d}", "b20": 0.45, "univ_ret_d10": -0.06 if i % 4 == 0 else 0.02, "feat": rng.normal(size=16).astype(np.float32)}
        for i in range(25)
    ]
    macro_va = [
        {"date": f"2026-06-{i+1:02d}", "b20": 0.42, "univ_ret_d10": -0.05 if i % 5 == 0 else 0.01, "feat": rng.normal(size=16).astype(np.float32)}
        for i in range(15)
    ]
    haz = g38.fit_crash_hazard_models(macro_tr, macro_va)
    assert len(haz["coef"]) == 16
    assert 0.0 < haz["train_pos_rate"] < 1.0


def test_wave4_all_bundles_have_no_missing_local_imports() -> None:
    import short_probe_common as spc

    common_extras = [
        "g2_kaggle_entry.py",
        "g4_base_kaggle_entry.py",
        "g16_pretrained_small_entry.py",
        "g24_common.py",
        "g26_mini2k_pts_alllayers_entry.py",
        "g28_l45_gentle_ft_entry.py",
        "g31_mini_mastery_c2pool_ft_entry.py",
        "g34_quad_dep_s1s2_logits_entry.py",
        "wave4_common.py",
    ]
    entries = [
        "g35_crash_shield_tailrisk_entry.py",
        "g36_lambdarank_topk_liquid_entry.py",
        "g37_quad_to_light_distill_entry.py",
        "g38_termstruct_hazard_net_entry.py",
    ]
    for entry in entries:
        zf = zipfile.ZipFile(io.BytesIO(spc.source_bundle(entry, common_extras)))
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
                    if (mod.startswith("g") and "_" in mod) or mod == "wave4_common":
                        assert mod in gpu_mods, f"{entry}: {n} missing {mod}"


def test_e1_breadth_velocity_gate_logic(tmp_path: Path) -> None:
    import eval_adaptive_execution_shield as e1

    dates = [f"2026-07-{i:02d}" for i in range(1, 28)]
    rows = []
    # 10 stocks: steady rise then sudden sharp 5d pullback -> breadth_ma20 stays >= 0.40 while breadth_ma5 collapses
    for i, d in enumerate(dates):
        for s in range(10):
            base_p = 10.0 + (i * 0.5 if i < 23 else 22 * 0.5 - (i - 22) * 1.1)
            rows.append({
                "symbol": f"sh.6000{s:02d}",
                "date": d,
                "open": base_p,
                "high": base_p,
                "low": base_p,
                "close": base_p if s < 6 else 10.0 - i * 0.1,
                "volume": 1000.0,
                "amount": 1e8,
                "market_cap": 1e10,
            })
    p = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    reg, _ = e1.compute_causal_breadth_and_vol_mask(p)
    assert "G_BREADTH_VEL" in reg.columns
    assert int(reg.iloc[-1]["G_BREADTH"]) >= int(reg.iloc[-1]["G_BREADTH_VEL"])


def _helper_double(x: int) -> int:
    return x * 2


def _helper_fail() -> None:
    raise RuntimeError("boom")


def test_run_in_isolated_process_success_and_failfast() -> None:
    import wave4_common as w4

    assert w4.run_in_isolated_process(_helper_double, 21) == 42
    with pytest.raises(RuntimeError, match="exited with code"):
        w4.run_in_isolated_process(_helper_fail)


def test_wave4_score_bundles_compile_and_complete() -> None:
    import short_probe_common as spc

    base_extras = [
        "g2_kaggle_entry.py",
        "g4_base_kaggle_entry.py",
        "g16_pretrained_small_entry.py",
        "g24_common.py",
        "g26_mini2k_pts_alllayers_entry.py",
        "g28_l45_gentle_ft_entry.py",
        "g31_mini_mastery_c2pool_ft_entry.py",
        "g34_quad_dep_s1s2_logits_entry.py",
        "wave4_common.py",
    ]
    pairs = [
        ("g35_score_oos_entry.py", "g35_crash_shield_tailrisk_entry.py"),
        ("g36_score_oos_entry.py", "g36_lambdarank_topk_liquid_entry.py"),
        ("g37_score_oos_entry.py", "g37_quad_to_light_distill_entry.py"),
        ("g38_score_oos_entry.py", "g38_termstruct_hazard_net_entry.py"),
    ]
    for entry, extra in pairs:
        zf = zipfile.ZipFile(io.BytesIO(spc.source_bundle(entry, base_extras + [extra])))
        names = set(zf.namelist())
        assert f"finetune/gpu_alpha/{entry}" in names
        assert f"finetune/gpu_alpha/{extra}" in names
        compile(zf.read(f"finetune/gpu_alpha/{entry}").decode(), entry, "exec")


def test_g38_hazard_gate_tiers_mapping() -> None:
    import eval_wave4_master_ensembles as ew4

    b20 = pd.Series([0.55, 0.55, 0.55, 0.38, 0.25])
    prob = pd.Series([0.20, 0.60, 0.75, 0.60, 0.10])
    tiers = ew4.compute_hazard_gate_tiers(b20, prob)
    assert tiers.tolist() == [2, 1, 0, 0, 0]
