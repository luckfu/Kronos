"""Unit tests for Wave 5 (G39-G40) STE adapter, distilled/tail-aware architectures, Study E2, and Secondary Track."""
from __future__ import annotations

import ast
import io
import json
import inspect
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))


def test_g40_ste_dep_layer_parity_and_gradient_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import types

    try:
        import einops  # noqa: F401
    except ImportError:
        fake_einops = types.ModuleType("einops")
        fake_einops.rearrange = lambda *a, **kw: None
        fake_einops.reduce = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "einops", fake_einops)
    import model
    importlib.reload(model)
    Kronos = model.Kronos
    import g40_mini_tailaware_gentle_ft_entry as g40

    torch.manual_seed(2026)
    mini = Kronos(
        s1_bits=5, s2_bits=5, n_layers=4, d_model=32, n_heads=2, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=False,
    )
    adapter = g40.TailAwareMiniDepAdapter(mini, freeze_until=2)
    adapter.eval()

    h_low = torch.randn(6, 12, 32)
    p_cont, p_floor, s1_last, s1_teacher = adapter.forward_top(h_low, with_teacher=True)
    assert p_cont.shape == (6,)
    assert p_floor.shape == (6,)
    assert s1_last.shape == (6, 32)
    assert s1_teacher is not None and s1_teacher.shape == (6, 32)

    # Verify exact forward parity between STE dep_layer(s2) and hard argmax emb_s1 lookup
    with torch.no_grad():
        h = h_low.clone()
        for idx in range(adapter.freeze_until, len(adapter.model.transformer)):
            h = adapter.model.transformer[idx](h, key_padding_mask=None)
        h_out = adapter.model.norm(h).float()
        s1_all = adapter.model.head(h_out).float()
        hard_sib = adapter.model.embedding.emb_s1(s1_all.argmax(dim=-1)).float()
        dep_hard = adapter.model.dep_layer(h_out, hard_sib)
        feat_hard = adapter.rank_norm(torch.cat([h_out[:, -1, :], h_out[:, -5:, :].mean(dim=1), dep_hard[:, -1, :]], dim=-1))
        cont_hard = adapter.head_cont(feat_hard).squeeze(-1)
        floor_hard = adapter.head_floor(feat_hard).squeeze(-1)
    assert torch.allclose(p_cont, cont_hard, atol=1e-5, rtol=1e-5)
    assert torch.allclose(p_floor, floor_hard, atol=1e-5, rtol=1e-5)

    # Verify gradients reach dep_layer, head.proj_s1, and top transformer[2..3], while lower blocks stay frozen
    adapter.train()
    adapter.zero_grad()
    yc = torch.randn(6)
    yf = torch.randn(6)
    pc, pf, s1_ft, s1_t = adapter.forward_top(h_low, with_teacher=True)
    loss = 0.7 * g40.tail_penalized_cosic_loss(pc, yc, yf) + 0.3 * g40.cosine_ic_loss_vec(pf, yf)
    loss.backward()

    assert adapter.model.dep_layer.cross_attn.q_proj.weight.grad is not None
    assert float(adapter.model.dep_layer.cross_attn.q_proj.weight.grad.abs().max()) > 0.0
    assert adapter.model.head.proj_s1.weight.grad is not None
    assert float(adapter.model.head.proj_s1.weight.grad.abs().max()) > 0.0
    assert adapter.model.transformer[2].self_attn.q_proj.weight.grad is not None
    assert adapter.model.transformer[3].ffn.w1.weight.grad is not None
    assert adapter.model.transformer[0].self_attn.q_proj.weight.grad is None
    assert adapter.model.transformer[1].self_attn.q_proj.weight.grad is None
    assert adapter.model.embedding.emb_s1.weight.grad is None


def test_g39_g40_architectures_and_zero_teacher_oos() -> None:
    import g39_tailshield_fwd_light_distill_entry as g39
    import g40_mini_tailaware_gentle_ft_entry as g40

    m_dual = g39._build_stu_nn("dual_mlp", g39.IN_DIM_C2, g39.N_ANC_C2)
    mc, mf, mfl = m_dual(torch.randn(24, g39.IN_DIM_C2))
    assert mc.shape == (24,) and mf.shape == (24,) and mfl.shape == (24,)
    loss_39 = g39.tail_penalized_cosic_loss(mc, torch.randn(24), mfl)
    assert torch.isfinite(loss_39)
    loss_39.backward()

    m_xs = g39._build_stu_nn("xstock", g39.IN_DIM_MC2, g39.N_ANC_MC2)
    out_xs = m_xs(torch.randn(24, g39.IN_DIM_MC2))
    assert out_xs.shape == (24,) and torch.isfinite(out_xs).all()

    m40_d = g40.G40DualTailShieldMLP()
    mu40, fl40 = m40_d(torch.randn(20, g40.IN_DIM))
    assert mu40.shape == (20,) and fl40.shape == (20,)
    m40_x = g40.G40CrossStockTailNet()
    out40 = m40_x(torch.randn(20, g40.IN_DIM))
    assert out40.shape == (20,) and torch.isfinite(out40).all()

    src39 = inspect.getsource(g39._score_oos_worker)
    assert "_load_student_only" in src39 and "_load_teacher_and_student" not in src39
    src40 = inspect.getsource(g40._load_joint_extractor)
    assert "_load_kronos_base" not in src40 and "_load_pretrained_small" not in src40


def test_wave5_bundles_compile_and_complete() -> None:
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
    specs = [
        ("g39_tailshield_fwd_light_distill_entry.py", []),
        ("g39_score_oos_entry.py", ["g39_tailshield_fwd_light_distill_entry.py"]),
        ("g40_mini_tailaware_gentle_ft_entry.py", []),
        ("g40_score_oos_entry.py", ["g40_mini_tailaware_gentle_ft_entry.py"]),
    ]
    for entry, extra in specs:
        zf = zipfile.ZipFile(io.BytesIO(spc.source_bundle(entry, base_extras + extra)))
        names = set(zf.namelist())
        gpu_mods = {
            n.split("/")[-1][:-3]
            for n in names
            if n.startswith("finetune/gpu_alpha/") and n.endswith(".py")
        }
        for n in names:
            if not n.startswith("finetune/gpu_alpha/") or not n.endswith(".py"):
                continue
            code = zf.read(n).decode()
            compile(code, n, "exec")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                    if (mod.startswith("g") and "_" in mod) or mod == "wave4_common":
                        assert mod in gpu_mods, f"{entry}: {n} missing {mod}"


def test_e2_mcb_parity_and_invariants(tmp_path: Path) -> None:
    oos_mod = ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"
    if str(oos_mod) not in sys.path:
        sys.path.insert(0, str(oos_mod))
    import eval_no_g32_and_midcycle_shield as e2
    import nonpv_and_regime_study as nrs
    import signal_usage_backtest as sub

    dates = [f"2026-08-{i:02d}" for i in range(1, 32)] + [f"2026-09-{i:02d}" for i in range(1, 21)]
    rows = []
    for di, d in enumerate(dates):
        for s in range(6):
            # Stocks smoothly trending then dropping sharply on day 35..40
            p = 10.0 + di * 0.25 if di < 35 else 10.0 + 34 * 0.25 - (di - 34) * 0.7
            if s == 0 and di == 37:
                # Force limit-down deferral (flat close == exact -10% limit_down from di=36 prev close 17.10)
                down_px = round((10.0 + 34 * 0.25 - 2 * 0.7) * 0.90, 2)
                rows.append({
                    "symbol": f"sh.6000{s:02d}", "date": d,
                    "open": down_px, "high": down_px, "low": down_px, "close": down_px,
                    "volume": 1000.0, "amount": 1e8, "market_cap": 1e10,
                })
            else:
                rows.append({
                    "symbol": f"sh.6000{s:02d}", "date": d,
                    "open": p * 0.995, "high": p * 1.02, "low": p * 0.985, "close": p,
                    "volume": 1000.0, "amount": 1e8, "market_cap": 1e10,
                })
    raw_p = tmp_path / "raw_mcb.csv"
    pd.DataFrame(rows).to_csv(raw_p, index=False)
    market = sub.Market(raw_p)
    scores = np.tile(np.arange(6, 0, -1, dtype=float), (len(market.dates), 1))
    rebalance_days = [22, 32, 42]
    last_day = len(market.dates) - 1
    gate_2 = np.full(len(market.dates), 2, dtype=int)

    # 1. Exact parity with nrs.run_gated_slots when enable_mcb=False
    ref = nrs.run_gated_slots(market, nrs.gated_plan(market, scores, 4, gate_2, True), rebalance_days, 4, last_day)
    got_off = e2.run_mcb_gated_slots(
        market, nrs.gated_plan(market, scores, 4, gate_2, True), rebalance_days, 4, last_day,
        mcb_tier=np.zeros(len(market.dates), dtype=int), enable_mcb=False,
    )
    diff = float((ref["daily"]["ret"] - got_off["daily"]["ret"]).abs().max())
    assert diff < 1e-12
    assert got_off["mcb_triggers"] == 0

    # 2. Downward-only ratchet + one_down deferral + max_decision_day boundary
    mcb_tier = np.full(len(market.dates), 2, dtype=int)
    mcb_tier[36] = 0   # panic day -> should sell at day 37 open except limit-down sh.600000
    mcb_tier[38] = 2   # whipsaw recovery day -> ratchet must NOT re-enter before rebalance 42
    mcb_tier[45] = 0   # beyond max_decision_day=42 -> must be ignored!
    got_on = e2.run_mcb_gated_slots(
        market, nrs.gated_plan(market, scores, 4, gate_2, True), rebalance_days, 4, last_day,
        mcb_tier=mcb_tier, enable_mcb=True, max_decision_day=42,
    )
    assert got_on["mcb_triggers"] >= 1
    # Day 37 open sells non-limit-down stocks; stock 0 stays held on day 37 and sells on day 38
    assert int(got_on["daily"].loc[market.dates[37], "names"]) == 1
    assert int(got_on["daily"].loc[market.dates[38], "names"]) == 0
    assert int(got_on["daily"].loc[market.dates[39], "names"]) == 0
    # After rebalance 42 opens 4 names, panic on day 45 (> max_decision_day=42) must NOT trigger!
    assert int(got_on["daily"].loc[market.dates[46], "names"]) == 4


def test_secondary_sealed_track_synthetic_fixture_and_bundle(tmp_path: Path) -> None:
    import eval_sealed_secondary_w4w5_dec as ess
    import short_probe_common as spc

    # 1. Verify push_sealed_w4w5_secondary bundle compiles cleanly with zero missing local imports
    extras = [
        "sealed_common.py",
        "sealed_w2w3_hidden_score_entry.py",
        "g2_kaggle_entry.py",
        "g4_base_kaggle_entry.py",
        "g16_pretrained_small_entry.py",
        "g24_common.py",
        "g25_base_fulldensity_l45_l52_entry.py",
        "g26_mini2k_pts_alllayers_entry.py",
        "g27_nonlinear_moe_lgbm_entry.py",
        "g28_l45_gentle_ft_entry.py",
        "g29_mini_pool_trifamily_entry.py",
        "g30_quadfam_nonlin_xstock_entry.py",
        "g31_mini_mastery_c2pool_ft_entry.py",
        "g33_base_dep_pool_gentle_ft_entry.py",
        "g34_quad_dep_s1s2_logits_entry.py",
        "wave4_common.py",
        "g35_crash_shield_tailrisk_entry.py",
        "g36_lambdarank_topk_liquid_entry.py",
        "g37_quad_to_light_distill_entry.py",
        "g38_termstruct_hazard_net_entry.py",
        "g39_tailshield_fwd_light_distill_entry.py",
        "g40_mini_tailaware_gentle_ft_entry.py",
    ]
    zf = zipfile.ZipFile(io.BytesIO(spc.source_bundle("sealed_w4w5_secondary_score_entry.py", extras)))
    names = set(zf.namelist())
    gpu_mods = {
        n.split("/")[-1][:-3]
        for n in names
        if n.startswith("finetune/gpu_alpha/") and n.endswith(".py")
    }
    for n in names:
        if not n.startswith("finetune/gpu_alpha/") or not n.endswith(".py"):
            continue
        code = zf.read(n).decode()
        compile(code, n, "exec")
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split(".")[0]
                if (mod.startswith("g") and "_" in mod) or mod in ("wave4_common", "sealed_common"):
                    assert mod in gpu_mods, f"secondary: {n} missing {mod}"

    # 2. Synthetic sealed directory exercising DEFER_AUTO_EXTEND + adjudication end-to-end
    rng = np.random.default_rng(20261128)
    cal = pd.bdate_range("2026-09-25", periods=46).strftime("%Y-%m-%d").tolist()
    codes = [f"600{s:03d}" for s in range(30)]
    kline_rows = []
    for code_i, code in enumerate(codes):
        px = 10.0
        for d in cal:
            px = max(2.0, px * float(np.exp(0.002 * (1 if code_i % 2 == 0 else -0.5) + rng.normal(0.0, 0.01))))
            kline_rows.append({
                "code": code, "date": d,
                "open": px * 0.995, "high": px * 1.015, "low": px * 0.985, "close": px,
                "volume": 1e6, "amount": 1e8, "adj_factor": 1.0,
            })
    pd.DataFrame(kline_rows).to_parquet(tmp_path / "sealed_kline_150d.parquet", index=False)

    sig_dates = cal[22:34]  # only 12 signal dates -> triggers DEFER_AUTO_EXTEND cleanly
    rank_rows, sec_rows = [], []
    req_cols = sorted({c[2:] for members in ess.SECONDARY_ENSEMBLES.values() for c in members if c.startswith("r_p_")} | {"p_g35_lgbm_cvar10"})
    for d in sig_dates:
        for i, code in enumerate(codes):
            c2 = rng.normal()
            v1e4 = rng.normal()
            rank_rows.append({
                "asof": d, "code": code, "predicted_return_d10": c2,
                "score_hidden": v1e4, "score_fused": 0.5,
                "sector_id": i % 5, "size_percentile": (i + 0.5) / len(codes),
            })
            r_dict = {"identity": f"{d}|{code}", "p_val_1e4": v1e4}
            for sc in req_cols:
                if sc != "p_val_1e4":
                    r_dict[sc] = float(rng.normal())
            sec_rows.append(r_dict)
    ranks_df = pd.DataFrame(rank_rows)
    # Exact fused value matching formula on last day
    last_d = sig_dates[-1]
    m_last = ranks_df["asof"] == last_d
    ranks_df.loc[m_last, "score_fused"] = (
        0.5 * ranks_df.loc[m_last, "predicted_return_d10"].rank(pct=True, method="average")
        + 0.5 * ranks_df.loc[m_last, "score_hidden"].rank(pct=True, method="average")
    )
    ranks_df.to_parquet(tmp_path / "sealed_shadow_rankings.parquet", index=False)
    pd.DataFrame(sec_rows).to_parquet(tmp_path / "sealed_w4w5_secondary_scores.parquet", index=False)

    out_json = tmp_path / "sec_out.json"
    rc = ess.main(["--sealed-dir", str(tmp_path), "--output", str(out_json)])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["parity_gate"]["score_hidden_corr"] > 0.9999
    assert payload["parity_gate"]["score_fused_corr"] > 0.9999
    assert payload["prerequisites"]["status"] == "DEFER_AUTO_EXTEND"
    assert payload["decision"]["final_action"] == "DEFER_AUTO_EXTEND"


def test_secondary_and_wave5_ensembles_schema_alignment() -> None:
    import re
    import eval_sealed_secondary_w4w5_dec as ess
    import eval_wave5_master_ensembles as ew5

    # Every column referenced by eval_wave5_master_ensembles and eval_sealed_secondary_w4w5_dec
    # must exist in the corresponding generator source / manifest, and all candidates must have complexity ranks.
    all_rank_cols = {
        m
        for ens in (ess.SECONDARY_ENSEMBLES, ew5.ENSEMBLES)
        for members in ens.values()
        for m in members
    }
    assert all(m == "r_c2" or m.startswith("r_p_") for m in all_rank_cols)

    declared_p_cols = {"p_val_1e4"}
    for _, cols in ew5.SOURCES.values():
        declared_p_cols.update(cols)
    for fname in (
        "g35_crash_shield_tailrisk_entry.py",
        "g37_quad_to_light_distill_entry.py",
        "g38_termstruct_hazard_net_entry.py",
        "g39_tailshield_fwd_light_distill_entry.py",
        "g40_mini_tailaware_gentle_ft_entry.py",
        "sealed_w2w3_hidden_score_entry.py",
    ):
        txt = (ROOT / "finetune" / "gpu_alpha" / fname).read_text()
        declared_p_cols.update(re.findall(r"p_g[0-9]{2}_[a-z0-9_]+", txt))
        # G39 uses f"{prefix}_..." dynamic column construction
        if "p_g39_c2only" in txt:
            for pfx in ("p_g39_c2only", "p_g39_minic2"):
                declared_p_cols.update({
                    f"{pfx}_mlp_cvar_shield",
                    f"{pfx}_mlp_cont_shield",
                    f"{pfx}_xstock_tail",
                })
            declared_p_cols.add("p_g39_minic2_mlp_fwd")

    missing = sorted(m[2:] for m in all_rank_cols if m != "r_c2" and m[2:] not in declared_p_cols)
    assert not missing, f"Missing ensemble columns: {missing}"

    for cand in ess.SECONDARY_ENSEMBLES:
        if cand != "FUSE_VAL_1E4":
            assert cand in ess.SECONDARY_COMPLEXITY_RANK, f"Missing complexity rank for {cand}"
            assert cand in ew5.ENSEMBLES, f"Secondary candidate {cand} must match Wave 5 master ensemble"
            assert ess.SECONDARY_ENSEMBLES[cand] == ew5.ENSEMBLES[cand], f"Mismatch in members for {cand}"



