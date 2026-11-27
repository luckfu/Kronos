"""G36: Liquid-Pool Top-10/20 LambdaRank & Focal-ListNet Precision Engine.

Directly targets real-world top-10 / top-20 liquid portfolio alpha (`liq10`, `liq20`, `k=10/20` net return):
- Scale-invariant liquid-pool mask via daily cross-sectional 20d median amount percentile (amt_pct >= 0.35).
- Exponentially upper-tail concentrated integer relevance grades (0..15), capping illiquid bottom-35% names <= 3.
- Trains LightGBM LGBMRanker (objective='lambdarank', ndcg_eval_at=[10, 20]) on r10 and multi-horizon targets,
  plus 3-seed Focal-ListNet MLP and 3-seed Top-K CrossStockFactorAttentionNet.
- Strictly leak-free: train <= 2026-04-30, val_sel 2026-05-18..2026-07-02, OOS 2026-07-17..2026-10-23.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from g1_kaggle_entry import build_oos_batches, find_dataset_file
from g1_multiscale_extract_and_fit import (
    TRAIN_FIT_END,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    daily_spearman_ic,
    phase,
    rank_pct_centered,
    standardize_1d,
)
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec
from short_probe_common import detect_gpu_count, run_gpu_workers_failfast
from wave4_common import (
    CTX_DIM,
    FEAT_DIM,
    K_PCA,
    PCA_DIM,
    SL_ALL,
    SL_BASE,
    SL_C2,
    SL_MINI,
    SL_PTS,
    compute_quad_context_14d,
    compute_wave4_date_weights,
    extract_10d_path_targets,
    fwd_date_batched,
    load_quad_models,
    make_c2_paths_dict,
    pca_quad_worker,
    project_pca_4,
    run_in_isolated_process,
    solve_ridge_specs_from_disk,
    top_evecs_from_pts,
)

OUTPUT = Path(os.environ.get("G36_OUTPUT", "/kaggle/working/g36_out"))
INPUT = Path(os.environ.get("G36_INPUT", "/kaggle/input"))
TMP_DIR = Path("/kaggle/working/_tmp_g36")

TRAIN_START_G36 = "2021-01-01"
TRAIN_STRIDE_G36 = 2
SCHEMES = ("std", "regime", "liq_regime")
TARGETS = ("raw", "multi", "ret5")
ALPHAS_SINGLE = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_QUAD = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)

ANCHOR_KEYS = [
    "mini_regime",
    "base_raw",
    "pts_multi",
    "c2_multi",
    "quad_multi",
    "quad_regime",
    "quad_ret5",
    "quad_liq_multi",
]
N_ANCHORS = len(ANCHOR_KEYS)
IN_DIM = N_ANCHORS + PCA_DIM + CTX_DIM  # 8 + 192 + 14 = 214

LABEL_GAIN = [0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.8, 5.0, 6.5, 8.5, 11.0, 14.0, 18.0, 23.0, 30.0]


def make_relevance_labels(values: np.ndarray, is_liq65: np.ndarray) -> np.ndarray:
    """Map cross-sectional target values into 0..15 upper-tail grades; cap illiquid names at <= 3."""
    pct = rank_pct_centered(values) + 0.5  # (0, 1]
    bins = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995])
    rel = np.digitize(pct, bins, right=False).astype(np.int32)
    rel = np.where(is_liq65, rel, np.minimum(rel, 3))
    return rel


def focal_listnet_loss(pred: torch.Tensor, target_z: torch.Tensor, is_liq: torch.Tensor, tau: float = 2.5) -> torch.Tensor:
    """Liquid-masked top-tail ListNet loss concentrating probability mass on tradable upper winners."""
    p_norm = (pred - pred.mean()) / pred.std(unbiased=False).clamp_min(1e-6)
    log_mask = torch.where(is_liq > 0.5, 0.0, -3.0)
    tgt_prob = F.softmax(tau * target_z + log_mask, dim=0)
    log_pred = F.log_softmax(tau * p_norm, dim=0)
    return -(tgt_prob * log_pred).sum()


class TopKFiLMMLP(nn.Module):
    """Regime & liquidity-aware residual FiLM MLP for Top-K ranking sharpness."""

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128, dropout: float = 0.45):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, N_ANCHORS))
        self.in_norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.film1 = nn.Linear(4, hidden * 2)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mkt = x[:, -4:]
        base = (x[:, :N_ANCHORS] * F.softmax(self.gate_net(mkt), dim=1)).sum(dim=1)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        g, b = self.film1(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g)) + 0.25 * b
        h = F.silu(self.fc2(self.drop(h)))
        return base + torch.tanh(self.res_scale) * self.head(self.drop(h)).squeeze(-1)


class CrossStockTopKAttnNet(nn.Module):
    """Permutation-equivariant latent-factor CrossStock Attention optimized for Top-K precision."""

    def __init__(self, in_dim: int = IN_DIM, d_attn: int = 64, n_factors: int = 8, dropout: float = 0.45):
        super().__init__()
        self.d_attn = d_attn
        self.gate_net = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, N_ANCHORS))
        self.in_norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.Linear(in_dim, d_attn), nn.SiLU(), nn.LayerNorm(d_attn))
        self.latent_q = nn.Parameter(torch.randn(n_factors, d_attn) * 0.08)
        self.k_stock = nn.Linear(d_attn, d_attn, bias=False)
        self.v_stock = nn.Linear(d_attn, d_attn, bias=False)
        self.q_stock = nn.Linear(d_attn, d_attn, bias=False)
        self.k_factor = nn.Linear(d_attn, d_attn, bias=False)
        self.v_factor = nn.Linear(d_attn, d_attn, bias=False)
        self.ctx_norm = nn.LayerNorm(d_attn)
        self.out_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_attn * 2, d_attn),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_attn, 1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mkt = x[:, -4:]
        base = (x[:, :N_ANCHORS] * F.softmax(self.gate_net(mkt), dim=1)).sum(dim=1)
        h = self.proj(self.drop(self.in_norm(x)))
        scale = float(self.d_attn) ** -0.5
        attn_pool = F.softmax((self.latent_q @ self.k_stock(h).T) * scale, dim=-1)
        factors = attn_pool @ self.v_stock(h)
        attn_back = F.softmax((self.q_stock(h) @ self.k_factor(factors).T) * scale, dim=-1)
        ctx = self.ctx_norm(attn_back @ self.v_factor(factors))
        feat = torch.cat([h, h - ctx], dim=1)
        return base + torch.tanh(self.res_scale) * self.out_mlp(feat).squeeze(-1)


def _build_nn(kind: str) -> nn.Module:
    if kind == "film_topk":
        return TopKFiLMMLP(IN_DIM)
    if kind == "xstock_topk":
        return CrossStockTopKAttnNet(IN_DIM)
    raise ValueError(kind)


@torch.inference_mode()
def _main_extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, ext = load_quad_models(rank, paths)
    try:
        u_m = torch.as_tensor(u_dict["m"], device=device, dtype=torch.float32)
        u_b = torch.as_tensor(u_dict["b"], device=device, dtype=torch.float32)
        u_p = torch.as_tensor(u_dict["p"], device=device, dtype=torch.float32)
        u_c = torch.as_tensor(u_dict["c"], device=device, dtype=torch.float32)

        grams = {s: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for s in SCHEMES}
        rhs = {s: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in TARGETS} for s in SCHEMES}

        tr_recs = []
        n_rows = 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            path_d = extract_10d_path_targets(sym_tr, s_idx, t_idx)
            ctx_np, st = compute_quad_context_14d(sym_tr, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(str(d_str), len(s_idx), st["b20"])
            is_liq = st["is_liq65"]

            x32 = fwd_date_batched(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            yr_np = standardize_1d(rank_pct_centered(r10)).astype(np.float32)
            y5_np = standardize_1d(rank_pct_centered(path_d["r5"])).astype(np.float32)
            y_map = {
                "raw": torch.as_tensor(yr_np, device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
                "ret5": torch.as_tensor(y5_np, device=device, dtype=torch.float64),
            }
            xx = x64.T @ x64
            xys = {t: x64.T @ y_map[t] for t in TARGETS}
            for s in ("std", "regime"):
                w = w_map[s]
                grams[s].add_(xx, alpha=w)
                for t in TARGETS:
                    rhs[s][t].add_(xys[t], alpha=w)

            # Row-weighted liquid-pool focus (1.0 for liquid top-65%, 0.25 for illiquid tail)
            rw = torch.as_tensor(np.where(is_liq, 1.0, 0.25), device=device, dtype=torch.float64) * w_map["regime"]
            x64_w = x64 * rw.sqrt().unsqueeze(1)
            grams["liq_regime"].addmm_(x64_w.T, x64_w)
            for t in TARGETS:
                rhs["liq_regime"][t].addmv_(x64.T, y_map[t] * rw)
            n_rows += int(x32.shape[0])

            z_pca = project_pca_4(x32, u_m, u_b, u_p, u_c).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str),
                "z_pca": z_pca,
                "ctx": ctx_np,
                "y_raw": yr_np,
                "y_multi": y_multi.astype(np.float32),
                "rel_r10": make_relevance_labels(r10, is_liq),
                "rel_multi": make_relevance_labels(y_multi, is_liq),
                "is_liq": is_liq.astype(np.float32),
            })
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g36_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        tr_g_paths, tr_r_paths = {}, {}
        for s in SCHEMES:
            gp = str(TMP_DIR / f"gtr_{s}_r{rank}.npy")
            np.save(gp, grams[s].cpu().numpy())
            tr_g_paths[s] = gp
            grams[s].zero_()
            tr_r_paths[s] = {}
            for t in TARGETS:
                rp = str(TMP_DIR / f"rtr_{s}_{t}_r{rank}.npy")
                np.save(rp, rhs[s][t].cpu().numpy())
                tr_r_paths[s][t] = rp
                rhs[s][t].zero_()

        val_sel_recs = []
        for b in [b for i, b in enumerate(val_batches) if i % n_gpus == rank]:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            path_d = extract_10d_path_targets(sym_val, s_idx, t_idx)
            ctx_np, st = compute_quad_context_14d(sym_val, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(str(d_str), len(s_idx), st["b20"])
            is_liq = st["is_liq65"]

            x32 = fwd_date_batched(ext, sym_val, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            yr_np = standardize_1d(rank_pct_centered(r10)).astype(np.float32)
            y5_np = standardize_1d(rank_pct_centered(path_d["r5"])).astype(np.float32)
            y_map = {
                "raw": torch.as_tensor(yr_np, device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
                "ret5": torch.as_tensor(y5_np, device=device, dtype=torch.float64),
            }
            xx = x64.T @ x64
            xys = {t: x64.T @ y_map[t] for t in TARGETS}
            for s in ("std", "regime"):
                w = w_map[s]
                grams[s].add_(xx, alpha=w)
                for t in TARGETS:
                    rhs[s][t].add_(xys[t], alpha=w)
            rw = torch.as_tensor(np.where(is_liq, 1.0, 0.25), device=device, dtype=torch.float64) * w_map["regime"]
            x64_w = x64 * rw.sqrt().unsqueeze(1)
            grams["liq_regime"].addmm_(x64_w.T, x64_w)
            for t in TARGETS:
                rhs["liq_regime"][t].addmv_(x64.T, y_map[t] * rw)

            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                z_pca = project_pca_4(x32, u_m, u_b, u_p, u_c).half().cpu().numpy()
                xf_path = str(TMP_DIR / f"vsel_{d_str}_r{rank}.npy")
                np.save(xf_path, x32.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "x_full_path": xf_path,
                    "z_pca": z_pca,
                    "ctx": ctx_np,
                    "r10": r10.astype(np.float64),
                    "r5": path_d["r5"].astype(np.float64),
                    "rel_r10": make_relevance_labels(r10, is_liq),
                    "rel_multi": make_relevance_labels(y_multi, is_liq),
                    "is_liq": is_liq,
                })

        va_g_paths, va_r_paths = {}, {}
        for s in SCHEMES:
            gp = str(TMP_DIR / f"gva_{s}_r{rank}.npy")
            np.save(gp, grams[s].cpu().numpy())
            va_g_paths[s] = gp
            va_r_paths[s] = {}
            for t in TARGETS:
                rp = str(TMP_DIR / f"rva_{s}_{t}_r{rank}.npy")
                np.save(rp, rhs[s][t].cpu().numpy())
                va_r_paths[s][t] = rp

        out = str(TMP_DIR / f"main_meta_r{rank}.pt")
        torch.save({
            "rank": rank, "n_rows": n_rows,
            "tr_g_paths": tr_g_paths, "tr_r_paths": tr_r_paths,
            "va_g_paths": va_g_paths, "va_r_paths": va_r_paths,
            "tr_recs": tr_recs, "val_sel_recs": val_sel_recs,
        }, out)
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _project_oof_worker(args: tuple) -> str:
    rank, n_gpus, paths, fold_heads_opp, sym_tr, tr_batches = args
    device, ext = load_quad_models(rank, paths)
    try:
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in fold_heads_opp.items()}
        out_scores: dict[str, np.ndarray] = {}
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x32 = fwd_date_batched(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            cols = []
            for k in ANCHOR_KEYS:
                sl, wt = heads_t[k]
                s = x32[:, sl] @ wt
                cols.append((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6))
            out_scores[str(d_str)] = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
            if (bi + 1) % 100 == 0 or bi + 1 == len(my_tr):
                phase("g36_oof_progress", rank=rank, done=bi + 1, total=len(my_tr))
        out = str(TMP_DIR / f"oof_r{rank}.pt")
        torch.save(out_scores, out)
        return out
    finally:
        ext.close()


def _val_topk_score(val_days: list, preds_list: list[np.ndarray]) -> float:
    """Blended Spear-IC + liquid top-20 excess z-score for epoch selection."""
    dates, preds, rets, top20_zs = [], [], [], []
    for (_d, _xv, r10, is_liq), p in zip(val_days, preds_list):
        dates.append(np.full(len(r10), _d))
        preds.append(p)
        rets.append(r10)
        rz = standardize_1d(rank_pct_centered(r10))
        liq_idx = np.flatnonzero(is_liq)
        if len(liq_idx) >= 20:
            top20 = liq_idx[np.argsort(-p[liq_idx])[:20]]
            top20_zs.append(float(np.mean(rz[top20])))
    ic = daily_spearman_ic(np.concatenate(dates), np.concatenate(preds), np.concatenate(rets))
    return float(0.65 * ic + 0.35 * (float(np.mean(top20_zs)) if top20_zs else 0.0) * 0.15)


def _train_topk_nn_worker(args: tuple) -> dict:
    rank, jobs, train_pt, val_pt = args
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    train_days = torch.load(train_pt, map_location="cpu", weights_only=False)
    val_days = torch.load(val_pt, map_location="cpu", weights_only=False)

    results = {}
    for name, kind, epochs, lr, wd in jobs:
        best_states, val_scores = [], []
        for seed in (42, 102, 202):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = _build_nn(kind).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
            best_sc, best_sd = -1e9, None
            for _ep in range(epochs):
                m.train()
                for idx in np.random.permutation(len(train_days)):
                    x_np, yr_np, ym_np, is_liq_np = train_days[idx][:4]
                    xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                    yr = torch.as_tensor(yr_np, device=device, dtype=torch.float32)
                    ym = torch.as_tensor(ym_np, device=device, dtype=torch.float32)
                    lq = torch.as_tensor(is_liq_np, device=device, dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    p = m(xt)
                    if kind == "film_topk":
                        loss = 0.65 * focal_listnet_loss(p, yr, lq, tau=2.5) + 0.35 * cosine_ic_loss_vec(p, ym)
                    else:
                        loss = 0.50 * focal_listnet_loss(p, ym, lq, tau=2.5) + 0.50 * cosine_ic_loss_vec(p, ym)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()

                m.eval()
                with torch.inference_mode():
                    plist = [m(torch.as_tensor(d[1], device=device, dtype=torch.float32)).cpu().numpy() for d in val_days]
                    sc = _val_topk_score(val_days, plist)
                if sc > best_sc:
                    best_sc = sc
                    best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            assert best_sd is not None
            best_states.append(best_sd)
            val_scores.append(best_sc)
        results[name] = {"kind": kind, "states": best_states, "val_score": float(np.mean(val_scores))}
        phase("g36_nn_done", rank=rank, model_name=name, val_score=round(float(np.mean(val_scores)), 4))
    return results


def fit_lambdarank(
    x_tr: np.ndarray,
    rel_tr: np.ndarray,
    grp_tr: list[int],
    x_va: np.ndarray,
    rel_va: np.ndarray,
    grp_va: list[int],
    d_va: np.ndarray,
    r10_va: np.ndarray,
    seed: int,
):
    import lightgbm as lgb

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10, 20],
        lambdarank_truncation_level=30,
        label_gain=LABEL_GAIN,
        n_estimators=550,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=600,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=200.0,
        reg_alpha=10.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(
        x_tr,
        rel_tr,
        group=grp_tr,
        eval_set=[(x_va, rel_va)],
        eval_group=[grp_va],
        eval_at=[10, 20],
        callbacks=[lgb.early_stopping(45, verbose=False)],
    )
    p_va = ranker.predict(x_va)
    return ranker, float(daily_spearman_ic(d_va, p_va, r10_va))


def _fit_lambdarank_child(
    x_tr: np.ndarray,
    rel_tr: np.ndarray,
    grp_tr: list[int],
    x_va: np.ndarray,
    rel_va: np.ndarray,
    grp_va: list[int],
    d_va: np.ndarray,
    r10_va: np.ndarray,
    out_model_path: str,
    seed: int,
) -> dict:
    ranker, val_ic = fit_lambdarank(x_tr, rel_tr, grp_tr, x_va, rel_va, grp_va, d_va, r10_va, seed=seed)
    ranker.booster_.save_model(out_model_path)
    return {"val_ic": float(val_ic), "best_iteration": int(ranker.best_iteration_ or 0)}


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_dict, full_heads, nn_specs, r10_path, rmulti_path, lgbm_cols, sym_oos, oos_batches = args
    import lightgbm as lgb

    device, ext = load_quad_models(rank, paths)
    try:
        u_m = torch.as_tensor(u_dict["m"], device=device, dtype=torch.float32)
        u_b = torch.as_tensor(u_dict["b"], device=device, dtype=torch.float32)
        u_p = torch.as_tensor(u_dict["p"], device=device, dtype=torch.float32)
        u_c = torch.as_tensor(u_dict["c"], device=device, dtype=torch.float32)
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in full_heads.items()}

        nn_loaded: dict[str, list[nn.Module]] = {}
        for name, spec in nn_specs.items():
            mods = []
            for sd in spec["states"]:
                m = _build_nn(spec["kind"]).to(device)
                m.load_state_dict(sd)
                m.eval()
                mods.append(m)
            nn_loaded[name] = mods

        b_r10 = lgb.Booster(model_file=r10_path)
        b_multi = lgb.Booster(model_file=rmulti_path)

        frames = []
        for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            ctx_np, _ = compute_quad_context_14d(sym_oos, s_idx, t_idx, pct_b)
            x32 = fwd_date_batched(ext, sym_oos, s_idx, t_idx, sec_b, pct_b)

            raw_s: dict[str, np.ndarray] = {}
            for k, (sl, wt) in heads_t.items():
                s = x32[:, sl] @ wt
                raw_s[k] = ((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)

            z_pca = project_pca_4(x32, u_m, u_b, u_p, u_c).cpu().numpy().astype(np.float32)
            anchors = np.column_stack([raw_s[k] for k in ANCHOR_KEYS]).astype(np.float32)
            x_feat = np.concatenate([anchors, z_pca, ctx_np], axis=1).astype(np.float32)
            xt = torch.as_tensor(x_feat, device=device, dtype=torch.float32)

            lr_r10 = standardize_1d(b_r10.predict(x_feat[:, lgbm_cols], num_threads=1)).astype(np.float32)
            lr_multi = standardize_1d(b_multi.predict(x_feat[:, lgbm_cols], num_threads=1)).astype(np.float32)
            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g36_quad_liq_ridge": raw_s["quad_liq_multi"],
                "p_g36_quad_lambdarank_r10": lr_r10,
                "p_g36_quad_lambdarank_multi": lr_multi,
            })
            for name, mods in nn_loaded.items():
                df_d[f"p_{name}"] = standardize_1d(np.mean([m(xt).cpu().numpy() for m in mods], axis=0)).astype(np.float32)
            df_d["p_g36_hybrid_lambdarank_lin"] = (0.5 * lr_r10 + 0.5 * raw_s["quad_regime"]).astype(np.float32)
            df_d["p_g36_hybrid_topk_all"] = (
                (lr_r10 + df_d["p_g36_quad_focal_listnet"].to_numpy() + df_d["p_g36_quad_xstock_topk"].to_numpy()) / 3.0
            ).astype(np.float32)
            frames.append(df_d)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = str(TMP_DIR / f"oos_r{rank}.csv.gz")
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
        return out_csv
    finally:
        ext.close()


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g36_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    paths = make_c2_paths_dict(INPUT)

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G36, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G36]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    step = max(1, len(tr_batches) // 40)
    pca_pts = run_gpu_workers_failfast(
        pca_quad_worker, [(r, n_gpus, paths, sym_tr, tr_batches[::step][:40], "g36") for r in range(n_gpus)], tag="g36_pca"
    )
    u_dict = top_evecs_from_pts(pca_pts, K_PCA)

    main_pts = run_gpu_workers_failfast(
        _main_extract_worker,
        [(r, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        tag="g36_main",
    )
    fold_meta = [torch.load(p, map_location="cpu", weights_only=False) for p in main_pts]
    for p in main_pts:
        Path(p).unlink(missing_ok=True)
    fold_meta.sort(key=lambda d: d["rank"])

    n_tr_rows = sum(d["n_rows"] for d in fold_meta)
    tr_recs, val_sel_recs = [], []
    for d in fold_meta:
        tr_recs.extend(d["tr_recs"])
        val_sel_recs.extend(d["val_sel_recs"])
        d["tr_recs"], d["val_sel_recs"] = [], []
    tr_recs.sort(key=lambda r: r["date"])
    val_sel_recs.sort(key=lambda r: r["date"])

    x_sel = np.concatenate([np.load(r["x_full_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["r10"]), r["date"]) for r in val_sel_recs])
    sel_targets = {
        "raw": np.concatenate([r["r10"] for r in val_sel_recs]),
        "multi": np.concatenate([r["r10"] for r in val_sel_recs]),
        "ret5": np.concatenate([r["r5"] for r in val_sel_recs]),
    }

    ridge_specs = [
        ("mini_regime", SL_MINI, "regime", "multi", ALPHAS_SINGLE),
        ("base_raw", SL_BASE, "std", "raw", ALPHAS_SINGLE),
        ("pts_multi", SL_PTS, "std", "multi", ALPHAS_SINGLE),
        ("c2_multi", SL_C2, "std", "multi", ALPHAS_SINGLE),
        ("quad_multi", SL_ALL, "std", "multi", ALPHAS_QUAD),
        ("quad_regime", SL_ALL, "regime", "multi", ALPHAS_QUAD),
        ("quad_ret5", SL_ALL, "std", "ret5", ALPHAS_QUAD),
        ("quad_liq_multi", SL_ALL, "liq_regime", "multi", ALPHAS_QUAD),
    ]
    selection, full_heads, tr_heads, fold_opp_heads = solve_ridge_specs_from_disk(
        fold_meta, SCHEMES, TARGETS, ridge_specs, ANCHOR_KEYS, x_sel, d_sel, sel_targets
    )
    del fold_meta, x_sel, d_sel, sel_targets
    gc.collect()

    oof_pts = run_gpu_workers_failfast(
        _project_oof_worker, [(r, n_gpus, paths, fold_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)], tag="g36_oof"
    )
    oof_map: dict[str, np.ndarray] = {}
    for p in oof_pts:
        oof_map.update(torch.load(p, map_location="cpu", weights_only=False))
        Path(p).unlink(missing_ok=True)
    del sym_tr, tr_batches, sym_val, val_batches
    gc.collect()

    train_days = []
    grp_tr, rel_r10_tr, rel_multi_tr = [], [], []
    for rec in tr_recs:
        xf = np.concatenate([oof_map[rec["date"]], rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        train_days.append((xf, rec["y_raw"], rec["y_multi"], rec["is_liq"]))
        grp_tr.append(len(xf))
        rel_r10_tr.append(rec["rel_r10"])
        rel_multi_tr.append(rec["rel_multi"])
    tr_recs.clear()
    oof_map.clear()

    val_days = []
    grp_va, rel_r10_va, rel_multi_va = [], [], []
    for rec in val_sel_recs:
        xf_full = np.load(rec["x_full_path"]).astype(np.float32)
        Path(rec["x_full_path"]).unlink(missing_ok=True)
        anc = np.column_stack([standardize_1d(xf_full[:, tr_heads[k][0]] @ tr_heads[k][1]).astype(np.float32) for k in ANCHOR_KEYS])
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        val_days.append((rec["date"], xf, rec["r10"], rec["is_liq"]))
        grp_va.append(len(xf))
        rel_r10_va.append(rec["rel_r10"])
        rel_multi_va.append(rec["rel_multi"])
    val_sel_recs.clear()

    train_pt, val_pt = str(TMP_DIR / "train_days.pt"), str(TMP_DIR / "val_days.pt")
    torch.save(train_days, train_pt)
    torch.save(val_days, val_pt)

    jobs_g0 = [("g36_quad_focal_listnet", "film_topk", 18, 1.5e-3, 2e-2)]
    jobs_g1 = [("g36_quad_xstock_topk", "xstock_topk", 18, 1.2e-3, 2e-2)]
    nn_jobs = [(0, jobs_g0, train_pt, val_pt), (1 % n_gpus, jobs_g1, train_pt, val_pt)] if n_gpus >= 2 else [(0, jobs_g0 + jobs_g1, train_pt, val_pt)]
    nn_parts = run_gpu_workers_failfast(_train_topk_nn_worker, nn_jobs, tag="g36_nn")
    nn_specs = {}
    for part in nn_parts:
        for name, res in part.items():
            nn_specs[name] = {"kind": res["kind"], "states": res["states"]}
            selection[name] = {"val_score": res["val_score"]}
    Path(train_pt).unlink(missing_ok=True)
    Path(val_pt).unlink(missing_ok=True)

    lgbm_cols = np.concatenate([
        np.arange(0, N_ANCHORS),
        np.arange(N_ANCHORS, N_ANCHORS + 16),
        np.arange(N_ANCHORS + K_PCA, N_ANCHORS + K_PCA + 16),
        np.arange(N_ANCHORS + 2 * K_PCA, N_ANCHORS + 2 * K_PCA + 16),
        np.arange(N_ANCHORS + 3 * K_PCA, N_ANCHORS + 3 * K_PCA + 16),
        np.arange(N_ANCHORS + PCA_DIM, IN_DIM),
    ])
    x_tr_lgb = np.concatenate([d[0][:, lgbm_cols] for d in train_days], axis=0)
    x_va_lgb = np.concatenate([d[1][:, lgbm_cols] for d in val_days], axis=0)
    d_va_lgb = np.concatenate([np.full(len(d[2]), d[0]) for d in val_days])
    r10_va_lgb = np.concatenate([d[2] for d in val_days])

    r10_path = str(OUTPUT / "g36_lambdarank_r10.txt")
    selection["g36_quad_lambdarank_r10"] = run_in_isolated_process(
        _fit_lambdarank_child,
        x_tr_lgb,
        np.concatenate(rel_r10_tr),
        grp_tr,
        x_va_lgb,
        np.concatenate(rel_r10_va),
        grp_va,
        d_va_lgb,
        r10_va_lgb,
        r10_path,
        20261127,
    )

    rmulti_path = str(OUTPUT / "g36_lambdarank_multi.txt")
    selection["g36_quad_lambdarank_multi"] = run_in_isolated_process(
        _fit_lambdarank_child,
        x_tr_lgb,
        np.concatenate(rel_multi_tr),
        grp_tr,
        x_va_lgb,
        np.concatenate(rel_multi_va),
        grp_va,
        d_va_lgb,
        r10_va_lgb,
        rmulti_path,
        20261128,
    )
    del train_days, val_days, x_tr_lgb, x_va_lgb
    gc.collect()

    np.savez(OUTPUT / "g36_heads.npz", u_mini=u_dict["m"], u_base=u_dict["b"], u_pts=u_dict["p"], u_c2=u_dict["c"], **{k: v[1] for k, v in full_heads.items()})
    torch.save({"nn_specs": nn_specs, "lgbm_cols": lgbm_cols}, OUTPUT / "g36_models.pt")
    (OUTPUT / "g36_selection_partial.json").write_text(
        json.dumps({"n_train_rows": n_tr_rows, "selection": selection}, indent=2, ensure_ascii=False)
    )

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [(r, n_gpus, paths, u_dict, full_heads, nn_specs, r10_path, rmulti_path, lgbm_cols, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g36_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g36_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g36_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for d in ("hf_kronos_mini", "hf_tok2k", "hf_kronos_base", "hf_kronos_small", "hf_tok_base"):
        shutil.rmtree(Path("/kaggle/working") / d, ignore_errors=True)
    phase("g36_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
