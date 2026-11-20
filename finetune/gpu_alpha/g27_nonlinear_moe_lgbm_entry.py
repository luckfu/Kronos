"""G27: Non-Linear Heads (Residual Cosine-IC MLP, Top-K ListNet MLP, Breadth-FiLM MLP, LightGBM, ICIR-Linear)
+ Causal Breadth-Gated 2-Expert MoE on Frozen PT-Small L=45 [L3,L5,L6_pre,L7,L7_m5] + Base L=45 [L5,L8,L10].

Strictly leak-free:
- PCA basis & OOF Ridge anchors fitted strictly on train_data.pkl (<= 2026-04-30) via even/odd day cross-fit.
- Model/epoch/alpha selection strictly on VAL_SEL (2026-05-18 .. 2026-07-02).
- OOS scoring strictly on 2026-07-17 .. 2026-10-23 (335,312 rows); 2026-10-24+ untouched.
"""
from __future__ import annotations

import json
import multiprocessing as mp
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
    solve_ridge_subblock,
    standardize_1d,
    zscore_cross_section_torch,
)
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec
from g4_base_kaggle_entry import _load_kronos_base
from g16_pretrained_small_entry import _load_pretrained_small
from g24_common import extract_r5_and_breadth
from short_probe_common import detect_gpu_count, norm_window

OUTPUT = Path(os.environ.get("G27_OUTPUT", "/kaggle/working/g27_out"))
INPUT = Path(os.environ.get("G27_INPUT", "/kaggle/input"))
D_SMALL = 512
D_BASE = 832
# PTS L=45: [L3, L5, L6_pre, L7_last, L7_m5] = 2560
# Base L=45: [L5, L8, L10_last] = 2496
PTS_DIM = D_SMALL * 5
BASE_DIM = D_BASE * 3
FEAT_DIM = PTS_DIM + BASE_DIM  # 5056
SL_PTS = slice(0, PTS_DIM)
SL_BASE = slice(PTS_DIM, FEAT_DIM)
SL_ALL = slice(0, FEAT_DIM)

K_PCA_PTS = 64
K_PCA_BASE = 64
PCA_DIM = K_PCA_PTS + K_PCA_BASE  # 128

TRAIN_START_G27 = "2021-01-01"
TRAIN_STRIDE_G27 = 2
ALPHAS_G27 = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
BREADTH_SPLIT = 0.35


class G27DualL45Extractor:
    """Single L=45 pass extracting PTS [L3,L5,L6_pre,L7,L7_m5] + Base [L5,L8,L10]."""

    def __init__(self, pts_model, base_model, tokenizer, device: torch.device):
        self.pts = pts_model
        self.base = base_model
        self.tok = tokenizer
        self.device = device
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            pts_model.transformer[3].register_forward_hook(self._make_hook("p3")),
            pts_model.transformer[5].register_forward_hook(self._make_hook("p5")),
            pts_model.transformer[6].register_forward_hook(self._make_hook("p6")),
            base_model.transformer[5].register_forward_hook(self._make_hook("b5")),
            base_model.transformer[8].register_forward_hook(self._make_hook("b8")),
        ]

    def _make_hook(self, key: str):
        def _hook(_mod, _inp, out):
            self._cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            tok45 = self.tok.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx_pts = self.pts.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            _, ctx_base = self.base.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
        p7_last = ctx_pts[:, -1, :].float()
        p7_m5 = ctx_pts[:, -5:, :].float().mean(dim=1)
        return torch.cat([
            self._cap["p3"], self._cap["p5"], self._cap["p6"], p7_last, p7_m5,
            self._cap["b5"], self._cap["b8"], ctx_base[:, -1, :].float(),
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, pts_dir: str, base_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    pts = _load_pretrained_small(pts_dir, device).eval()
    base = _load_kronos_base(base_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, G27DualL45Extractor(pts, base, tok, device)


def compute_row_context(sym_arr, s_idx: np.ndarray, t_idx: np.ndarray, pct_np: np.ndarray) -> tuple[np.ndarray, float]:
    """Return strictly causal [N, 4] context = [size_z, rev10_z, vol20_z, breadth_c] and scalar breadth."""
    n = len(s_idx)
    rev10 = np.empty(n, dtype=np.float64)
    vol20 = np.empty(n, dtype=np.float64)
    above = 0
    for i in range(n):
        c = np.nan_to_num(sym_arr[int(s_idx[i])][0][int(t_idx[i]) - 20 : int(t_idx[i]) + 1, 3].astype(np.float64), nan=1.0)
        c_now = c[-1]
        c_10 = c[-11]
        rev10[i] = -(c_now / max(c_10, 1e-8) - 1.0)
        rets = np.diff(np.log(np.maximum(c, 1e-8)))
        vol20[i] = float(np.std(rets))
        if c_now > float(c[1:].mean()):
            above += 1
    breadth = float(above) / float(max(n, 1))
    ctx = np.column_stack([
        standardize_1d(rank_pct_centered(pct_np.astype(np.float64))),
        standardize_1d(rank_pct_centered(rev10)),
        standardize_1d(rank_pct_centered(vol20)),
        np.full(n, breadth - 0.40, dtype=np.float64),
    ]).astype(np.float32)
    return ctx, breadth


def _fwd_date(ext, sym_arr, s_idx, t_idx, sec_np, pct_np, batch_size: int = 384) -> torch.Tensor:
    n = len(s_idx)
    chunks = []
    for off in range(0, n, batch_size):
        sl = slice(off, min(off + batch_size, n))
        raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl]))
    return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)


@torch.inference_mode()
def _pca_basis_worker(args: tuple) -> str:
    """Mini-pass on ~60 evenly spaced train dates to fit orthonormal PCA basis for PTS & Base."""
    rank, n_gpus, pts_dir, base_dir, tok_dir, sym_tr, sub_batches = args
    device, ext = _load_models(rank, pts_dir, base_dir, tok_dir)
    try:
        gp = torch.zeros((PTS_DIM, PTS_DIM), dtype=torch.float64, device=device)
        gb = torch.zeros((BASE_DIM, BASE_DIM), dtype=torch.float64, device=device)
        my_b = [b for i, b in enumerate(sub_batches) if i % n_gpus == rank]
        for b in my_b:
            _, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x64 = _fwd_date(ext, sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            xp, xb = x64[:, :PTS_DIM], x64[:, PTS_DIM:]
            gp.addmm_(xp.T, xp)
            gb.addmm_(xb.T, xb)
        out = f"/tmp/g27_pca_sub_r{rank}.pt"
        torch.save({"gp": gp.cpu().numpy(), "gb": gb.cpu().numpy()}, out)
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _main_extract_worker(args: tuple) -> str:
    """Main pass: accumulate fold Gram/RHS (all/bull/bear) AND cache compact PCA-128 + context per date."""
    rank, n_gpus, pts_dir, base_dir, tok_dir, u_pts_np, u_base_np, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, pts_dir, base_dir, tok_dir)
    try:
        u_pts = torch.as_tensor(u_pts_np, device=device, dtype=torch.float32)
        u_base = torch.as_tensor(u_base_np, device=device, dtype=torch.float32)

        grams = {k: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for k in ("all", "bull", "bear")}
        rhs = {
            k: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in ("raw", "multi")}
            for k in ("all", "bull", "bear")
        }
        tr_recs = []
        n_rows = 0

        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            ctx_np, breadth = compute_row_context(sym_tr, s_idx, t_idx, pct_np)
            x32 = _fwd_date(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            yr_np = standardize_1d(rank_pct_centered(r10)).astype(np.float32)
            yr = torch.as_tensor(yr_np, device=device, dtype=torch.float64)
            ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
            xx = x64.T @ x64
            xr = x64.T @ yr
            xm = x64.T @ ym
            grams["all"].add_(xx)
            rhs["all"]["raw"].add_(xr)
            rhs["all"]["multi"].add_(xm)
            rk = "bear" if breadth < BREADTH_SPLIT else "bull"
            grams[rk].add_(xx)
            rhs[rk]["raw"].add_(xr)
            rhs[rk]["multi"].add_(xm)
            n_rows += int(x32.shape[0])

            z_pca = torch.cat([x32[:, :PTS_DIM] @ u_pts, x32[:, PTS_DIM:] @ u_base], dim=1)
            z_pca = zscore_cross_section_torch(z_pca, clip_val=5.0).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str),
                "fold": rank,
                "breadth": float(breadth),
                "z_pca": z_pca,
                "ctx": ctx_np,
                "y_raw": yr_np,
                "y_multi": y_multi.astype(np.float32),
                "ret10": r10.astype(np.float32),
            })
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g27_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        val_recs = []
        my_val = [b for i, b in enumerate(val_batches) if i % n_gpus == rank]
        for b in my_val:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            ctx_np, breadth = compute_row_context(sym_val, s_idx, t_idx, pct_np)
            x32 = _fwd_date(ext, sym_val, s_idx, t_idx, sec_np, pct_np)
            z_pca = torch.cat([x32[:, :PTS_DIM] @ u_pts, x32[:, PTS_DIM:] @ u_base], dim=1)
            z_pca = zscore_cross_section_torch(z_pca, clip_val=5.0).half().cpu().numpy()
            val_recs.append({
                "date": str(d_str),
                "breadth": float(breadth),
                "x_full": x32.half().cpu().numpy(),
                "z_pca": z_pca,
                "ctx": ctx_np,
                "y_raw": standardize_1d(rank_pct_centered(r10)).astype(np.float32),
                "y_multi": y_multi.astype(np.float32),
                "ret10": r10.astype(np.float32),
            })

        out = f"/tmp/g27_main_r{rank}.pt"
        torch.save({
            "rank": rank,
            "n_rows": n_rows,
            "grams": {k: v.cpu().numpy() for k, v in grams.items()},
            "rhs": {k: {t: v.cpu().numpy() for t, v in d.items()} for k, d in rhs.items()},
            "tr_recs": tr_recs,
            "val_recs": val_recs,
        }, out)
        return out
    finally:
        ext.close()


def _project_ridge_oof_worker(args: tuple) -> str:
    """Compute exact 2-fold OOF linear anchor scores on train dates using opposite fold's Ridge weights."""
    rank, n_gpus, pts_dir, base_dir, tok_dir, fold_heads_opp, sym_tr, tr_batches = args
    device, ext = _load_models(rank, pts_dir, base_dir, tok_dir)
    try:
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in fold_heads_opp.items()}
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        out_scores: dict[str, np.ndarray] = {}
        keys = list(heads_t.keys())
        for b in my_tr:
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x32 = _fwd_date(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            cols = []
            for k in keys:
                sl, wt = heads_t[k]
                s = x32[:, sl] @ wt
                s = (s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)
                cols.append(s)
            out_scores[str(d_str)] = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
        out = f"/tmp/g27_oof_anchors_r{rank}.pt"
        torch.save({"keys": keys, "scores": out_scores}, out)
        return out
    finally:
        ext.close()


class ResidualAlphaMLP(nn.Module):
    """Ridge-anchored residual MLP with zero-init residual scale."""

    def __init__(self, in_dim: int, anchor_idx: int = 0, hidden: int = 96, dropout: float = 0.5):
        super().__init__()
        self.anchor_idx = anchor_idx
        self.anchor_w = nn.Parameter(torch.zeros(6))
        self.anchor_w.data[anchor_idx] = 1.0
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.anchor_w, dim=0)
        base = (x[:, :6] * w.unsqueeze(0)).sum(dim=1)
        return base + torch.tanh(self.res_scale) * self.net(x).squeeze(-1)


class BreadthFiLMMLP(nn.Module):
    """Market-breadth & volatility FiLM-modulated non-linear head anchored to Ridge experts."""

    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.5):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 6))
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.film = nn.Linear(2, hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Last 2 cols of x: [vol20_z (mean=0, std=1), breadth_c]; compute market state [breadth_c, abs_vol_mean]
        breadth_c = x[:, -1:]
        mkt_vol = x[:, -2:-1].abs().mean(dim=0, keepdim=True).expand_as(breadth_c)
        mkt = torch.cat([breadth_c, mkt_vol], dim=1)
        w = F.softmax(self.gate_net(mkt), dim=1)
        base = (x[:, :6] * w).sum(dim=1)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        gb = self.film(mkt)
        gamma, beta = gb.chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta
        return base + torch.tanh(self.res_scale) * self.fc2(self.drop(h)).squeeze(-1)


def topk_listnet_loss(pred: torch.Tensor, target_z: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    """Asymmetric top-tail ListNet KL loss focusing gradient on top-decile winners."""
    p_norm = (pred - pred.mean()) / pred.std(unbiased=False).clamp_min(1e-6)
    tgt_prob = F.softmax(tau * target_z, dim=0)
    log_pred = F.log_softmax(tau * p_norm, dim=0)
    return -(tgt_prob * log_pred).sum()


def train_daily_model(
    model_fn,
    train_days: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    val_sel_days: list[tuple[str, np.ndarray, np.ndarray]],
    loss_mode: str,
    device: torch.device,
    epochs: int = 18,
    lr: float = 1.5e-3,
    wd: float = 2e-2,
    seeds: tuple[int, ...] = (42, 102, 202),
) -> tuple[list[dict], float]:
    """Train multi-seed ensemble strictly selecting best epoch on VAL_SEL."""
    best_states = []
    val_ics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = model_fn().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        best_ic = -1e9
        best_sd = None
        for ep in range(epochs):
            m.train()
            order = np.random.permutation(len(train_days))
            if loss_mode == "icir":
                # Accumulate chunks of 16 days to optimize mean(IC) - 0.15 * std(IC)
                for off in range(0, len(order), 16):
                    opt.zero_grad()
                    ics = []
                    for idx in order[off : off + 16]:
                        x_np, y_np, _ = train_days[idx]
                        xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                        yt = torch.as_tensor(y_np, device=device, dtype=torch.float32)
                        p = m(xt).squeeze(-1)
                        p = p - p.mean()
                        t = yt - yt.mean()
                        ics.append(F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).squeeze(0))
                    ic_vec = torch.stack(ics)
                    loss = -ic_vec.mean() + 0.15 * ic_vec.std(unbiased=False)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
            else:
                for idx in order:
                    x_np, y_np, _ = train_days[idx]
                    xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                    yt = torch.as_tensor(y_np, device=device, dtype=torch.float32)
                    opt.zero_grad()
                    p = m(xt).squeeze(-1)
                    if loss_mode == "cosic":
                        loss = cosine_ic_loss_vec(p, yt)
                    elif loss_mode == "topk":
                        loss = 0.5 * cosine_ic_loss_vec(p, yt) + 0.5 * topk_listnet_loss(p, yt, tau=2.0)
                    else:
                        loss = cosine_ic_loss_vec(p, yt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()

            m.eval()
            with torch.inference_mode():
                preds, dates, rets = [], [], []
                for d_str, xv, rv in val_sel_days:
                    pv = m(torch.as_tensor(xv, device=device, dtype=torch.float32)).squeeze(-1).cpu().numpy()
                    preds.append(pv)
                    dates.append(np.full(len(rv), d_str))
                    rets.append(rv)
                ic = daily_spearman_ic(np.concatenate(dates), np.concatenate(preds), np.concatenate(rets))
            if ic > best_ic:
                best_ic = ic
                best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        assert best_sd is not None
        best_states.append(best_sd)
        val_ics.append(best_ic)
    return best_states, float(np.mean(val_ics))


def fit_lgbm_model(
    train_days: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    val_sel_days: list[tuple[str, np.ndarray, np.ndarray]],
    keep_cols: np.ndarray,
):
    import lightgbm as lgb

    x_tr = np.concatenate([d[0][:, keep_cols] for d in train_days], axis=0)
    y_tr = np.concatenate([d[1] for d in train_days], axis=0)
    x_va = np.concatenate([d[1][:, keep_cols] for d in val_sel_days], axis=0)
    y_va = np.concatenate([standardize_1d(rank_pct_centered(d[2])) for d in val_sel_days], axis=0)
    d_va = np.concatenate([np.full(len(d[2]), d[0]) for d in val_sel_days])
    r_va = np.concatenate([d[2] for d in val_sel_days])

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=800,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=200.0,
        reg_alpha=10.0,
        random_state=20261119,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    p_va = model.predict(x_va)
    val_ic = daily_spearman_ic(d_va, p_va, r_va)
    return model, float(val_ic)


def _select_ridge(gram_tr, rhs_tr, gram_all, rhs_all, sl, x_sel, d_sel, r_sel):
    xs = np.asarray(x_sel[:, sl], dtype=np.float64)
    best_a, best_ic = None, -1e9
    for a in ALPHAS_G27:
        w = solve_ridge_subblock(gram_tr, rhs_tr, sl, a)
        ic = daily_spearman_ic(d_sel, xs @ w, r_sel)
        if ic > best_ic:
            best_ic, best_a = ic, a
    assert best_a is not None
    w_tr = solve_ridge_subblock(gram_tr, rhs_tr, sl, best_a).astype(np.float32)
    w_all = solve_ridge_subblock(gram_all, rhs_all, sl, best_a).astype(np.float32)
    return best_a, float(best_ic), w_tr, w_all


def _select_breadth_gate(
    s_bull_sel: np.ndarray,
    s_bear_sel: np.ndarray,
    b_sel: np.ndarray,
    d_sel: np.ndarray,
    r_sel: np.ndarray,
) -> tuple[float, float, float]:
    best_ic, best_pair = -1e9, (10.0, 0.35)
    for slope in (5.0, 10.0, 20.0, 40.0):
        for center in (0.25, 0.30, 0.35, 0.40, 0.45):
            g = 1.0 / (1.0 + np.exp(-slope * (b_sel - center)))
            pred = g * s_bull_sel + (1.0 - g) * s_bear_sel
            ic = daily_spearman_ic(d_sel, pred, r_sel)
            if ic > best_ic:
                best_ic = ic
                best_pair = (slope, center)
    return best_pair[0], best_pair[1], float(best_ic)


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    (
        rank, n_gpus, pts_dir, base_dir, tok_dir,
        u_pts_np, u_base_np, full_heads, moe_gates,
        nn_specs, lgbm_path, lgbm_cols, sym_oos, oos_batches,
    ) = args
    import lightgbm as lgb

    device, ext = _load_models(rank, pts_dir, base_dir, tok_dir)
    try:
        u_pts = torch.as_tensor(u_pts_np, device=device, dtype=torch.float32)
        u_base = torch.as_tensor(u_base_np, device=device, dtype=torch.float32)
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in full_heads.items()}
        anchor_keys = ["pts_all", "pts_bull", "pts_bear", "base_all", "base_bull", "base_bear"]
        in_dim = 6 + PCA_DIM + 4

        nn_loaded: dict[str, list[nn.Module]] = {}
        for name, (kind, states) in nn_specs.items():
            mods = []
            for sd in states:
                if kind == "resid":
                    m = ResidualAlphaMLP(in_dim, anchor_idx=0).to(device)
                elif kind == "film":
                    m = BreadthFiLMMLP(in_dim).to(device)
                elif kind == "linear":
                    m = nn.Linear(in_dim, 1).to(device)
                else:
                    raise ValueError(kind)
                m.load_state_dict(sd)
                m.eval()
                mods.append(m)
            nn_loaded[name] = mods

        booster = lgb.Booster(model_file=lgbm_path)

        frames = []
        my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]
        for d_str, recs in my_oos:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            ctx_np, breadth = compute_row_context(sym_oos, s_idx, t_idx, pct_b)
            x32 = _fwd_date(ext, sym_oos, s_idx, t_idx, sec_b, pct_b)

            raw_s: dict[str, np.ndarray] = {}
            for k, (sl, wt) in heads_t.items():
                s = x32[:, sl] @ wt
                s_z = ((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)
                raw_s[k] = s_z

            z_pca = torch.cat([x32[:, :PTS_DIM] @ u_pts, x32[:, PTS_DIM:] @ u_base], dim=1)
            z_pca = zscore_cross_section_torch(z_pca, clip_val=5.0).cpu().numpy().astype(np.float32)
            anchors = np.column_stack([raw_s[k] for k in anchor_keys]).astype(np.float32)
            x_feat = np.concatenate([anchors, z_pca, ctx_np], axis=1).astype(np.float32)
            xt = torch.as_tensor(x_feat, device=device, dtype=torch.float32)

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g27_ridge_pts_all": raw_s["pts_all"],
                "p_g27_ridge_base_all": raw_s["base_all"],
                "p_g27_ridge_joint_all": raw_s["joint_all"],
            })
            for moe_name, (kbull, kbear, slope, center) in moe_gates.items():
                g = float(1.0 / (1.0 + np.exp(-slope * (breadth - center))))
                df_d[f"p_{moe_name}"] = (g * raw_s[kbull] + (1.0 - g) * raw_s[kbear]).astype(np.float32)

            for name, mods in nn_loaded.items():
                ps = [m(xt).squeeze(-1).cpu().numpy() for m in mods]
                df_d[f"p_{name}"] = np.mean(ps, axis=0).astype(np.float32)

            df_d["p_g27_lgbm_nonlin"] = booster.predict(x_feat[:, lgbm_cols]).astype(np.float32)
            frames.append(df_d)

        out_csv = f"/tmp/g27_oos_r{rank}.csv.gz"
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
        return out_csv
    finally:
        ext.close()


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g27_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer"),
    ]
    pts_dir, base_dir, tok_dir = hf_dirs
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G27, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G27]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    # Pass 0: PCA basis from 60 evenly spaced training dates (<= 2026-04-30)
    step = max(1, len(tr_batches) // 60)
    sub_batches = tr_batches[::step][:60]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pca_pts = pool.map(_pca_basis_worker, [(r, n_gpus, pts_dir, base_dir, tok_dir, sym_tr, sub_batches) for r in range(n_gpus)])
    gp = sum(torch.load(p, map_location="cpu", weights_only=False)["gp"] for p in pca_pts)
    gb = sum(torch.load(p, map_location="cpu", weights_only=False)["gb"] for p in pca_pts)
    for p in pca_pts:
        Path(p).unlink(missing_ok=True)
    _, vp = np.linalg.eigh(gp)
    _, vb = np.linalg.eigh(gb)
    u_pts = vp[:, -K_PCA_PTS:][:, ::-1].astype(np.float32).copy()
    u_base = vb[:, -K_PCA_BASE:][:, ::-1].astype(np.float32).copy()
    phase("g27_pca_ready", elapsed_sec=round(time.time() - t0, 1))

    # Pass 1: Main extraction (2-fold split by rank) + val extraction
    with ctx.Pool(n_gpus) as pool:
        main_pts = pool.map(
            _main_extract_worker,
            [(r, n_gpus, pts_dir, base_dir, tok_dir, u_pts, u_base, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        )

    fold_data = [torch.load(p, map_location="cpu", weights_only=False) for p in main_pts]
    for p in main_pts:
        Path(p).unlink(missing_ok=True)

    n_tr_rows = sum(d["n_rows"] for d in fold_data)
    grams_tr = {k: sum(d["grams"][k] for d in fold_data) for k in ("all", "bull", "bear")}
    rhs_tr = {
        k: {t: sum(d["rhs"][k][t] for d in fold_data) for t in ("raw", "multi")}
        for k in ("all", "bull", "bear")
    }
    tr_recs = []
    val_recs = []
    for d in fold_data:
        tr_recs.extend(d["tr_recs"])
        val_recs.extend(d["val_recs"])
    val_recs.sort(key=lambda r: r["date"])

    # Accumulate val into grams_all/rhs_all and gather VAL_SEL
    grams_all = {k: grams_tr[k].copy() for k in ("all", "bull", "bear")}
    rhs_all = {k: {t: v.copy() for t, v in d.items()} for k, d in rhs_tr.items()}
    sel_x, sel_d, sel_r, sel_b = [], [], [], []
    val_sel_meta = []
    for rec in val_recs:
        x64 = rec["x_full"].astype(np.float64)
        xx = x64.T @ x64
        xr = x64.T @ rec["y_raw"].astype(np.float64)
        xm = x64.T @ rec["y_multi"].astype(np.float64)
        grams_all["all"] += xx
        rhs_all["all"]["raw"] += xr
        rhs_all["all"]["multi"] += xm
        rk = "bear" if rec["breadth"] < BREADTH_SPLIT else "bull"
        grams_all[rk] += xx
        rhs_all[rk]["raw"] += xr
        rhs_all[rk]["multi"] += xm
        if VAL_SEL_START <= rec["date"] <= VAL_SEL_END:
            sel_x.append(rec["x_full"].astype(np.float32))
            sel_d.append(np.full(len(rec["ret10"]), rec["date"]))
            sel_r.append(rec["ret10"].astype(np.float64))
            sel_b.append(np.full(len(rec["ret10"]), rec["breadth"], dtype=np.float64))
            val_sel_meta.append(rec)
        else:
            rec["x_full"] = None
    x_sel = np.concatenate(sel_x, axis=0)
    d_sel = np.concatenate(sel_d)
    r_sel = np.concatenate(sel_r)
    b_sel = np.concatenate(sel_b)

    # Fit closed-form Ridge anchors & Bull/Bear experts
    selection: dict[str, dict] = {}
    full_heads: dict[str, tuple[slice, np.ndarray]] = {}
    tr_heads: dict[str, tuple[slice, np.ndarray]] = {}
    anchor_specs = [
        ("pts_all", SL_PTS, "all", "multi"),
        ("pts_bull", SL_PTS, "bull", "multi"),
        ("pts_bear", SL_PTS, "bear", "multi"),
        ("base_all", SL_BASE, "all", "raw"),
        ("base_bull", SL_BASE, "bull", "raw"),
        ("base_bear", SL_BASE, "bear", "raw"),
        ("joint_all", SL_ALL, "all", "multi"),
        ("joint_bull", SL_ALL, "bull", "multi"),
        ("joint_bear", SL_ALL, "bear", "multi"),
    ]
    sel_alphas: dict[str, float] = {}
    for name, sl, reg, tkey in anchor_specs:
        a, ic, w_tr, w_all = _select_ridge(
            grams_tr[reg], rhs_tr[reg][tkey], grams_all[reg], rhs_all[reg][tkey], sl, x_sel, d_sel, r_sel,
        )
        sel_alphas[name] = a
        tr_heads[name] = (sl, w_tr)
        full_heads[name] = (sl, w_all)
        selection[f"ridge_{name}"] = {"alpha": a, "val_ic": ic}

    # Select Causal Breadth-Gated 2-Expert MoE gates on VAL_SEL
    moe_gates: dict[str, tuple[str, str, float, float]] = {}
    for moe_name, kbull, kbear in (
        ("g27_moe_breadth_pts", "pts_bull", "pts_bear"),
        ("g27_moe_breadth_base", "base_bull", "base_bear"),
        ("g27_moe_breadth_joint", "joint_bull", "joint_bear"),
    ):
        sl_b, w_bull = tr_heads[kbull]
        _, w_bear = tr_heads[kbear]
        sb = x_sel[:, sl_b] @ w_bull
        sr = x_sel[:, sl_b] @ w_bear
        # Cross-sectionally standardize per date before gating
        sb_z = np.empty_like(sb)
        sr_z = np.empty_like(sr)
        for dv in np.unique(d_sel):
            m = d_sel == dv
            sb_z[m] = standardize_1d(sb[m])
            sr_z[m] = standardize_1d(sr[m])
        slope, center, ic = _select_breadth_gate(sb_z, sr_z, b_sel, d_sel, r_sel)
        moe_gates[moe_name] = (kbull, kbear, slope, center)
        selection[moe_name] = {"slope": slope, "center": center, "val_ic": ic}

    # Build 2-fold OOF linear anchors for train dates (using opposite fold's Gram/RHS!)
    anchor_keys = ["pts_all", "pts_bull", "pts_bear", "base_all", "base_bull", "base_bear"]
    fold_opp_heads = []
    for r in range(n_gpus):
        opp = (r + 1) % n_gpus if n_gpus > 1 else 0
        h_opp = {}
        for k in anchor_keys:
            _, sl, reg, tkey = next(sp for sp in anchor_specs if sp[0] == k)
            # Half-sample alpha scaling
            a_half = sel_alphas[k] * (0.5 if n_gpus > 1 else 1.0)
            w_opp = solve_ridge_subblock(fold_data[opp]["grams"][reg], fold_data[opp]["rhs"][reg][tkey], sl, a_half).astype(np.float32)
            h_opp[k] = (sl, w_opp)
        fold_opp_heads.append(h_opp)
    del fold_data, sel_x, x_sel

    # Compute exact 2-fold OOF anchor scores for train_recs
    with ctx.Pool(n_gpus) as pool:
        oof_pts = pool.map(
            _project_ridge_oof_worker,
            [(r, n_gpus, pts_dir, base_dir, tok_dir, fold_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)],
        )
    oof_map: dict[str, np.ndarray] = {}
    for p in oof_pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        oof_map.update(d["scores"])
        Path(p).unlink(missing_ok=True)
    del sym_tr, tr_batches, sym_val, val_batches
    phase("g27_oof_anchors_ready", elapsed_sec=round(time.time() - t0, 1))

    # Assemble daily [6 OOF anchors + 128 PCA + 4 causal ctx = 138d] arrays
    train_days = []
    for rec in tr_recs:
        anc = oof_map[rec["date"]]
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        train_days.append((xf, rec["y_multi"], rec["ret10"]))
    tr_recs.clear()

    val_sel_days = []
    for rec in val_sel_meta:
        xf_full = rec["x_full"].astype(np.float32)
        cols = []
        for k in anchor_keys:
            sl, wt = tr_heads[k]
            cols.append(standardize_1d(xf_full[:, sl] @ wt).astype(np.float32))
        anc = np.column_stack(cols)
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        val_sel_days.append((rec["date"], xf, rec["ret10"].astype(np.float64)))
        rec["x_full"] = None
    val_recs.clear()

    # Train 4 Non-Linear / ICIR Models on GPU 0 (strictly selecting on VAL_SEL)
    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_dim = 6 + PCA_DIM + 4
    nn_specs: dict[str, tuple[str, list[dict]]] = {}

    st_cos, ic_cos = train_daily_model(
        lambda: ResidualAlphaMLP(in_dim, anchor_idx=0), train_days, val_sel_days, "cosic", dev0,
    )
    nn_specs["g27_resid_mlp_cosic"] = ("resid", st_cos)
    selection["g27_resid_mlp_cosic"] = {"val_ic": ic_cos}

    st_topk, ic_topk = train_daily_model(
        lambda: ResidualAlphaMLP(in_dim, anchor_idx=0), train_days, val_sel_days, "topk", dev0,
    )
    nn_specs["g27_resid_mlp_topk"] = ("resid", st_topk)
    selection["g27_resid_mlp_topk"] = {"val_ic": ic_topk}

    st_film, ic_film = train_daily_model(
        lambda: BreadthFiLMMLP(in_dim), train_days, val_sel_days, "cosic", dev0,
    )
    nn_specs["g27_film_regime_mlp"] = ("film", st_film)
    selection["g27_film_regime_mlp"] = {"val_ic": ic_film}

    def _make_icir_lin():
        lin = nn.Linear(in_dim, 1)
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        lin.weight.data[0, 0] = 0.5
        lin.weight.data[0, 3] = 0.5
        return lin

    st_icir, ic_icir = train_daily_model(
        _make_icir_lin, train_days, val_sel_days, "icir", dev0, epochs=22, lr=2e-3, wd=5e-2,
    )
    nn_specs["g27_icir_linear"] = ("linear", st_icir)
    selection["g27_icir_linear"] = {"val_ic": ic_icir}

    # LightGBM on [6 anchors + top-32 PTS PCA + top-32 Base PCA + 4 causal ctx = 74 features]
    lgbm_cols = np.concatenate([
        np.arange(0, 6),
        np.arange(6, 6 + 32),
        np.arange(6 + K_PCA_PTS, 6 + K_PCA_PTS + 32),
        np.arange(6 + PCA_DIM, in_dim),
    ])
    lgbm_model, ic_lgbm = fit_lgbm_model(train_days, val_sel_days, lgbm_cols)
    lgbm_path = str(OUTPUT / "g27_lgbm.txt")
    lgbm_model.booster_.save_model(lgbm_path)
    selection["g27_lgbm_nonlin"] = {"val_ic": ic_lgbm, "best_iteration": int(lgbm_model.best_iteration_ or 0)}
    del train_days, val_sel_days
    phase("g27_models_trained", selection_summary={k: round(v["val_ic"], 4) for k, v in selection.items()})

    # Save weights & score OOS
    np.savez(OUTPUT / "g27_heads.npz", u_pts=u_pts, u_base=u_base, **{k: v[1] for k, v in full_heads.items()})
    torch.save({"nn_specs": nn_specs, "moe_gates": moe_gates, "lgbm_cols": lgbm_cols}, OUTPUT / "g27_models.pt")

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [(r, n_gpus, pts_dir, base_dir, tok_dir, u_pts, u_base, full_heads, moe_gates,
              nn_specs, lgbm_path, lgbm_cols, sym_oos, oos_batches) for r in range(n_gpus)],
        )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g27_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g27_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g27_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
