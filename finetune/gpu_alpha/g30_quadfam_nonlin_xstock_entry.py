"""G30: Quad-Family (Mini-2k + Base + PT-Small + C2_L45) Non-Linear + Cross-Stock Latent Factor Attention
+ 4-Signal Market-Regime FiLM (Cosine-IC & Upper-Tail Rank-Weighted) + Dual-Horizon LightGBM.

Strictly leak-free:
- PCA-192 (48x4) & 2-fold OOF Ridge anchors fitted strictly on train_data.pkl (<= 2026-04-30).
- Model/epoch/alpha selection strictly on VAL_SEL (2026-05-18 .. 2026-07-02).
- OOS scoring strictly on 2026-07-17 .. 2026-10-23 (335,312 rows); 2026-10-24+ untouched.
- Parent process NEVER initializes CUDA so all fork pools (PCA, extract, OOF, NN training, OOS scoring) run cleanly.
"""
from __future__ import annotations

import gc
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
from g24_common import compute_date_weights, extract_r5_and_breadth
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window, run_gpu_workers_failfast

OUTPUT = Path(os.environ.get("G30_OUTPUT", "/kaggle/working/g30_out"))
INPUT = Path(os.environ.get("G30_INPUT", "/kaggle/input"))

D_MINI = 256
D_BASE = 832
D_SMALL = 512

# Contiguous block layout (FEAT_DIM = 1280 + 2496 + 2560 + 2560 = 8896):
# Mini (1280):   [m45_l1, m45_l3_last, m45_l3_m5, m45_l3_diff5, m30_l3_last]
# Base (2496):   [b45_l5, b45_l8, b45_l10_last]
# PTS  (2560):   [p45_l3, p45_l5, p45_l6, p45_l7_last, p45_l7_m5]
# C2   (2560):   [c45_l3, c45_l5, c45_l6, c45_l7_last, c45_l7_m5]
MINI_DIM = D_MINI * 5
BASE_DIM = D_BASE * 3
PTS_DIM = D_SMALL * 5
C2_DIM = D_SMALL * 5
FEAT_DIM = MINI_DIM + BASE_DIM + PTS_DIM + C2_DIM  # 8896

SL_MINI = slice(0, MINI_DIM)
SL_BASE = slice(MINI_DIM, MINI_DIM + BASE_DIM)
SL_PTS = slice(MINI_DIM + BASE_DIM, MINI_DIM + BASE_DIM + PTS_DIM)
SL_C2 = slice(MINI_DIM + BASE_DIM + PTS_DIM, FEAT_DIM)
SL_ALL = slice(0, FEAT_DIM)

K_PCA = 48
PCA_DIM = K_PCA * 4  # 192
N_ANCHORS = 8
CTX_DIM = 12
IN_DIM = N_ANCHORS + PCA_DIM + CTX_DIM  # 212

TRAIN_START_G30 = "2021-01-01"
TRAIN_STRIDE_G30 = 2
SCHEMES = ("std", "decay", "regime")
ALPHAS_SINGLE = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_QUAD = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)

ANCHOR_KEYS = [
    "mini_multi",
    "mini_regime",
    "base_raw",
    "pts_multi",
    "c2_multi",
    "quad_multi",
    "quad_regime",
    "quad_ret5",
]


class QuadFamilyL45Extractor:
    """Single L=45 pass (plus tiny Mini L=30) across Mini-2k, Base, PT-Small, and C2."""

    def __init__(self, mini_m, tok2k, base_m, pts_m, c2_m, tok_base, device: torch.device):
        self.mini = mini_m
        self.tok2k = tok2k
        self.base = base_m
        self.pts = pts_m
        self.c2 = c2_m
        self.tok_base = tok_base
        self.device = device
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            mini_m.transformer[1].register_forward_hook(self._mk("m1")),
            base_m.transformer[5].register_forward_hook(self._mk("b5")),
            base_m.transformer[8].register_forward_hook(self._mk("b8")),
            pts_m.transformer[3].register_forward_hook(self._mk("p3")),
            pts_m.transformer[5].register_forward_hook(self._mk("p5")),
            pts_m.transformer[6].register_forward_hook(self._mk("p6")),
            c2_m.transformer[3].register_forward_hook(self._mk("c3")),
            c2_m.transformer[5].register_forward_hook(self._mk("c5")),
            c2_m.transformer[6].register_forward_hook(self._mk("c6")),
        ]

    def _mk(self, key: str):
        def _hook(_m, _i, out):
            self._cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            t30 = self.tok2k.encode(norm_window(raw_t[:, -30:]), half=True)
            _, m30 = self.mini.decode_s1(t30[0], t30[1], st_t[:, -30:])
            m30_last = m30[:, -1, :].float()

            t45 = self.tok2k.encode(norm_window(raw_t[:, -45:]), half=True)
            _, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])
            m1_45 = self._cap["m1"]

            tb45 = self.tok_base.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx_b = self.base.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            _, ctx_p = self.pts.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            _, ctx_c = self.c2.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)

        m45_last = m45[:, -1, :].float()
        m45_m5 = m45[:, -5:, :].float().mean(dim=1)
        m45_diff5 = m45_last - m45_m5
        p7_last = ctx_p[:, -1, :].float()
        p7_m5 = ctx_p[:, -5:, :].float().mean(dim=1)
        c7_last = ctx_c[:, -1, :].float()
        c7_m5 = ctx_c[:, -5:, :].float().mean(dim=1)

        return torch.cat([
            m1_45, m45_last, m45_m5, m45_diff5, m30_last,
            self._cap["b5"], self._cap["b8"], ctx_b[:, -1, :].float(),
            self._cap["p3"], self._cap["p5"], self._cap["p6"], p7_last, p7_m5,
            self._cap["c3"], self._cap["c5"], self._cap["c6"], c7_last, c7_m5,
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, paths: dict[str, str]):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    base = _load_kronos_base(paths["base"], device).eval()
    pts = _load_pretrained_small(paths["pts"], device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    return device, QuadFamilyL45Extractor(mini, tok2k, base, pts, c2, tok_base, device)


def compute_rich_context(
    sym_arr, s_idx: np.ndarray, t_idx: np.ndarray, pct_np: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Strictly causal [N, 12] context: 8 stock cross-sectional features + 4 daily market-regime features."""
    n = len(s_idx)
    rev5 = np.empty(n, dtype=np.float64)
    rev10 = np.empty(n, dtype=np.float64)
    rev20 = np.empty(n, dtype=np.float64)
    vol20 = np.empty(n, dtype=np.float64)
    vrat = np.empty(n, dtype=np.float64)
    lamt = np.empty(n, dtype=np.float64)
    pos20 = np.empty(n, dtype=np.float64)
    above20 = 0
    above5 = 0

    for i in range(n):
        vals = sym_arr[int(s_idx[i])][0]
        ti = int(t_idx[i])
        win = np.nan_to_num(vals[ti - 20 : ti + 1].astype(np.float64), nan=1.0)
        c = np.maximum(win[:, 3], 1e-8)
        h = win[1:, 1]
        lo = win[1:, 2]
        amt = np.maximum(win[1:, 5], 0.0)
        c_now = c[-1]
        rev5[i] = -(c_now / c[-6] - 1.0)
        rev10[i] = -(c_now / c[-11] - 1.0)
        rev20[i] = -(c_now / c[0] - 1.0)
        lrets = np.diff(np.log(c))
        v20 = float(np.std(lrets))
        v5 = float(np.std(lrets[-5:]))
        vol20[i] = v20
        vrat[i] = v5 / max(v20, 1e-4)
        lamt[i] = float(np.log1p(np.mean(amt)))
        hi20, lo20 = float(np.max(h)), float(np.min(lo))
        pos20[i] = (c_now - lo20) / max(hi20 - lo20, 1e-6)
        if c_now > float(c[1:].mean()):
            above20 += 1
        if c_now > float(c[-5:].mean()):
            above5 += 1

    b20 = float(above20) / float(max(n, 1))
    b5 = float(above5) / float(max(n, 1))
    mkt_disp = float(np.clip(np.std(rev10), 0.0, 0.5)) - 0.10
    mkt_vol = float(np.clip(np.mean(vol20), 0.0, 0.2)) - 0.025

    ctx = np.column_stack([
        standardize_1d(rank_pct_centered(pct_np.astype(np.float64))),
        standardize_1d(rank_pct_centered(rev5)),
        standardize_1d(rank_pct_centered(rev10)),
        standardize_1d(rank_pct_centered(rev20)),
        standardize_1d(rank_pct_centered(vol20)),
        standardize_1d(rank_pct_centered(vrat)),
        standardize_1d(rank_pct_centered(lamt)),
        standardize_1d(rank_pct_centered(pos20)),
        np.full(n, b20 - 0.40, dtype=np.float64),
        np.full(n, b5 - 0.50, dtype=np.float64),
        np.full(n, mkt_disp * 10.0, dtype=np.float64),
        np.full(n, mkt_vol * 20.0, dtype=np.float64),
    ]).astype(np.float32)
    return ctx, b20


def _fwd_date(ext, sym_arr, s_idx, t_idx, sec_np, pct_np, batch_size: int = 320) -> torch.Tensor:
    n = len(s_idx)
    chunks = []
    for off in range(0, n, batch_size):
        sl = slice(off, min(off + batch_size, n))
        raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl]))
    return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)


@torch.inference_mode()
def _pca_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_tr, sub_batches = args
    device, ext = _load_models(rank, paths)
    try:
        gm = torch.zeros((MINI_DIM, MINI_DIM), dtype=torch.float64, device=device)
        gb = torch.zeros((BASE_DIM, BASE_DIM), dtype=torch.float64, device=device)
        gp = torch.zeros((PTS_DIM, PTS_DIM), dtype=torch.float64, device=device)
        gc2 = torch.zeros((C2_DIM, C2_DIM), dtype=torch.float64, device=device)
        for b in [b for i, b in enumerate(sub_batches) if i % n_gpus == rank]:
            _, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x64 = _fwd_date(ext, sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            xm, xb, xp, xc = x64[:, SL_MINI], x64[:, SL_BASE], x64[:, SL_PTS], x64[:, SL_C2]
            gm.addmm_(xm.T, xm)
            gb.addmm_(xb.T, xb)
            gp.addmm_(xp.T, xp)
            gc2.addmm_(xc.T, xc)
        out = f"/tmp/g30_pca_r{rank}.pt"
        torch.save({
            "gm": gm.cpu().numpy(), "gb": gb.cpu().numpy(),
            "gp": gp.cpu().numpy(), "gc2": gc2.cpu().numpy(),
        }, out)
        return out
    finally:
        ext.close()


def _project_pca_4(x32: torch.Tensor, u_m: torch.Tensor, u_b: torch.Tensor, u_p: torch.Tensor, u_c: torch.Tensor) -> torch.Tensor:
    z = torch.cat([
        x32[:, SL_MINI] @ u_m,
        x32[:, SL_BASE] @ u_b,
        x32[:, SL_PTS] @ u_p,
        x32[:, SL_C2] @ u_c,
    ], dim=1)
    return zscore_cross_section_torch(z, clip_val=5.0)


@torch.inference_mode()
def _main_extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, paths)
    try:
        u_m = torch.as_tensor(u_dict["m"], device=device, dtype=torch.float32)
        u_b = torch.as_tensor(u_dict["b"], device=device, dtype=torch.float32)
        u_p = torch.as_tensor(u_dict["p"], device=device, dtype=torch.float32)
        u_c = torch.as_tensor(u_dict["c"], device=device, dtype=torch.float32)

        grams = {s: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for s in SCHEMES}
        rhs = {s: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in ("raw", "multi", "ret5")} for s in SCHEMES}

        tr_recs = []
        n_rows = 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            r5, breadth = extract_r5_and_breadth(sym_tr, s_idx, t_idx)
            ctx_np, _ = compute_rich_context(sym_tr, s_idx, t_idx, pct_np)
            w_decay, w_regime = compute_date_weights(str(d_str), len(s_idx), breadth)
            w_map = {"std": 1.0, "decay": w_decay, "regime": w_regime}

            x32 = _fwd_date(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            yr_np = standardize_1d(rank_pct_centered(r10)).astype(np.float32)
            y5_np = standardize_1d(rank_pct_centered(r5)).astype(np.float32)
            yr = torch.as_tensor(yr_np, device=device, dtype=torch.float64)
            ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
            y5 = torch.as_tensor(y5_np, device=device, dtype=torch.float64)

            xx = x64.T @ x64
            xr = x64.T @ yr
            xm = x64.T @ ym
            x5 = x64.T @ y5
            for s in SCHEMES:
                w = w_map[s]
                grams[s].add_(xx, alpha=w)
                rhs[s]["raw"].add_(xr, alpha=w)
                rhs[s]["multi"].add_(xm, alpha=w)
                rhs[s]["ret5"].add_(x5, alpha=w)
            n_rows += int(x32.shape[0])

            z_pca = _project_pca_4(x32, u_m, u_b, u_p, u_c).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str),
                "z_pca": z_pca,
                "ctx": ctx_np,
                "y_multi": y_multi.astype(np.float32),
                "y_ret5": y5_np,
                "ret10": r10.astype(np.float32),
            })
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g30_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        # Flush train Gram/RHS one matrix at a time to /tmp/*.npy and zero in-place for val (halves peak RAM!)
        tr_g_paths, tr_r_paths = {}, {}
        for s in SCHEMES:
            gp = f"/tmp/g30_gtr_{s}_r{rank}.npy"
            np.save(gp, grams[s].cpu().numpy())
            tr_g_paths[s] = gp
            grams[s].zero_()
            tr_r_paths[s] = {}
            for t in ("raw", "multi", "ret5"):
                rp = f"/tmp/g30_rtr_{s}_{t}_r{rank}.npy"
                np.save(rp, rhs[s][t].cpu().numpy())
                tr_r_paths[s][t] = rp
                rhs[s][t].zero_()

        val_sel_recs = []
        my_val = [b for i, b in enumerate(val_batches) if i % n_gpus == rank]
        for b in my_val:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            r5, breadth = extract_r5_and_breadth(sym_val, s_idx, t_idx)
            w_decay, w_regime = compute_date_weights(str(d_str), len(s_idx), breadth)
            w_map = {"std": 1.0, "decay": w_decay, "regime": w_regime}

            x32 = _fwd_date(ext, sym_val, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            yr_np = standardize_1d(rank_pct_centered(r10)).astype(np.float32)
            y5_np = standardize_1d(rank_pct_centered(r5)).astype(np.float32)
            yr = torch.as_tensor(yr_np, device=device, dtype=torch.float64)
            ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
            y5 = torch.as_tensor(y5_np, device=device, dtype=torch.float64)

            xx = x64.T @ x64
            xr = x64.T @ yr
            xm = x64.T @ ym
            x5 = x64.T @ y5
            for s in SCHEMES:
                w = w_map[s]
                grams[s].add_(xx, alpha=w)
                rhs[s]["raw"].add_(xr, alpha=w)
                rhs[s]["multi"].add_(xm, alpha=w)
                rhs[s]["ret5"].add_(x5, alpha=w)

            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                ctx_np, _ = compute_rich_context(sym_val, s_idx, t_idx, pct_np)
                z_pca = _project_pca_4(x32, u_m, u_b, u_p, u_c).half().cpu().numpy()
                xf_path = f"/tmp/g30_vsel_{d_str}_r{rank}.npy"
                np.save(xf_path, x32.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "x_full_path": xf_path,
                    "z_pca": z_pca,
                    "ctx": ctx_np,
                    "y_multi": y_multi.astype(np.float32),
                    "y_ret5": y5_np,
                    "ret10": r10.astype(np.float32),
                    "ret5": r5.astype(np.float32),
                })

        va_g_paths, va_r_paths = {}, {}
        for s in SCHEMES:
            gp = f"/tmp/g30_gva_{s}_r{rank}.npy"
            np.save(gp, grams[s].cpu().numpy())
            va_g_paths[s] = gp
            va_r_paths[s] = {}
            for t in ("raw", "multi", "ret5"):
                rp = f"/tmp/g30_rva_{s}_{t}_r{rank}.npy"
                np.save(rp, rhs[s][t].cpu().numpy())
                va_r_paths[s][t] = rp

        out = f"/tmp/g30_main_meta_r{rank}.pt"
        torch.save({
            "rank": rank,
            "n_rows": n_rows,
            "tr_g_paths": tr_g_paths,
            "tr_r_paths": tr_r_paths,
            "va_g_paths": va_g_paths,
            "va_r_paths": va_r_paths,
            "tr_recs": tr_recs,
            "val_sel_recs": val_sel_recs,
        }, out)
        phase("g30_extract_saved", rank=rank, n_rows=n_rows, n_vsel=len(val_sel_recs))
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _project_oof_worker(args: tuple) -> str:
    rank, n_gpus, paths, fold_heads_opp, sym_tr, tr_batches = args
    device, ext = _load_models(rank, paths)
    try:
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in fold_heads_opp.items()}
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        out_scores: dict[str, np.ndarray] = {}
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x32 = _fwd_date(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            cols = []
            for k in ANCHOR_KEYS:
                sl, wt = heads_t[k]
                s = x32[:, sl] @ wt
                s = (s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)
                cols.append(s)
            out_scores[str(d_str)] = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
            if (bi + 1) % 100 == 0 or bi + 1 == len(my_tr):
                phase("g30_oof_progress", rank=rank, done=bi + 1, total=len(my_tr))
        out = f"/tmp/g30_oof_r{rank}.pt"
        torch.save(out_scores, out)
        return out
    finally:
        ext.close()


class QuadResidualMLP(nn.Module):
    """8-anchor residual MLP with zero-init residual gate."""

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.anchor_w = nn.Parameter(torch.zeros(N_ANCHORS))
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
        base = (x[:, :N_ANCHORS] * w.unsqueeze(0)).sum(dim=1)
        return base + torch.tanh(self.res_scale) * self.net(x).squeeze(-1)


class QuadRegimeFiLMMLP(nn.Module):
    """4-signal market-regime FiLM MLP modulating both 8-anchor gates and hidden activations."""

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, N_ANCHORS))
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.film1 = nn.Linear(4, hidden * 2)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.film2 = nn.Linear(4, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden // 2, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mkt = x[:, -4:]
        w = F.softmax(self.gate_net(mkt), dim=1)
        base = (x[:, :N_ANCHORS] * w).sum(dim=1)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        g1, b1 = self.film1(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g1)) + 0.25 * b1
        h = F.silu(self.fc2(self.drop(h)))
        g2, b2 = self.film2(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g2)) + 0.25 * b2
        return base + torch.tanh(self.res_scale) * self.head(self.drop(h)).squeeze(-1)


class CrossStockFactorAttentionNet(nn.Module):
    """Permutation-equivariant O(N*K) Cross-Stock Latent Factor Attention over daily cross-section."""

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
        w = F.softmax(self.gate_net(mkt), dim=1)
        base = (x[:, :N_ANCHORS] * w).sum(dim=1)

        h = self.proj(self.drop(self.in_norm(x)))  # [N, D]
        scale = float(self.d_attn) ** -0.5
        # Step 1: K learned cross-sectional queries pool all N stocks into K daily latent factors [K, D]
        attn_pool = F.softmax((self.latent_q @ self.k_stock(h).T) * scale, dim=-1)  # [K, N]
        factors = attn_pool @ self.v_stock(h)  # [K, D]
        # Step 2: Every stock attends back to the K daily latent factors [N, D]
        attn_back = F.softmax((self.q_stock(h) @ self.k_factor(factors).T) * scale, dim=-1)  # [N, K]
        ctx = self.ctx_norm(attn_back @ self.v_factor(factors))
        feat = torch.cat([h, h - ctx], dim=1)
        return base + torch.tanh(self.res_scale) * self.out_mlp(feat).squeeze(-1)


def rank_weighted_cosic_loss(pred: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """Smooth bounded upper-tail weighted Pearson correlation loss (weights in [1.0, 1.6])."""
    w = 1.0 + 0.6 * torch.sigmoid(2.0 * target_z)
    w = w / w.sum()
    p_m = (w * pred).sum()
    t_m = (w * target_z).sum()
    dp = pred - p_m
    dt = target_z - t_m
    cov = (w * dp * dt).sum()
    vp = (w * dp * dp).sum().clamp_min(1e-8).sqrt()
    vt = (w * dt * dt).sum().clamp_min(1e-8).sqrt()
    return 1.0 - cov / (vp * vt)


def _build_nn_model(kind: str) -> nn.Module:
    if kind == "resid":
        return QuadResidualMLP(IN_DIM)
    if kind == "film":
        return QuadRegimeFiLMMLP(IN_DIM)
    if kind == "xstock":
        return CrossStockFactorAttentionNet(IN_DIM)
    raise ValueError(f"Unknown model kind: {kind}")


def _train_nn_worker(args: tuple) -> dict:
    """Train a subset of NN models on a dedicated GPU inside a forked worker (keeps parent CUDA-free!)."""
    rank, job_list, train_path, val_path = args
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    train_days = torch.load(train_path, map_location="cpu", weights_only=False)
    val_sel_days = torch.load(val_path, map_location="cpu", weights_only=False)

    results = {}
    for name, kind, loss_mode, epochs, lr, wd in job_list:
        best_states = []
        val_ics = []
        for seed in (42, 102, 202):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = _build_nn_model(kind).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
            best_ic = -1e9
            best_sd = None
            for _ep in range(epochs):
                m.train()
                order = np.random.permutation(len(train_days))
                for idx in order:
                    x_np, ym_np, y5_np = train_days[idx]
                    xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                    ym = torch.as_tensor(ym_np, device=device, dtype=torch.float32)
                    y5 = torch.as_tensor(y5_np, device=device, dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    p = m(xt)
                    if loss_mode == "multi_cosic":
                        loss = 0.7 * cosine_ic_loss_vec(p, ym) + 0.3 * cosine_ic_loss_vec(p, y5)
                    elif loss_mode == "rankw_cosic":
                        loss = 0.7 * rank_weighted_cosic_loss(p, ym) + 0.3 * cosine_ic_loss_vec(p, y5)
                    else:
                        loss = cosine_ic_loss_vec(p, ym)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()

                m.eval()
                with torch.inference_mode():
                    preds, dates, rets = [], [], []
                    for d_str, xv, rv in val_sel_days:
                        pv = m(torch.as_tensor(xv, device=device, dtype=torch.float32)).cpu().numpy()
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
        results[name] = {"kind": kind, "states": best_states, "val_ic": float(np.mean(val_ics))}
        phase("g30_nn_done", rank=rank, model_name=name, val_ic=round(float(np.mean(val_ics)), 4))
    return results


def fit_lgbm(train_days: list, val_sel_days: list, keep_cols: np.ndarray, target_idx: int, seed: int):
    import lightgbm as lgb

    x_tr = np.concatenate([d[0][:, keep_cols] for d in train_days], axis=0)
    y_tr = np.concatenate([d[target_idx] for d in train_days], axis=0)
    x_va = np.concatenate([d[1][:, keep_cols] for d in val_sel_days], axis=0)
    y_va_raw = np.concatenate([standardize_1d(rank_pct_centered(d[2])) for d in val_sel_days], axis=0)
    d_va = np.concatenate([np.full(len(d[2]), d[0]) for d in val_sel_days])
    r_va = np.concatenate([d[2] for d in val_sel_days])

    model = lgb.LGBMRegressor(
        n_estimators=550,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=800,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=250.0,
        reg_alpha=10.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va_raw)],
        callbacks=[lgb.early_stopping(45, verbose=False)],
    )
    p_va = model.predict(x_va)
    return model, float(daily_spearman_ic(d_va, p_va, r_va))


def _select_ridge(gram_tr, rhs_tr, gram_all, rhs_all, sl, alphas, x_sel, d_sel, r10_sel, r5_sel, is_ret5: bool):
    xs = np.asarray(x_sel[:, sl], dtype=np.float64)
    best_a, best_ic, grid = None, -1e9, []
    for a in alphas:
        w = solve_ridge_subblock(gram_tr, rhs_tr, sl, a)
        pred = xs @ w
        ic10 = daily_spearman_ic(d_sel, pred, r10_sel)
        ic5 = daily_spearman_ic(d_sel, pred, r5_sel)
        sc = ic5 if is_ret5 else ic10
        grid.append({"alpha": a, "val_ic_r10": ic10, "val_ic_r5": ic5})
        if sc > best_ic:
            best_ic, best_a = sc, a
    assert best_a is not None
    w_tr = solve_ridge_subblock(gram_tr, rhs_tr, sl, best_a).astype(np.float32)
    w_all = solve_ridge_subblock(gram_all, rhs_all, sl, best_a).astype(np.float32)
    return best_a, float(best_ic), grid, w_tr, w_all


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    (
        rank, n_gpus, paths, u_dict, full_heads, nn_specs,
        lgbm_multi_path, lgbm_ret5_path, lgbm_cols, sym_oos, oos_batches,
    ) = args
    import lightgbm as lgb

    device, ext = _load_models(rank, paths)
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
                m = _build_nn_model(spec["kind"]).to(device)
                m.load_state_dict(sd)
                m.eval()
                mods.append(m)
            nn_loaded[name] = mods

        b_multi = lgb.Booster(model_file=lgbm_multi_path)
        b_ret5 = lgb.Booster(model_file=lgbm_ret5_path)

        frames = []
        my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]
        for d_str, recs in my_oos:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            ctx_np, _ = compute_rich_context(sym_oos, s_idx, t_idx, pct_b)
            x32 = _fwd_date(ext, sym_oos, s_idx, t_idx, sec_b, pct_b)

            raw_s: dict[str, np.ndarray] = {}
            for k, (sl, wt) in heads_t.items():
                s = x32[:, sl] @ wt
                s_z = ((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)
                raw_s[k] = s_z

            z_pca = _project_pca_4(x32, u_m, u_b, u_p, u_c).cpu().numpy().astype(np.float32)
            anchors = np.column_stack([raw_s[k] for k in ANCHOR_KEYS]).astype(np.float32)
            x_feat = np.concatenate([anchors, z_pca, ctx_np], axis=1).astype(np.float32)
            xt = torch.as_tensor(x_feat, device=device, dtype=torch.float32)

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g30_quad_ridge_multi": raw_s["quad_multi"],
                "p_g30_quad_ridge_decay": raw_s["quad_decay"],
                "p_g30_quad_ridge_regime": raw_s["quad_regime"],
                "p_g30_quad_ridge_ret5": raw_s["quad_ret5"],
                "p_g30_c2l45_ridge_multi": raw_s["c2_multi"],
                "p_g30_mini_ridge_regime": raw_s["mini_regime"],
            })
            for name, mods in nn_loaded.items():
                ps = [m(xt).cpu().numpy() for m in mods]
                df_d[f"p_{name}"] = np.mean(ps, axis=0).astype(np.float32)

            df_d["p_g30_quad_lgbm_multi"] = b_multi.predict(x_feat[:, lgbm_cols]).astype(np.float32)
            df_d["p_g30_quad_lgbm_ret5"] = b_ret5.predict(x_feat[:, lgbm_cols]).astype(np.float32)
            frames.append(df_d)

        out_csv = f"/tmp/g30_oos_r{rank}.csv.gz"
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
    phase("g30_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    paths = {
        "mini": hf_dirs[0], "tok2k": hf_dirs[1], "base": hf_dirs[2],
        "pts": hf_dirs[3], "tok_base": hf_dirs[4], "c2": c2_dir,
    }
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G30, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G30]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    # Pass 0: PCA-48 per backbone on 50 evenly spaced train dates (<= 2026-04-30)
    step = max(1, len(tr_batches) // 50)
    sub_batches = tr_batches[::step][:50]
    pca_pts = run_gpu_workers_failfast(_pca_worker, [(r, n_gpus, paths, sym_tr, sub_batches) for r in range(n_gpus)], tag="g30_pca")
    gm = sum(torch.load(p, map_location="cpu", weights_only=False)["gm"] for p in pca_pts)
    gb = sum(torch.load(p, map_location="cpu", weights_only=False)["gb"] for p in pca_pts)
    gp = sum(torch.load(p, map_location="cpu", weights_only=False)["gp"] for p in pca_pts)
    gc2 = sum(torch.load(p, map_location="cpu", weights_only=False)["gc2"] for p in pca_pts)
    for p in pca_pts:
        Path(p).unlink(missing_ok=True)

    def _top_evec(mat: np.ndarray, k: int) -> np.ndarray:
        _, v = np.linalg.eigh(mat)
        return v[:, -k:][:, ::-1].astype(np.float32).copy()

    u_dict = {"m": _top_evec(gm, K_PCA), "b": _top_evec(gb, K_PCA), "p": _top_evec(gp, K_PCA), "c": _top_evec(gc2, K_PCA)}
    del gm, gb, gp, gc2
    phase("g30_pca_ready", elapsed_sec=round(time.time() - t0, 1))

    # Pass 1: Main extraction (2 folds) + val stats (fail-fast & raw .npy disk dump)
    main_pts = run_gpu_workers_failfast(
        _main_extract_worker,
        [(r, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        tag="g30_main",
    )
    fold_meta = [torch.load(p, map_location="cpu", weights_only=False) for p in main_pts]
    for p in main_pts:
        Path(p).unlink(missing_ok=True)
    fold_meta.sort(key=lambda d: d["rank"])

    n_tr_rows = sum(d["n_rows"] for d in fold_meta)
    tr_recs = []
    val_sel_recs = []
    for d in fold_meta:
        tr_recs.extend(d["tr_recs"])
        val_sel_recs.extend(d["val_sel_recs"])
        d["tr_recs"] = []
        d["val_sel_recs"] = []
    val_sel_recs.sort(key=lambda r: r["date"])

    x_sel = np.concatenate([np.load(r["x_full_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["ret10"]), r["date"]) for r in val_sel_recs])
    r10_sel = np.concatenate([r["ret10"].astype(np.float64) for r in val_sel_recs])
    r5_sel = np.concatenate([r["ret5"].astype(np.float64) for r in val_sel_recs])

    ridge_specs = [
        ("mini_multi", SL_MINI, "std", "multi", ALPHAS_SINGLE),
        ("mini_regime", SL_MINI, "regime", "multi", ALPHAS_SINGLE),
        ("base_raw", SL_BASE, "std", "raw", ALPHAS_SINGLE),
        ("pts_multi", SL_PTS, "std", "multi", ALPHAS_SINGLE),
        ("c2_multi", SL_C2, "std", "multi", ALPHAS_SINGLE),
        ("quad_multi", SL_ALL, "std", "multi", ALPHAS_QUAD),
        ("quad_decay", SL_ALL, "decay", "multi", ALPHAS_QUAD),
        ("quad_regime", SL_ALL, "regime", "multi", ALPHAS_QUAD),
        ("quad_ret5", SL_ALL, "std", "ret5", ALPHAS_QUAD),
    ]
    selection: dict[str, dict] = {}
    full_heads: dict[str, tuple[slice, np.ndarray]] = {}
    tr_heads: dict[str, tuple[slice, np.ndarray]] = {}
    sel_alphas: dict[str, float] = {}

    # Solve one weighting scheme at a time (keeps only 1x 633MB Gram in RAM!)
    for sch in SCHEMES:
        sch_specs = [sp for sp in ridge_specs if sp[2] == sch]
        if not sch_specs:
            continue
        g_tr_s = sum(np.load(d["tr_g_paths"][sch]) for d in fold_meta)
        g_all_s = g_tr_s + sum(np.load(d["va_g_paths"][sch]) for d in fold_meta)
        for name, sl, _s, tkey, alphas in sch_specs:
            r_tr_s = sum(np.load(d["tr_r_paths"][sch][tkey]) for d in fold_meta)
            r_all_s = r_tr_s + sum(np.load(d["va_r_paths"][sch][tkey]) for d in fold_meta)
            a, ic, grid, w_tr, w_all = _select_ridge(
                g_tr_s, r_tr_s, g_all_s, r_all_s,
                sl, alphas, x_sel, d_sel, r10_sel, r5_sel, is_ret5=(tkey == "ret5"),
            )
            sel_alphas[name] = a
            tr_heads[name] = (sl, w_tr)
            full_heads[name] = (sl, w_all)
            selection[f"ridge_{name}"] = {"alpha": a, "val_ic": ic, "grid": grid}
        del g_tr_s, g_all_s
        for d in fold_meta:
            Path(d["va_g_paths"][sch]).unlink(missing_ok=True)
            for t in ("raw", "multi", "ret5"):
                Path(d["va_r_paths"][sch][t]).unlink(missing_ok=True)
        gc.collect()

    # Prepare opposite-fold OOF anchor heads (one fold & scheme at a time from disk!)
    fold_opp_heads = [{} for _ in range(n_gpus)]
    for sch in SCHEMES:
        sch_anchors = [k for k in ANCHOR_KEYS if next(sp for sp in ridge_specs if sp[0] == k)[2] == sch]
        if not sch_anchors:
            continue
        for opp in range(n_gpus):
            g_opp = np.load(fold_meta[opp]["tr_g_paths"][sch])
            for r in range(n_gpus):
                if ((r + 1) % n_gpus if n_gpus > 1 else 0) != opp:
                    continue
                for k in sch_anchors:
                    _, sl, _s, tkey, _ = next(sp for sp in ridge_specs if sp[0] == k)
                    r_opp = np.load(fold_meta[opp]["tr_r_paths"][sch][tkey])
                    a_half = sel_alphas[k] * (0.5 if n_gpus > 1 else 1.0)
                    w_opp = solve_ridge_subblock(g_opp, r_opp, sl, a_half).astype(np.float32)
                    fold_opp_heads[r][k] = (sl, w_opp)
            del g_opp
        for d in fold_meta:
            Path(d["tr_g_paths"][sch]).unlink(missing_ok=True)
            for t in ("raw", "multi", "ret5"):
                Path(d["tr_r_paths"][sch][t]).unlink(missing_ok=True)
        gc.collect()

    del fold_meta, x_sel, d_sel, r10_sel, r5_sel
    gc.collect()
    phase("g30_ridge_ready", elapsed_sec=round(time.time() - t0, 1))

    # Pass 2: Strictly 2-fold OOF linear anchor projection on train dates
    oof_pts = run_gpu_workers_failfast(
        _project_oof_worker,
        [(r, n_gpus, paths, fold_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)],
        tag="g30_oof",
    )
    oof_map: dict[str, np.ndarray] = {}
    for p in oof_pts:
        oof_map.update(torch.load(p, map_location="cpu", weights_only=False))
        Path(p).unlink(missing_ok=True)
    del sym_tr, tr_batches, sym_val, val_batches
    gc.collect()
    phase("g30_oof_ready", elapsed_sec=round(time.time() - t0, 1))

    # Assemble compact daily [8 anchors + 192 PCA + 12 ctx = 212d] datasets
    train_days = []
    for rec in tr_recs:
        anc = oof_map[rec["date"]]
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        train_days.append((xf, rec["y_multi"], rec["y_ret5"]))
    tr_recs.clear()
    oof_map.clear()

    val_sel_days = []
    for rec in val_sel_recs:
        xf_full = np.load(rec["x_full_path"]).astype(np.float32)
        Path(rec["x_full_path"]).unlink(missing_ok=True)
        anc = np.column_stack([standardize_1d(xf_full[:, tr_heads[k][0]] @ tr_heads[k][1]).astype(np.float32) for k in ANCHOR_KEYS])
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        val_sel_days.append((rec["date"], xf, rec["ret10"].astype(np.float64)))
    val_sel_recs.clear()

    train_pt = "/tmp/g30_train_days.pt"
    val_pt = "/tmp/g30_val_days.pt"
    torch.save(train_days, train_pt)
    torch.save(val_sel_days, val_pt)

    # Train 4 NN architectures in parallel across GPU 0 and GPU 1 inside fail-fast workers (parent stays CUDA-free!)
    jobs_g0 = [
        ("g30_quad_resid_cosic", "resid", "multi_cosic", 18, 1.5e-3, 2e-2),
        ("g30_quad_film_cosic", "film", "multi_cosic", 18, 1.5e-3, 2e-2),
    ]
    jobs_g1 = [
        ("g30_quad_film_rankw", "film", "rankw_cosic", 18, 1.5e-3, 2e-2),
        ("g30_quad_xstock_attn", "xstock", "multi_cosic", 18, 1.2e-3, 2e-2),
    ]
    nn_jobs = [(0, jobs_g0, train_pt, val_pt), (1 % n_gpus, jobs_g1, train_pt, val_pt)] if n_gpus >= 2 else [(0, jobs_g0 + jobs_g1, train_pt, val_pt)]
    nn_parts = run_gpu_workers_failfast(_train_nn_worker, nn_jobs, tag="g30_nn")
    nn_specs: dict[str, dict] = {}
    for part in nn_parts:
        for name, res in part.items():
            nn_specs[name] = {"kind": res["kind"], "states": res["states"]}
            selection[name] = {"val_ic": res["val_ic"]}
    Path(train_pt).unlink(missing_ok=True)
    Path(val_pt).unlink(missing_ok=True)

    # Train 2 LightGBM specialists on [8 anchors + top-16 PCA x 4 + 12 ctx = 84d]
    lgbm_cols = np.concatenate([
        np.arange(0, N_ANCHORS),
        np.arange(N_ANCHORS, N_ANCHORS + 16),
        np.arange(N_ANCHORS + K_PCA, N_ANCHORS + K_PCA + 16),
        np.arange(N_ANCHORS + 2 * K_PCA, N_ANCHORS + 2 * K_PCA + 16),
        np.arange(N_ANCHORS + 3 * K_PCA, N_ANCHORS + 3 * K_PCA + 16),
        np.arange(N_ANCHORS + PCA_DIM, IN_DIM),
    ])
    lgb_m, ic_lm = fit_lgbm(train_days, val_sel_days, lgbm_cols, target_idx=1, seed=20261121)
    lgbm_multi_path = str(OUTPUT / "g30_lgbm_multi.txt")
    lgb_m.booster_.save_model(lgbm_multi_path)
    selection["g30_quad_lgbm_multi"] = {"val_ic": ic_lm, "best_iteration": int(lgb_m.best_iteration_ or 0)}

    lgb_5, ic_l5 = fit_lgbm(train_days, val_sel_days, lgbm_cols, target_idx=2, seed=20261122)
    lgbm_ret5_path = str(OUTPUT / "g30_lgbm_ret5.txt")
    lgb_5.booster_.save_model(lgbm_ret5_path)
    selection["g30_quad_lgbm_ret5"] = {"val_ic": ic_l5, "best_iteration": int(lgb_5.best_iteration_ or 0)}
    del train_days, val_sel_days
    gc.collect()

    phase("g30_models_trained", selection_summary={k: round(v["val_ic"], 4) for k, v in selection.items()})

    # Persist artifacts BEFORE OOS scoring
    np.savez(
        OUTPUT / "g30_heads.npz",
        u_mini=u_dict["m"], u_base=u_dict["b"], u_pts=u_dict["p"], u_c2=u_dict["c"],
        **{k: v[1] for k, v in full_heads.items()},
    )
    torch.save({"nn_specs": nn_specs, "lgbm_cols": lgbm_cols}, OUTPUT / "g30_models.pt")

    # Pass 3: Score 65d OOS cleanly on 2x T4 (parent has ZERO CUDA state!)
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [(r, n_gpus, paths, u_dict, full_heads, nn_specs, lgbm_multi_path, lgbm_ret5_path, lgbm_cols, sym_oos, oos_batches)
         for r in range(n_gpus)],
        tag="g30_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g30_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g30_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g30_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
