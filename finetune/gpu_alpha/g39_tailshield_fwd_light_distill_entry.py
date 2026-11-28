"""G39: Quad-Family 3-Target Crash-Shield & Continuation (CVaR10 + Fwd5-10 + Floor10) Cross-Architecture Distillation.

Strictly leak-free:
- Train <= 2026-04-30 (stride=2, ~2.92M rows): 2-fold OOF Super-Teacher combines Base(102M) + PT-Small(24.7M) +
  C2 + Mini-2k Ridge + 2-fold OOF Heavy Teacher LightGBMs across [cvar10, nowhipsaw, fwd5_10, floor10].
- Physical Zero-Teacher OOS Inference: Distills into C2-Only (zero new weights) and Ultra-Light C2 + 4.1M Mini
  multi-head students (`LightDualTailDistillMLP`, `LightCrossStockTailNet`, isolated Student LGBMs).
  `_score_oos_worker` NEVER loads Base or PT-Small!
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
    solve_ridge_subblock,
    standardize_1d,
    zscore_cross_section_torch,
)
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec
from g4_base_kaggle_entry import _load_kronos_base
from g16_pretrained_small_entry import _load_pretrained_small
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from g31_mini_mastery_c2pool_ft_entry import build_s1_token_prototypes
from g34_quad_dep_s1s2_logits_entry import _s1_s2_stats
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window, run_gpu_workers_failfast
from wave4_common import (
    CTX_DIM,
    compute_quad_context_14d,
    compute_wave4_date_weights,
    extract_10d_path_targets,
    fit_lgbm_regressor_isolated,
    run_in_isolated_process,
)

OUTPUT = Path(os.environ.get("G39_OUTPUT", "/kaggle/working/g39_out"))
INPUT = Path(os.environ.get("G39_INPUT", "/kaggle/input"))
TMP_DIR = Path("/kaggle/working/_tmp_g39")

D_MINI = 256
D_SMALL = 512
D_BASE = 832

# Compact Student feature layout (STUDENT_DIM = 5130 + 2058 = 7188, drops slow C2 L=120):
#   C2   (5130): [c20_last, c45_l3, c45_l5, c45_l6, c45_last, c45_m5, c45_diff5, dep_c,
#                 c90_last, c90_m5, st_c(10)]
#   Mini (2058): [m30_last, m30_m5, m45_l1, m45_last, m45_m5, m45_m10, m45_diff5, dep_m, st_m(10)]
C2_ONLY_DIM = D_SMALL * 10 + 10   # 5130
MINI_ONLY_DIM = D_MINI * 8 + 10   # 2058
STUDENT_DIM = C2_ONLY_DIM + MINI_ONLY_DIM  # 7188
SL_C2 = slice(0, C2_ONLY_DIM)
SL_C2_L45 = slice(D_SMALL * 1, D_SMALL * 8)  # c45_l3..dep_c (3584)
SL_MINI = slice(C2_ONLY_DIM, STUDENT_DIM)
SL_MINIC2 = slice(0, STUDENT_DIM)

# Heavy Teacher addon (NEVER loaded during OOS scoring, TEACHER_EXTRA_DIM = 5396):
#   Base (3328): [b45_l5, b45_l8, b45_last, dep_b]
#   PTS  (2048): [p45_l5, p45_l6, p45_last, dep_p]
#   Stats  (20): [st_b(10), st_p(10)]
TEACHER_EXTRA_DIM = D_BASE * 4 + D_SMALL * 4 + 20  # 5396

K_PCA_C2 = 64
K_PCA_MINI = 32
K_PCA_TEA = 64

C2_ANC_KEYS = ["c2_cvar", "c2_nowhipsaw", "c2_fwd5_10", "c2_crashentry"]
MC2_ANC_KEYS = C2_ANC_KEYS + ["mini_cvar", "mc2_cvar", "mc2_fwd5_10", "mc2_liqcvar"]
TEA_ANC_KEYS = ["tea_heavy_cvar", "tea_heavy_nowhipsaw", "tea_heavy_fwd5_10", "tea_heavy_floor10", "tea_crashentry_cvar"]

N_ANC_C2 = len(C2_ANC_KEYS)    # 4
N_ANC_MC2 = len(MC2_ANC_KEYS)  # 8
IN_DIM_C2 = N_ANC_C2 + K_PCA_C2 + CTX_DIM                 # 4 + 64 + 14 = 82
IN_DIM_MC2 = N_ANC_MC2 + K_PCA_C2 + K_PCA_MINI + CTX_DIM  # 8 + 64 + 32 + 14 = 118

TRAIN_START_G39 = "2021-01-01"
TRAIN_STRIDE_G39 = 2
ALPHAS_STU = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)
ALPHAS_TEA = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)


def tail_penalized_cosic_loss(pred: torch.Tensor, target_z: torch.Tensor, floor_z: torch.Tensor) -> torch.Tensor:
    """Asymmetric Cosine-IC focusing gradient weight on bottom crash tail and upper winners."""
    w = 1.0 + 0.6 * torch.sigmoid(-2.0 * floor_z) + 0.4 * torch.sigmoid(2.0 * target_z)
    w = w / w.sum()
    dp = pred - (w * pred).sum()
    dt = target_z - (w * target_z).sum()
    vp = (w * dp * dp).sum().clamp_min(1e-8).sqrt()
    vt = (w * dt * dt).sum().clamp_min(1e-8).sqrt()
    return 1.0 - (w * dp * dt).sum() / (vp * vt)


def build_g39_targets(path_d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute standardized path-risk and continuation targets strictly from [ti..ti+10]."""
    r5 = path_d["r5"]
    r10 = path_d["r10"]
    floor10 = path_d["floor10"]
    r_fwd5_10 = (1.0 + r10) / np.maximum(1.0 + r5, 1e-4) - 1.0
    cvar_raw = r10 - 0.75 * np.maximum(0.0, -floor10)
    nowhipsaw_raw = r10 - 0.60 * np.maximum(0.0, r5 - r10)
    return {
        "cvar": standardize_1d(rank_pct_centered(cvar_raw)).astype(np.float32),
        "nowhipsaw": standardize_1d(rank_pct_centered(nowhipsaw_raw)).astype(np.float32),
        "fwd5_10": standardize_1d(rank_pct_centered(r_fwd5_10)).astype(np.float32),
        "floor10": standardize_1d(rank_pct_centered(floor10)).astype(np.float32),
        "cvar_raw": cvar_raw.astype(np.float64),
        "fwd_raw": r_fwd5_10.astype(np.float64),
        "r10_raw": r10.astype(np.float64),
    }


class LightStudentCompactExtractor:
    """Physical zero-Base/PTS extractor using ONLY C2 (L=20,45,90) + 4.1M Mini-2k (L=30,45)."""

    def __init__(self, c2_m, tok_base, mini_m, tok2k, device: torch.device):
        self.c2 = c2_m
        self.tok_base = tok_base
        self.mini = mini_m
        self.tok2k = tok2k
        self.device = device
        self.proto_base = build_s1_token_prototypes(tok_base, device)
        self.proto_2k = build_s1_token_prototypes(tok2k, device)
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            c2_m.transformer[3].register_forward_hook(self._mk("c3")),
            c2_m.transformer[5].register_forward_hook(self._mk("c5")),
            c2_m.transformer[6].register_forward_hook(self._mk("c6")),
            mini_m.transformer[1].register_forward_hook(self._mk("m1")),
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

        x45_norm = norm_window(raw_t[:, -45:])
        cur_c = x45_norm[:, -1, 3]

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            tb20 = self.tok_base.encode(norm_window(raw_t[:, -20:]), half=True)
            _, c20 = self.c2.decode_s1(tb20[0], tb20[1], st_t[:, -20:], sector_id=sec_t, size_percentile=pct_t)
            c20_last = c20[:, -1, :].float()

            tb90 = self.tok_base.encode(norm_window(raw_t[:, -90:]), half=True)
            _, c90 = self.c2.decode_s1(tb90[0], tb90[1], st_t[:, -90:], sector_id=sec_t, size_percentile=pct_t)
            c90_f = c90.float()
            c90_last, c90_m5 = c90_f[:, -1, :], c90_f[:, -5:, :].mean(dim=1)

            # Run C2 L=45 LAST so hooks c3/c5/c6 belong strictly to L=45!
            tb45 = self.tok_base.encode(x45_norm, half=True)
            s1_c, c45 = self.c2.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_c, st_c = _s1_s2_stats(self.c2, s1_c, c45, cur_c, self.proto_base)
            c45_f = c45.float()
            c45_last = c45_f[:, -1, :]
            c45_m5 = c45_f[:, -5:, :].mean(dim=1)
            c45_diff5 = c45_last - c45_m5

            t30 = self.tok2k.encode(norm_window(raw_t[:, -30:]), half=True)
            _, m30 = self.mini.decode_s1(t30[0], t30[1], st_t[:, -30:])
            m30_f = m30.float()
            m30_last, m30_m5 = m30_f[:, -1, :], m30_f[:, -5:, :].mean(dim=1)

            # Run Mini L=45 LAST so hook m1 belongs strictly to L=45!
            t45 = self.tok2k.encode(x45_norm, half=True)
            s1_m, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])
            dep_m, st_m = _s1_s2_stats(self.mini, s1_m, m45, cur_c, self.proto_2k)
            m45_f = m45.float()
            m45_last = m45_f[:, -1, :]
            m45_m5 = m45_f[:, -5:, :].mean(dim=1)
            m45_m10 = m45_f[:, -10:, :].mean(dim=1)
            m45_diff5 = m45_last - m45_m5

        return torch.cat([
            c20_last,
            self._cap["c3"], self._cap["c5"], self._cap["c6"], c45_last, c45_m5, c45_diff5, dep_c,
            c90_last, c90_m5, st_c,
            m30_last, m30_m5,
            self._cap["m1"], m45_last, m45_m5, m45_m10, m45_diff5, dep_m, st_m,
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


class HeavyTeacherAddon:
    """Extracts Base (102M) + PT-Small (24.7M) L=45 teacher features (ONLY used <= 2026-07-02)."""

    def __init__(self, base_m, pts_m, tok_base, device: torch.device):
        self.base = base_m
        self.pts = pts_m
        self.tok_base = tok_base
        self.device = device
        self.proto_base = build_s1_token_prototypes(tok_base, device)
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            base_m.transformer[5].register_forward_hook(self._mk("b5")),
            base_m.transformer[8].register_forward_hook(self._mk("b8")),
            pts_m.transformer[5].register_forward_hook(self._mk("p5")),
            pts_m.transformer[6].register_forward_hook(self._mk("p6")),
        ]

    def _mk(self, key: str):
        def _hook(_m, _i, out):
            self._cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b[:, -45:], device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b[:, -45:], device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)
        x45_norm = norm_window(raw_t)
        cur_c = x45_norm[:, -1, 3]
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            tb45 = self.tok_base.encode(x45_norm, half=True)
            s1_b, c_b = self.base.decode_s1(tb45[0], tb45[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            dep_b, st_b_out = _s1_s2_stats(self.base, s1_b, c_b, cur_c, self.proto_base)
            s1_p, c_p = self.pts.decode_s1(tb45[0], tb45[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            dep_p, st_p = _s1_s2_stats(self.pts, s1_p, c_p, cur_c, self.proto_base)
        return torch.cat([
            self._cap["b5"], self._cap["b8"], c_b[:, -1, :].float(), dep_b,
            self._cap["p5"], self._cap["p6"], c_p[:, -1, :].float(), dep_p,
            st_b_out, st_p,
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_student_only(rank: int, paths: dict[str, str]):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    return device, LightStudentCompactExtractor(c2, tok_base, mini, tok2k, device)


def _load_teacher_and_student(rank: int, paths: dict[str, str]):
    device, stu = _load_student_only(rank, paths)
    base = _load_kronos_base(paths["base"], device).eval()
    pts = _load_pretrained_small(paths["pts"], device).eval()
    return device, stu, HeavyTeacherAddon(base, pts, stu.tok_base, device)


class LightDualTailDistillMLP(nn.Module):
    """Regime-modulated 3-head distilled student jointly predicting (mu_cvar, mu_fwd, fl_tail)."""

    def __init__(self, in_dim: int, n_anc: int, hidden: int = 128, dropout: float = 0.45):
        super().__init__()
        self.n_anc = n_anc
        self.gate_cvar = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, n_anc))
        self.gate_fwd = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, n_anc))
        self.gate_fl = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, n_anc))
        self.in_norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.film = nn.Linear(4, hidden * 2)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head_cvar = nn.Linear(hidden // 2, 1)
        self.head_fwd = nn.Linear(hidden // 2, 1)
        self.head_fl = nn.Linear(hidden // 2, 1)
        self.res_cvar = nn.Parameter(torch.tensor(0.0))
        self.res_fwd = nn.Parameter(torch.tensor(0.0))
        self.res_fl = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mkt = x[:, -4:]
        anc = x[:, : self.n_anc]
        base_cvar = (anc * F.softmax(self.gate_cvar(mkt), dim=1)).sum(dim=1)
        base_fwd = (anc * F.softmax(self.gate_fwd(mkt), dim=1)).sum(dim=1)
        base_fl = (anc * F.softmax(self.gate_fl(mkt), dim=1)).sum(dim=1)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        g, b = self.film(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g)) + 0.25 * b
        d = self.drop(F.silu(self.fc2(self.drop(h))))
        mu_cvar = base_cvar + torch.tanh(self.res_cvar) * self.head_cvar(d).squeeze(-1)
        mu_fwd = base_fwd + torch.tanh(self.res_fwd) * self.head_fwd(d).squeeze(-1)
        fl_tail = base_fl + torch.tanh(self.res_fl) * self.head_fl(d).squeeze(-1)
        return mu_cvar, mu_fwd, fl_tail


class LightCrossStockTailNet(nn.Module):
    """Permutation-equivariant induced set-attention student trained on CVaR + continuation blend."""

    def __init__(self, in_dim: int, n_anc: int, d_attn: int = 64, n_factors: int = 8, dropout: float = 0.45):
        super().__init__()
        self.n_anc = n_anc
        self.d_attn = d_attn
        self.gate = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, n_anc))
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
            nn.Dropout(dropout), nn.Linear(d_attn * 2, d_attn), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_attn, 1)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mkt = x[:, -4:]
        base = (x[:, : self.n_anc] * F.softmax(self.gate(mkt), dim=1)).sum(dim=1)
        h = self.proj(self.drop(self.in_norm(x)))
        scale = float(self.d_attn) ** -0.5
        factors = F.softmax((self.latent_q @ self.k_stock(h).T) * scale, dim=-1) @ self.v_stock(h)
        ctx = self.ctx_norm(F.softmax((self.q_stock(h) @ self.k_factor(factors).T) * scale, dim=-1) @ self.v_factor(factors))
        return base + torch.tanh(self.res_scale) * self.out_mlp(torch.cat([h, h - ctx], dim=1)).squeeze(-1)


def _build_stu_nn(kind: str, in_dim: int, n_anc: int) -> nn.Module:
    if kind == "dual_mlp":
        return LightDualTailDistillMLP(in_dim, n_anc)
    if kind == "xstock":
        return LightCrossStockTailNet(in_dim, n_anc)
    raise ValueError(kind)


def _fwd_ext(ext, sym_arr, s_idx, t_idx, sec_np, pct_np, bs: int = 256) -> torch.Tensor:
    chunks = []
    for off in range(0, len(s_idx), bs):
        sl = slice(off, min(off + bs, len(s_idx)))
        raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl]))
    return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)


@torch.inference_mode()
def _pca_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_tr, sub_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, stu, tea = _load_teacher_and_student(rank, paths)
    try:
        gc2 = torch.zeros((C2_ONLY_DIM, C2_ONLY_DIM), dtype=torch.float64, device=device)
        gmn = torch.zeros((MINI_ONLY_DIM, MINI_ONLY_DIM), dtype=torch.float64, device=device)
        gt = torch.zeros((TEACHER_EXTRA_DIM, TEACHER_EXTRA_DIM), dtype=torch.float64, device=device)
        for b in [b for i, b in enumerate(sub_batches) if i % n_gpus == rank]:
            _, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            xs = _fwd_ext(stu, sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            xt = _fwd_ext(tea, sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            gc2.addmm_(xs[:, SL_C2].T, xs[:, SL_C2])
            gmn.addmm_(xs[:, SL_MINI].T, xs[:, SL_MINI])
            gt.addmm_(xt.T, xt)
        out = str(TMP_DIR / f"pca_r{rank}.pt")
        torch.save({"gc2": gc2.cpu().numpy(), "gmn": gmn.cpu().numpy(), "gt": gt.cpu().numpy()}, out)
        return out
    finally:
        stu.close()
        tea.close()


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_c2, u_mn, u_tea, sym_tr, tr_batches, sym_val, val_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, stu, tea = _load_teacher_and_student(rank, paths)
    try:
        uc2_t = torch.as_tensor(u_c2, device=device, dtype=torch.float32)
        umn_t = torch.as_tensor(u_mn, device=device, dtype=torch.float32)
        utea_t = torch.as_tensor(u_tea, device=device, dtype=torch.float32)

        tea_targets = ("cvar", "nowhipsaw", "fwd5_10", "floor10")
        tea_schemes = ("regime", "crashentry")
        stu_targets = ("cvar", "nowhipsaw", "fwd5_10")
        stu_schemes = ("regime", "crashentry", "liqcvar")

        g_tea = {s: torch.zeros((TEACHER_EXTRA_DIM, TEACHER_EXTRA_DIM), dtype=torch.float64, device=device) for s in tea_schemes}
        r_tea = {s: {t: torch.zeros(TEACHER_EXTRA_DIM, dtype=torch.float64, device=device) for t in tea_targets} for s in tea_schemes}
        g_stu = {s: torch.zeros((STUDENT_DIM, STUDENT_DIM), dtype=torch.float64, device=device) for s in stu_schemes}
        r_stu = {s: {t: torch.zeros(STUDENT_DIM, dtype=torch.float64, device=device) for t in stu_targets} for s in stu_schemes}

        tr_recs = []
        n_rows = 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, _ym, _yo, _r10 = b
            path_d = extract_10d_path_targets(sym_tr, s_idx, t_idx)
            tgt = build_g39_targets(path_d)
            ctx_np, st = compute_quad_context_14d(sym_tr, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(
                str(d_str), len(s_idx), st["b20"], float(np.mean(path_d["r10"])), float(np.mean(path_d["r5"])),
            )
            w_liq = np.where(st["amt_pct"] < 0.35, 0.25, 1.0).astype(np.float64) * w_map["regime"]

            xs32 = _fwd_ext(stu, sym_tr, s_idx, t_idx, sec_np, pct_np)
            xt32 = _fwd_ext(tea, sym_tr, s_idx, t_idx, sec_np, pct_np)
            xs64, xt64 = xs32.to(torch.float64), xt32.to(torch.float64)

            y_tea_t = {t: torch.as_tensor(tgt[t], device=device, dtype=torch.float64) for t in tea_targets}
            xx_t = xt64.T @ xt64
            xty = {t: xt64.T @ y_tea_t[t] for t in tea_targets}
            for s in tea_schemes:
                w = w_map[s]
                g_tea[s].add_(xx_t, alpha=w)
                for t in tea_targets:
                    r_tea[s][t].add_(xty[t], alpha=w)

            xx_s = xs64.T @ xs64
            xsy = {t: xs64.T @ y_tea_t[t] for t in stu_targets}
            for s in ("regime", "crashentry"):
                w = w_map[s]
                g_stu[s].add_(xx_s, alpha=w)
                for t in stu_targets:
                    r_stu[s][t].add_(xsy[t], alpha=w)

            # Soft-liquidity weighted Gram & RHS on CVaR target (zero truncation distortion)
            wl_t = torch.as_tensor(w_liq, device=device, dtype=torch.float64)
            xs64_wl = xs64 * wl_t.unsqueeze(1)
            g_stu["liqcvar"].addmm_(xs64_wl.T, xs64)
            r_stu["liqcvar"]["cvar"].addmv_(xs64.T, wl_t * y_tea_t["cvar"])

            n_rows += int(xs32.shape[0])
            zc2 = zscore_cross_section_torch(xs32[:, SL_C2] @ uc2_t, clip_val=5.0).half().cpu().numpy()
            zmn = zscore_cross_section_torch(xs32[:, SL_MINI] @ umn_t, clip_val=5.0).half().cpu().numpy()
            ztea = zscore_cross_section_torch(xt32 @ utea_t, clip_val=5.0).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str),
                "zc2": zc2,
                "zmn": zmn,
                "ztea": ztea,
                "ctx": ctx_np,
                "y_cvar": tgt["cvar"],
                "y_nowhipsaw": tgt["nowhipsaw"],
                "y_fwd": tgt["fwd5_10"],
                "y_floor": tgt["floor10"],
                "w_crash": float(w_map["crashentry"]),
            })
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g39_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        p_gtea_tr, p_rtea_tr = {}, {}
        for s in tea_schemes:
            gp = str(TMP_DIR / f"gtea_tr_{s}_r{rank}.npy")
            np.save(gp, g_tea[s].cpu().numpy())
            p_gtea_tr[s] = gp
            g_tea[s].zero_()
            p_rtea_tr[s] = {}
            for t in tea_targets:
                rp = str(TMP_DIR / f"rtea_tr_{s}_{t}_r{rank}.npy")
                np.save(rp, r_tea[s][t].cpu().numpy())
                p_rtea_tr[s][t] = rp
                r_tea[s][t].zero_()

        p_gstu_tr, p_rstu_tr = {}, {}
        for s in stu_schemes:
            gp = str(TMP_DIR / f"gstu_tr_{s}_r{rank}.npy")
            np.save(gp, g_stu[s].cpu().numpy())
            p_gstu_tr[s] = gp
            g_stu[s].zero_()
            p_rstu_tr[s] = {}
            for t in stu_targets:
                rp = str(TMP_DIR / f"rstu_tr_{s}_{t}_r{rank}.npy")
                np.save(rp, r_stu[s][t].cpu().numpy())
                p_rstu_tr[s][t] = rp
                r_stu[s][t].zero_()

        val_sel_recs = []
        for b in [b for i, b in enumerate(val_batches) if i % n_gpus == rank]:
            d_str, s_idx, t_idx, sec_np, pct_np, _ym, _yo, _r10 = b
            path_d = extract_10d_path_targets(sym_val, s_idx, t_idx)
            tgt = build_g39_targets(path_d)
            ctx_np, st = compute_quad_context_14d(sym_val, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(
                str(d_str), len(s_idx), st["b20"], float(np.mean(path_d["r10"])), float(np.mean(path_d["r5"])),
            )
            w_liq = np.where(st["amt_pct"] < 0.35, 0.25, 1.0).astype(np.float64) * w_map["regime"]

            xs32 = _fwd_ext(stu, sym_val, s_idx, t_idx, sec_np, pct_np)
            xt32 = _fwd_ext(tea, sym_val, s_idx, t_idx, sec_np, pct_np)
            xs64, xt64 = xs32.to(torch.float64), xt32.to(torch.float64)

            y_tea_t = {t: torch.as_tensor(tgt[t], device=device, dtype=torch.float64) for t in tea_targets}
            xx_t = xt64.T @ xt64
            xty = {t: xt64.T @ y_tea_t[t] for t in tea_targets}
            for s in tea_schemes:
                w = w_map[s]
                g_tea[s].add_(xx_t, alpha=w)
                for t in tea_targets:
                    r_tea[s][t].add_(xty[t], alpha=w)

            xx_s = xs64.T @ xs64
            xsy = {t: xs64.T @ y_tea_t[t] for t in stu_targets}
            for s in ("regime", "crashentry"):
                w = w_map[s]
                g_stu[s].add_(xx_s, alpha=w)
                for t in stu_targets:
                    r_stu[s][t].add_(xsy[t], alpha=w)

            wl_t = torch.as_tensor(w_liq, device=device, dtype=torch.float64)
            g_stu["liqcvar"].addmm_((xs64 * wl_t.unsqueeze(1)).T, xs64)
            r_stu["liqcvar"]["cvar"].addmv_(xs64.T, wl_t * y_tea_t["cvar"])

            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                zc2 = zscore_cross_section_torch(xs32[:, SL_C2] @ uc2_t, clip_val=5.0).half().cpu().numpy()
                zmn = zscore_cross_section_torch(xs32[:, SL_MINI] @ umn_t, clip_val=5.0).half().cpu().numpy()
                ztea = zscore_cross_section_torch(xt32 @ utea_t, clip_val=5.0).half().cpu().numpy()
                ps = str(TMP_DIR / f"vsel_s_{d_str}_r{rank}.npy")
                pt = str(TMP_DIR / f"vsel_t_{d_str}_r{rank}.npy")
                np.save(ps, xs32.half().cpu().numpy())
                np.save(pt, xt32.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "xs_path": ps,
                    "xt_path": pt,
                    "zc2": zc2,
                    "zmn": zmn,
                    "ztea": ztea,
                    "ctx": ctx_np,
                    "cvar_raw": tgt["cvar_raw"],
                    "fwd_raw": tgt["fwd_raw"],
                    "floor_raw": path_d["floor10"].astype(np.float64),
                })

        p_gtea_va, p_rtea_va = {}, {}
        for s in tea_schemes:
            gp = str(TMP_DIR / f"gtea_va_{s}_r{rank}.npy")
            np.save(gp, g_tea[s].cpu().numpy())
            p_gtea_va[s] = gp
            p_rtea_va[s] = {}
            for t in tea_targets:
                rp = str(TMP_DIR / f"rtea_va_{s}_{t}_r{rank}.npy")
                np.save(rp, r_tea[s][t].cpu().numpy())
                p_rtea_va[s][t] = rp

        p_gstu_va, p_rstu_va = {}, {}
        for s in stu_schemes:
            gp = str(TMP_DIR / f"gstu_va_{s}_r{rank}.npy")
            np.save(gp, g_stu[s].cpu().numpy())
            p_gstu_va[s] = gp
            p_rstu_va[s] = {}
            for t in stu_targets:
                rp = str(TMP_DIR / f"rstu_va_{s}_{t}_r{rank}.npy")
                np.save(rp, r_stu[s][t].cpu().numpy())
                p_rstu_va[s][t] = rp

        out = str(TMP_DIR / f"meta_r{rank}.pt")
        torch.save({
            "rank": rank, "n_rows": n_rows,
            "p_gtea_tr": p_gtea_tr, "p_rtea_tr": p_rtea_tr, "p_gtea_va": p_gtea_va, "p_rtea_va": p_rtea_va,
            "p_gstu_tr": p_gstu_tr, "p_rstu_tr": p_rstu_tr, "p_gstu_va": p_gstu_va, "p_rstu_va": p_rstu_va,
            "tr_recs": tr_recs, "val_sel_recs": val_sel_recs,
        }, out)
        return out
    finally:
        stu.close()
        tea.close()


@torch.inference_mode()
def _project_oof_worker(args: tuple) -> str:
    rank, n_gpus, paths, stu_opp, tea_opp, sym_tr, tr_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, stu, tea = _load_teacher_and_student(rank, paths)
    try:
        stu_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in stu_opp.items()}
        tea_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in tea_opp.items()}
        out_map = {}
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            xs = _fwd_ext(stu, sym_tr, s_idx, t_idx, sec_np, pct_np)
            xt = _fwd_ext(tea, sym_tr, s_idx, t_idx, sec_np, pct_np)
            sd = {}
            for k, (sl, wt) in stu_t.items():
                v = xs[:, sl] @ wt
                sd[k] = ((v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)
            for k, (sl, wt) in tea_t.items():
                v = xt[:, sl] @ wt
                sd[k] = ((v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)
            out_map[str(d_str)] = sd
            if (bi + 1) % 100 == 0 or bi + 1 == len(my_tr):
                phase("g39_oof_progress", rank=rank, done=bi + 1, total=len(my_tr))
        out = str(TMP_DIR / f"oof_r{rank}.pt")
        torch.save(out_map, out)
        return out
    finally:
        stu.close()
        tea.close()


def _solve_2fold(g_tr_list, r_tr_list, g_va_list, r_va_list, sl, alphas, x_sel, d_sel, y_sel):
    g_tr = sum(g_tr_list)
    r_tr = sum(r_tr_list)
    g_all = g_tr + sum(g_va_list)
    r_all = r_tr + sum(r_va_list)
    xs = np.asarray(x_sel[:, sl], dtype=np.float64)
    best_a, best_ic = None, -1e9
    for a in alphas:
        w = solve_ridge_subblock(g_tr, r_tr, sl, a)
        ic = daily_spearman_ic(d_sel, xs @ w, y_sel)
        if ic > best_ic:
            best_ic, best_a = ic, a
    assert best_a is not None
    w_tr = solve_ridge_subblock(g_tr, r_tr, sl, best_a).astype(np.float32)
    w_all = solve_ridge_subblock(g_all, r_all, sl, best_a).astype(np.float32)
    w_folds = [
        solve_ridge_subblock(g_tr_list[i], r_tr_list[i], sl, best_a * (0.5 if len(g_tr_list) > 1 else 1.0)).astype(np.float32)
        for i in range(len(g_tr_list))
    ]
    return best_a, float(best_ic), w_tr, w_all, w_folds


def _fit_teacher_3target_oof_lgbm(
    tea_feats_all: list[np.ndarray],
    tea_ycvar_all: list[np.ndarray],
    tea_yfwd_all: list[np.ndarray],
    tea_dates: list[str],
    oof_tea_anc: dict[str, dict[str, np.ndarray]],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Train 2-fold OOF Heavy Teacher LightGBMs (cvar & fwd5_10) inside isolated child process."""
    import lightgbm as lgb

    half = len(tea_feats_all) // 2
    out: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, (tr_sl, ev_sl) in enumerate([(slice(half, None), slice(0, half)), (slice(0, half), slice(half, None))]):
        xt_f = np.concatenate(tea_feats_all[tr_sl], axis=0)
        yc_f = np.concatenate(tea_ycvar_all[tr_sl], axis=0)
        yf_f = np.concatenate(tea_yfwd_all[tr_sl], axis=0)
        kw = dict(
            n_estimators=260, learning_rate=0.04, max_depth=4, num_leaves=15,
            min_child_samples=800, subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
            reg_lambda=250.0, n_jobs=-1, verbose=-1,
        )
        m_cv = lgb.LGBMRegressor(random_state=20261128 + fold_idx * 10, **kw)
        m_cv.fit(xt_f, yc_f)
        m_fw = lgb.LGBMRegressor(random_state=20261129 + fold_idx * 10, **kw)
        m_fw.fit(xt_f, yf_f)
        for j in range(*ev_sl.indices(len(tea_feats_all))):
            pc = standardize_1d(m_cv.predict(tea_feats_all[j])).astype(np.float32)
            pf = standardize_1d(m_fw.predict(tea_feats_all[j])).astype(np.float32)
            om = oof_tea_anc[tea_dates[j]]
            t_cvar = standardize_1d(0.45 * pc + 0.30 * om["tea_heavy_cvar"] + 0.25 * om["tea_heavy_nowhipsaw"]).astype(np.float32)
            t_fwd = standardize_1d(0.55 * pf + 0.45 * om["tea_heavy_fwd5_10"]).astype(np.float32)
            t_floor = standardize_1d(om["tea_heavy_floor10"]).astype(np.float32)
            out.append((j, t_cvar, t_fwd, t_floor))
    out.sort(key=lambda x: x[0])
    return out


def _train_student_nn_worker(args: tuple) -> dict:
    rank, jobs, data_pt, val_pt = args
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    train_days = torch.load(data_pt, map_location="cpu", weights_only=False)
    val_days = torch.load(val_pt, map_location="cpu", weights_only=False)

    results = {}
    for name, kind, arm_idx, in_dim, n_anc, epochs, lr, wd in jobs:
        best_states, val_ics = [], []
        for seed in (42, 102, 202):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = _build_stu_nn(kind, in_dim, n_anc).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
            best_ic, best_sd = -1e9, None
            for _ep in range(epochs):
                m.train()
                for idx in np.random.permutation(len(train_days)):
                    xc2, xmc2, yc_np, yfw_np, yfl_np, tc_np, tf_np, tfl_np, w_crash = train_days[idx]
                    xt = torch.as_tensor(xc2 if arm_idx == 0 else xmc2, device=device, dtype=torch.float32)
                    yc = torch.as_tensor(yc_np, device=device, dtype=torch.float32)
                    yfw = torch.as_tensor(yfw_np, device=device, dtype=torch.float32)
                    yfl = torch.as_tensor(yfl_np, device=device, dtype=torch.float32)
                    tc = torch.as_tensor(tc_np, device=device, dtype=torch.float32)
                    tf = torch.as_tensor(tf_np, device=device, dtype=torch.float32)
                    tfl = torch.as_tensor(tfl_np, device=device, dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    if kind == "dual_mlp":
                        mu_cvar, mu_fwd, fl_tail = m(xt)
                        tgt_c = 0.65 * tc + 0.35 * yc
                        tgt_f = 0.65 * tf + 0.35 * yfw
                        tgt_fl = 0.50 * tfl + 0.50 * yfl
                        loss = (
                            0.45 * tail_penalized_cosic_loss(mu_cvar, tgt_c, yfl)
                            + 0.35 * cosine_ic_loss_vec(mu_fwd, tgt_f)
                            + 0.20 * cosine_ic_loss_vec(fl_tail, tgt_fl)
                        ) * min(float(w_crash), 2.5)
                    else:
                        pred = m(xt)
                        blend = 0.55 * (0.65 * tc + 0.35 * yc) + 0.45 * (0.65 * tf + 0.35 * yfw)
                        blend = (blend - blend.mean()) / blend.std(unbiased=False).clamp_min(1e-6)
                        loss = tail_penalized_cosic_loss(pred, blend, yfl) * min(float(w_crash), 2.5)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()

                m.eval()
                with torch.inference_mode():
                    pc_list, pf_list, dates, cv_r, fw_r = [], [], [], [], []
                    for d_str, xc2_v, xmc2_v, cvar_r, fwd_r in val_days:
                        xv = torch.as_tensor(xc2_v if arm_idx == 0 else xmc2_v, device=device, dtype=torch.float32)
                        if kind == "dual_mlp":
                            mc, mf, _ = m(xv)
                            pc_list.append(mc.cpu().numpy())
                            pf_list.append(mf.cpu().numpy())
                        else:
                            out = m(xv).cpu().numpy()
                            pc_list.append(out)
                            pf_list.append(out)
                        dates.append(np.full(len(cvar_r), d_str))
                        cv_r.append(cvar_r)
                        fw_r.append(fwd_r)
                    d_cat = np.concatenate(dates)
                    ic = 0.5 * daily_spearman_ic(d_cat, np.concatenate(pc_list), np.concatenate(cv_r)) + 0.5 * daily_spearman_ic(
                        d_cat, np.concatenate(pf_list), np.concatenate(fw_r)
                    )
                if ic > best_ic:
                    best_ic = ic
                    best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            assert best_sd is not None
            best_states.append(best_sd)
            val_ics.append(best_ic)
        results[name] = {
            "kind": kind,
            "arm": arm_idx,
            "in_dim": in_dim,
            "n_anc": n_anc,
            "states": best_states,
            "val_ic": float(np.mean(val_ics)),
        }
        phase("g39_nn_done", rank=rank, model_name=name, val_ic=round(float(np.mean(val_ics)), 4))
    return results


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    """Physical zero-teacher OOS scoring: ONLY loads C2 + Mini-2k (NO Base, NO PT-Small!)."""
    (
        rank, n_gpus, paths, u_c2, u_mn, stu_heads, nn_specs,
        lgbm_paths, c2_anc_keys, mc2_anc_keys, sym_oos, oos_batches,
    ) = args
    import lightgbm as lgb

    device, stu = _load_student_only(rank, paths)
    try:
        uc2_t = torch.as_tensor(u_c2, device=device, dtype=torch.float32)
        umn_t = torch.as_tensor(u_mn, device=device, dtype=torch.float32)
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in stu_heads.items()}

        nn_loaded: dict[str, tuple[str, int, list[nn.Module]]] = {}
        for name, spec in nn_specs.items():
            mods = []
            for sd in spec["states"]:
                m = _build_stu_nn(spec["kind"], spec["in_dim"], spec["n_anc"]).to(device)
                m.load_state_dict(sd)
                m.eval()
                mods.append(m)
            nn_loaded[name] = (spec["kind"], spec["arm"], mods)

        boosters = {k: lgb.Booster(model_file=p) for k, p in lgbm_paths.items()}

        frames = []
        for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            ctx_np, _ = compute_quad_context_14d(sym_oos, s_idx, t_idx, pct_b)
            xs = _fwd_ext(stu, sym_oos, s_idx, t_idx, sec_b, pct_b)

            raw_s: dict[str, np.ndarray] = {}
            for k, (sl, wt) in heads_t.items():
                v = xs[:, sl] @ wt
                raw_s[k] = ((v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)

            zc2 = zscore_cross_section_torch(xs[:, SL_C2] @ uc2_t, clip_val=5.0).cpu().numpy().astype(np.float32)
            zmn = zscore_cross_section_torch(xs[:, SL_MINI] @ umn_t, clip_val=5.0).cpu().numpy().astype(np.float32)

            x_c2 = np.concatenate([np.column_stack([raw_s[k] for k in c2_anc_keys]), zc2, ctx_np], axis=1).astype(np.float32)
            x_mc2 = np.concatenate([np.column_stack([raw_s[k] for k in mc2_anc_keys]), zc2, zmn, ctx_np], axis=1).astype(np.float32)
            xt_c2 = torch.as_tensor(x_c2, device=device, dtype=torch.float32)
            xt_mc2 = torch.as_tensor(x_mc2, device=device, dtype=torch.float32)

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g39_c2only_ridge_cvar": raw_s["c2_cvar"],
                "p_g39_c2only_ridge_fwd5_10": raw_s["c2_fwd5_10"],
                "p_g39_minic2_ridge_cvar": raw_s["mc2_cvar"],
                "p_g39_minic2_ridge_liqcvar": raw_s["mc2_liqcvar"],
                "p_g39_minic2_ridge_fwd5_10": raw_s["mc2_fwd5_10"],
            })

            for name, (kind, arm, mods) in nn_loaded.items():
                xt = xt_c2 if arm == 0 else xt_mc2
                prefix = "p_g39_c2only" if arm == 0 else "p_g39_minic2"
                if kind == "dual_mlp":
                    cs, fws, fls = [], [], []
                    for m in mods:
                        mc, mf, mfl = m(xt)
                        cs.append(mc.cpu().numpy())
                        fws.append(mf.cpu().numpy())
                        fls.append(mfl.cpu().numpy())
                    mc_z = standardize_1d(np.mean(cs, axis=0))
                    mf_z = standardize_1d(np.mean(fws, axis=0))
                    mfl_z = standardize_1d(np.mean(fls, axis=0))
                    df_d[f"{prefix}_mlp_cvar_shield"] = standardize_1d(mc_z + 0.35 * mfl_z).astype(np.float32)
                    if arm == 1:
                        df_d[f"{prefix}_mlp_fwd"] = mf_z.astype(np.float32)
                    df_d[f"{prefix}_mlp_cont_shield"] = standardize_1d(0.55 * mc_z + 0.45 * mf_z + 0.25 * mfl_z).astype(np.float32)
                else:
                    df_d[f"{prefix}_xstock_tail"] = standardize_1d(np.mean([m(xt).cpu().numpy() for m in mods], axis=0)).astype(np.float32)

            df_d["p_g39_c2only_lgbm_cvar"] = standardize_1d(boosters["c2_cvar"].predict(x_c2, num_threads=1)).astype(np.float32)
            df_d["p_g39_c2only_lgbm_fwd5_10"] = standardize_1d(boosters["c2_fwd"].predict(x_c2, num_threads=1)).astype(np.float32)
            df_d["p_g39_minic2_lgbm_cvar"] = standardize_1d(boosters["mc2_cvar"].predict(x_mc2, num_threads=1)).astype(np.float32)
            df_d["p_g39_minic2_lgbm_fwd5_10"] = standardize_1d(boosters["mc2_fwd"].predict(x_mc2, num_threads=1)).astype(np.float32)
            frames.append(df_d)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = str(TMP_DIR / f"oos_r{rank}.csv.gz")
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
        return out_csv
    finally:
        stu.close()


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g39_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        "base": snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        "pts": snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    }
    paths = {**hf_dirs, "c2": c2_dir}

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G39, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G39]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    step = max(1, len(tr_batches) // 40)
    pca_pts = run_gpu_workers_failfast(_pca_worker, [(r, n_gpus, paths, sym_tr, tr_batches[::step][:40]) for r in range(n_gpus)], tag="g39_pca")
    gc2 = sum(torch.load(p, map_location="cpu", weights_only=False)["gc2"] for p in pca_pts)
    gmn = sum(torch.load(p, map_location="cpu", weights_only=False)["gmn"] for p in pca_pts)
    gt = sum(torch.load(p, map_location="cpu", weights_only=False)["gt"] for p in pca_pts)
    for p in pca_pts:
        Path(p).unlink(missing_ok=True)

    def _evec(mat, k):
        _, v = np.linalg.eigh(mat)
        return v[:, -k:][:, ::-1].astype(np.float32).copy()

    u_c2, u_mn, u_tea = _evec(gc2, K_PCA_C2), _evec(gmn, K_PCA_MINI), _evec(gt, K_PCA_TEA)
    del gc2, gmn, gt

    ext_pts = run_gpu_workers_failfast(
        _extract_worker, [(r, n_gpus, paths, u_c2, u_mn, u_tea, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)], tag="g39_ext"
    )
    metas = [torch.load(p, map_location="cpu", weights_only=False) for p in ext_pts]
    for p in ext_pts:
        Path(p).unlink(missing_ok=True)
    metas.sort(key=lambda d: d["rank"])

    n_tr_rows = sum(d["n_rows"] for d in metas)
    tr_recs, val_sel_recs = [], []
    for d in metas:
        tr_recs.extend(d["tr_recs"])
        val_sel_recs.extend(d["val_sel_recs"])
        d["tr_recs"], d["val_sel_recs"] = [], []
    tr_recs.sort(key=lambda r: r["date"])
    val_sel_recs.sort(key=lambda r: r["date"])

    xs_sel = np.concatenate([np.load(r["xs_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    xt_sel = np.concatenate([np.load(r["xt_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["cvar_raw"]), r["date"]) for r in val_sel_recs])
    sel_eval = {
        "cvar": np.concatenate([r["cvar_raw"] for r in val_sel_recs]),
        "nowhipsaw": np.concatenate([r["cvar_raw"] for r in val_sel_recs]),
        "fwd5_10": np.concatenate([r["fwd_raw"] for r in val_sel_recs]),
        "floor10": np.concatenate([r["floor_raw"] for r in val_sel_recs]),
    }

    # Solve Heavy Teacher Ridge heads per scheme and release matrices cleanly
    tea_specs = [
        ("tea_heavy_cvar", slice(0, TEACHER_EXTRA_DIM), "regime", "cvar"),
        ("tea_heavy_nowhipsaw", slice(0, TEACHER_EXTRA_DIM), "regime", "nowhipsaw"),
        ("tea_heavy_fwd5_10", slice(0, TEACHER_EXTRA_DIM), "regime", "fwd5_10"),
        ("tea_heavy_floor10", slice(0, TEACHER_EXTRA_DIM), "regime", "floor10"),
        ("tea_crashentry_cvar", slice(0, TEACHER_EXTRA_DIM), "crashentry", "cvar"),
    ]
    tea_opp_heads = [{} for _ in range(n_gpus)]
    selection = {}
    for sch in ("regime", "crashentry"):
        g_tr = [np.load(d["p_gtea_tr"][sch]) for d in metas]
        g_va = [np.load(d["p_gtea_va"][sch]) for d in metas]
        for name, sl, _s, tkey in [sp for sp in tea_specs if sp[2] == sch]:
            r_tr = [np.load(d["p_rtea_tr"][sch][tkey]) for d in metas]
            r_va = [np.load(d["p_rtea_va"][sch][tkey]) for d in metas]
            a, ic, _wtr, _wall, w_folds = _solve_2fold(g_tr, r_tr, g_va, r_va, sl, ALPHAS_TEA, xt_sel, d_sel, sel_eval[tkey])
            for r in range(n_gpus):
                opp = (r + 1) % n_gpus if n_gpus > 1 else 0
                tea_opp_heads[r][name] = (sl, w_folds[opp])
            selection[name] = {"alpha": a, "val_ic": ic}
        del g_tr, g_va
        for d in metas:
            Path(d["p_gtea_tr"][sch]).unlink(missing_ok=True)
            Path(d["p_gtea_va"][sch]).unlink(missing_ok=True)
            for t in ("cvar", "nowhipsaw", "fwd5_10", "floor10"):
                Path(d["p_rtea_tr"][sch][t]).unlink(missing_ok=True)
                Path(d["p_rtea_va"][sch][t]).unlink(missing_ok=True)
        gc.collect()

    # Solve Student Ridge heads one weighting scheme at a time
    stu_specs = [
        ("c2_cvar", SL_C2, "regime", "cvar"),
        ("c2_nowhipsaw", SL_C2, "regime", "nowhipsaw"),
        ("c2_fwd5_10", SL_C2, "regime", "fwd5_10"),
        ("c2_crashentry", SL_C2, "crashentry", "cvar"),
        ("mini_cvar", SL_MINI, "regime", "cvar"),
        ("mc2_cvar", SL_MINIC2, "regime", "cvar"),
        ("mc2_fwd5_10", SL_MINIC2, "regime", "fwd5_10"),
        ("mc2_liqcvar", SL_MINIC2, "liqcvar", "cvar"),
    ]
    stu_full_heads, stu_tr_heads, stu_opp_heads = {}, {}, [{} for _ in range(n_gpus)]
    for sch in ("regime", "crashentry", "liqcvar"):
        g_tr = [np.load(d["p_gstu_tr"][sch]) for d in metas]
        g_va = [np.load(d["p_gstu_va"][sch]) for d in metas]
        for name, sl, _s, tkey in [sp for sp in stu_specs if sp[2] == sch]:
            r_tr = [np.load(d["p_rstu_tr"][sch][tkey]) for d in metas]
            r_va = [np.load(d["p_rstu_va"][sch][tkey]) for d in metas]
            a, ic, w_tr, w_all, w_folds = _solve_2fold(g_tr, r_tr, g_va, r_va, sl, ALPHAS_STU, xs_sel, d_sel, sel_eval[tkey])
            stu_tr_heads[name] = (sl, w_tr)
            stu_full_heads[name] = (sl, w_all)
            for r in range(n_gpus):
                opp = (r + 1) % n_gpus if n_gpus > 1 else 0
                stu_opp_heads[r][name] = (sl, w_folds[opp])
            selection[f"stu_{name}"] = {"alpha": a, "val_ic": ic}
        del g_tr, g_va
        for d in metas:
            Path(d["p_gstu_tr"][sch]).unlink(missing_ok=True)
            Path(d["p_gstu_va"][sch]).unlink(missing_ok=True)
            for t in ("cvar", "nowhipsaw", "fwd5_10"):
                Path(d["p_rstu_tr"][sch][t]).unlink(missing_ok=True)
                Path(d["p_rstu_va"][sch][t]).unlink(missing_ok=True)
        gc.collect()
    del metas, xs_sel, xt_sel, d_sel, sel_eval
    gc.collect()

    oof_pts = run_gpu_workers_failfast(
        _project_oof_worker, [(r, n_gpus, paths, stu_opp_heads[r], tea_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)], tag="g39_oof"
    )
    oof_map = {}
    for p in oof_pts:
        oof_map.update(torch.load(p, map_location="cpu", weights_only=False))
        Path(p).unlink(missing_ok=True)
    del sym_tr, tr_batches, sym_val, val_batches
    gc.collect()

    # Build 2-fold OOF 3-Target Super-Teacher (CVaR + Fwd5-10 + Floor10) strictly in isolated child process
    tea_feats_all, tea_ycvar_all, tea_yfwd_all, tea_dates = [], [], [], []
    for rec in tr_recs:
        d_str = rec["date"]
        om = oof_map[d_str]
        t_anc = np.column_stack([om[k] for k in (
            "tea_heavy_cvar", "tea_heavy_nowhipsaw", "tea_heavy_fwd5_10", "tea_crashentry_cvar", "mc2_cvar", "mc2_fwd5_10"
        )])
        tf = np.concatenate([
            t_anc,
            rec["ztea"].astype(np.float32),
            rec["zc2"].astype(np.float32)[:, :24],
            rec["zmn"].astype(np.float32)[:, :16],
            rec["ctx"],
        ], axis=1).astype(np.float32)
        tea_feats_all.append(tf)
        tea_ycvar_all.append(rec["y_cvar"])
        tea_yfwd_all.append(rec["y_fwd"])
        tea_dates.append(d_str)

    oof_tea_anc = {
        d: {k: oof_map[d][k] for k in ("tea_heavy_cvar", "tea_heavy_nowhipsaw", "tea_heavy_fwd5_10", "tea_heavy_floor10")}
        for d in tea_dates
    }
    t_super_3 = run_in_isolated_process(_fit_teacher_3target_oof_lgbm, tea_feats_all, tea_ycvar_all, tea_yfwd_all, tea_dates, oof_tea_anc)
    tea_feats_all.clear()
    tea_ycvar_all.clear()
    tea_yfwd_all.clear()
    oof_tea_anc.clear()

    train_days = []
    for i, rec in enumerate(tr_recs):
        om = oof_map[rec["date"]]
        xc2 = np.concatenate([np.column_stack([om[k] for k in C2_ANC_KEYS]), rec["zc2"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        xmc2 = np.concatenate([np.column_stack([om[k] for k in MC2_ANC_KEYS]), rec["zc2"].astype(np.float32), rec["zmn"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        _, tc, tf, tfl = t_super_3[i]
        train_days.append((xc2, xmc2, rec["y_cvar"], rec["y_fwd"], rec["y_floor"], tc, tf, tfl, rec["w_crash"]))
    tr_recs.clear()
    oof_map.clear()
    t_super_3.clear()

    val_days = []
    for rec in val_sel_recs:
        xs_f = np.load(rec["xs_path"]).astype(np.float32)
        Path(rec["xs_path"]).unlink(missing_ok=True)
        Path(rec["xt_path"]).unlink(missing_ok=True)
        xc2 = np.concatenate([
            np.column_stack([standardize_1d(xs_f[:, stu_tr_heads[k][0]] @ stu_tr_heads[k][1]) for k in C2_ANC_KEYS]).astype(np.float32),
            rec["zc2"].astype(np.float32),
            rec["ctx"],
        ], axis=1).astype(np.float32)
        xmc2 = np.concatenate([
            np.column_stack([standardize_1d(xs_f[:, stu_tr_heads[k][0]] @ stu_tr_heads[k][1]) for k in MC2_ANC_KEYS]).astype(np.float32),
            rec["zc2"].astype(np.float32),
            rec["zmn"].astype(np.float32),
            rec["ctx"],
        ], axis=1).astype(np.float32)
        val_days.append((rec["date"], xc2, xmc2, rec["cvar_raw"], rec["fwd_raw"]))
    val_sel_recs.clear()

    train_pt, val_pt = str(TMP_DIR / "train_days.pt"), str(TMP_DIR / "val_days.pt")
    torch.save(train_days, train_pt)
    torch.save(val_days, val_pt)

    jobs_g0 = [
        ("g39_c2only_dual_mlp", "dual_mlp", 0, IN_DIM_C2, N_ANC_C2, 18, 1.4e-3, 2e-2),
        ("g39_c2only_xstock", "xstock", 0, IN_DIM_C2, N_ANC_C2, 18, 1.2e-3, 2e-2),
    ]
    jobs_g1 = [
        ("g39_minic2_dual_mlp", "dual_mlp", 1, IN_DIM_MC2, N_ANC_MC2, 18, 1.4e-3, 2e-2),
        ("g39_minic2_xstock", "xstock", 1, IN_DIM_MC2, N_ANC_MC2, 18, 1.2e-3, 2e-2),
    ]
    nn_jobs = [(0, jobs_g0, train_pt, val_pt), (1 % n_gpus, jobs_g1, train_pt, val_pt)] if n_gpus >= 2 else [(0, jobs_g0 + jobs_g1, train_pt, val_pt)]
    nn_parts = run_gpu_workers_failfast(_train_student_nn_worker, nn_jobs, tag="g39_nn")
    nn_specs = {}
    for part in nn_parts:
        for name, res in part.items():
            nn_specs[name] = {k: res[k] for k in ("kind", "arm", "in_dim", "n_anc", "states")}
            selection[name] = {"val_ic": res["val_ic"]}
    Path(train_pt).unlink(missing_ok=True)
    Path(val_pt).unlink(missing_ok=True)

    # 4 Distilled Student LightGBMs trained strictly inside isolated child processes
    xc2_tr = np.concatenate([d[0] for d in train_days], axis=0)
    xmc2_tr = np.concatenate([d[1] for d in train_days], axis=0)
    yc_tr = np.concatenate([0.65 * d[5] + 0.35 * d[2] for d in train_days], axis=0)
    yf_tr = np.concatenate([0.65 * d[6] + 0.35 * d[3] for d in train_days], axis=0)

    xc2_va = np.concatenate([d[1] for d in val_days], axis=0)
    xmc2_va = np.concatenate([d[2] for d in val_days], axis=0)
    d_va = np.concatenate([np.full(len(d[3]), d[0]) for d in val_days])
    rc_va = np.concatenate([d[3] for d in val_days])
    rf_va = np.concatenate([d[4] for d in val_days])
    yc_va_z = np.concatenate([standardize_1d(rank_pct_centered(d[3])) for d in val_days], axis=0)
    yf_va_z = np.concatenate([standardize_1d(rank_pct_centered(d[4])) for d in val_days], axis=0)

    lgbm_paths = {
        "c2_cvar": str(OUTPUT / "g39_lgbm_c2only_cvar.txt"),
        "c2_fwd": str(OUTPUT / "g39_lgbm_c2only_fwd.txt"),
        "mc2_cvar": str(OUTPUT / "g39_lgbm_minic2_cvar.txt"),
        "mc2_fwd": str(OUTPUT / "g39_lgbm_minic2_fwd.txt"),
    }
    selection["g39_c2only_lgbm_cvar"] = fit_lgbm_regressor_isolated(xc2_tr, yc_tr, xc2_va, yc_va_z, d_va, rc_va, lgbm_paths["c2_cvar"], seed=20261128)
    selection["g39_c2only_lgbm_fwd5_10"] = fit_lgbm_regressor_isolated(xc2_tr, yf_tr, xc2_va, yf_va_z, d_va, rf_va, lgbm_paths["c2_fwd"], seed=20261129)
    selection["g39_minic2_lgbm_cvar"] = fit_lgbm_regressor_isolated(xmc2_tr, yc_tr, xmc2_va, yc_va_z, d_va, rc_va, lgbm_paths["mc2_cvar"], seed=20261130)
    selection["g39_minic2_lgbm_fwd5_10"] = fit_lgbm_regressor_isolated(xmc2_tr, yf_tr, xmc2_va, yf_va_z, d_va, rf_va, lgbm_paths["mc2_fwd"], seed=20261131)
    del train_days, val_days, xc2_tr, xmc2_tr, xc2_va, xmc2_va
    gc.collect()

    np.savez(OUTPUT / "g39_heads.npz", u_c2=u_c2, u_mini=u_mn, **{k: v[1] for k, v in stu_full_heads.items()})
    torch.save({"nn_specs": nn_specs, "c2_anc_keys": C2_ANC_KEYS, "mc2_anc_keys": MC2_ANC_KEYS}, OUTPUT / "g39_models.pt")
    (OUTPUT / "g39_selection_partial.json").write_text(
        json.dumps({"n_train_rows": n_tr_rows, "selection": selection}, indent=2, ensure_ascii=False)
    )

    # Physical zero-Teacher guarantee: delete Base & PT-Small BEFORE OOS scoring!
    for key in ("base", "pts"):
        shutil.rmtree(hf_dirs[key], ignore_errors=True)

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [(r, n_gpus, paths, u_c2, u_mn, stu_full_heads, nn_specs, lgbm_paths, C2_ANC_KEYS, MC2_ANC_KEYS, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g39_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g39_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g39_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for key in ("mini", "tok2k", "tok_base"):
        shutil.rmtree(hf_dirs[key], ignore_errors=True)
    phase("g39_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
