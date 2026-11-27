"""G38: Multi-Horizon Term-Structure Continuation (r3/r5/r10 + Days 6-10 Drift) + Leading Crash-Hazard Net.

1. Stock-level term-structure & anti-reversal continuation:
   Explicitly separates days 1..5 (r5) from days 6..10 continuation (r_fwd5_10 = C[t+10]/C[t+5] - 1).
   Trains Ridge + 3-seed TermStructContNet + LightGBM specialists predicting [r3, r5, r_fwd5_10, r10],
   exporting `consistent_d5_d10` that penalizes short-lived 5d spikes likely to reverse in days 6..10.
2. Market-level causal leading crash-hazard detector (`P_crash(t)`):
   Combines causal market-breadth momentum (delta_b20_3d/5d, b5-b20, ew_ret5/20) with daily
   Kronos s1 logit entropy / expected-return dispersion / hidden-state cosine dispersion across 2019-2026
   to score forward 10-day crash probability on `breadth >= 0.30` active entry days.
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
    train_lgbm_regressor,
)

OUTPUT = Path(os.environ.get("G38_OUTPUT", "/kaggle/working/g38_out"))
INPUT = Path(os.environ.get("G38_INPUT", "/kaggle/input"))
TMP_DIR = Path("/kaggle/working/_tmp_g38")

D_MINI = 256
D_SMALL = 512
D_BASE = 832

# Layout (FEAT_DIM = 1024 + 2560 + 3328 + 30 = 6942):
#   Mini  (1024): [m20_last, m45_l1, m45_last, m45_dep]
#   C2    (2560): [c20_last, c45_l5, c45_last, c45_dep, c90_last]
#   Base  (3328): [b45_l5, b45_l8, b45_last, b45_dep]
#   Stats   (30): [st_m(10), st_c(10), st_b(10)]
MINI_DIM = D_MINI * 4    # 1024
C2_DIM = D_SMALL * 5     # 2560
BASE_DIM = D_BASE * 4    # 3328
STATS_DIM = 30
FEAT_DIM = MINI_DIM + C2_DIM + BASE_DIM + STATS_DIM  # 6942

SL_MINI = slice(0, MINI_DIM)
SL_C2 = slice(MINI_DIM, MINI_DIM + C2_DIM)
SL_BASE = slice(MINI_DIM + C2_DIM, MINI_DIM + C2_DIM + BASE_DIM)
SL_STATS = slice(MINI_DIM + C2_DIM + BASE_DIM, FEAT_DIM)
SL_ALL = slice(0, FEAT_DIM)

K_PCA = 48
PCA_DIM = K_PCA * 3  # 144 (Mini + C2 + Base)

TRAIN_START_G38 = "2019-01-01"
TRAIN_STRIDE_G38 = 2
SCHEMES = ("std", "regime")
TARGETS = ("r3", "r5", "fwd5_10", "r10", "multi")
ALPHAS_SINGLE = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_TRI = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)

ANCHOR_KEYS = ["mini_r5", "c2_r5", "c2_fwd5_10", "base_r10", "tri_r3", "tri_r5", "tri_fwd5_10", "tri_r10", "tri_regime"]
N_ANCHORS = len(ANCHOR_KEYS)
IN_DIM = N_ANCHORS + PCA_DIM + CTX_DIM  # 9 + 144 + 14 = 167


class TermStructExtractor:
    """Multi-lookback (L=20, 45, 90) extractor across Mini-2k, C2, and Base with dep_s2 + s1/s2 stats."""

    def __init__(self, mini_m, tok2k, c2_m, base_m, tok_base, device: torch.device):
        self.mini = mini_m
        self.tok2k = tok2k
        self.c2 = c2_m
        self.base = base_m
        self.tok_base = tok_base
        self.device = device
        self.proto_2k = build_s1_token_prototypes(tok2k, device)
        self.proto_base = build_s1_token_prototypes(tok_base, device)
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            mini_m.transformer[1].register_forward_hook(self._mk("m1")),
            c2_m.transformer[5].register_forward_hook(self._mk("c5")),
            base_m.transformer[5].register_forward_hook(self._mk("b5")),
            base_m.transformer[8].register_forward_hook(self._mk("b8")),
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
            t20 = self.tok2k.encode(norm_window(raw_t[:, -20:]), half=True)
            _, m20 = self.mini.decode_s1(t20[0], t20[1], st_t[:, -20:])
            m20_last = m20[:, -1, :].float()

            t45 = self.tok2k.encode(x45_norm, half=True)
            s1_m, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])
            dep_m, st_m = _s1_s2_stats(self.mini, s1_m, m45, cur_c, self.proto_2k)

            tb20 = self.tok_base.encode(norm_window(raw_t[:, -20:]), half=True)
            _, c20 = self.c2.decode_s1(tb20[0], tb20[1], st_t[:, -20:], sector_id=sec_t, size_percentile=pct_t)
            c20_last = c20[:, -1, :].float()

            tb90 = self.tok_base.encode(norm_window(raw_t[:, -90:]), half=True)
            _, c90 = self.c2.decode_s1(tb90[0], tb90[1], st_t[:, -90:], sector_id=sec_t, size_percentile=pct_t)
            c90_last = c90[:, -1, :].float()

            # L=45 last so hooks m1, c5, b5, b8 correspond to L=45!
            tb45 = self.tok_base.encode(x45_norm, half=True)
            s1_c, c45 = self.c2.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_c, st_c = _s1_s2_stats(self.c2, s1_c, c45, cur_c, self.proto_base)

            s1_b, c_b = self.base.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_b, st_b_out = _s1_s2_stats(self.base, s1_b, c_b, cur_c, self.proto_base)

        return torch.cat([
            m20_last, self._cap["m1"], m45[:, -1, :].float(), dep_m,
            c20_last, self._cap["c5"], c45[:, -1, :].float(), dep_c, c90_last,
            self._cap["b5"], self._cap["b8"], c_b[:, -1, :].float(), dep_b,
            st_m, st_c, st_b_out,
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, paths: dict[str, str]):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    base = _load_kronos_base(paths["base"], device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    return device, TermStructExtractor(mini, tok2k, c2, base, tok_base, device)


def _fwd_raw(ext, sym_arr, s_idx, t_idx, sec_np, pct_np, bs: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (raw_unnormalized_feat, cross_section_z_feat)."""
    chunks = []
    for off in range(0, len(s_idx), bs):
        sl = slice(off, min(off + bs, len(s_idx)))
        raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl]))
    x_raw = torch.cat(chunks, dim=0)
    return x_raw, zscore_cross_section_torch(x_raw, clip_val=5.0)


def compute_macro_snapshot(
    sym_arr, s_idx: np.ndarray, t_idx: np.ndarray, x_raw: torch.Tensor
) -> np.ndarray:
    """Strictly causal 16-dim daily market-state & Kronos uncertainty vector for date t (<= ti)."""
    n = len(s_idx)
    b20_cnt = b5_cnt = b20_m3_cnt = b20_m5_cnt = stall_cnt = 0
    r1d = np.empty(n, dtype=np.float64)
    r5d = np.empty(n, dtype=np.float64)
    r20d = np.empty(n, dtype=np.float64)
    vol20 = np.empty(n, dtype=np.float64)
    vrat = np.empty(n, dtype=np.float64)

    for i in range(n):
        vals = sym_arr[int(s_idx[i])][0]
        ti = int(t_idx[i])
        c = np.maximum(np.nan_to_num(vals[ti - 25 : ti + 1, 3].astype(np.float64), nan=1.0), 1e-8)
        c_now = c[-1]
        r1d[i] = c_now / c[-2] - 1.0
        r5d[i] = c_now / c[-6] - 1.0
        r20d[i] = c_now / c[-21] - 1.0
        lrets = np.diff(np.log(c[-21:]))
        v20 = float(np.std(lrets))
        v5 = float(np.std(lrets[-5:]))
        vol20[i] = v20
        vrat[i] = v5 / max(v20, 1e-4)
        if c_now > float(c[-20:].mean()):
            b20_cnt += 1
        if c_now > float(c[-5:].mean()):
            b5_cnt += 1
        if c[-4] > float(c[-23:-3].mean()):
            b20_m3_cnt += 1
        if c[-6] > float(c[-25:-5].mean()):
            b20_m5_cnt += 1
        # High-price / high-vol stall flag
        if r20d[i] > 0.12 and r1d[i] < -0.01:
            stall_cnt += 1

    b20 = b20_cnt / max(n, 1)
    b5 = b5_cnt / max(n, 1)
    db3 = b20 - (b20_m3_cnt / max(n, 1))
    db5 = b20 - (b20_m5_cnt / max(n, 1))

    # Daily Kronos internal state diagnostics (strictly <= asof)
    st_m = x_raw[:, SL_STATS.start : SL_STATS.start + 10].cpu().numpy()
    st_c = x_raw[:, SL_STATS.start + 10 : SL_STATS.start + 20].cpu().numpy()
    c45 = x_raw[:, MINI_DIM + D_SMALL * 2 : MINI_DIM + D_SMALL * 3]
    c45_u = F.normalize(c45, dim=1)
    c45_mean = F.normalize(c45_u.mean(dim=0, keepdim=True), dim=1)
    hidden_cos_disp = float((1.0 - (c45_u @ c45_mean.T).squeeze(1)).mean().item())

    return np.asarray([
        b20,
        b5,
        b5 - b20,
        db3,
        db5,
        float(np.mean(r5d)),
        float(np.mean(r20d)),
        float(np.mean(vol20)),
        float(np.mean(vrat)),
        float(np.std(r5d)),
        float(stall_cnt) / max(n, 1),
        float(np.mean(st_m[:, 5])),   # mean_s1_ent_mini
        float(np.quantile(st_m[:, 5], 0.90)),  # p90_s1_ent_mini
        float(np.std(st_m[:, 0])),    # std_s1_exp_ret_mini
        float(np.mean(st_c[:, 5])),   # mean_s1_ent_c2
        hidden_cos_disp,
    ], dtype=np.float32)


class TermStructContNet(nn.Module):
    """Multi-horizon residual MLP jointly predicting [r5, fwd5_10, r10] term structure."""

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, N_ANCHORS))
        self.in_norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.film1 = nn.Linear(4, hidden * 2)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head3 = nn.Linear(hidden // 2, 3)  # [hat_r5, hat_fwd5_10, hat_r10]
        self.anc_proj = nn.Linear(N_ANCHORS, 3, bias=False)
        self.res_scale = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mkt = x[:, -4:]
        gw = F.softmax(self.gate(mkt), dim=1)
        base = self.anc_proj(x[:, :N_ANCHORS] * gw)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        g, b = self.film1(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g)) + 0.25 * b
        h = F.silu(self.fc2(self.drop(h)))
        return base + torch.tanh(self.res_scale).unsqueeze(0) * self.head3(self.drop(h))


def _solve_and_delete(fold_meta, x_sel, d_sel, sel_targets, ridge_specs):
    n_gpus = len(fold_meta)
    selection, full_heads, tr_heads, fold_opp_heads = {}, {}, {}, [{} for _ in range(n_gpus)]
    sel_alphas = {}
    for sch in SCHEMES:
        g_tr_s = sum(np.load(d["tr_g_paths"][sch]) for d in fold_meta)
        g_all_s = g_tr_s + sum(np.load(d["va_g_paths"][sch]) for d in fold_meta)
        for name, sl, _s, tkey, alphas in [sp for sp in ridge_specs if sp[2] == sch]:
            r_tr_s = sum(np.load(d["tr_r_paths"][sch][tkey]) for d in fold_meta)
            r_all_s = r_tr_s + sum(np.load(d["va_r_paths"][sch][tkey]) for d in fold_meta)
            xs = np.asarray(x_sel[:, sl], dtype=np.float64)
            best_a, best_ic = None, -1e9
            for a in alphas:
                w = solve_ridge_subblock(g_tr_s, r_tr_s, sl, a)
                ic = daily_spearman_ic(d_sel, xs @ w, sel_targets[tkey])
                if ic > best_ic:
                    best_ic, best_a = ic, a
            assert best_a is not None
            sel_alphas[name] = best_a
            tr_heads[name] = (sl, solve_ridge_subblock(g_tr_s, r_tr_s, sl, best_a).astype(np.float32))
            full_heads[name] = (sl, solve_ridge_subblock(g_all_s, r_all_s, sl, best_a).astype(np.float32))
            selection[f"ridge_{name}"] = {"alpha": best_a, "val_ic": float(best_ic)}
        del g_tr_s, g_all_s
        for d in fold_meta:
            Path(d["va_g_paths"][sch]).unlink(missing_ok=True)
            for t in TARGETS:
                Path(d["va_r_paths"][sch][t]).unlink(missing_ok=True)
        gc.collect()

    for sch in SCHEMES:
        sch_anchors = [k for k in ANCHOR_KEYS if next(sp for sp in ridge_specs if sp[0] == k)[2] == sch]
        if not sch_anchors:
            for d in fold_meta:
                Path(d["tr_g_paths"][sch]).unlink(missing_ok=True)
                for t in TARGETS:
                    Path(d["tr_r_paths"][sch][t]).unlink(missing_ok=True)
            continue
        for opp in range(n_gpus):
            g_opp = np.load(fold_meta[opp]["tr_g_paths"][sch])
            for r in range(n_gpus):
                if ((r + 1) % n_gpus if n_gpus > 1 else 0) != opp:
                    continue
                for k in sch_anchors:
                    _, sl, _s, tkey, _ = next(sp for sp in ridge_specs if sp[0] == k)
                    r_opp = np.load(fold_meta[opp]["tr_r_paths"][sch][tkey])
                    w_opp = solve_ridge_subblock(g_opp, r_opp, sl, sel_alphas[k] * (0.5 if n_gpus > 1 else 1.0)).astype(np.float32)
                    fold_opp_heads[r][k] = (sl, w_opp)
            del g_opp
        for d in fold_meta:
            Path(d["tr_g_paths"][sch]).unlink(missing_ok=True)
            for t in TARGETS:
                Path(d["tr_r_paths"][sch][t]).unlink(missing_ok=True)
        gc.collect()
    return selection, full_heads, tr_heads, fold_opp_heads


@torch.inference_mode()
def _pca_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_tr, sub_batches = args
    device, ext = _load_models(rank, paths)
    try:
        gm = torch.zeros((MINI_DIM, MINI_DIM), dtype=torch.float64, device=device)
        gc2 = torch.zeros((C2_DIM, C2_DIM), dtype=torch.float64, device=device)
        gb = torch.zeros((BASE_DIM, BASE_DIM), dtype=torch.float64, device=device)
        for b in [b for i, b in enumerate(sub_batches) if i % n_gpus == rank]:
            _, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            _, x64 = _fwd_raw(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            x64 = x64.to(torch.float64)
            gm.addmm_(x64[:, SL_MINI].T, x64[:, SL_MINI])
            gc2.addmm_(x64[:, SL_C2].T, x64[:, SL_C2])
            gb.addmm_(x64[:, SL_BASE].T, x64[:, SL_BASE])
        out = str(TMP_DIR / f"pca_r{rank}.pt")
        torch.save({"gm": gm.cpu().numpy(), "gc2": gc2.cpu().numpy(), "gb": gb.cpu().numpy()}, out)
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _main_extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, paths)
    try:
        um_t = torch.as_tensor(u_dict["m"], device=device, dtype=torch.float32)
        uc_t = torch.as_tensor(u_dict["c"], device=device, dtype=torch.float32)
        ub_t = torch.as_tensor(u_dict["b"], device=device, dtype=torch.float32)

        grams = {s: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for s in SCHEMES}
        rhs = {s: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in TARGETS} for s in SCHEMES}

        tr_recs, macro_tr = [], []
        n_rows = 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, _r10 = b
            path_d = extract_10d_path_targets(sym_tr, s_idx, t_idx)
            ctx_np, st = compute_quad_context_14d(sym_tr, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(str(d_str), len(s_idx), st["b20"])

            x_raw, x32 = _fwd_raw(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            macro_vec = compute_macro_snapshot(sym_tr, s_idx, t_idx, x_raw)
            macro_tr.append({
                "date": str(d_str),
                "feat": macro_vec,
                "univ_ret_d10": float(np.mean(path_d["r10"])),
                "b20": float(macro_vec[0]),
            })

            x64 = x32.to(torch.float64)
            y3_z = standardize_1d(rank_pct_centered(path_d["r3"])).astype(np.float32)
            y5_z = standardize_1d(rank_pct_centered(path_d["r5"])).astype(np.float32)
            y5_10_z = standardize_1d(rank_pct_centered(path_d["r_fwd5_10"])).astype(np.float32)
            y10_z = standardize_1d(rank_pct_centered(path_d["r10"])).astype(np.float32)
            y_t = {
                "r3": torch.as_tensor(y3_z, device=device, dtype=torch.float64),
                "r5": torch.as_tensor(y5_z, device=device, dtype=torch.float64),
                "fwd5_10": torch.as_tensor(y5_10_z, device=device, dtype=torch.float64),
                "r10": torch.as_tensor(y10_z, device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
            }
            xx = x64.T @ x64
            xys = {t: x64.T @ y_t[t] for t in TARGETS}
            for s in SCHEMES:
                grams[s].add_(xx, alpha=w_map[s])
                for t in TARGETS:
                    rhs[s][t].add_(xys[t], alpha=w_map[s])
            n_rows += int(x32.shape[0])

            z_pca = zscore_cross_section_torch(
                torch.cat([x32[:, SL_MINI] @ um_t, x32[:, SL_C2] @ uc_t, x32[:, SL_BASE] @ ub_t], dim=1),
                clip_val=5.0,
            ).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str), "z_pca": z_pca, "ctx": ctx_np,
                "y5": y5_z, "y5_10": y5_10_z, "ym": y_multi.astype(np.float32),
            })
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g38_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

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

        val_sel_recs, macro_va = [], []
        for b in [b for i, b in enumerate(val_batches) if i % n_gpus == rank]:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, _r10 = b
            path_d = extract_10d_path_targets(sym_val, s_idx, t_idx)
            ctx_np, st = compute_quad_context_14d(sym_val, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(str(d_str), len(s_idx), st["b20"])

            x_raw, x32 = _fwd_raw(ext, sym_val, s_idx, t_idx, sec_np, pct_np)
            macro_vec = compute_macro_snapshot(sym_val, s_idx, t_idx, x_raw)
            macro_va.append({
                "date": str(d_str), "feat": macro_vec,
                "univ_ret_d10": float(np.mean(path_d["r10"])), "b20": float(macro_vec[0]),
            })

            x64 = x32.to(torch.float64)
            y3_z = standardize_1d(rank_pct_centered(path_d["r3"])).astype(np.float32)
            y5_z = standardize_1d(rank_pct_centered(path_d["r5"])).astype(np.float32)
            y5_10_z = standardize_1d(rank_pct_centered(path_d["r_fwd5_10"])).astype(np.float32)
            y10_z = standardize_1d(rank_pct_centered(path_d["r10"])).astype(np.float32)
            y_t = {
                "r3": torch.as_tensor(y3_z, device=device, dtype=torch.float64),
                "r5": torch.as_tensor(y5_z, device=device, dtype=torch.float64),
                "fwd5_10": torch.as_tensor(y5_10_z, device=device, dtype=torch.float64),
                "r10": torch.as_tensor(y10_z, device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
            }
            xx = x64.T @ x64
            xys = {t: x64.T @ y_t[t] for t in TARGETS}
            for s in SCHEMES:
                grams[s].add_(xx, alpha=w_map[s])
                for t in TARGETS:
                    rhs[s][t].add_(xys[t], alpha=w_map[s])

            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                z_pca = zscore_cross_section_torch(
                    torch.cat([x32[:, SL_MINI] @ um_t, x32[:, SL_C2] @ uc_t, x32[:, SL_BASE] @ ub_t], dim=1),
                    clip_val=5.0,
                ).half().cpu().numpy()
                xf_path = str(TMP_DIR / f"vsel_{d_str}_r{rank}.npy")
                np.save(xf_path, x32.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str), "x_full_path": xf_path, "z_pca": z_pca, "ctx": ctx_np,
                    "r3": path_d["r3"], "r5": path_d["r5"], "r_fwd5_10": path_d["r_fwd5_10"], "r10": path_d["r10"],
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

        out = str(TMP_DIR / f"meta_r{rank}.pt")
        torch.save({
            "rank": rank, "n_rows": n_rows,
            "tr_g_paths": tr_g_paths, "tr_r_paths": tr_r_paths,
            "va_g_paths": va_g_paths, "va_r_paths": va_r_paths,
            "tr_recs": tr_recs, "val_sel_recs": val_sel_recs,
            "macro_tr": macro_tr, "macro_va": macro_va,
        }, out)
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _project_oof_worker(args: tuple) -> str:
    rank, n_gpus, paths, fold_heads_opp, sym_tr, tr_batches = args
    device, ext = _load_models(rank, paths)
    try:
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in fold_heads_opp.items()}
        out_scores = {}
        for bi, b in enumerate([b for i, b in enumerate(tr_batches) if i % n_gpus == rank]):
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            _, x32 = _fwd_raw(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            cols = []
            for k in ANCHOR_KEYS:
                sl, wt = heads_t[k]
                s = x32[:, sl] @ wt
                cols.append((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6))
            out_scores[str(d_str)] = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
            if (bi + 1) % 100 == 0 or bi + 1 == len(tr_batches) // max(n_gpus, 1):
                phase("g38_oof_progress", rank=rank, done=bi + 1)
        out = str(TMP_DIR / f"oof_r{rank}.pt")
        torch.save(out_scores, out)
        return out
    finally:
        ext.close()


def _train_termstruct_mlp(args: tuple) -> dict:
    rank, train_pt, val_pt = args
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    train_days = torch.load(train_pt, map_location="cpu", weights_only=False)
    val_days = torch.load(val_pt, map_location="cpu", weights_only=False)

    best_states, val_ics = [], []
    for seed in (42, 102, 202):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = TermStructContNet(IN_DIM).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1.4e-3, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=18, eta_min=1.4e-4)
        best_ic, best_sd = -1e9, None
        for _ep in range(18):
            m.train()
            for idx in np.random.permutation(len(train_days)):
                x_np, y5_np, y5_10_np, ym_np = train_days[idx]
                xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                y5 = torch.as_tensor(y5_np, device=device, dtype=torch.float32)
                y510 = torch.as_tensor(y5_10_np, device=device, dtype=torch.float32)
                ym = torch.as_tensor(ym_np, device=device, dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                out = m(xt)
                loss = 0.25 * cosine_ic_loss_vec(out[:, 0], y5) + 0.40 * cosine_ic_loss_vec(out[:, 1], y510) + 0.35 * cosine_ic_loss_vec(out[:, 2], ym)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sched.step()

            m.eval()
            with torch.inference_mode():
                preds, dates, rets = [], [], []
                for d_str, xv, r10, r_cont in val_days:
                    out = m(torch.as_tensor(xv, device=device, dtype=torch.float32))
                    preds.append((out[:, 1] + out[:, 2]).cpu().numpy())
                    dates.append(np.full(len(r10), d_str))
                    rets.append(0.5 * standardize_1d(rank_pct_centered(r_cont)) + 0.5 * standardize_1d(rank_pct_centered(r10)))
                ic = daily_spearman_ic(np.concatenate(dates), np.concatenate(preds), np.concatenate(rets))
            if ic > best_ic:
                best_ic = ic
                best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        assert best_sd is not None
        best_states.append(best_sd)
        val_ics.append(best_ic)
    return {"states": best_states, "val_ic": float(np.mean(val_ics))}


def _fit_weighted_l2_logistic(x_z: np.ndarray, y: np.ndarray, c_val: float, max_iter: int = 25) -> tuple[np.ndarray, float]:
    """Deterministic balanced L2-regularized logistic regression via damped Newton-IRLS (zero sklearn dep)."""
    n, d = x_z.shape
    n_pos = max(float((y == 1).sum()), 1.0)
    n_neg = max(float((y == 0).sum()), 1.0)
    sw = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    x_aug = np.column_stack([np.ones(n, dtype=np.float64), x_z])
    reg = np.diag([0.0] + [1.0 / max(float(c_val), 1e-6)] * d)
    theta = np.zeros(d + 1, dtype=np.float64)
    for _ in range(max_iter):
        logits = np.clip(x_aug @ theta, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        wt = sw * np.maximum(p * (1.0 - p), 1e-6)
        grad = x_aug.T @ (sw * (p - y)) + reg @ theta
        hess = (x_aug * wt[:, None]).T @ x_aug + reg
        delta = np.linalg.solve(hess, grad)
        theta -= delta
        if float(np.max(np.abs(delta))) < 1e-6:
            break
    return theta[1:], float(theta[0])


def fit_crash_hazard_models(macro_tr: list[dict], macro_va: list[dict]) -> dict:
    """Fit strictly causal L2-Logistic + Ridge crash-hazard classifiers on active days (breadth >= 0.30)."""

    def _label_series(recs: list[dict], horizon_steps: int = 5) -> pd.DataFrame:
        recs_s = sorted(recs, key=lambda r: r["date"])
        b20s = np.array([r["b20"] for r in recs_s], dtype=np.float64)
        events = np.zeros(len(recs_s), dtype=np.int32)
        for i in range(len(recs_s)):
            hi = min(len(recs_s), i + horizon_steps + 1)
            max_drop = float(np.min(b20s[i:hi]) - b20s[i]) if hi > i + 1 else 0.0
            if recs_s[i]["univ_ret_d10"] <= -0.04 or max_drop <= -0.15:
                events[i] = 1
        return pd.DataFrame({
            "date": [r["date"] for r in recs_s],
            "b20": b20s,
            "event": events,
            "univ_ret_d10": [r["univ_ret_d10"] for r in recs_s],
            "feat": [r["feat"] for r in recs_s],
        })

    df_tr = _label_series(macro_tr, horizon_steps=5)
    df_va = _label_series(macro_va, horizon_steps=10)
    tr_act = df_tr[df_tr["b20"] >= 0.30].reset_index(drop=True)
    va_act = df_va[(df_va["date"] >= VAL_SEL_START) & (df_va["b20"] >= 0.30)].reset_index(drop=True)

    x_tr = np.stack(tr_act["feat"].to_list()).astype(np.float64)
    y_tr = tr_act["event"].to_numpy(dtype=np.float64)
    mu_tr = x_tr.mean(axis=0)
    sd_tr = np.maximum(x_tr.std(axis=0), 1e-6)
    x_tr_z = np.clip((x_tr - mu_tr) / sd_tr, -4.0, 4.0)

    best_c, best_brier = 0.1, 1e9
    if len(va_act) >= 5 and va_act["event"].nunique() >= 2:
        x_va = np.stack(va_act["feat"].to_list()).astype(np.float64)
        x_va_z = np.clip((x_va - mu_tr) / sd_tr, -4.0, 4.0)
        y_va = va_act["event"].to_numpy(dtype=np.float64)
        for c_val in (0.02, 0.05, 0.1, 0.3, 1.0):
            w_c, b_c = _fit_weighted_l2_logistic(x_tr_z, y_tr, c_val)
            p = 1.0 / (1.0 + np.exp(-np.clip(x_va_z @ w_c + b_c, -20.0, 20.0)))
            brier = float(np.mean((p - y_va) ** 2))
            if brier < best_brier:
                best_brier, best_c = brier, c_val

    df_all = pd.concat([tr_act, va_act], ignore_index=True)
    x_all = np.stack(df_all["feat"].to_list()).astype(np.float64)
    y_all = df_all["event"].to_numpy(dtype=np.float64)
    mu_all = x_all.mean(axis=0)
    sd_all = np.maximum(x_all.std(axis=0), 1e-6)
    x_all_z = np.clip((x_all - mu_all) / sd_all, -4.0, 4.0)

    w_final, b_final = _fit_weighted_l2_logistic(x_all_z, y_all, best_c)
    y_ret = df_all["univ_ret_d10"].to_numpy(dtype=np.float64)
    y_ret_m = float(y_ret.mean())
    x_c = x_all_z - x_all_z.mean(axis=0, keepdims=True)
    w_ridge = np.linalg.solve(x_c.T @ x_c + 50.0 * np.eye(x_c.shape[1]), x_c.T @ (y_ret - y_ret_m))
    b_ridge = float(y_ret_m - x_all_z.mean(axis=0) @ w_ridge)
    return {
        "best_c": best_c,
        "val_brier": best_brier,
        "n_train_active": int(len(tr_act)),
        "train_pos_rate": float(y_tr.mean()) if len(y_tr) else 0.0,
        "scaler_mean": mu_all.tolist(),
        "scaler_scale": sd_all.tolist(),
        "coef": w_final.tolist(),
        "intercept": float(b_final),
        "ridge_coef": w_ridge.tolist(),
        "ridge_intercept": float(b_ridge),
    }


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_dict, full_heads, mlp_states, lgb_cont_path, lgb_r10_path, lgbm_cols, sym_oos, oos_batches = args
    import lightgbm as lgb

    device, ext = _load_models(rank, paths)
    try:
        um_t = torch.as_tensor(u_dict["m"], device=device, dtype=torch.float32)
        uc_t = torch.as_tensor(u_dict["c"], device=device, dtype=torch.float32)
        ub_t = torch.as_tensor(u_dict["b"], device=device, dtype=torch.float32)
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in full_heads.items()}

        mods = []
        for sd in mlp_states:
            m = TermStructContNet(IN_DIM).to(device)
            m.load_state_dict(sd)
            m.eval()
            mods.append(m)
        b_cont = lgb.Booster(model_file=lgb_cont_path)
        b_r10 = lgb.Booster(model_file=lgb_r10_path)

        frames, macro_rows = [], []
        for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            ctx_np, _ = compute_quad_context_14d(sym_oos, s_idx, t_idx, pct_b)
            x_raw, x32 = _fwd_raw(ext, sym_oos, s_idx, t_idx, sec_b, pct_b)
            macro_vec = compute_macro_snapshot(sym_oos, s_idx, t_idx, x_raw)
            macro_rows.append({"asof_date": d_str, "feat": macro_vec.tolist()})

            raw_s = {}
            for k, (sl, wt) in heads_t.items():
                s = x32[:, sl] @ wt
                raw_s[k] = ((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)

            z_pca = zscore_cross_section_torch(
                torch.cat([x32[:, SL_MINI] @ um_t, x32[:, SL_C2] @ uc_t, x32[:, SL_BASE] @ ub_t], dim=1), clip_val=5.0
            ).cpu().numpy().astype(np.float32)
            x_feat = np.concatenate([np.column_stack([raw_s[k] for k in ANCHOR_KEYS]), z_pca, ctx_np], axis=1).astype(np.float32)
            xt = torch.as_tensor(x_feat, device=device, dtype=torch.float32)

            outs = np.mean([m(xt).cpu().numpy() for m in mods], axis=0)
            p_r5 = standardize_1d(outs[:, 0]).astype(np.float32)
            p_cont = standardize_1d(outs[:, 1]).astype(np.float32)
            p_r10 = standardize_1d(outs[:, 2]).astype(np.float32)
            consistent = (p_r5 + p_cont - 0.4 * np.maximum(0.0, p_r5 - p_cont)).astype(np.float32)

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g38_ridge_r3": raw_s["tri_r3"],
                "p_g38_ridge_r5": raw_s["tri_r5"],
                "p_g38_ridge_fwd5_10": raw_s["tri_fwd5_10"],
                "p_g38_ridge_r10": raw_s["tri_r10"],
                "p_g38_mlp_r5": p_r5,
                "p_g38_mlp_fwd5_10": p_cont,
                "p_g38_mlp_r10": p_r10,
                "p_g38_consistent_d5_d10": standardize_1d(consistent).astype(np.float32),
                "p_g38_lgbm_fwd5_10": b_cont.predict(x_feat[:, lgbm_cols], num_threads=1).astype(np.float32),
                "p_g38_lgbm_r10": b_r10.predict(x_feat[:, lgbm_cols], num_threads=1).astype(np.float32),
            })
            frames.append(df_d)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = str(TMP_DIR / f"oos_r{rank}.csv.gz")
        out_mac = str(TMP_DIR / f"oos_macro_r{rank}.json")
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
        Path(out_mac).write_text(json.dumps(macro_rows))
        return f"{out_csv}||{out_mac}"
    finally:
        ext.close()


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
    phase("g38_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        "base": snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    }
    paths = {**hf_dirs, "c2": c2_dir}

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G38, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G38]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    step = max(1, len(tr_batches) // 40)
    pca_pts = run_gpu_workers_failfast(_pca_worker, [(r, n_gpus, paths, sym_tr, tr_batches[::step][:40]) for r in range(n_gpus)], tag="g38_pca")
    gm = sum(torch.load(p, map_location="cpu", weights_only=False)["gm"] for p in pca_pts)
    gc2 = sum(torch.load(p, map_location="cpu", weights_only=False)["gc2"] for p in pca_pts)
    gb = sum(torch.load(p, map_location="cpu", weights_only=False)["gb"] for p in pca_pts)
    for p in pca_pts:
        Path(p).unlink(missing_ok=True)

    def _top(mat):
        _, v = np.linalg.eigh(mat)
        return v[:, -K_PCA:][:, ::-1].astype(np.float32).copy()

    u_dict = {"m": _top(gm), "c": _top(gc2), "b": _top(gb)}
    del gm, gc2, gb

    main_pts = run_gpu_workers_failfast(
        _main_extract_worker, [(r, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)], tag="g38_main"
    )
    fold_meta = [torch.load(p, map_location="cpu", weights_only=False) for p in main_pts]
    for p in main_pts:
        Path(p).unlink(missing_ok=True)
    fold_meta.sort(key=lambda d: d["rank"])

    n_tr_rows = sum(d["n_rows"] for d in fold_meta)
    tr_recs, val_sel_recs, macro_tr, macro_va = [], [], [], []
    for d in fold_meta:
        tr_recs.extend(d["tr_recs"])
        val_sel_recs.extend(d["val_sel_recs"])
        macro_tr.extend(d["macro_tr"])
        macro_va.extend(d["macro_va"])
        d["tr_recs"], d["val_sel_recs"], d["macro_tr"], d["macro_va"] = [], [], [], []
    tr_recs.sort(key=lambda r: r["date"])
    val_sel_recs.sort(key=lambda r: r["date"])

    hazard_meta = fit_crash_hazard_models(macro_tr, macro_va)
    phase("g38_hazard_fitted", best_c=hazard_meta["best_c"], val_brier=round(hazard_meta["val_brier"], 4))

    x_sel = np.concatenate([np.load(r["x_full_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["r10"]), r["date"]) for r in val_sel_recs])
    sel_targets = {
        "r3": np.concatenate([r["r3"] for r in val_sel_recs]),
        "r5": np.concatenate([r["r5"] for r in val_sel_recs]),
        "fwd5_10": np.concatenate([r["r_fwd5_10"] for r in val_sel_recs]),
        "r10": np.concatenate([r["r10"] for r in val_sel_recs]),
        "multi": np.concatenate([r["r10"] for r in val_sel_recs]),
    }

    ridge_specs = [
        ("mini_r5", SL_MINI, "std", "r5", ALPHAS_SINGLE),
        ("c2_r5", SL_C2, "std", "r5", ALPHAS_SINGLE),
        ("c2_fwd5_10", SL_C2, "std", "fwd5_10", ALPHAS_SINGLE),
        ("base_r10", SL_BASE, "std", "r10", ALPHAS_SINGLE),
        ("tri_r3", SL_ALL, "std", "r3", ALPHAS_TRI),
        ("tri_r5", SL_ALL, "std", "r5", ALPHAS_TRI),
        ("tri_fwd5_10", SL_ALL, "std", "fwd5_10", ALPHAS_TRI),
        ("tri_r10", SL_ALL, "std", "r10", ALPHAS_TRI),
        ("tri_regime", SL_ALL, "regime", "multi", ALPHAS_TRI),
    ]
    selection, full_heads, tr_heads, fold_opp_heads = _solve_and_delete(fold_meta, x_sel, d_sel, sel_targets, ridge_specs)
    del fold_meta, x_sel, d_sel, sel_targets
    gc.collect()

    oof_pts = run_gpu_workers_failfast(
        _project_oof_worker, [(r, n_gpus, paths, fold_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)], tag="g38_oof"
    )
    oof_map = {}
    for p in oof_pts:
        oof_map.update(torch.load(p, map_location="cpu", weights_only=False))
        Path(p).unlink(missing_ok=True)
    del sym_tr, tr_batches, sym_val, val_batches
    gc.collect()

    train_days = []
    for rec in tr_recs:
        xf = np.concatenate([oof_map[rec["date"]], rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        train_days.append((xf, rec["y5"], rec["y5_10"], rec["ym"]))
    tr_recs.clear()
    oof_map.clear()

    val_days = []
    for rec in val_sel_recs:
        xf_full = np.load(rec["x_full_path"]).astype(np.float32)
        Path(rec["x_full_path"]).unlink(missing_ok=True)
        anc = np.column_stack([standardize_1d(xf_full[:, tr_heads[k][0]] @ tr_heads[k][1]).astype(np.float32) for k in ANCHOR_KEYS])
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        val_days.append((rec["date"], xf, rec["r10"], rec["r_fwd5_10"]))
    val_sel_recs.clear()

    train_pt, val_pt = str(TMP_DIR / "train_days.pt"), str(TMP_DIR / "val_days.pt")
    torch.save(train_days, train_pt)
    torch.save(val_days, val_pt)
    mlp_res = run_gpu_workers_failfast(_train_termstruct_mlp, [(0, train_pt, val_pt)], tag="g38_nn")[0]
    mlp_states = mlp_res["states"]
    selection["g38_termstruct_mlp"] = {"val_ic": mlp_res["val_ic"]}
    Path(train_pt).unlink(missing_ok=True)
    Path(val_pt).unlink(missing_ok=True)

    lgbm_cols = np.concatenate([
        np.arange(0, N_ANCHORS),
        np.arange(N_ANCHORS, N_ANCHORS + 16),
        np.arange(N_ANCHORS + K_PCA, N_ANCHORS + K_PCA + 16),
        np.arange(N_ANCHORS + 2 * K_PCA, N_ANCHORS + 2 * K_PCA + 16),
        np.arange(N_ANCHORS + PCA_DIM, IN_DIM),
    ])
    x_tr_lgb = np.concatenate([d[0][:, lgbm_cols] for d in train_days], axis=0)
    x_va_lgb = np.concatenate([d[1][:, lgbm_cols] for d in val_days], axis=0)
    d_va_lgb = np.concatenate([np.full(len(d[2]), d[0]) for d in val_days])

    lgb_cont_path = str(OUTPUT / "g38_lgbm_cont.txt")
    selection["g38_lgbm_fwd5_10"] = fit_lgbm_regressor_isolated(
        x_tr_lgb,
        np.concatenate([d[2] for d in train_days]),
        x_va_lgb,
        np.concatenate([standardize_1d(rank_pct_centered(d[3])) for d in val_days]),
        d_va_lgb,
        np.concatenate([d[3] for d in val_days]),
        lgb_cont_path,
        seed=20261127,
    )

    lgb_r10_path = str(OUTPUT / "g38_lgbm_r10.txt")
    selection["g38_lgbm_r10"] = fit_lgbm_regressor_isolated(
        x_tr_lgb,
        np.concatenate([d[3] for d in train_days]),
        x_va_lgb,
        np.concatenate([standardize_1d(rank_pct_centered(d[2])) for d in val_days]),
        d_va_lgb,
        np.concatenate([d[2] for d in val_days]),
        lgb_r10_path,
        seed=20261128,
    )
    del train_days, val_days, x_tr_lgb, x_va_lgb
    gc.collect()

    np.savez(OUTPUT / "g38_heads.npz", u_mini=u_dict["m"], u_c2=u_dict["c"], u_base=u_dict["b"], **{k: v[1] for k, v in full_heads.items()})
    torch.save({"mlp_states": mlp_states, "lgbm_cols": lgbm_cols, "hazard": hazard_meta}, OUTPUT / "g38_models.pt")
    (OUTPUT / "g38_selection_partial.json").write_text(
        json.dumps({"n_train_rows": n_tr_rows, "hazard": hazard_meta, "selection": selection}, indent=2, ensure_ascii=False)
    )

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    res_pairs = run_gpu_workers_failfast(
        _score_oos_worker,
        [(r, n_gpus, paths, u_dict, full_heads, mlp_states, lgb_cont_path, lgb_r10_path, lgbm_cols, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g38_oos",
    )
    csv_paths, mac_paths = zip(*(p.split("||") for p in res_pairs))
    oos_df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in csv_paths:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g38_scores.csv.gz", index=False, compression="gzip")

    # Score causal OOS crash-hazard probabilities
    macro_oos = []
    for mp in mac_paths:
        macro_oos.extend(json.loads(Path(mp).read_text()))
        Path(mp).unlink(missing_ok=True)
    macro_oos.sort(key=lambda r: r["asof_date"])
    sc_mean = np.asarray(hazard_meta["scaler_mean"], dtype=np.float64)
    sc_scale = np.maximum(np.asarray(hazard_meta["scaler_scale"], dtype=np.float64), 1e-8)
    w_clf = np.asarray(hazard_meta["coef"], dtype=np.float64)
    b_clf = float(hazard_meta["intercept"])
    w_rdg = np.asarray(hazard_meta["ridge_coef"], dtype=np.float64)
    b_rdg = float(hazard_meta["ridge_intercept"])
    haz_rows = []
    for r in macro_oos:
        f = np.asarray(r["feat"], dtype=np.float64)
        fz = np.clip((f - sc_mean) / sc_scale, -4.0, 4.0)
        logit = float(fz @ w_clf + b_clf)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
        pred_ret = float(fz @ w_rdg + b_rdg)
        haz_rows.append({
            "asof_date": r["asof_date"],
            "p_crash_hazard": prob,
            "pred_univ_ret10": pred_ret,
            "breadth_ma20": float(f[0]),
            "breadth_ma5": float(f[1]),
            "delta_b20_3d": float(f[3]),
            "delta_b20_5d": float(f[4]),
            "mean_s1_ent_mini": float(f[11]),
            "hidden_cos_disp_c2": float(f[15]),
        })
    pd.DataFrame(haz_rows).to_csv(OUTPUT / "oos_g38_regime_hazard.csv", index=False)

    summary = {"n_train_rows": n_tr_rows, "hazard": hazard_meta, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g38_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for d in hf_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    phase("g38_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
