"""G31: Kronos-mini (4.1M, Tok-2k) Deep Mastery + C2 Multi-Scale Pooling + Mini Ultra-Gentle Top-Block FT.

Part 1 (Full-density 2021-2026 stride=1, 5.83M rows):
- Mini-2k 6-window (L=15, 22, 30, 45, 60, 90) + rich temporal pooling (m3, m5, m10, diff5, diff10, std5)
  + FIRST-EVER dep_layer (s2) hidden states + deterministic 1-step s1 LM logit expectation/entropy/uncertainty!
- C2 zero-new-weight multi-scale rich pooling (L=30, 45, 60) across std / decay730 / decay365 / regime
  and raw / multi / ret5 / volnorm10 (Sharpe-normalized) targets!
- Compact Mini+C2 joint Ridge heads (ultra-lightweight production candidate: only +4.1M params!).

Part 2 (Parallel 2x T4 Ultra-Gentle Top-Block Cosine-IC FT of Kronos-mini 4.1M):
- Arm 0 (GPU 0): Mini L=45 (freeze_until=2, KL+L2-SP anchor) + full-density post-FT pooled Ridge.
- Arm 1 (GPU 1): Mini L=30 (freeze_until=2, KL+L2-SP anchor) + full-density post-FT pooled Ridge.

Strictly leak-free: train <= 2026-04-30, val_sel 2026-05-18..2026-07-02, OOS 2026-07-17..2026-10-23.
"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from g2_kaggle_entry import evaluate_adapter_on_batches
from g2_rank_finetune import TopBlockRankAdapter, build_g2_date_batches, cosine_ic_loss_vec, kl_logits_anchor
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from g28_l45_gentle_ft_entry import _apply_top_state
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window, run_gpu_workers_failfast

OUTPUT = Path(os.environ.get("G31_OUTPUT", "/kaggle/working/g31_out"))
INPUT = Path(os.environ.get("G31_INPUT", "/kaggle/input"))

D_MINI = 256
D_SMALL = 512

# Mini layout (MINI_DIM = 18 * 256 + 8 = 4616):
#  0.. 256: m15_last
#  1.. 512: m22_last
#  2..1280: m30_last, m30_m5, m30_diff5 (3*256=768)
#  5..3328: m45_l1, m45_last, m45_m3, m45_m5, m45_m10, m45_diff5, m45_diff10, m45_std5 (8*256=2048)
# 13..3584: m45_dep_s2 (256)
# 14..4352: m60_last, m60_m5, m60_diff5 (3*256=768)
# 17..4608: m90_last (256)
# 4608..4616: s1_logit_stats (8d)
MINI_DIM = D_MINI * 18 + 8  # 4616
SL_M_45_RICH = slice(D_MINI * 5, D_MINI * 13)         # 2048
SL_M_45_DEP = slice(D_MINI * 13, D_MINI * 14)         # 256
SL_M_45_RICH_DEP = slice(D_MINI * 5, D_MINI * 14)     # 2304
SL_M_30_45_RICH = slice(D_MINI * 2, D_MINI * 14)      # 3072
SL_M_S1_STATS = slice(D_MINI * 18, MINI_DIM)          # 8
SL_M_ALL = slice(0, MINI_DIM)                         # 4616

# C2 layout (C2_DIM = 9 * 512 = 4608):
# 0..1024: c30_last, c30_m5
# 2..4608: c45_l3, c45_l5, c45_l6, c45_last, c45_m5, c45_m10, c45_diff5 (7*512=3584)
C2_DIM = D_SMALL * 9  # 4608
SL_C_45_RICH = slice(D_SMALL * 2, D_SMALL * 9)  # 3584
SL_C_ALL = slice(0, C2_DIM)                     # 4608

# Joint compact slice: Mini [m30_pool + m45_rich + m45_dep] (3072) + Mini s1_stats (8) + C2 [c45_rich] (3584) = 6664
JOINT_DIM = 3072 + 8 + 3584  # 6664
SL_J_ALL = slice(0, JOINT_DIM)

SCHEMES = ("std", "decay365", "regime")
TARGETS = ("raw", "multi", "ret5", "volnorm10")
TRAIN_START_G31 = "2021-01-01"
ALPHAS_MINI = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_LARGE = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)
ALPHAS_STATS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)


def compute_date_weights_g31(d_str: str, n_stocks: int, breadth: float) -> dict[str, float]:
    dt = datetime.strptime(d_str, "%Y-%m-%d")
    dt_ref = datetime.strptime("2026-07-02", "%Y-%m-%d")
    age = max(0.0, float((dt_ref - dt).days))
    eq_day = 4500.0 / float(max(n_stocks, 1))
    return {
        "std": 1.0,
        "decay730": float(0.5 ** (age / 730.0)),
        "decay365": float(0.5 ** (age / 365.0)),
        "regime": float(eq_day * (3.0 if breadth < 0.35 else 1.0)),
    }


def extract_r5_vol_breadth(sym_arr, s_idx: np.ndarray, t_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(s_idx)
    r5 = np.empty(n, dtype=np.float64)
    vol20 = np.empty(n, dtype=np.float64)
    above = 0
    for i in range(n):
        vals = sym_arr[int(s_idx[i])][0]
        ti = int(t_idx[i])
        c = np.maximum(np.nan_to_num(vals[ti - 20 : ti + 1, 3].astype(np.float64), nan=1.0), 1e-8)
        c0 = c[-1]
        c5 = max(float(vals[ti + 5, 3]), 1e-8)
        r5[i] = c5 / c0 - 1.0
        vol20[i] = max(float(np.std(np.diff(np.log(c)))), 0.005)
        if c0 > float(c[1:].mean()):
            above += 1
    r5 = np.nan_to_num(r5, nan=0.0, posinf=0.0, neginf=0.0)
    return r5, vol20, float(above) / float(max(n, 1))


@torch.inference_mode()
def build_s1_token_prototypes(tok2k, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode all 2^s1_bits coarse s1 tokens via post_quant_embed_pre -> decoder -> head once."""
    vocab = 2 ** tok2k.s1_bits
    idx = torch.arange(vocab, device=device, dtype=torch.long)
    mask = 2 ** torch.arange(tok2k.s1_bits, device=device, dtype=torch.long)
    bits = ((idx.unsqueeze(-1) & mask) != 0).float() * 2.0 - 1.0
    bits = bits * (1.0 / (tok2k.codebook_dim ** 0.5))
    z = tok2k.post_quant_embed_pre(bits).unsqueeze(1)  # [V, 1, D]
    for layer in tok2k.decoder:
        z = layer(z)
    out = tok2k.head(z).squeeze(1).float()  # [V, 6] = [O, H, L, C, V, A]
    p_close = out[:, 3]
    p_oc = out[:, 3] - out[:, 0]
    p_hl = (out[:, 1] - out[:, 2]).clamp_min(0.0)
    return p_close, p_oc, p_hl


class MiniC2MasteryExtractor:
    """Extracts Mini-2k 6-window rich pool + dep_layer + s1 analytical stats AND C2 multi-scale rich pool."""

    def __init__(self, mini_m, tok2k, c2_m, tok_base, device: torch.device):
        self.mini = mini_m
        self.tok2k = tok2k
        self.c2 = c2_m
        self.tok_base = tok_base
        self.device = device
        self.p_close, self.p_oc, self.p_hl = build_s1_token_prototypes(tok2k, device)
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            mini_m.transformer[1].register_forward_hook(self._mk("m1")),
            c2_m.transformer[3].register_forward_hook(self._mk("c3")),
            c2_m.transformer[5].register_forward_hook(self._mk("c5")),
            c2_m.transformer[6].register_forward_hook(self._mk("c6")),
        ]

    def _mk(self, key: str):
        def _hook(_m, _i, out):
            self._cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            t15 = self.tok2k.encode(norm_window(raw_t[:, -15:]), half=True)
            _, m15 = self.mini.decode_s1(t15[0], t15[1], st_t[:, -15:])
            m15_last = m15[:, -1, :].float()

            t22 = self.tok2k.encode(norm_window(raw_t[:, -22:]), half=True)
            _, m22 = self.mini.decode_s1(t22[0], t22[1], st_t[:, -22:])
            m22_last = m22[:, -1, :].float()

            t30 = self.tok2k.encode(norm_window(raw_t[:, -30:]), half=True)
            _, m30 = self.mini.decode_s1(t30[0], t30[1], st_t[:, -30:])
            m30_last = m30[:, -1, :].float()
            m30_m5 = m30[:, -5:, :].float().mean(dim=1)
            m30_diff5 = m30_last - m30_m5

            t60 = self.tok2k.encode(norm_window(raw_t[:, -60:]), half=True)
            _, m60 = self.mini.decode_s1(t60[0], t60[1], st_t[:, -60:])
            m60_last = m60[:, -1, :].float()
            m60_m5 = m60[:, -5:, :].float().mean(dim=1)
            m60_diff5 = m60_last - m60_m5

            t90 = self.tok2k.encode(norm_window(raw_t[:, -90:]), half=True)
            _, m90 = self.mini.decode_s1(t90[0], t90[1], st_t[:, -90:])
            m90_last = m90[:, -1, :].float()

            # Mini L=45 last so hook self._cap["m1"] belongs to L=45!
            t45 = self.tok2k.encode(norm_window(raw_t[:, -45:]), half=True)
            s1_logits, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])
            m45_l1 = self._cap["m1"]

            # First-ever dep_layer (s2) hidden state conditioned on argmax predicted next s1 token!
            pred_s1_ids = s1_logits.argmax(dim=-1)
            sib = self.mini.embedding.emb_s1(pred_s1_ids)
            dep_out = self.mini.dep_layer(m45, sib)
            m45_dep = dep_out[:, -1, :].float()

            # C2 multi-scale: L=30, then L=45 last so hooks c3/c5/c6 belong to L=45!
            tb30 = self.tok_base.encode(norm_window(raw_t[:, -30:]), half=True)
            _, c30 = self.c2.decode_s1(tb30[0], tb30[1], st_t[:, -30:], sector_id=sec_t, size_percentile=pct_t)
            c30_last = c30[:, -1, :].float()
            c30_m5 = c30[:, -5:, :].float().mean(dim=1)

            tb45 = self.tok_base.encode(norm_window(raw_t[:, -45:]), half=True)
            _, c45 = self.c2.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)

        # Mini L=45 rich temporal pooling
        m45_f = m45.float()
        m45_last = m45_f[:, -1, :]
        m45_m3 = m45_f[:, -3:, :].mean(dim=1)
        m45_m5 = m45_f[:, -5:, :].mean(dim=1)
        m45_m10 = m45_f[:, -10:, :].mean(dim=1)
        m45_diff5 = m45_last - m45_m5
        m45_diff10 = m45_last - m45_m10
        m45_std5 = m45_f[:, -5:, :].std(dim=1, unbiased=False)

        # Deterministic 1-step s1 LM distribution stats (8d)
        logits_last = s1_logits[:, -1, :].float()
        probs = F.softmax(logits_last, dim=-1)
        log_p = F.log_softmax(logits_last, dim=-1)
        exp_c = probs @ self.p_close
        var_c = (probs @ (self.p_close ** 2) - exp_c ** 2).clamp_min(1e-6)
        std_c = var_c.sqrt()
        exp_oc = probs @ self.p_oc
        exp_hl = probs @ self.p_hl
        cur_c_norm = norm_window(raw_t[:, -45:])[:, -1, 3]
        exp_ret = exp_c - cur_c_norm
        prob_up = (probs * (self.p_close.unsqueeze(0) > cur_c_norm.unsqueeze(1)).float()).sum(dim=-1)
        ent = -(probs * log_p).sum(dim=-1)
        max_p = probs.max(dim=-1).values
        top5_p = probs.topk(5, dim=-1).values.sum(dim=-1)
        s1_stats = torch.stack([exp_ret, exp_oc, std_c, exp_hl, prob_up, ent, max_p, top5_p], dim=1)

        # C2 L=45 rich temporal pooling
        c45_f = c45.float()
        c45_last = c45_f[:, -1, :]
        c45_m5 = c45_f[:, -5:, :].mean(dim=1)
        c45_m10 = c45_f[:, -10:, :].mean(dim=1)
        c45_diff5 = c45_last - c45_m5

        xm = torch.cat([
            m15_last, m22_last,
            m30_last, m30_m5, m30_diff5,
            m45_l1, m45_last, m45_m3, m45_m5, m45_m10, m45_diff5, m45_diff10, m45_std5,
            m45_dep,
            m60_last, m60_m5, m60_diff5,
            m90_last,
            s1_stats,
        ], dim=1)
        xc = torch.cat([
            c30_last, c30_m5,
            self._cap["c3"], self._cap["c5"], self._cap["c6"], c45_last, c45_m5, c45_m10, c45_diff5,
        ], dim=1)
        return xm, xc

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _assemble_joint(xm: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
    return torch.cat([xm[:, SL_M_30_45_RICH], xm[:, SL_M_S1_STATS], xc[:, SL_C_45_RICH]], dim=1)


def _load_part1_models(rank: int, paths: dict[str, str]):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    return device, MiniC2MasteryExtractor(mini, tok2k, c2, tok_base, device)


@torch.inference_mode()
def _part1_extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_part1_models(rank, paths)
    try:
        dims = {"m": MINI_DIM, "c": C2_DIM, "j": JOINT_DIM}
        grams = {b: {s: torch.zeros((d, d), dtype=torch.float64, device=device) for s in SCHEMES} for b, d in dims.items()}
        rhss = {b: {s: {t: torch.zeros(d, dtype=torch.float64, device=device) for t in TARGETS} for s in SCHEMES} for b, d in dims.items()}

        def _fwd(sym_arr, s_idx, t_idx, sec_np, pct_np, batch_size: int = 384):
            ms, cs = [], []
            for off in range(0, len(s_idx), batch_size):
                sl = slice(off, min(off + batch_size, len(s_idx)))
                raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                xm, xc = ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl])
                ms.append(xm)
                cs.append(xc)
            xm_z = zscore_cross_section_torch(torch.cat(ms, dim=0), clip_val=5.0)
            xc_z = zscore_cross_section_torch(torch.cat(cs, dim=0), clip_val=5.0)
            return xm_z, xc_z

        def _accum(xm_z, xc_z, y_map, w_map):
            xj_z = _assemble_joint(xm_z, xc_z)
            for blk, xt in (("m", xm_z), ("c", xc_z), ("j", xj_z)):
                x64 = xt.to(torch.float64)
                xx = x64.T @ x64
                xys = {t: x64.T @ y_map[t] for t in TARGETS}
                for s in SCHEMES:
                    w = w_map[s]
                    grams[blk][s].add_(xx, alpha=w)
                    for t in TARGETS:
                        rhss[blk][s][t].add_(xys[t], alpha=w)

        def _flush_mats(prefix: str) -> tuple[dict, dict]:
            gp, rp = {}, {}
            for b in dims:
                gp[b], rp[b] = {}, {}
                for s in SCHEMES:
                    p = f"/tmp/g31_{prefix}_g_{b}_{s}_r{rank}.npy"
                    np.save(p, grams[b][s].cpu().numpy())
                    gp[b][s] = p
                    grams[b][s].zero_()
                    rp[b][s] = {}
                    for t in TARGETS:
                        pt = f"/tmp/g31_{prefix}_r_{b}_{s}_{t}_r{rank}.npy"
                        np.save(pt, rhss[b][s][t].cpu().numpy())
                        rp[b][s][t] = pt
                        rhss[b][s][t].zero_()
            return gp, rp

        n_rows = 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            r5, vol20, breadth = extract_r5_vol_breadth(sym_tr, s_idx, t_idx)
            w_map = compute_date_weights_g31(str(d_str), len(s_idx), breadth)
            xm_z, xc_z = _fwd(sym_tr, s_idx, t_idx, sec_np, pct_np)
            y_map = {
                "raw": torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
                "ret5": torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64),
                "volnorm10": torch.as_tensor(standardize_1d(rank_pct_centered(r10 / vol20)), device=device, dtype=torch.float64),
            }
            _accum(xm_z, xc_z, y_map, w_map)
            n_rows += int(xm_z.shape[0])
            if (bi + 1) % 100 == 0 or bi + 1 == len(my_tr):
                phase("g31_p1_train", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        tr_gp, tr_rp = _flush_mats("tr")

        val_sel_recs = []
        my_val = [b for i, b in enumerate(val_batches) if i % n_gpus == rank]
        for b in my_val:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            r5, vol20, breadth = extract_r5_vol_breadth(sym_val, s_idx, t_idx)
            w_map = compute_date_weights_g31(str(d_str), len(s_idx), breadth)
            xm_z, xc_z = _fwd(sym_val, s_idx, t_idx, sec_np, pct_np)
            y_map = {
                "raw": torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
                "ret5": torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64),
                "volnorm10": torch.as_tensor(standardize_1d(rank_pct_centered(r10 / vol20)), device=device, dtype=torch.float64),
            }
            _accum(xm_z, xc_z, y_map, w_map)
            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                xm_p = f"/tmp/g31_vsel_xm_{d_str}_r{rank}.npy"
                xc_p = f"/tmp/g31_vsel_xc_{d_str}_r{rank}.npy"
                np.save(xm_p, xm_z.half().cpu().numpy())
                np.save(xc_p, xc_z.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "xm_path": xm_p,
                    "xc_path": xc_p,
                    "ret10": r10.astype(np.float32),
                    "ret5": r5.astype(np.float32),
                })

        va_gp, va_rp = _flush_mats("va")
        out = f"/tmp/g31_p1_meta_r{rank}.pt"
        torch.save({
            "n_rows": n_rows,
            "tr_gp": tr_gp, "tr_rp": tr_rp,
            "va_gp": va_gp, "va_rp": va_rp,
            "val_sel_recs": val_sel_recs,
        }, out)
        phase("g31_p1_saved", rank=rank, n_rows=n_rows, n_vsel=len(val_sel_recs))
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _part1_oos_worker(args: tuple) -> str:
    rank, n_gpus, paths, heads_m, heads_c, heads_j, sym_oos, oos_batches = args
    device, ext = _load_part1_models(rank, paths)
    try:
        hm_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_m.items()}
        hc_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_c.items()}
        hj_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_j.items()}

        frames = []
        my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]
        for d_str, recs in my_oos:
            ms, cs = [], []
            for off in range(0, len(recs), 384):
                sl = recs[off : off + 384]
                raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119 : r[1] + 1] for r in sl])
                st_b = np.stack([sym_oos[r[0]][1][r[1] - 119 : r[1] + 1] for r in sl])
                sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
                pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
                xm, xc = ext.extract_batch(raw_b, st_b, sec_b, pct_b)
                ms.append(xm)
                cs.append(xc)
            xm_z = zscore_cross_section_torch(torch.cat(ms, dim=0), clip_val=5.0)
            xc_z = zscore_cross_section_torch(torch.cat(cs, dim=0), clip_val=5.0)
            xj_z = _assemble_joint(xm_z, xc_z)

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
            })
            for k, (sl, wt) in hm_t.items():
                df_d[f"p_{k}"] = (xm_z[:, sl] @ wt).cpu().numpy().astype(np.float32)
            for k, (sl, wt) in hc_t.items():
                df_d[f"p_{k}"] = (xc_z[:, sl] @ wt).cpu().numpy().astype(np.float32)
            for k, (sl, wt) in hj_t.items():
                df_d[f"p_{k}"] = (xj_z[:, sl] @ wt).cpu().numpy().astype(np.float32)
            frames.append(df_d)

        out = f"/tmp/g31_p1_oos_r{rank}.csv.gz"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False, compression="gzip")
        return out
    finally:
        ext.close()


# ---- Part 2: Parallel 2x T4 Ultra-Gentle Top-Block Cosine-IC FT of Kronos-mini (4.1M) ----

@torch.no_grad()
def _warm_start_mini(adapter: TopBlockRankAdapter, tok2k, sym_tr, tr_batches, lb: int, device) -> None:
    adapter.eval()
    xs, ys = [], []
    step = max(1, len(tr_batches) // 60)
    rng = np.random.default_rng(20261121 + lb)
    for b in tr_batches[::step][:60]:
        _, s_idx, t_idx, _, _, y_multi, _, _ = b
        n = len(s_idx)
        take = rng.choice(n, size=min(128, n), replace=False)
        raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
        st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
        h_low = adapter.encode_frozen_lower(
            tok2k, torch.as_tensor(raw_b, device=device, dtype=torch.float32),
            torch.as_tensor(st_b, device=device, dtype=torch.float32),
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = h_low
            for idx in range(adapter.freeze_until, len(adapter.model.transformer)):
                h = adapter.model.transformer[idx](h, key_padding_mask=None)
            h_out = adapter.model.norm(h).float()
        feat = adapter.rank_norm(torch.cat([h_out[:, -1, :], h_out[:, -5:, :].mean(dim=1)], dim=-1)).cpu().numpy().astype(np.float64)
        xs.append(feat)
        ys.append(y_multi[take].astype(np.float64))
    X, Y = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    w = np.linalg.solve(X.T @ X + 1e5 * np.eye(X.shape[1]), X.T @ Y)
    scale = 1.0 / max(float((X @ w).std()), 1e-6)
    adapter.rank_head.weight.copy_(torch.as_tensor(w * scale, dtype=torch.float32, device=device).unsqueeze(0))
    adapter.rank_head.bias.zero_()


def _mini_ft_arm_worker(args: tuple) -> str:
    rank, arm_name, lb, mini_dir, tok2k_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = _load_pretrained_mini(mini_dir, device)
    tok2k = KronosTokenizer.from_pretrained(tok2k_dir).to(device).eval()
    for p in tok2k.parameters():
        p.requires_grad = False

    val_sel_batches = [b for b in val_batches if VAL_SEL_START <= b[0] <= VAL_SEL_END]
    ft_batches = tr_batches[::2]
    adapter = TopBlockRankAdapter(model, freeze_until=2).to(device)
    _warm_start_mini(adapter, tok2k, sym_tr, ft_batches, lb, device)
    init_m, _ = evaluate_adapter_on_batches(adapter, tok2k, sym_val, val_sel_batches, device, (lb,))
    phase("g31_ft_init", arm=arm_name, lb=lb, **init_m)

    top_params = [p for _, p in adapter.model.named_parameters() if p.requires_grad]
    head_params = list(adapter.rank_norm.parameters()) + list(adapter.rank_head.parameters())
    opt = torch.optim.AdamW([
        {"params": top_params, "lr": 2.0e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 1.5e-4, "weight_decay": 1e-2},
    ])
    epochs = 3
    total_steps = epochs * len(ft_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=2e-6)

    rng = np.random.default_rng(20261121 + rank)
    best_ic = init_m["val_ic_raw"]
    best_state = adapter.extract_top_state_dict()
    history = [{"step": 0, **init_m}]

    step = 0
    val_every = max(1, len(ft_batches) // 2)
    for _ep in range(epochs):
        order = rng.permutation(len(ft_batches))
        for bi in order:
            _, s_idx, t_idx, sec_np, pct_np, y_multi, _, _ = ft_batches[bi]
            n = len(s_idx)
            take = rng.choice(n, size=min(384, n), replace=False) if n > 384 else np.arange(n)
            raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
            st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
            h_low = adapter.encode_frozen_lower(
                tok2k, torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
            )
            adapter.train()
            adapter.model.embedding.eval()
            adapter.model.time_emb.eval()
            for idx in range(adapter.freeze_until):
                adapter.model.transformer[idx].eval()

            opt.zero_grad(set_to_none=True)
            score, s1_ft, s1_t = adapter.forward_top(
                h_low,
                torch.as_tensor(sec_np[take], device=device, dtype=torch.long),
                torch.as_tensor(pct_np[take], device=device, dtype=torch.float32),
                with_teacher=True,
            )
            ym = torch.as_tensor(y_multi[take], device=device, dtype=torch.float32)
            loss = cosine_ic_loss_vec(score, ym) + 0.08 * kl_logits_anchor(s1_t, s1_ft, temperature=1.5) + 2000.0 * adapter.weight_l2_anchor()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(top_params + head_params, 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % val_every == 0 or step == total_steps:
                vm, _ = evaluate_adapter_on_batches(adapter, tok2k, sym_val, val_sel_batches, device, (lb,))
                rec = {"epoch": round(step / len(ft_batches), 2), "step": step, **vm}
                history.append(rec)
                if vm["val_ic_raw"] > best_ic:
                    best_ic = vm["val_ic_raw"]
                    best_state = adapter.extract_top_state_dict()
                phase("g31_ft_val", arm=arm_name, **rec)

    _apply_top_state(adapter, best_state)
    adapter.eval()
    model.eval()
    torch.save(best_state, OUTPUT / f"{arm_name}_best_top.pt")

    # Post-FT full-density pooled Ridge on [L1, last, m5, m10, diff5] = 1280d (very fast for 4.1M Mini!)
    fdim = D_MINI * 5
    cap: dict[str, torch.Tensor] = {}
    hook = model.transformer[1].register_forward_hook(lambda _m, _i, o: cap.__setitem__("l1", o[:, -1, :].detach().float()))

    @torch.inference_mode()
    def _extract_ft(sym_arr, s_idx, t_idx, batch_size: int = 512) -> torch.Tensor:
        chunks = []
        for off in range(0, len(s_idx), batch_size):
            sl = slice(off, min(off + batch_size, len(s_idx)))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
            st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                tok = tok2k.encode(norm_window(raw_t), half=True)
                _, ctx = model.decode_s1(tok[0], tok[1], st_t)
            cf = ctx.float()
            last = cf[:, -1, :]
            m5 = cf[:, -5:, :].mean(dim=1)
            m10 = cf[:, -10:, :].mean(dim=1)
            chunks.append(torch.cat([cap["l1"], last, m5, m10, last - m5], dim=1))
        return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    g_tr = torch.zeros((fdim, fdim), dtype=torch.float64, device=device)
    b_tr = torch.zeros(fdim, dtype=torch.float64, device=device)
    for b in tr_batches:
        _, s_idx, t_idx, _, _, y_multi, _, _ = b
        x64 = _extract_ft(sym_tr, s_idx, t_idx).to(torch.float64)
        ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
        g_tr.addmm_(x64.T, x64)
        b_tr.addmv_(x64.T, ym)

    g_all = g_tr.clone()
    b_all = b_tr.clone()
    sel_x, sel_d, sel_r = [], [], []
    for b in val_batches:
        d_str, s_idx, t_idx, _, _, y_multi, _, r10 = b
        x32 = _extract_ft(sym_val, s_idx, t_idx)
        x64 = x32.to(torch.float64)
        ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
        g_all.addmm_(x64.T, x64)
        b_all.addmv_(x64.T, ym)
        if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
            sel_x.append(x32.cpu().numpy())
            sel_d.append(np.full(len(r10), str(d_str)))
            sel_r.append(r10.astype(np.float64))

    g_tr_np, b_tr_np = g_tr.cpu().numpy(), b_tr.cpu().numpy()
    g_all_np, b_all_np = g_all.cpu().numpy(), b_all.cpu().numpy()
    x_sel = np.concatenate(sel_x, axis=0).astype(np.float64)
    d_sel = np.concatenate(sel_d)
    r_sel = np.concatenate(sel_r)

    best_a, best_ric = None, -1e9
    for a in (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9):
        w = solve_ridge_subblock(g_tr_np, b_tr_np, slice(0, fdim), a)
        ic = daily_spearman_ic(d_sel, x_sel @ w, r_sel)
        if ic > best_ric:
            best_ric, best_a = ic, a
    assert best_a is not None
    w_ft = solve_ridge_subblock(g_all_np, b_all_np, slice(0, fdim), best_a).astype(np.float32)
    wt = torch.as_tensor(w_ft, device=device, dtype=torch.float32)

    # Score 65d OOS for both post-FT pooled Ridge and E2E rank head
    frames = []
    with torch.inference_mode():
        for d_str, recs in oos_batches:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            x32 = _extract_ft(sym_oos, s_idx, t_idx)
            p_ridge = (x32 @ wt).cpu().numpy().astype(np.float32)

            # E2E score
            e2e_parts = []
            for off in range(0, len(recs), 512):
                sl = recs[off : off + 512]
                raw_b = np.stack([sym_oos[r[0]][0][r[1] - lb + 1 : r[1] + 1] for r in sl])
                st_b = np.stack([sym_oos[r[0]][1][r[1] - lb + 1 : r[1] + 1] for r in sl])
                raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
                st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    tok = tok2k.encode(norm_window(raw_t), half=True)
                    _, ctx = model.decode_s1(tok[0], tok[1], st_t)
                cf = ctx.float()
                feat = torch.cat([cf[:, -1, :], cf[:, -5:, :].mean(dim=1)], dim=-1)
                e2e_parts.append(adapter.rank_head(adapter.rank_norm(feat)).squeeze(-1).cpu().numpy())
            p_e2e = np.concatenate(e2e_parts).astype(np.float32)
            frames.append(pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                f"p_{arm_name}_ridge_pool": p_ridge,
                f"p_{arm_name}_e2e": p_e2e,
            }))
    hook.remove()
    out_csv = f"/tmp/g31_ft_{arm_name}_oos.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
    out_meta = f"/tmp/g31_ft_{arm_name}_meta.pt"
    torch.save({
        "arm_name": arm_name, "lb": lb, "best_ft_val_ic": best_ic,
        "ridge_alpha": best_a, "ridge_val_ic": best_ric,
        "w_ft": w_ft, "history": history, "oos_csv": out_csv,
    }, out_meta)
    return out_meta


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g31_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    paths = {"mini": hf_dirs[0], "tok2k": hf_dirs[1], "tok_base": hf_dirs[2], "c2": c2_dir}
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G31, TRAIN_FIT_END, min_cross_section=200)
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    # ---- PART 1: Full-density stride=1 Mini Mastery + C2 Multi-Scale Pooling (fail-fast & raw .npy!) ----
    p1_pts = run_gpu_workers_failfast(
        _part1_extract_worker,
        [(r, n_gpus, paths, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        tag="g31_p1",
    )
    metas = [torch.load(p, map_location="cpu", weights_only=False) for p in p1_pts]
    for p in p1_pts:
        Path(p).unlink(missing_ok=True)
    n_tr_rows = sum(d["n_rows"] for d in metas)
    val_sel_recs = []
    for d in metas:
        val_sel_recs.extend(d["val_sel_recs"])
        d["val_sel_recs"] = []
    val_sel_recs.sort(key=lambda r: r["date"])

    xm_sel = np.concatenate([np.load(r["xm_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    xc_sel = np.concatenate([np.load(r["xc_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["ret10"]), r["date"]) for r in val_sel_recs])
    r10_sel = np.concatenate([r["ret10"].astype(np.float64) for r in val_sel_recs])
    r5_sel = np.concatenate([r["ret5"].astype(np.float64) for r in val_sel_recs])
    for r in val_sel_recs:
        Path(r["xm_path"]).unlink(missing_ok=True)
        Path(r["xc_path"]).unlink(missing_ok=True)
    val_sel_recs.clear()
    gc.collect()

    def _fit_specs(blk: str, x_mat: np.ndarray, specs: list):
        sel_out, heads_out = {}, {}
        for sch in SCHEMES:
            sch_specs = [sp for sp in specs if sp[2] == sch]
            if not sch_specs:
                continue
            g_tr_s = sum(np.load(d["tr_gp"][blk][sch]) for d in metas)
            g_all_s = g_tr_s + sum(np.load(d["va_gp"][blk][sch]) for d in metas)
            for name, sl, _s, tkey, alphas in sch_specs:
                r_tr_s = sum(np.load(d["tr_rp"][blk][sch][tkey]) for d in metas)
                r_all_s = r_tr_s + sum(np.load(d["va_rp"][blk][sch][tkey]) for d in metas)
                xs = np.asarray(x_mat[:, sl], dtype=np.float64)
                best_a, best_ic = None, -1e9
                for a in alphas:
                    w = solve_ridge_subblock(g_tr_s, r_tr_s, sl, a)
                    pred = xs @ w
                    sc = daily_spearman_ic(d_sel, pred, r5_sel if tkey == "ret5" else r10_sel)
                    if sc > best_ic:
                        best_ic, best_a = sc, a
                assert best_a is not None
                wf = solve_ridge_subblock(g_all_s, r_all_s, sl, best_a).astype(np.float32)
                heads_out[name] = (sl, wf)
                sel_out[name] = {"scheme": sch, "target": tkey, "dim": int(wf.shape[0]), "alpha": best_a, "val_ic": float(best_ic)}
            del g_tr_s, g_all_s
            for d in metas:
                Path(d["tr_gp"][blk][sch]).unlink(missing_ok=True)
                Path(d["va_gp"][blk][sch]).unlink(missing_ok=True)
                for t in TARGETS:
                    Path(d["tr_rp"][blk][sch][t]).unlink(missing_ok=True)
                    Path(d["va_rp"][blk][sch][t]).unlink(missing_ok=True)
            gc.collect()
        return sel_out, heads_out

    mini_specs = [
        ("g31_mini_s1_stats", SL_M_S1_STATS, "std", "raw", ALPHAS_STATS),
        ("g31_mini_l45_dep_s2", SL_M_45_DEP, "std", "raw", ALPHAS_MINI),
        ("g31_mini_l45_rich_raw", SL_M_45_RICH, "std", "raw", ALPHAS_MINI),
        ("g31_mini_l45_rich_dep", SL_M_45_RICH_DEP, "std", "multi", ALPHAS_MINI),
        ("g31_mini_l30_45_rich_multi", SL_M_30_45_RICH, "std", "multi", ALPHAS_MINI),
        ("g31_mini_l30_45_rich_regime", SL_M_30_45_RICH, "regime", "multi", ALPHAS_MINI),
        ("g31_mini_l30_45_rich_volnorm", SL_M_30_45_RICH, "std", "volnorm10", ALPHAS_MINI),
        ("g31_mini_6win_all_multi", SL_M_ALL, "std", "multi", ALPHAS_LARGE),
        ("g31_mini_6win_all_decay365", SL_M_ALL, "decay365", "multi", ALPHAS_LARGE),
        ("g31_mini_6win_all_regime", SL_M_ALL, "regime", "multi", ALPHAS_LARGE),
    ]
    c2_specs = [
        ("g31_c2_l45_rich_multi", SL_C_45_RICH, "std", "multi", ALPHAS_LARGE),
        ("g31_c2_l45_rich_regime", SL_C_45_RICH, "regime", "multi", ALPHAS_LARGE),
        ("g31_c2_l45_rich_volnorm", SL_C_45_RICH, "std", "volnorm10", ALPHAS_LARGE),
        ("g31_c2_2win_all_multi", SL_C_ALL, "std", "multi", ALPHAS_LARGE),
        ("g31_c2_2win_all_decay365", SL_C_ALL, "decay365", "multi", ALPHAS_LARGE),
    ]
    joint_specs = [
        ("g31_minic2_joint_multi", SL_J_ALL, "std", "multi", ALPHAS_LARGE),
        ("g31_minic2_joint_decay365", SL_J_ALL, "decay365", "multi", ALPHAS_LARGE),
        ("g31_minic2_joint_regime", SL_J_ALL, "regime", "multi", ALPHAS_LARGE),
        ("g31_minic2_joint_ret5", SL_J_ALL, "std", "ret5", ALPHAS_LARGE),
        ("g31_minic2_joint_volnorm", SL_J_ALL, "std", "volnorm10", ALPHAS_LARGE),
    ]

    selection = {}
    s_m, heads_m = _fit_specs("m", xm_sel, mini_specs)
    s_c, heads_c = _fit_specs("c", xc_sel, c2_specs)
    xj_sel = np.concatenate([xm_sel[:, SL_M_30_45_RICH], xm_sel[:, SL_M_S1_STATS], xc_sel[:, SL_C_45_RICH]], axis=1)
    del xm_sel, xc_sel
    gc.collect()
    s_j, heads_j = _fit_specs("j", xj_sel, joint_specs)
    selection.update(s_m)
    selection.update(s_c)
    selection.update(s_j)
    del xj_sel, metas
    gc.collect()
    phase("g31_p1_heads_fitted", n_heads=len(selection), elapsed_sec=round(time.time() - t0, 1))

    # Score Part 1 on 65d OOS
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    p1_csvs = run_gpu_workers_failfast(
        _part1_oos_worker,
        [(r, n_gpus, paths, heads_m, heads_c, heads_j, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g31_p1_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in p1_csvs], ignore_index=True).sort_values(["asof_date", "symbol"]).reset_index(drop=True)
    for p in p1_csvs:
        Path(p).unlink(missing_ok=True)
    phase("g31_p1_oos_done", rows=len(oos_df), elapsed_sec=round(time.time() - t0, 1))

    # ---- PART 2: Parallel 2x T4 Gentle Top-Block Cosine-IC FT of Kronos-mini (L=45 & L=30) ----
    ft_args = [
        (0, "g31_mini_l45_ft", 45, paths["mini"], paths["tok2k"], sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),
        (1 % n_gpus, "g31_mini_l30_ft", 30, paths["mini"], paths["tok2k"], sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),
    ]
    ft_metas = run_gpu_workers_failfast(_mini_ft_arm_worker, ft_args, tag="g31_ft")

    all_heads = {k: v[1] for k, v in {**heads_m, **heads_c, **heads_j}.items()}
    ft_summary = {}
    for mp_path in ft_metas:
        meta = torch.load(mp_path, map_location="cpu", weights_only=False)
        arm = meta["arm_name"]
        all_heads[f"{arm}_ridge_pool"] = meta["w_ft"]
        ft_summary[arm] = {
            "lb": meta["lb"], "best_ft_val_ic": meta["best_ft_val_ic"],
            "ridge_alpha": meta["ridge_alpha"], "ridge_val_ic": meta["ridge_val_ic"],
            "history": meta["history"],
        }
        df_ft = pd.read_csv(meta["oos_csv"]).sort_values(["asof_date", "symbol"]).reset_index(drop=True)
        for col in df_ft.columns:
            if col.startswith("p_"):
                oos_df[col] = df_ft[col].values
        Path(meta["oos_csv"]).unlink(missing_ok=True)
        Path(mp_path).unlink(missing_ok=True)

    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    np.savez(OUTPUT / "g31_heads.npz", **all_heads)
    oos_df.to_csv(OUTPUT / "oos_g31_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_train_rows": n_tr_rows,
        "selection": selection,
        "ft_summary": ft_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g31_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g31_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
