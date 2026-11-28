"""G40: Kronos-mini (4.1M) Differentiable Straight-Through dep_layer(s2) + Top-Block Tail-Aware Gentle FT
& Joint FT-Mini (L=30+45) + Frozen C2 Crash-Shield Continuation Mastery (Physical Zero Base/PTS).

Part 1: Parallel 2x T4 gentle fine-tuning (1.5 epochs, freeze_until=2, KL + L2 anchor) with Straight-Through
Estimator propagating gradients through dep_layer(s2) and proj_s1 onto tail-penalized cvar10 + fwd5_10 + floor10.
Part 2: 2-fold OOF extraction + PCA-96 + DualRewardTailShieldMLP + CrossStockTailNet + 2 isolated LightGBMs
combining FT-Mini (L=30,45) + Frozen C2 (L=30,45,90), zero Base(102M), zero PT-Small(24.7M).
"""
from __future__ import annotations

import copy
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
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec, kl_logits_anchor
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
)

OUTPUT = Path(os.environ.get("G40_OUTPUT", "/kaggle/working/g40_out"))
INPUT = Path(os.environ.get("G40_INPUT", "/kaggle/input"))
TMP_DIR = Path("/kaggle/working/_tmp_g40")

D_MINI = 256
D_SMALL = 512

# Joint feature layout (FEAT_DIM = 2324 + 3594 = 5918, ZERO Base/PTS):
#   FT-Mini L=30 (778) : [m30_last, m30_m5, dep_m30, st_m30(10)]
#   FT-Mini L=45 (1546): [m45_l1, m45_last, m45_m5, m45_m10, m45_diff5, dep_m45, st_m45(10)]
#   Frozen C2    (3594): [c30_last, c45_l5, c45_l6, c45_last, c45_m5, dep_c45, c90_last, st_c45(10)]
FTMINI_L30_DIM = D_MINI * 3 + 10   # 778
FTMINI_L45_DIM = D_MINI * 6 + 10   # 1546
FTMINI_DIM = FTMINI_L30_DIM + FTMINI_L45_DIM  # 2324
C2_DIM = D_SMALL * 7 + 10          # 3594
FEAT_DIM = FTMINI_DIM + C2_DIM     # 5918

SL_FTMINI = slice(0, FTMINI_DIM)
SL_C2 = slice(FTMINI_DIM, FEAT_DIM)
SL_ALL = slice(0, FEAT_DIM)

K_PCA = 48
PCA_DIM = K_PCA * 2  # 96
ANCHOR_KEYS = [
    "ftmini_cvar",
    "ftmini_fwd5_10",
    "ftmini_nowhipsaw",
    "joint_cvar",
    "joint_nowhipsaw",
    "joint_fwd5_10",
    "joint_crashentry",
    "joint_liqcvar",
]
N_ANCHORS = len(ANCHOR_KEYS)  # 8
IN_DIM = N_ANCHORS + PCA_DIM + CTX_DIM  # 8 + 96 + 14 = 118

TRAIN_START_G40 = "2021-01-01"
TRAIN_STRIDE_G40 = 2
ALPHAS = (1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)


def tail_penalized_cosic_loss(pred: torch.Tensor, target_z: torch.Tensor, floor_z: torch.Tensor) -> torch.Tensor:
    """Asymmetric Cosine-IC focusing gradient weight on bottom crash tail and upper winners."""
    w = 1.0 + 0.6 * torch.sigmoid(-2.0 * floor_z) + 0.4 * torch.sigmoid(2.0 * target_z)
    w = w / w.sum()
    dp = pred - (w * pred).sum()
    dt = target_z - (w * target_z).sum()
    vp = (w * dp * dp).sum().clamp_min(1e-8).sqrt()
    vt = (w * dt * dt).sum().clamp_min(1e-8).sqrt()
    return 1.0 - (w * dp * dt).sum() / (vp * vt)


def build_g40_targets(path_d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r5 = path_d["r5"]
    r10 = path_d["r10"]
    floor10 = path_d["floor10"]
    r_fwd5_10 = (1.0 + r10) / np.maximum(1.0 + r5, 1e-4) - 1.0
    cvar_raw = r10 - 0.75 * np.maximum(0.0, -floor10)
    nowhipsaw_raw = r10 - 0.60 * np.maximum(0.0, r5 - r10)
    yc = standardize_1d(rank_pct_centered(cvar_raw)).astype(np.float32)
    yfw = standardize_1d(rank_pct_centered(r_fwd5_10)).astype(np.float32)
    return {
        "cvar": yc,
        "nowhipsaw": standardize_1d(rank_pct_centered(nowhipsaw_raw)).astype(np.float32),
        "fwd5_10": yfw,
        "floor10": standardize_1d(rank_pct_centered(floor10)).astype(np.float32),
        "blend": standardize_1d(0.55 * yc + 0.45 * yfw).astype(np.float32),
        "cvar_raw": cvar_raw.astype(np.float64),
        "fwd_raw": r_fwd5_10.astype(np.float64),
    }


class TailAwareMiniDepAdapter(nn.Module):
    """Top-block Kronos-mini adapter with Straight-Through Estimator through dep_layer(s2) & proj_s1."""

    def __init__(self, mini_model, freeze_until: int = 2):
        super().__init__()
        self.freeze_until = int(freeze_until)
        self.model = mini_model
        for p in self.model.parameters():
            p.requires_grad = False
        for idx in range(self.freeze_until, len(self.model.transformer)):
            for p in self.model.transformer[idx].parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True
        for p in self.model.dep_layer.parameters():
            p.requires_grad = True
        for p in self.model.head.proj_s1.parameters():
            p.requires_grad = True

        self.teacher_blocks = nn.ModuleList([
            copy.deepcopy(self.model.transformer[idx]).eval()
            for idx in range(self.freeze_until, len(self.model.transformer))
        ])
        self.teacher_norm = copy.deepcopy(self.model.norm).eval()
        self.teacher_proj_s1 = copy.deepcopy(self.model.head.proj_s1).eval()
        for m in (self.teacher_blocks, self.teacher_norm, self.teacher_proj_s1):
            for p in m.parameters():
                p.requires_grad = False

        self.l2_anchor = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        d_model = int(self.model.d_model)
        self.rank_norm = nn.LayerNorm(3 * d_model)
        self.rank_drop = nn.Dropout(0.30)
        self.head_cont = nn.Linear(3 * d_model, 1)
        self.head_floor = nn.Linear(3 * d_model, 1)

    @torch.no_grad()
    def encode_frozen_lower(self, tokenizer, raw_hist: torch.Tensor, stamp_hist: torch.Tensor) -> torch.Tensor:
        device = raw_hist.device
        x_norm = norm_window(raw_hist)
        was_training = self.model.training
        self.model.eval()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            tokens = tokenizer.encode(x_norm, half=True)
            h = self.model.embedding([tokens[0], tokens[1]])
            if stamp_hist is not None:
                h = h + self.model.time_emb(stamp_hist)
            for idx in range(self.freeze_until):
                h = self.model.transformer[idx](h, key_padding_mask=None)
        if was_training:
            self.model.train()
            self.model.embedding.eval()
            self.model.time_emb.eval()
            for idx in range(self.freeze_until):
                self.model.transformer[idx].eval()
        return h.detach()

    def forward_top(
        self,
        h_lower: torch.Tensor,
        with_teacher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device = h_lower.device
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = h_lower
            for idx in range(self.freeze_until, len(self.model.transformer)):
                h = self.model.transformer[idx](h, key_padding_mask=None)
            h_out = self.model.norm(h)
            s1_ft = self.model.head(h_out)

        with torch.autocast(device.type, enabled=False):
            h_f = h_out.float()
            s1_f = s1_ft.float()
            p_s1 = F.softmax(s1_f, dim=-1)
            emb_w = self.model.embedding.emb_s1.weight.float()
            soft_sib = p_s1 @ emb_w
            with torch.no_grad():
                hard_sib = self.model.embedding.emb_s1(s1_f.detach().argmax(dim=-1)).float()
            sib = soft_sib + (hard_sib - soft_sib).detach()
            dep_out = self.model.dep_layer(h_f, sib)
            feat = torch.cat([h_f[:, -1, :], h_f[:, -5:, :].mean(dim=1), dep_out[:, -1, :]], dim=-1)
            fn = self.rank_drop(self.rank_norm(feat))
            p_cont = self.head_cont(fn).squeeze(-1)
            p_floor = self.head_floor(fn).squeeze(-1)

        s1_t = None
        if with_teacher:
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ht = h_lower
                for layer in self.teacher_blocks:
                    ht = layer(ht, key_padding_mask=None)
                ht = self.teacher_norm(ht)
                s1_t = self.teacher_proj_s1(ht[:, -1, :]).float()

        return p_cont, p_floor, s1_f[:, -1, :], s1_t

    def weight_l2_anchor(self) -> torch.Tensor:
        total = None
        count = 0
        for name, p in self.model.named_parameters():
            if name in self.l2_anchor:
                diff = (p - self.l2_anchor[name].to(p.device)).pow(2).sum()
                total = diff if total is None else (total + diff)
                count += p.numel()
        if total is None or count == 0:
            return torch.tensor(0.0, device=next(self.head_cont.parameters()).device)
        return total / float(count)

    def extract_top_state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if name in self.l2_anchor:
                out[f"model.{name}"] = p.detach().float().cpu()
        for k, v in self.rank_norm.state_dict().items():
            out[f"rank_norm.{k}"] = v.detach().float().cpu()
        for k, v in self.head_cont.state_dict().items():
            out[f"head_cont.{k}"] = v.detach().float().cpu()
        for k, v in self.head_floor.state_dict().items():
            out[f"head_floor.{k}"] = v.detach().float().cpu()
        return out


def apply_ft_top_to_mini(mini_model, top_sd: dict[str, torch.Tensor]) -> None:
    """Apply saved FT model.* weights into a Kronos-mini instance."""
    msd = mini_model.state_dict()
    for k, v in top_sd.items():
        if k.startswith("model."):
            mk = k[6:]
            msd[mk] = v.to(msd[mk].device, dtype=msd[mk].dtype)
    mini_model.load_state_dict(msd)


@torch.no_grad()
def _warm_start_g40_adapter(adapter: TailAwareMiniDepAdapter, tok2k, sym_tr, tr_batches, lb: int, device: torch.device) -> None:
    adapter.eval()
    xs, y_b, y_f = [], [], []
    step = max(1, len(tr_batches) // 60)
    rng = np.random.default_rng(20261128 + lb)
    for b in tr_batches[::step][:60]:
        _, s_idx, t_idx, _, _, _, _, _ = b
        n = len(s_idx)
        take = rng.choice(n, size=min(128, n), replace=False)
        raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
        st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
        path_d = extract_10d_path_targets(sym_tr, s_idx[take], t_idx[take])
        tgt = build_g40_targets(path_d)
        h_low = adapter.encode_frozen_lower(
            tok2k,
            torch.as_tensor(raw_b, device=device, dtype=torch.float32),
            torch.as_tensor(st_b, device=device, dtype=torch.float32),
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = h_low
            for idx in range(adapter.freeze_until, len(adapter.model.transformer)):
                h = adapter.model.transformer[idx](h, key_padding_mask=None)
            h_out = adapter.model.norm(h)
            s1_ft = adapter.model.head(h_out)
        h_f = h_out.float()
        sib = adapter.model.embedding.emb_s1(s1_ft.float().argmax(dim=-1)).float()
        dep_out = adapter.model.dep_layer(h_f, sib).float()
        feat = adapter.rank_norm(torch.cat([h_f[:, -1, :], h_f[:, -5:, :].mean(dim=1), dep_out[:, -1, :]], dim=-1))
        xs.append(feat.cpu().numpy().astype(np.float64))
        y_b.append(tgt["blend"].astype(np.float64))
        y_f.append(tgt["floor10"].astype(np.float64))

    X = np.concatenate(xs, axis=0)
    eye = 1e5 * np.eye(X.shape[1])
    xtx = X.T @ X + eye
    for head, y_arr in [(adapter.head_cont, np.concatenate(y_b, axis=0)), (adapter.head_floor, np.concatenate(y_f, axis=0))]:
        w = np.linalg.solve(xtx, X.T @ y_arr)
        scale = 1.0 / max(float((X @ w).std()), 1e-6)
        head.weight.copy_(torch.as_tensor(w * scale, dtype=torch.float32, device=device).unsqueeze(0))
        head.bias.zero_()


@torch.inference_mode()
def _eval_g40_adapter(
    adapter: TailAwareMiniDepAdapter, tok2k, sym_val, val_sel_batches: list, lb: int, device: torch.device
) -> dict[str, float]:
    adapter.eval()
    preds, dates, cv_list, fw_list = [], [], [], []
    for d_str, s_idx, t_idx, _sec, _pct, _ym, _yo, _r10 in val_sel_batches:
        path_d = extract_10d_path_targets(sym_val, s_idx, t_idx)
        tgt = build_g40_targets(path_d)
        pc_chunks = []
        for off in range(0, len(s_idx), 512):
            sl = slice(off, min(off + 512, len(s_idx)))
            raw_b = np.stack([sym_val[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_val[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            h_low = adapter.encode_frozen_lower(
                tok2k,
                torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
            )
            pc, pf, _, _ = adapter.forward_top(h_low, with_teacher=False)
            pc_chunks.append((pc + 0.25 * pf).cpu().numpy())
        preds.append(np.concatenate(pc_chunks))
        dates.append(np.full(len(s_idx), str(d_str)))
        cv_list.append(tgt["cvar_raw"])
        fw_list.append(tgt["fwd_raw"])
    d_cat = np.concatenate(dates)
    p_cat = np.concatenate(preds)
    ic_cv = float(daily_spearman_ic(d_cat, p_cat, np.concatenate(cv_list)))
    ic_fw = float(daily_spearman_ic(d_cat, p_cat, np.concatenate(fw_list)))
    return {"val_ic_cvar": ic_cv, "val_ic_fwd": ic_fw, "val_ic_comb": 0.5 * (ic_cv + ic_fw)}


def _ft_mini_arm_worker(args: tuple) -> str:
    rank, arm_name, lb, mini_dir, tok2k_dir, sym_tr, tr_batches, sym_val, val_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(mini_dir, device)
    tok2k = KronosTokenizer.from_pretrained(tok2k_dir).to(device).eval()
    for p in tok2k.parameters():
        p.requires_grad = False

    val_sel_batches = [b for b in val_batches if VAL_SEL_START <= str(b[0]) <= VAL_SEL_END]
    ft_batches = tr_batches[::2]
    adapter = TailAwareMiniDepAdapter(mini, freeze_until=2).to(device)
    _warm_start_g40_adapter(adapter, tok2k, sym_tr, ft_batches, lb, device)
    init_m = _eval_g40_adapter(adapter, tok2k, sym_val, val_sel_batches, lb, device)
    phase("g40_ft_init", arm=arm_name, lb=lb, **init_m)

    top_params = [p for _, p in adapter.model.named_parameters() if p.requires_grad]
    head_params = list(adapter.rank_norm.parameters()) + list(adapter.head_cont.parameters()) + list(adapter.head_floor.parameters())
    opt = torch.optim.AdamW([
        {"params": top_params, "lr": 1.5e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 1.2e-4, "weight_decay": 1e-2},
    ])
    total_steps = max(1, int(round(1.5 * len(ft_batches))))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1.5e-6)
    val_every = max(1, len(ft_batches) // 4)

    rng = np.random.default_rng(20261128 + rank)
    best_ic = init_m["val_ic_comb"]
    best_state = adapter.extract_top_state_dict()
    history = [{"step": 0, "epoch": 0.0, **init_m}]

    step = 0
    for _ep in range(2):
        for bi in rng.permutation(len(ft_batches)):
            if step >= total_steps:
                break
            _, s_idx, t_idx, _, _, _, _, _ = ft_batches[bi]
            n = len(s_idx)
            take = rng.choice(n, size=min(384, n), replace=False) if n > 384 else np.arange(n)
            path_d = extract_10d_path_targets(sym_tr, s_idx[take], t_idx[take])
            tgt = build_g40_targets(path_d)
            raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
            st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
            h_low = adapter.encode_frozen_lower(
                tok2k,
                torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
            )
            adapter.train()
            adapter.model.embedding.eval()
            adapter.model.time_emb.eval()
            for idx in range(adapter.freeze_until):
                adapter.model.transformer[idx].eval()

            opt.zero_grad(set_to_none=True)
            p_cont, p_floor, s1_ft, s1_t = adapter.forward_top(h_low, with_teacher=True)
            yb = torch.as_tensor(tgt["blend"], device=device, dtype=torch.float32)
            yf = torch.as_tensor(tgt["floor10"], device=device, dtype=torch.float32)
            loss = (
                0.70 * tail_penalized_cosic_loss(p_cont, yb, yf)
                + 0.30 * cosine_ic_loss_vec(p_floor, yf)
                + 0.08 * kl_logits_anchor(s1_t, s1_ft, temperature=1.5)
                + 2500.0 * adapter.weight_l2_anchor()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(top_params + head_params, 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % val_every == 0 or step == total_steps:
                vm = _eval_g40_adapter(adapter, tok2k, sym_val, val_sel_batches, lb, device)
                rec = {"step": step, "epoch": round(step / len(ft_batches), 2), **vm}
                history.append(rec)
                if vm["val_ic_comb"] > best_ic:
                    best_ic = vm["val_ic_comb"]
                    best_state = adapter.extract_top_state_dict()
                phase("g40_ft_val", arm=arm_name, **rec)
        if step >= total_steps:
            break

    out_pt = str(OUTPUT / f"ft_mini_l{lb}_top.pt")
    torch.save({"arm_name": arm_name, "lb": lb, "best_val_ic": best_ic, "history": history, "top_state": best_state}, out_pt)
    return out_pt


class JointFTMiniC2Extractor:
    """Physical zero-Base/PTS extractor combining FT-Mini (L=30,45) + Frozen C2 (L=30,45,90)."""

    def __init__(self, ft_m30, ft_m45, tok2k, c2_m, tok_base, device: torch.device, top_l45_sd: dict[str, torch.Tensor]):
        self.m30 = ft_m30
        self.m45 = ft_m45
        self.tok2k = tok2k
        self.c2 = c2_m
        self.tok_base = tok_base
        self.device = device
        self.proto_2k = build_s1_token_prototypes(tok2k, device)
        self.proto_base = build_s1_token_prototypes(tok_base, device)
        self.rank_norm = nn.LayerNorm(D_MINI * 3).to(device)
        self.head_cont = nn.Linear(D_MINI * 3, 1).to(device)
        self.head_floor = nn.Linear(D_MINI * 3, 1).to(device)
        self.rank_norm.load_state_dict({k[10:]: v.to(device) for k, v in top_l45_sd.items() if k.startswith("rank_norm.")})
        self.head_cont.load_state_dict({k[10:]: v.to(device) for k, v in top_l45_sd.items() if k.startswith("head_cont.")})
        self.head_floor.load_state_dict({k[11:]: v.to(device) for k, v in top_l45_sd.items() if k.startswith("head_floor.")})
        self.rank_norm.eval()
        self.head_cont.eval()
        self.head_floor.eval()

        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            ft_m45.transformer[1].register_forward_hook(self._mk("m1")),
            c2_m.transformer[5].register_forward_hook(self._mk("c5")),
            c2_m.transformer[6].register_forward_hook(self._mk("c6")),
        ]

    def _mk(self, key: str):
        def _hook(_m, _i, out):
            self._cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(
        self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray, return_e2e: bool = False
    ):
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)

        x30_norm = norm_window(raw_t[:, -30:])
        x45_norm = norm_window(raw_t[:, -45:])
        cur_c30 = x30_norm[:, -1, 3]
        cur_c45 = x45_norm[:, -1, 3]

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            t30 = self.tok2k.encode(x30_norm, half=True)
            s1_m30, m30 = self.m30.decode_s1(t30[0], t30[1], st_t[:, -30:])
            dep_m30, st_m30 = _s1_s2_stats(self.m30, s1_m30, m30, cur_c30, self.proto_2k)
            m30_f = m30.float()
            m30_last, m30_m5 = m30_f[:, -1, :], m30_f[:, -5:, :].mean(dim=1)

            # FT-Mini L=45 last so hook m1 belongs to L=45
            t45 = self.tok2k.encode(x45_norm, half=True)
            s1_m45, m45 = self.m45.decode_s1(t45[0], t45[1], st_t[:, -45:])
            dep_m45, st_m45 = _s1_s2_stats(self.m45, s1_m45, m45, cur_c45, self.proto_2k)
            m45_f = m45.float()
            m45_last = m45_f[:, -1, :]
            m45_m5 = m45_f[:, -5:, :].mean(dim=1)
            m45_m10 = m45_f[:, -10:, :].mean(dim=1)
            m45_diff5 = m45_last - m45_m5

            # Frozen C2: L=30, L=90, then L=45 LAST so hooks c5/c6 belong strictly to L=45!
            tb30 = self.tok_base.encode(x30_norm, half=True)
            _, c30 = self.c2.decode_s1(tb30[0], tb30[1], st_t[:, -30:], sector_id=sec_t, size_percentile=pct_t)
            c30_last = c30[:, -1, :].float()

            tb90 = self.tok_base.encode(norm_window(raw_t[:, -90:]), half=True)
            _, c90 = self.c2.decode_s1(tb90[0], tb90[1], st_t[:, -90:], sector_id=sec_t, size_percentile=pct_t)
            c90_last = c90[:, -1, :].float()

            tb45 = self.tok_base.encode(x45_norm, half=True)
            s1_c45, c45 = self.c2.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_c45, st_c45 = _s1_s2_stats(self.c2, s1_c45, c45, cur_c45, self.proto_base)
            c45_f = c45.float()
            c45_last, c45_m5 = c45_f[:, -1, :], c45_f[:, -5:, :].mean(dim=1)

        feats = torch.cat([
            m30_last, m30_m5, dep_m30, st_m30,
            self._cap["m1"], m45_last, m45_m5, m45_m10, m45_diff5, dep_m45, st_m45,
            c30_last, self._cap["c5"], self._cap["c6"], c45_last, c45_m5, dep_c45, c90_last, st_c45,
        ], dim=1)
        if not return_e2e:
            return feats
        e2e_in = self.rank_norm(torch.cat([m45_last, m45_m5, dep_m45], dim=-1))
        e2e_score = self.head_cont(e2e_in).squeeze(-1) + 0.25 * self.head_floor(e2e_in).squeeze(-1)
        return feats, e2e_score

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_joint_extractor(rank: int, paths: dict[str, str], top_l30_sd: dict, top_l45_sd: dict):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ft_m30 = _load_pretrained_mini(paths["mini"], device)
    apply_ft_top_to_mini(ft_m30, top_l30_sd)
    ft_m30.eval()

    ft_m45 = _load_pretrained_mini(paths["mini"], device)
    apply_ft_top_to_mini(ft_m45, top_l45_sd)
    ft_m45.eval()

    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    return device, JointFTMiniC2Extractor(ft_m30, ft_m45, tok2k, c2, tok_base, device, top_l45_sd)


def _fwd_joint(ext: JointFTMiniC2Extractor, sym_arr, s_idx, t_idx, sec_np, pct_np, bs: int = 256, return_e2e: bool = False):
    chunks, e2e_chunks = [], []
    for off in range(0, len(s_idx), bs):
        sl = slice(off, min(off + bs, len(s_idx)))
        raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        if return_e2e:
            f, sc = ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl], return_e2e=True)
            chunks.append(f)
            e2e_chunks.append(sc)
        else:
            chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl], return_e2e=False))
    x_z = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
    if not return_e2e:
        return x_z
    e2e = torch.cat(e2e_chunks, dim=0)
    e2e_z = (e2e - e2e.mean()) / e2e.std(unbiased=False).clamp_min(1e-6)
    return x_z, e2e_z.cpu().numpy().astype(np.float32)


class G40DualTailShieldMLP(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, n_anc: int = N_ANCHORS, hidden: int = 128, dropout: float = 0.45):
        super().__init__()
        self.n_anc = n_anc
        self.gate_mu = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, n_anc))
        self.gate_fl = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, n_anc))
        self.in_norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.film1 = nn.Linear(4, hidden * 2)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head_mu = nn.Linear(hidden // 2, 1)
        self.head_fl = nn.Linear(hidden // 2, 1)
        self.res_mu = nn.Parameter(torch.tensor(0.0))
        self.res_fl = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mkt = x[:, -4:]
        anc = x[:, : self.n_anc]
        base_mu = (anc * F.softmax(self.gate_mu(mkt), dim=1)).sum(dim=1)
        base_fl = (anc * F.softmax(self.gate_fl(mkt), dim=1)).sum(dim=1)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        g, b = self.film1(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g)) + 0.25 * b
        d = self.drop(F.silu(self.fc2(self.drop(h))))
        mu = base_mu + torch.tanh(self.res_mu) * self.head_mu(d).squeeze(-1)
        fl = base_fl + torch.tanh(self.res_fl) * self.head_fl(d).squeeze(-1)
        return mu, fl


class G40CrossStockTailNet(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, n_anc: int = N_ANCHORS, d_attn: int = 64, n_factors: int = 8, dropout: float = 0.45):
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


@torch.inference_mode()
def _pca_worker(args: tuple) -> str:
    rank, n_gpus, paths, top30, top45, sym_tr, sub_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, ext = _load_joint_extractor(rank, paths, top30, top45)
    try:
        gf = torch.zeros((FTMINI_DIM, FTMINI_DIM), dtype=torch.float64, device=device)
        gc2 = torch.zeros((C2_DIM, C2_DIM), dtype=torch.float64, device=device)
        for b in [b for i, b in enumerate(sub_batches) if i % n_gpus == rank]:
            _, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x64 = _fwd_joint(ext, sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            gf.addmm_(x64[:, SL_FTMINI].T, x64[:, SL_FTMINI])
            gc2.addmm_(x64[:, SL_C2].T, x64[:, SL_C2])
        out = str(TMP_DIR / f"pca_r{rank}.pt")
        torch.save({"gf": gf.cpu().numpy(), "gc2": gc2.cpu().numpy()}, out)
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, top30, top45, u_ftm, u_c2, sym_tr, tr_batches, sym_val, val_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, ext = _load_joint_extractor(rank, paths, top30, top45)
    try:
        uftm_t = torch.as_tensor(u_ftm, device=device, dtype=torch.float32)
        uc2_t = torch.as_tensor(u_c2, device=device, dtype=torch.float32)
        targets = ("cvar", "nowhipsaw", "fwd5_10")
        schemes = ("regime", "crashentry", "liqcvar")

        g_tr = {s: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for s in schemes}
        r_tr = {s: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in targets} for s in schemes}
        tr_recs, n_rows = [], 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            path_d = extract_10d_path_targets(sym_tr, s_idx, t_idx)
            tgt = build_g40_targets(path_d)
            ctx_np, st = compute_quad_context_14d(sym_tr, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(
                str(d_str), len(s_idx), st["b20"], float(np.mean(path_d["r10"])), float(np.mean(path_d["r5"])),
            )
            w_liq = np.where(st["amt_pct"] < 0.35, 0.25, 1.0).astype(np.float64) * w_map["regime"]

            x32 = _fwd_joint(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            xx = x64.T @ x64
            yt = {t: torch.as_tensor(tgt[t], device=device, dtype=torch.float64) for t in targets}
            xys = {t: x64.T @ yt[t] for t in targets}
            for s in ("regime", "crashentry"):
                w = w_map[s]
                g_tr[s].add_(xx, alpha=w)
                for t in targets:
                    r_tr[s][t].add_(xys[t], alpha=w)

            wl_t = torch.as_tensor(w_liq, device=device, dtype=torch.float64)
            g_tr["liqcvar"].addmm_((x64 * wl_t.unsqueeze(1)).T, x64)
            r_tr["liqcvar"]["cvar"].addmv_(x64.T, wl_t * yt["cvar"])

            z_pca = zscore_cross_section_torch(
                torch.cat([x32[:, SL_FTMINI] @ uftm_t, x32[:, SL_C2] @ uc2_t], dim=1), clip_val=5.0
            ).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str),
                "z_pca": z_pca,
                "ctx": ctx_np,
                "y_cvar": tgt["cvar"],
                "y_fwd": tgt["fwd5_10"],
                "y_nowhipsaw": tgt["nowhipsaw"],
                "y_floor": tgt["floor10"],
                "w_crash": float(w_map["crashentry"]),
            })
            n_rows += int(x32.shape[0])
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g40_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        p_gtr, p_rtr = {}, {}
        for s in schemes:
            gp = str(TMP_DIR / f"gtr_{s}_r{rank}.npy")
            np.save(gp, g_tr[s].cpu().numpy())
            p_gtr[s] = gp
            g_tr[s].zero_()
            p_rtr[s] = {}
            for t in targets:
                rp = str(TMP_DIR / f"rtr_{s}_{t}_r{rank}.npy")
                np.save(rp, r_tr[s][t].cpu().numpy())
                p_rtr[s][t] = rp
                r_tr[s][t].zero_()

        val_sel_recs = []
        for b in [b for i, b in enumerate(val_batches) if i % n_gpus == rank]:
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            path_d = extract_10d_path_targets(sym_val, s_idx, t_idx)
            tgt = build_g40_targets(path_d)
            ctx_np, st = compute_quad_context_14d(sym_val, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(
                str(d_str), len(s_idx), st["b20"], float(np.mean(path_d["r10"])), float(np.mean(path_d["r5"])),
            )
            w_liq = np.where(st["amt_pct"] < 0.35, 0.25, 1.0).astype(np.float64) * w_map["regime"]

            x32 = _fwd_joint(ext, sym_val, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            xx = x64.T @ x64
            yt = {t: torch.as_tensor(tgt[t], device=device, dtype=torch.float64) for t in targets}
            xys = {t: x64.T @ yt[t] for t in targets}
            for s in ("regime", "crashentry"):
                w = w_map[s]
                g_tr[s].add_(xx, alpha=w)
                for t in targets:
                    r_tr[s][t].add_(xys[t], alpha=w)

            wl_t = torch.as_tensor(w_liq, device=device, dtype=torch.float64)
            g_tr["liqcvar"].addmm_((x64 * wl_t.unsqueeze(1)).T, x64)
            r_tr["liqcvar"]["cvar"].addmv_(x64.T, wl_t * yt["cvar"])

            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                z_pca = zscore_cross_section_torch(
                    torch.cat([x32[:, SL_FTMINI] @ uftm_t, x32[:, SL_C2] @ uc2_t], dim=1), clip_val=5.0
                ).half().cpu().numpy()
                xp = str(TMP_DIR / f"vsel_{d_str}_r{rank}.npy")
                np.save(xp, x32.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "x_path": xp,
                    "z_pca": z_pca,
                    "ctx": ctx_np,
                    "cvar_raw": tgt["cvar_raw"],
                    "fwd_raw": tgt["fwd_raw"],
                })

        p_gva, p_rva = {}, {}
        for s in schemes:
            gp = str(TMP_DIR / f"gva_{s}_r{rank}.npy")
            np.save(gp, g_tr[s].cpu().numpy())
            p_gva[s] = gp
            p_rva[s] = {}
            for t in targets:
                rp = str(TMP_DIR / f"rva_{s}_{t}_r{rank}.npy")
                np.save(rp, r_tr[s][t].cpu().numpy())
                p_rva[s][t] = rp

        out = str(TMP_DIR / f"meta_r{rank}.pt")
        torch.save({
            "rank": rank, "n_rows": n_rows,
            "p_gtr": p_gtr, "p_rtr": p_rtr, "p_gva": p_gva, "p_rva": p_rva,
            "tr_recs": tr_recs, "val_sel_recs": val_sel_recs,
        }, out)
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _project_oof_worker(args: tuple) -> str:
    rank, n_gpus, paths, top30, top45, fold_opp_heads, sym_tr, tr_batches = args
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    device, ext = _load_joint_extractor(rank, paths, top30, top45)
    try:
        ht = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in fold_opp_heads.items()}
        out_map = {}
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x32 = _fwd_joint(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            cols = []
            for k in ANCHOR_KEYS:
                sl, wt = ht[k]
                v = x32[:, sl] @ wt
                cols.append((v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6))
            out_map[str(d_str)] = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
            if (bi + 1) % 100 == 0 or bi + 1 == len(my_tr):
                phase("g40_oof_progress", rank=rank, done=bi + 1, total=len(my_tr))
        out = str(TMP_DIR / f"oof_r{rank}.pt")
        torch.save(out_map, out)
        return out
    finally:
        ext.close()


def _train_joint_nn_worker(args: tuple) -> dict:
    rank, jobs, train_pt, val_pt = args
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    train_days = torch.load(train_pt, map_location="cpu", weights_only=False)
    val_days = torch.load(val_pt, map_location="cpu", weights_only=False)

    results = {}
    for name, kind, loss_mode, epochs, lr, wd in jobs:
        best_states, val_ics = [], []
        for seed in (42, 102, 202):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = (G40DualTailShieldMLP() if kind == "dual" else G40CrossStockTailNet()).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
            best_ic, best_sd = -1e9, None
            for _ep in range(epochs):
                m.train()
                for idx in np.random.permutation(len(train_days)):
                    x_np, yc_np, yfw_np, ynw_np, yfl_np, w_crash = train_days[idx]
                    xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                    yc = torch.as_tensor(yc_np, device=device, dtype=torch.float32)
                    yfw = torch.as_tensor(yfw_np, device=device, dtype=torch.float32)
                    ynw = torch.as_tensor(ynw_np, device=device, dtype=torch.float32)
                    yfl = torch.as_tensor(yfl_np, device=device, dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    if kind == "dual":
                        mu, fl = m(xt)
                        if loss_mode == "cvar":
                            loss = (0.65 * tail_penalized_cosic_loss(mu, yc, yfl) + 0.35 * cosine_ic_loss_vec(fl, yfl)) * min(float(w_crash), 2.5)
                        else:
                            blend = 0.50 * ynw + 0.50 * yfw
                            blend = (blend - blend.mean()) / blend.std(unbiased=False).clamp_min(1e-6)
                            loss = 0.65 * tail_penalized_cosic_loss(mu, blend, yfl) + 0.35 * cosine_ic_loss_vec(fl, yfl)
                    else:
                        pred = m(xt)
                        blend = 0.55 * yc + 0.45 * yfw
                        blend = (blend - blend.mean()) / blend.std(unbiased=False).clamp_min(1e-6)
                        loss = tail_penalized_cosic_loss(pred, blend, yfl) * min(float(w_crash), 2.5)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()

                m.eval()
                with torch.inference_mode():
                    preds, dates, rets = [], [], []
                    for d_str, xv, cvar_r, fwd_r in val_days:
                        xv_t = torch.as_tensor(xv, device=device, dtype=torch.float32)
                        if kind == "dual":
                            mu, fl = m(xv_t)
                            sc = (mu + 0.35 * fl).cpu().numpy()
                        else:
                            sc = m(xv_t).cpu().numpy()
                        preds.append(sc)
                        dates.append(np.full(len(cvar_r), d_str))
                        rets.append(cvar_r if loss_mode != "cont" else fwd_r)
                    ic = daily_spearman_ic(np.concatenate(dates), np.concatenate(preds), np.concatenate(rets))
                if ic > best_ic:
                    best_ic = ic
                    best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            assert best_sd is not None
            best_states.append(best_sd)
            val_ics.append(best_ic)
        results[name] = {"kind": kind, "states": best_states, "val_ic": float(np.mean(val_ics))}
        phase("g40_nn_done", rank=rank, model_name=name, val_ic=round(float(np.mean(val_ics)), 4))
    return results


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    """Physical zero-Base/PTS OOS scoring using ONLY FT-Mini (4.1M) + Frozen C2."""
    (
        rank, n_gpus, paths, top30, top45, u_ftm, u_c2,
        full_heads, nn_specs, lgbm_cvar_path, lgbm_fwd_path, sym_oos, oos_batches,
    ) = args
    import lightgbm as lgb

    device, ext = _load_joint_extractor(rank, paths, top30, top45)
    try:
        uftm_t = torch.as_tensor(u_ftm, device=device, dtype=torch.float32)
        uc2_t = torch.as_tensor(u_c2, device=device, dtype=torch.float32)
        ht = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in full_heads.items()}

        nn_loaded: dict[str, tuple[str, list[nn.Module]]] = {}
        for name, spec in nn_specs.items():
            mods = []
            for sd in spec["states"]:
                m = (G40DualTailShieldMLP() if spec["kind"] == "dual" else G40CrossStockTailNet()).to(device)
                m.load_state_dict(sd)
                m.eval()
                mods.append(m)
            nn_loaded[name] = (spec["kind"], mods)

        b_cvar = lgb.Booster(model_file=lgbm_cvar_path)
        b_fwd = lgb.Booster(model_file=lgbm_fwd_path)

        frames = []
        for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            ctx_np, _ = compute_quad_context_14d(sym_oos, s_idx, t_idx, pct_b)
            x32, e2e_s = _fwd_joint(ext, sym_oos, s_idx, t_idx, sec_b, pct_b, return_e2e=True)

            raw_s: dict[str, np.ndarray] = {}
            for k, (sl, wt) in ht.items():
                v = x32[:, sl] @ wt
                raw_s[k] = ((v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)).cpu().numpy().astype(np.float32)

            z_pca = zscore_cross_section_torch(
                torch.cat([x32[:, SL_FTMINI] @ uftm_t, x32[:, SL_C2] @ uc2_t], dim=1), clip_val=5.0
            ).cpu().numpy().astype(np.float32)
            x_feat = np.concatenate([np.column_stack([raw_s[k] for k in ANCHOR_KEYS]), z_pca, ctx_np], axis=1).astype(np.float32)
            xt = torch.as_tensor(x_feat, device=device, dtype=torch.float32)

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g40_ftmini_l45_e2e": e2e_s,
                "p_g40_ftmini_ridge_cvar": raw_s["ftmini_cvar"],
                "p_g40_ftmini_ridge_fwd5_10": raw_s["ftmini_fwd5_10"],
                "p_g40_ftmini_ridge_nowhipsaw": raw_s["ftmini_nowhipsaw"],
                "p_g40_joint_ridge_cvar": raw_s["joint_cvar"],
                "p_g40_joint_ridge_nowhipsaw": raw_s["joint_nowhipsaw"],
                "p_g40_joint_ridge_fwd5_10": raw_s["joint_fwd5_10"],
                "p_g40_joint_ridge_crashentry": raw_s["joint_crashentry"],
                "p_g40_joint_ridge_liqcvar": raw_s["joint_liqcvar"],
            })

            for name, (kind, mods) in nn_loaded.items():
                if kind == "dual":
                    mus, fls = [], []
                    for m in mods:
                        mu, fl = m(xt)
                        mus.append(mu.cpu().numpy())
                        fls.append(fl.cpu().numpy())
                    mu_z = standardize_1d(np.mean(mus, axis=0))
                    fl_z = standardize_1d(np.mean(fls, axis=0))
                    if name.endswith("_cvar"):
                        df_d["p_g40_joint_tailshield_cvar_mu"] = mu_z.astype(np.float32)
                        df_d["p_g40_joint_tailshield_cvar_shield35"] = standardize_1d(mu_z + 0.35 * fl_z).astype(np.float32)
                    else:
                        df_d["p_g40_joint_tailshield_cont_shield35"] = standardize_1d(mu_z + 0.35 * fl_z).astype(np.float32)
                else:
                    df_d["p_g40_joint_xstock_tail"] = standardize_1d(np.mean([m(xt).cpu().numpy() for m in mods], axis=0)).astype(np.float32)

            df_d["p_g40_joint_lgbm_cvar10"] = standardize_1d(b_cvar.predict(x_feat, num_threads=1)).astype(np.float32)
            df_d["p_g40_joint_lgbm_fwd5_10"] = standardize_1d(b_fwd.predict(x_feat, num_threads=1)).astype(np.float32)
            frames.append(df_d)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = str(TMP_DIR / f"oos_r{rank}.csv.gz")
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
        return out_csv
    finally:
        ext.close()


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
    phase("g40_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    }
    paths = {**hf_dirs, "c2": c2_dir}

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G40, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G40]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    # Part 1: Parallel 2x T4 Tail-Aware STE dep_layer Gentle Fine-Tuning (L=45 & L=30)
    ft_arms = [
        (0, "g40_mini_l45_ft", 45, paths["mini"], paths["tok2k"], sym_tr, tr_batches, sym_val, val_batches),
        (1 % n_gpus, "g40_mini_l30_ft", 30, paths["mini"], paths["tok2k"], sym_tr, tr_batches, sym_val, val_batches),
    ]
    ft_pts = run_gpu_workers_failfast(_ft_mini_arm_worker, ft_arms, tag="g40_ft")
    ft_info = {}
    for p in ft_pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        ft_info[int(d["lb"])] = d
    top30 = ft_info[30]["top_state"]
    top45 = ft_info[45]["top_state"]
    selection = {
        "ft_mini_l45": {"best_val_ic": ft_info[45]["best_val_ic"], "history": ft_info[45]["history"]},
        "ft_mini_l30": {"best_val_ic": ft_info[30]["best_val_ic"], "history": ft_info[30]["history"]},
    }
    phase("g40_part1_ft_done", elapsed_sec=round(time.time() - t0, 1))

    # Part 2: PCA + Joint FT-Mini + Frozen C2 Mastery
    step = max(1, len(tr_batches) // 40)
    pca_pts = run_gpu_workers_failfast(
        _pca_worker, [(r, n_gpus, paths, top30, top45, sym_tr, tr_batches[::step][:40]) for r in range(n_gpus)], tag="g40_pca"
    )
    gf = sum(torch.load(p, map_location="cpu", weights_only=False)["gf"] for p in pca_pts)
    gc2 = sum(torch.load(p, map_location="cpu", weights_only=False)["gc2"] for p in pca_pts)
    for p in pca_pts:
        Path(p).unlink(missing_ok=True)

    def _evec(mat, k):
        _, v = np.linalg.eigh(mat)
        return v[:, -k:][:, ::-1].astype(np.float32).copy()

    u_ftm, u_c2 = _evec(gf, K_PCA), _evec(gc2, K_PCA)
    del gf, gc2

    ext_pts = run_gpu_workers_failfast(
        _extract_worker, [(r, n_gpus, paths, top30, top45, u_ftm, u_c2, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)], tag="g40_ext"
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

    x_sel = np.concatenate([np.load(r["x_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["cvar_raw"]), r["date"]) for r in val_sel_recs])
    sel_eval = {
        "cvar": np.concatenate([r["cvar_raw"] for r in val_sel_recs]),
        "nowhipsaw": np.concatenate([r["cvar_raw"] for r in val_sel_recs]),
        "fwd5_10": np.concatenate([r["fwd_raw"] for r in val_sel_recs]),
    }

    ridge_specs = [
        ("ftmini_cvar", SL_FTMINI, "regime", "cvar"),
        ("ftmini_fwd5_10", SL_FTMINI, "regime", "fwd5_10"),
        ("ftmini_nowhipsaw", SL_FTMINI, "regime", "nowhipsaw"),
        ("joint_cvar", SL_ALL, "regime", "cvar"),
        ("joint_nowhipsaw", SL_ALL, "regime", "nowhipsaw"),
        ("joint_fwd5_10", SL_ALL, "regime", "fwd5_10"),
        ("joint_crashentry", SL_ALL, "crashentry", "cvar"),
        ("joint_liqcvar", SL_ALL, "liqcvar", "cvar"),
    ]
    full_heads, tr_heads, fold_opp_heads = {}, {}, [{} for _ in range(n_gpus)]
    for sch in ("regime", "crashentry", "liqcvar"):
        g_tr = [np.load(d["p_gtr"][sch]) for d in metas]
        g_va = [np.load(d["p_gva"][sch]) for d in metas]
        for name, sl, _s, tkey in [sp for sp in ridge_specs if sp[2] == sch]:
            r_tr = [np.load(d["p_rtr"][sch][tkey]) for d in metas]
            r_va = [np.load(d["p_rva"][sch][tkey]) for d in metas]
            a, ic, w_tr, w_all, w_folds = _solve_2fold(g_tr, r_tr, g_va, r_va, sl, ALPHAS, x_sel, d_sel, sel_eval[tkey])
            tr_heads[name] = (sl, w_tr)
            full_heads[name] = (sl, w_all)
            for r in range(n_gpus):
                opp = (r + 1) % n_gpus if n_gpus > 1 else 0
                fold_opp_heads[r][name] = (sl, w_folds[opp])
            selection[f"ridge_{name}"] = {"alpha": a, "val_ic": ic}
        del g_tr, g_va
        for d in metas:
            Path(d["p_gtr"][sch]).unlink(missing_ok=True)
            Path(d["p_gva"][sch]).unlink(missing_ok=True)
            for t in ("cvar", "nowhipsaw", "fwd5_10"):
                Path(d["p_rtr"][sch][t]).unlink(missing_ok=True)
                Path(d["p_rva"][sch][t]).unlink(missing_ok=True)
        gc.collect()
    del metas, x_sel, d_sel, sel_eval
    gc.collect()

    oof_pts = run_gpu_workers_failfast(
        _project_oof_worker, [(r, n_gpus, paths, top30, top45, fold_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)], tag="g40_oof"
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
        train_days.append((xf, rec["y_cvar"], rec["y_fwd"], rec["y_nowhipsaw"], rec["y_floor"], rec["w_crash"]))
    tr_recs.clear()
    oof_map.clear()

    val_days = []
    for rec in val_sel_recs:
        xf_full = np.load(rec["x_path"]).astype(np.float32)
        Path(rec["x_path"]).unlink(missing_ok=True)
        anc = np.column_stack([standardize_1d(xf_full[:, tr_heads[k][0]] @ tr_heads[k][1]).astype(np.float32) for k in ANCHOR_KEYS])
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        val_days.append((rec["date"], xf, rec["cvar_raw"], rec["fwd_raw"]))
    val_sel_recs.clear()

    train_pt, val_pt = str(TMP_DIR / "train_days.pt"), str(TMP_DIR / "val_days.pt")
    torch.save(train_days, train_pt)
    torch.save(val_days, val_pt)

    jobs_g0 = [
        ("g40_joint_dual_cvar", "dual", "cvar", 18, 1.4e-3, 2e-2),
        ("g40_joint_xstock", "xstock", "xstock", 18, 1.2e-3, 2e-2),
    ]
    jobs_g1 = [("g40_joint_dual_cont", "dual", "cont", 18, 1.4e-3, 2e-2)]
    nn_jobs = [(0, jobs_g0, train_pt, val_pt), (1 % n_gpus, jobs_g1, train_pt, val_pt)] if n_gpus >= 2 else [(0, jobs_g0 + jobs_g1, train_pt, val_pt)]
    nn_parts = run_gpu_workers_failfast(_train_joint_nn_worker, nn_jobs, tag="g40_nn")
    nn_specs = {}
    for part in nn_parts:
        for name, res in part.items():
            nn_specs[name] = {"kind": res["kind"], "states": res["states"]}
            selection[name] = {"val_ic": res["val_ic"]}
    Path(train_pt).unlink(missing_ok=True)
    Path(val_pt).unlink(missing_ok=True)

    x_tr_lgb = np.concatenate([d[0] for d in train_days], axis=0)
    x_va_lgb = np.concatenate([d[1] for d in val_days], axis=0)
    d_va_lgb = np.concatenate([np.full(len(d[2]), d[0]) for d in val_days])

    y_tr_cvar = np.concatenate([d[1] for d in train_days], axis=0)
    rc_va = np.concatenate([d[2] for d in val_days])
    yc_va_z = np.concatenate([standardize_1d(rank_pct_centered(d[2])) for d in val_days], axis=0)
    lgbm_cvar_path = str(OUTPUT / "g40_lgbm_cvar10.txt")
    selection["g40_lgbm_cvar10"] = fit_lgbm_regressor_isolated(x_tr_lgb, y_tr_cvar, x_va_lgb, yc_va_z, d_va_lgb, rc_va, lgbm_cvar_path, seed=20261128)

    y_tr_fwd = np.concatenate([d[2] for d in train_days], axis=0)
    rf_va = np.concatenate([d[3] for d in val_days])
    yf_va_z = np.concatenate([standardize_1d(rank_pct_centered(d[3])) for d in val_days], axis=0)
    lgbm_fwd_path = str(OUTPUT / "g40_lgbm_fwd5_10.txt")
    selection["g40_lgbm_fwd5_10"] = fit_lgbm_regressor_isolated(x_tr_lgb, y_tr_fwd, x_va_lgb, yf_va_z, d_va_lgb, rf_va, lgbm_fwd_path, seed=20261129)
    del train_days, val_days, x_tr_lgb, x_va_lgb
    gc.collect()

    np.savez(OUTPUT / "g40_heads.npz", u_ftm=u_ftm, u_c2=u_c2, **{k: v[1] for k, v in full_heads.items()})
    torch.save({"nn_specs": nn_specs, "top30": top30, "top45": top45, "anchor_keys": ANCHOR_KEYS}, OUTPUT / "g40_models.pt")
    (OUTPUT / "g40_selection_partial.json").write_text(
        json.dumps({"n_train_rows": n_tr_rows, "selection": selection}, indent=2, ensure_ascii=False)
    )

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [
            (r, n_gpus, paths, top30, top45, u_ftm, u_c2, full_heads, nn_specs, lgbm_cvar_path, lgbm_fwd_path, sym_oos, oos_batches)
            for r in range(n_gpus)
        ],
        tag="g40_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g40_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g40_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for d in hf_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    phase("g40_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
