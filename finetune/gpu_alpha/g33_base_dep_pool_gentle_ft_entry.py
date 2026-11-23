"""G33: Kronos-base (102M) Deep Mastery — dep_layer (s2) + Rich Temporal Pooling (L=45 & L=52)
+ ret20/volnorm10/decay365/regime Targets + FIRST-EVER Ultra-Gentle Top-2-Block Cosine-IC FT of Kronos-base!

Part 1 (Full-density 2021-2026 stride=1, 5.83M rows):
- Base L=45 (8 x 832 = 6656d): [L5, L8, L10_pre, L11_last, L11_m5, L11_m10, L11_diff5, dep_s2]
- Base L=52 (4 x 832 = 3328d): [L8, L11_last, L11_m5, L11_diff5]
- Targets: raw (r10), multi, ret5, volnorm10 (Sharpe-normalized), ret20 (20-day forward return)
- Schemes: std, decay365, regime

Part 2 (Parallel 2x T4 Ultra-Gentle Top-2-Block Cosine-IC FT of Kronos-base 102M):
- Arm 0 (GPU 0): Base L=45 (freeze_until=10, KL+L2-SP anchor) + post-FT pooled Ridge (4160d).
- Arm 1 (GPU 1): Base L=52 (freeze_until=10, KL+L2-SP anchor) + post-FT pooled Ridge (4160d).

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
from g4_base_kaggle_entry import _load_kronos_base
from g28_l45_gentle_ft_entry import _apply_top_state
from short_probe_common import detect_gpu_count, norm_window, run_gpu_workers_failfast

OUTPUT = Path(os.environ.get("G33_OUTPUT", "/kaggle/working/g33_out"))
INPUT = Path(os.environ.get("G33_INPUT", "/kaggle/input"))

D_BASE = 832
# B45 layout (8 * 832 = 6656): [L5, L8, L10_pre, L11_last, L11_m5, L11_m10, L11_diff5, dep_s2]
B45_DIM = D_BASE * 8  # 6656
SL_B45_DEP = slice(D_BASE * 7, D_BASE * 8)          # 832
SL_B45_RICH = slice(0, D_BASE * 7)                  # 5824
SL_B45_ALL = slice(0, B45_DIM)                      # 6656

# B52 layout (4 * 832 = 3328): [L8, L11_last, L11_m5, L11_diff5]
B52_DIM = D_BASE * 4  # 3328
SL_B52_ALL = slice(0, B52_DIM)

SCHEMES = ("std", "decay365", "regime")
TARGETS = ("raw", "multi", "ret5", "volnorm10", "ret20")
ALPHAS_BASE = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)
TRAIN_START_G33 = "2021-01-01"


def compute_weights_g33(d_str: str, n_stocks: int, breadth: float) -> dict[str, float]:
    dt = datetime.strptime(d_str, "%Y-%m-%d")
    dt_ref = datetime.strptime("2026-07-02", "%Y-%m-%d")
    age = max(0.0, float((dt_ref - dt).days))
    eq_day = 4500.0 / float(max(n_stocks, 1))
    return {
        "std": 1.0,
        "decay365": float(0.5 ** (age / 365.0)),
        "regime": float(eq_day * (3.0 if breadth < 0.35 else 1.0)),
    }


def extract_r5_r20_vol_breadth(sym_arr, s_idx: np.ndarray, t_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(s_idx)
    r5 = np.empty(n, dtype=np.float64)
    r20 = np.empty(n, dtype=np.float64)
    vol20 = np.empty(n, dtype=np.float64)
    above = 0
    for i in range(n):
        vals = sym_arr[int(s_idx[i])][0]
        ti = int(t_idx[i])
        max_i = len(vals) - 1
        c = np.maximum(np.nan_to_num(vals[ti - 20 : ti + 1, 3].astype(np.float64), nan=1.0), 1e-8)
        c0 = c[-1]
        c5 = max(float(vals[min(ti + 5, max_i), 3]), 1e-8)
        c20 = max(float(vals[min(ti + 20, max_i), 3]), 1e-8)
        r5[i] = c5 / c0 - 1.0
        r20[i] = c20 / c0 - 1.0
        vol20[i] = max(float(np.std(np.diff(np.log(c)))), 0.005)
        if c0 > float(c[1:].mean()):
            above += 1
    r5 = np.nan_to_num(r5, nan=0.0, posinf=0.0, neginf=0.0)
    r20 = np.nan_to_num(r20, nan=0.0, posinf=0.0, neginf=0.0)
    return r5, r20, vol20, float(above) / float(max(n, 1))


class BaseDepPoolExtractor:
    """Extract Kronos-base L=45 (7 rich blocks + dep_s2) and L=52 (4 rich blocks)."""

    def __init__(self, base_m, tok_base, device: torch.device):
        self.base = base_m
        self.tok = tok_base
        self.device = device
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            base_m.transformer[5].register_forward_hook(self._mk("b5")),
            base_m.transformer[8].register_forward_hook(self._mk("b8")),
            base_m.transformer[10].register_forward_hook(self._mk("b10")),
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
            tb52 = self.tok.encode(norm_window(raw_t[:, -52:]), half=True)
            _, c52 = self.base.decode_s1(tb52[0], tb52[1], st_t[:, -52:], sector_id=sec_t, size_percentile=pct_t)
            b52_l8 = self._cap["b8"]
            c52_f = c52.float()
            b52_last = c52_f[:, -1, :]
            b52_m5 = c52_f[:, -5:, :].mean(dim=1)
            b52_diff5 = b52_last - b52_m5

            tb45 = self.tok.encode(norm_window(raw_t[:, -45:]), half=True)
            s1_logits, c45 = self.base.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            pred_s1 = s1_logits.argmax(dim=-1)
            sib = self.base.embedding.emb_s1(pred_s1)
            dep45 = self.base.dep_layer(c45, sib)[:, -1, :].float()

        c45_f = c45.float()
        b45_last = c45_f[:, -1, :]
        b45_m5 = c45_f[:, -5:, :].mean(dim=1)
        b45_m10 = c45_f[:, -10:, :].mean(dim=1)
        b45_diff5 = b45_last - b45_m5

        x45 = torch.cat([self._cap["b5"], self._cap["b8"], self._cap["b10"], b45_last, b45_m5, b45_m10, b45_diff5, dep45], dim=1)
        x52 = torch.cat([b52_l8, b52_last, b52_m5, b52_diff5], dim=1)
        return x45, x52

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_base(rank: int, base_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    base = _load_kronos_base(base_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, BaseDepPoolExtractor(base, tok, device)


@torch.inference_mode()
def _part1_extract_worker(args: tuple) -> str:
    rank, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_base(rank, base_dir, tok_dir)
    try:
        dims = {"b45": B45_DIM, "b52": B52_DIM}
        grams = {b: {s: torch.zeros((d, d), dtype=torch.float64, device=device) for s in SCHEMES} for b, d in dims.items()}
        rhss = {b: {s: {t: torch.zeros(d, dtype=torch.float64, device=device) for t in TARGETS} for s in SCHEMES} for b, d in dims.items()}

        def _fwd(sym_arr, s_idx, t_idx, sec_np, pct_np, batch_size: int = 256):
            p45, p52 = [], []
            for off in range(0, len(s_idx), batch_size):
                sl = slice(off, min(off + batch_size, len(s_idx)))
                raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                x45, x52 = ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl])
                p45.append(x45)
                p52.append(x52)
            return (
                zscore_cross_section_torch(torch.cat(p45, dim=0), clip_val=5.0),
                zscore_cross_section_torch(torch.cat(p52, dim=0), clip_val=5.0),
            )

        def _accum(x45, x52, y_map, w_map):
            for blk, xt in (("b45", x45), ("b52", x52)):
                x64 = xt.to(torch.float64)
                xx = x64.T @ x64
                xys = {t: x64.T @ y_map[t] for t in TARGETS}
                for s in SCHEMES:
                    w = w_map[s]
                    grams[blk][s].add_(xx, alpha=w)
                    for t in TARGETS:
                        rhss[blk][s][t].add_(xys[t], alpha=w)

        def _flush(prefix: str) -> tuple[dict, dict]:
            gp, rp = {}, {}
            for b in dims:
                gp[b], rp[b] = {}, {}
                for s in SCHEMES:
                    p = f"/tmp/g33_{prefix}_g_{b}_{s}_r{rank}.npy"
                    np.save(p, grams[b][s].cpu().numpy())
                    gp[b][s] = p
                    grams[b][s].zero_()
                    rp[b][s] = {}
                    for t in TARGETS:
                        pt = f"/tmp/g33_{prefix}_r_{b}_{s}_{t}_r{rank}.npy"
                        np.save(pt, rhss[b][s][t].cpu().numpy())
                        rp[b][s][t] = pt
                        rhss[b][s][t].zero_()
            return gp, rp

        n_rows = 0
        my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
        for bi, b in enumerate(my_tr):
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            r5, r20, vol20, breadth = extract_r5_r20_vol_breadth(sym_tr, s_idx, t_idx)
            w_map = compute_weights_g33(str(d_str), len(s_idx), breadth)
            x45, x52 = _fwd(sym_tr, s_idx, t_idx, sec_np, pct_np)
            y_map = {
                "raw": torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
                "ret5": torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64),
                "volnorm10": torch.as_tensor(standardize_1d(rank_pct_centered(r10 / vol20)), device=device, dtype=torch.float64),
                "ret20": torch.as_tensor(standardize_1d(rank_pct_centered(r20)), device=device, dtype=torch.float64),
            }
            _accum(x45, x52, y_map, w_map)
            n_rows += int(x45.shape[0])
            if (bi + 1) % 80 == 0 or bi + 1 == len(my_tr):
                phase("g33_p1_train", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

        tr_gp, tr_rp = _flush("tr")

        val_sel_recs = []
        for b in [b for i, b in enumerate(val_batches) if i % n_gpus == rank]:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
            r5, r20, vol20, breadth = extract_r5_r20_vol_breadth(sym_val, s_idx, t_idx)
            w_map = compute_weights_g33(str(d_str), len(s_idx), breadth)
            x45, x52 = _fwd(sym_val, s_idx, t_idx, sec_np, pct_np)
            y_map = {
                "raw": torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64),
                "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
                "ret5": torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64),
                "volnorm10": torch.as_tensor(standardize_1d(rank_pct_centered(r10 / vol20)), device=device, dtype=torch.float64),
                "ret20": torch.as_tensor(standardize_1d(rank_pct_centered(r20)), device=device, dtype=torch.float64),
            }
            _accum(x45, x52, y_map, w_map)
            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                p45 = f"/tmp/g33_vsel_45_{d_str}_r{rank}.npy"
                p52 = f"/tmp/g33_vsel_52_{d_str}_r{rank}.npy"
                np.save(p45, x45.half().cpu().numpy())
                np.save(p52, x52.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "x45_path": p45, "x52_path": p52,
                    "ret10": r10.astype(np.float32),
                    "ret5": r5.astype(np.float32),
                })

        va_gp, va_rp = _flush("va")
        out = f"/tmp/g33_p1_meta_r{rank}.pt"
        torch.save({
            "n_rows": n_rows,
            "tr_gp": tr_gp, "tr_rp": tr_rp,
            "va_gp": va_gp, "va_rp": va_rp,
            "val_sel_recs": val_sel_recs,
        }, out)
        phase("g33_p1_saved", rank=rank, n_rows=n_rows, n_vsel=len(val_sel_recs))
        return out
    finally:
        ext.close()


@torch.inference_mode()
def _part1_oos_worker(args: tuple) -> str:
    rank, n_gpus, base_dir, tok_dir, heads_45, heads_52, sym_oos, oos_batches = args
    device, ext = _load_base(rank, base_dir, tok_dir)
    try:
        h45_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_45.items()}
        h52_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_52.items()}
        frames = []
        for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
            p45, p52 = [], []
            for off in range(0, len(recs), 256):
                sl = recs[off : off + 256]
                raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119 : r[1] + 1] for r in sl])
                st_b = np.stack([sym_oos[r[0]][1][r[1] - 119 : r[1] + 1] for r in sl])
                sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
                pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
                x45, x52 = ext.extract_batch(raw_b, st_b, sec_b, pct_b)
                p45.append(x45)
                p52.append(x52)
            x45_z = zscore_cross_section_torch(torch.cat(p45, dim=0), clip_val=5.0)
            x52_z = zscore_cross_section_torch(torch.cat(p52, dim=0), clip_val=5.0)
            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
            })
            for k, (sl, wt) in h45_t.items():
                df_d[f"p_{k}"] = (x45_z[:, sl] @ wt).cpu().numpy().astype(np.float32)
            for k, (sl, wt) in h52_t.items():
                df_d[f"p_{k}"] = (x52_z[:, sl] @ wt).cpu().numpy().astype(np.float32)
            frames.append(df_d)
        out = f"/tmp/g33_p1_oos_r{rank}.csv.gz"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False, compression="gzip")
        return out
    finally:
        ext.close()


# ---- Part 2: Parallel 2x T4 Ultra-Gentle Top-2-Block Cosine-IC FT of Kronos-base (102M) ----

@torch.no_grad()
def _warm_start_base(adapter: TopBlockRankAdapter, tok, sym_tr, tr_batches, lb: int, device) -> None:
    adapter.eval()
    xs, ys = [], []
    step = max(1, len(tr_batches) // 50)
    rng = np.random.default_rng(20261121 + lb)
    for b in tr_batches[::step][:50]:
        _, s_idx, t_idx, sec_np, pct_np, y_multi, _, _ = b
        n = len(s_idx)
        take = rng.choice(n, size=min(96, n), replace=False)
        raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
        st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
        h_low = adapter.encode_frozen_lower(
            tok, torch.as_tensor(raw_b, device=device, dtype=torch.float32),
            torch.as_tensor(st_b, device=device, dtype=torch.float32),
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = h_low
            sec_t = torch.as_tensor(sec_np[take], device=device, dtype=torch.long)
            pct_t = torch.as_tensor(pct_np[take], device=device, dtype=torch.float32)
            for idx in range(adapter.freeze_until, len(adapter.model.transformer)):
                if adapter.model.context_layer > 0 and idx == adapter.model.context_layer:
                    h = adapter.model._add_context(h, sector_id=sec_t, size_percentile=pct_t)
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


def _base_ft_arm_worker(args: tuple) -> str:
    rank, arm_name, lb, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = _load_kronos_base(base_dir, device)
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    for p in tok.parameters():
        p.requires_grad = False

    val_sel_batches = [b for b in val_batches if VAL_SEL_START <= b[0] <= VAL_SEL_END]
    ft_batches = tr_batches[::3]
    adapter = TopBlockRankAdapter(model, freeze_until=10).to(device)
    _warm_start_base(adapter, tok, sym_tr, ft_batches, lb, device)
    init_m, _ = evaluate_adapter_on_batches(adapter, tok, sym_val, val_sel_batches, device, (lb,), eval_batch=192)
    phase("g33_ft_init", arm=arm_name, lb=lb, **init_m)

    top_params = [p for _, p in adapter.model.named_parameters() if p.requires_grad]
    head_params = list(adapter.rank_norm.parameters()) + list(adapter.rank_head.parameters())
    opt = torch.optim.AdamW([
        {"params": top_params, "lr": 1.0e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 1.0e-4, "weight_decay": 1e-2},
    ])
    epochs = 2
    total_steps = epochs * len(ft_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=1e-6)

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
            take = rng.choice(n, size=min(192, n), replace=False) if n > 192 else np.arange(n)
            raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
            st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in take])
            h_low = adapter.encode_frozen_lower(
                tok, torch.as_tensor(raw_b, device=device, dtype=torch.float32),
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
                vm, _ = evaluate_adapter_on_batches(adapter, tok, sym_val, val_sel_batches, device, (lb,), eval_batch=192)
                rec = {"epoch": round(step / len(ft_batches), 2), "step": step, **vm}
                history.append(rec)
                if vm["val_ic_raw"] > best_ic:
                    best_ic = vm["val_ic_raw"]
                    best_state = adapter.extract_top_state_dict()
                phase("g33_ft_val", arm=arm_name, **rec)

    _apply_top_state(adapter, best_state)
    adapter.eval()
    model.eval()
    torch.save(best_state, OUTPUT / f"{arm_name}_best_top.pt")

    # Post-FT pooled Ridge on [L5, L8, L11_last, L11_m5, L11_diff5] = 4160d (stride=2 for speed)
    fdim = D_BASE * 5
    cap: dict[str, torch.Tensor] = {}
    hooks = [
        model.transformer[5].register_forward_hook(lambda _m, _i, o: cap.__setitem__("b5", o[:, -1, :].detach().float())),
        model.transformer[8].register_forward_hook(lambda _m, _i, o: cap.__setitem__("b8", o[:, -1, :].detach().float())),
    ]

    @torch.inference_mode()
    def _extract_ft(sym_arr, s_idx, t_idx, sec_np, pct_np, batch_size: int = 256) -> torch.Tensor:
        chunks = []
        for off in range(0, len(s_idx), batch_size):
            sl = slice(off, min(off + batch_size, len(s_idx)))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - lb + 1 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
            st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
            sec_t = torch.as_tensor(sec_np[sl], device=device, dtype=torch.long)
            pct_t = torch.as_tensor(pct_np[sl], device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                tb = tok.encode(norm_window(raw_t), half=True)
                _, ctx = model.decode_s1(tb[0], tb[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            cf = ctx.float()
            last = cf[:, -1, :]
            m5 = cf[:, -5:, :].mean(dim=1)
            chunks.append(torch.cat([cap["b5"], cap["b8"], last, m5, last - m5], dim=1))
        return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    g_tr = torch.zeros((fdim, fdim), dtype=torch.float64, device=device)
    b_tr = torch.zeros(fdim, dtype=torch.float64, device=device)
    for b in tr_batches[::2]:
        _, s_idx, t_idx, sec_np, pct_np, y_multi, _, _ = b
        x64 = _extract_ft(sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
        ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
        g_tr.addmm_(x64.T, x64)
        b_tr.addmv_(x64.T, ym)

    g_all, b_all = g_tr.clone(), b_tr.clone()
    sel_x, sel_d, sel_r = [], [], []
    for b in val_batches:
        d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _, r10 = b
        x32 = _extract_ft(sym_val, s_idx, t_idx, sec_np, pct_np)
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
    d_sel, r_sel = np.concatenate(sel_d), np.concatenate(sel_r)

    best_a, best_ric = None, -1e9
    for a in (1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9):
        w = solve_ridge_subblock(g_tr_np, b_tr_np, slice(0, fdim), a)
        ic = daily_spearman_ic(d_sel, x_sel @ w, r_sel)
        if ic > best_ric:
            best_ric, best_a = ic, a
    assert best_a is not None
    w_ft = solve_ridge_subblock(g_all_np, b_all_np, slice(0, fdim), best_a).astype(np.float32)
    wt = torch.as_tensor(w_ft, device=device, dtype=torch.float32)

    frames = []
    with torch.inference_mode():
        for d_str, recs in oos_batches:
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_b = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in recs], dtype=np.float32)
            x32 = _extract_ft(sym_oos, s_idx, t_idx, sec_b, pct_b)
            p_ridge = (x32 @ wt).cpu().numpy().astype(np.float32)
            frames.append(pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                f"p_{arm_name}_ridge_pool": p_ridge,
            }))
    for h in hooks:
        h.remove()
    out_csv = f"/tmp/g33_ft_{arm_name}_oos.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
    out_meta = f"/tmp/g33_ft_{arm_name}_meta.pt"
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
    phase("g33_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    base_dir, tok_dir = hf_dirs
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G33, TRAIN_FIT_END, min_cross_section=200)
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    # Part 1: Full-density Base L=45 (dep_s2 + rich pool) + L=52 (fail-fast & raw .npy!)
    p1_pts = run_gpu_workers_failfast(
        _part1_extract_worker,
        [(r, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        tag="g33_p1",
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

    x45_sel = np.concatenate([np.load(r["x45_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    x52_sel = np.concatenate([np.load(r["x52_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["ret10"]), r["date"]) for r in val_sel_recs])
    r10_sel = np.concatenate([r["ret10"].astype(np.float64) for r in val_sel_recs])
    r5_sel = np.concatenate([r["ret5"].astype(np.float64) for r in val_sel_recs])
    for r in val_sel_recs:
        Path(r["x45_path"]).unlink(missing_ok=True)
        Path(r["x52_path"]).unlink(missing_ok=True)
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
            for name, sl, _s, tkey in sch_specs:
                r_tr_s = sum(np.load(d["tr_rp"][blk][sch][tkey]) for d in metas)
                r_all_s = r_tr_s + sum(np.load(d["va_rp"][blk][sch][tkey]) for d in metas)
                xs = np.asarray(x_mat[:, sl], dtype=np.float64)
                best_a, best_ic = None, -1e9
                for a in ALPHAS_BASE:
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

    specs_45 = [
        ("g33_base_l45_dep_s2", SL_B45_DEP, "std", "raw"),
        ("g33_base_l45_rich_raw", SL_B45_RICH, "std", "raw"),
        ("g33_base_l45_rich_multi", SL_B45_RICH, "std", "multi"),
        ("g33_base_l45_all_dep_raw", SL_B45_ALL, "std", "raw"),
        ("g33_base_l45_all_dep_multi", SL_B45_ALL, "std", "multi"),
        ("g33_base_l45_all_dep_decay365", SL_B45_ALL, "decay365", "multi"),
        ("g33_base_l45_all_dep_regime", SL_B45_ALL, "regime", "multi"),
        ("g33_base_l45_all_dep_ret5", SL_B45_ALL, "std", "ret5"),
        ("g33_base_l45_all_dep_volnorm", SL_B45_ALL, "std", "volnorm10"),
        ("g33_base_l45_all_dep_ret20", SL_B45_ALL, "std", "ret20"),
    ]
    specs_52 = [
        ("g33_base_l52_rich_raw", SL_B52_ALL, "std", "raw"),
        ("g33_base_l52_rich_multi", SL_B52_ALL, "std", "multi"),
        ("g33_base_l52_rich_decay365", SL_B52_ALL, "decay365", "multi"),
        ("g33_base_l52_rich_regime", SL_B52_ALL, "regime", "multi"),
        ("g33_base_l52_rich_volnorm", SL_B52_ALL, "std", "volnorm10"),
        ("g33_base_l52_rich_ret20", SL_B52_ALL, "std", "ret20"),
    ]
    selection = {}
    s45, heads_45 = _fit_specs("b45", x45_sel, specs_45)
    s52, heads_52 = _fit_specs("b52", x52_sel, specs_52)
    selection.update(s45)
    selection.update(s52)
    del x45_sel, x52_sel, metas
    gc.collect()
    phase("g33_p1_heads_fitted", n_heads=len(selection), elapsed_sec=round(time.time() - t0, 1))

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    p1_csvs = run_gpu_workers_failfast(
        _part1_oos_worker,
        [(r, n_gpus, base_dir, tok_dir, heads_45, heads_52, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g33_p1_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in p1_csvs], ignore_index=True).sort_values(["asof_date", "symbol"]).reset_index(drop=True)
    for p in p1_csvs:
        Path(p).unlink(missing_ok=True)
    phase("g33_p1_oos_done", rows=len(oos_df), elapsed_sec=round(time.time() - t0, 1))

    # Part 2: Parallel 2x T4 Ultra-Gentle Top-2-Block FT of Kronos-base (L=45 & L=52)
    ft_args = [
        (0, "g33_base_l45_ft", 45, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),
        (1 % n_gpus, "g33_base_l52_ft", 52, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),
    ]
    ft_metas = run_gpu_workers_failfast(_base_ft_arm_worker, ft_args, tag="g33_ft")

    all_heads = {k: v[1] for k, v in {**heads_45, **heads_52}.items()}
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
    np.savez(OUTPUT / "g33_heads.npz", **all_heads)
    oos_df.to_csv(OUTPUT / "oos_g33_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_train_rows": n_tr_rows,
        "selection": selection,
        "ft_summary": ft_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g33_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g33_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
