"""G35: Crash-Shield, Downside Left-Tail Floor & Anti-Whipsaw Mastery (Quad-Family L=45 + dep_s2 + s1/s2 stats).

Directly targets the V-shape crash vulnerability observed during global risk-asset selloffs:
- Fits strictly <= 2026-07-02 (train <= 2026-04-30, val_sel 2026-05-18..2026-07-02) on 4 path-dependent targets:
  1. floor10    : min_{k=1..10}(close[t+k]/close[t] - 1)   (left-tail crash survival floor)
  2. cvar10     : r10 - 0.75 * max(0, -floor10)             (downside-penalized return)
  3. nowhipsaw  : r10 - 0.60 * max(0, r5 - r10)             (anti-whipsaw vs days 6..10 crash)
  4. calmar10   : r10 / max(maxdd10, 0.03)                  (10-day path Calmar ratio)
- Introduces `crashentry` weighting scheme: 4x weight on pre-crash bull-trap days (breadth >= 0.30 where market drops >= 3.5%).
- Trains 3-seed DualRewardTailShieldMLP (predicting both expected return & tail floor) + 2 LightGBM tail specialists.
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
    zscore_cross_section_torch,
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
    SL_STATS,
    compute_quad_context_14d,
    compute_wave4_date_weights,
    extract_10d_path_targets,
    fwd_date_batched,
    load_quad_models,
    make_c2_paths_dict,
    pca_quad_worker,
    project_pca_4,
    solve_ridge_specs_from_disk,
    top_evecs_from_pts,
    train_lgbm_regressor,
    fit_lgbm_regressor_isolated,
)

OUTPUT = Path(os.environ.get("G35_OUTPUT", "/kaggle/working/g35_out"))
INPUT = Path(os.environ.get("G35_INPUT", "/kaggle/input"))
TMP_DIR = Path("/kaggle/working/_tmp_g35")

TRAIN_START_G35 = "2021-01-01"
TRAIN_STRIDE_G35 = 2
SCHEMES = ("std", "regime", "crashentry")
TARGETS = ("multi", "floor10", "cvar10", "nowhipsaw", "calmar10")
ALPHAS_SINGLE = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_QUAD = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)
ALPHAS_STATS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)

ANCHOR_KEYS = [
    "mini_cvar",
    "base_cvar",
    "c2_cvar",
    "quad_multi",
    "quad_floor10",
    "quad_cvar10",
    "quad_nowhipsaw",
    "quad_calmar10",
    "quad_crashentry_cvar",
]
N_ANCHORS = len(ANCHOR_KEYS)
IN_DIM = N_ANCHORS + PCA_DIM + CTX_DIM  # 9 + 192 + 14 = 215


def build_tail_targets(path_d: dict[str, np.ndarray], y_multi: np.ndarray) -> dict[str, np.ndarray]:
    """Compute standardized cross-sectional targets strictly from [ti..ti+10] path outcomes."""
    r5 = path_d["r5"]
    r10 = path_d["r10"]
    floor10 = path_d["floor10"]
    maxdd10 = path_d["maxdd10"]
    cvar_raw = r10 - 0.75 * np.maximum(0.0, -floor10)
    nowhipsaw_raw = r10 - 0.60 * np.maximum(0.0, r5 - r10)
    calmar_raw = r10 / np.maximum(maxdd10, 0.03)
    return {
        "multi": y_multi.astype(np.float32),
        "floor10": standardize_1d(rank_pct_centered(floor10)).astype(np.float32),
        "cvar10": standardize_1d(rank_pct_centered(cvar_raw)).astype(np.float32),
        "nowhipsaw": standardize_1d(rank_pct_centered(nowhipsaw_raw)).astype(np.float32),
        "calmar10": standardize_1d(rank_pct_centered(calmar_raw)).astype(np.float32),
        "cvar_raw": cvar_raw.astype(np.float64),
        "nowhipsaw_raw": nowhipsaw_raw.astype(np.float64),
        "floor10_raw": floor10.astype(np.float64),
    }


class DualRewardTailShieldMLP(nn.Module):
    """Regime-modulated dual-head residual MLP jointly predicting expected reward and left-tail floor."""

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.gate_mu = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, N_ANCHORS))
        self.gate_fl = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, N_ANCHORS))
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
        base_mu = (x[:, :N_ANCHORS] * F.softmax(self.gate_mu(mkt), dim=1)).sum(dim=1)
        base_fl = (x[:, :N_ANCHORS] * F.softmax(self.gate_fl(mkt), dim=1)).sum(dim=1)
        h = F.silu(self.fc1(self.drop(self.in_norm(x))))
        g, b = self.film1(mkt).chunk(2, dim=1)
        h = h * (1.0 + 0.25 * torch.tanh(g)) + 0.25 * b
        h = F.silu(self.fc2(self.drop(h)))
        d = self.drop(h)
        mu = base_mu + torch.tanh(self.res_mu) * self.head_mu(d).squeeze(-1)
        fl = base_fl + torch.tanh(self.res_fl) * self.head_fl(d).squeeze(-1)
        return mu, fl


def tail_penalized_cosic_loss(pred: torch.Tensor, target_z: torch.Tensor, floor_z: torch.Tensor) -> torch.Tensor:
    """Asymmetric Cosine-IC focusing extra gradient weight on both bottom crash tail and upper winners."""
    w = 1.0 + 0.6 * torch.sigmoid(-2.0 * floor_z) + 0.4 * torch.sigmoid(2.0 * target_z)
    w = w / w.sum()
    dp = pred - (w * pred).sum()
    dt = target_z - (w * target_z).sum()
    vp = (w * dp * dp).sum().clamp_min(1e-8).sqrt()
    vt = (w * dt * dt).sum().clamp_min(1e-8).sqrt()
    return 1.0 - (w * dp * dt).sum() / (vp * vt)


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
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, _r10 = b
            path_d = extract_10d_path_targets(sym_tr, s_idx, t_idx)
            tgt = build_tail_targets(path_d, y_multi)
            ctx_np, st = compute_quad_context_14d(sym_tr, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(
                str(d_str), len(s_idx), st["b20"], float(np.mean(path_d["r10"])), float(np.mean(path_d["r5"])),
            )

            x32 = fwd_date_batched(ext, sym_tr, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            xx = x64.T @ x64
            xys = {t: x64.T @ torch.as_tensor(tgt[t], device=device, dtype=torch.float64) for t in TARGETS}
            for s in SCHEMES:
                w = w_map[s]
                grams[s].add_(xx, alpha=w)
                for t in TARGETS:
                    rhs[s][t].add_(xys[t], alpha=w)
            n_rows += int(x32.shape[0])

            z_pca = project_pca_4(x32, u_m, u_b, u_p, u_c).half().cpu().numpy()
            tr_recs.append({
                "date": str(d_str),
                "z_pca": z_pca,
                "ctx": ctx_np,
                "y_cvar": tgt["cvar10"],
                "y_floor": tgt["floor10"],
                "y_nowhipsaw": tgt["nowhipsaw"],
                "y_multi": tgt["multi"],
                "weight_crash": float(w_map["crashentry"]),
            })
            if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
                phase("g35_extract_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

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
        my_val = [b for i, b in enumerate(val_batches) if i % n_gpus == rank]
        for b in my_val:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, _r10 = b
            path_d = extract_10d_path_targets(sym_val, s_idx, t_idx)
            tgt = build_tail_targets(path_d, y_multi)
            ctx_np, st = compute_quad_context_14d(sym_val, s_idx, t_idx, pct_np)
            w_map = compute_wave4_date_weights(
                str(d_str), len(s_idx), st["b20"], float(np.mean(path_d["r10"])), float(np.mean(path_d["r5"])),
            )

            x32 = fwd_date_batched(ext, sym_val, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.to(torch.float64)
            xx = x64.T @ x64
            xys = {t: x64.T @ torch.as_tensor(tgt[t], device=device, dtype=torch.float64) for t in TARGETS}
            for s in SCHEMES:
                w = w_map[s]
                grams[s].add_(xx, alpha=w)
                for t in TARGETS:
                    rhs[s][t].add_(xys[t], alpha=w)

            if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
                z_pca = project_pca_4(x32, u_m, u_b, u_p, u_c).half().cpu().numpy()
                xf_path = str(TMP_DIR / f"vsel_{d_str}_r{rank}.npy")
                np.save(xf_path, x32.half().cpu().numpy())
                val_sel_recs.append({
                    "date": str(d_str),
                    "x_full_path": xf_path,
                    "z_pca": z_pca,
                    "ctx": ctx_np,
                    "r10": path_d["r10"].astype(np.float32),
                    "cvar_raw": tgt["cvar_raw"],
                    "nowhipsaw_raw": tgt["nowhipsaw_raw"],
                    "floor10_raw": tgt["floor10_raw"],
                    "calmar_raw": (path_d["r10"] / np.maximum(path_d["maxdd10"], 0.03)).astype(np.float64),
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
                phase("g35_oof_progress", rank=rank, done=bi + 1, total=len(my_tr))
        out = str(TMP_DIR / f"oof_r{rank}.pt")
        torch.save(out_scores, out)
        return out
    finally:
        ext.close()


def _train_dual_nn_worker(args: tuple) -> dict:
    rank, jobs, train_pt, val_pt = args
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    train_days = torch.load(train_pt, map_location="cpu", weights_only=False)
    val_days = torch.load(val_pt, map_location="cpu", weights_only=False)

    results = {}
    for name, loss_mode, epochs, lr, wd in jobs:
        best_states, val_ics = [], []
        for seed in (42, 102, 202):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = DualRewardTailShieldMLP(IN_DIM).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
            best_ic, best_sd = -1e9, None
            for _ep in range(epochs):
                m.train()
                for idx in np.random.permutation(len(train_days)):
                    x_np, yc_np, yf_np, ynw_np, ym_np, w_crash = train_days[idx]
                    xt = torch.as_tensor(x_np, device=device, dtype=torch.float32)
                    yc = torch.as_tensor(yc_np, device=device, dtype=torch.float32)
                    yf = torch.as_tensor(yf_np, device=device, dtype=torch.float32)
                    ynw = torch.as_tensor(ynw_np, device=device, dtype=torch.float32)
                    ym = torch.as_tensor(ym_np, device=device, dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    mu, fl = m(xt)
                    if loss_mode == "cvar_shield":
                        loss = 0.65 * tail_penalized_cosic_loss(mu, yc, yf) + 0.35 * cosine_ic_loss_vec(fl, yf)
                        loss = loss * min(w_crash, 2.5)
                    else:
                        loss = 0.50 * cosine_ic_loss_vec(mu, ynw) + 0.25 * cosine_ic_loss_vec(mu, ym) + 0.25 * cosine_ic_loss_vec(fl, yf)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()

                m.eval()
                with torch.inference_mode():
                    preds, dates, rets = [], [], []
                    for d_str, xv, cvar_r, _nw_r in val_days:
                        mu, fl = m(torch.as_tensor(xv, device=device, dtype=torch.float32))
                        comb = (mu + 0.35 * fl).cpu().numpy()
                        preds.append(comb)
                        dates.append(np.full(len(cvar_r), d_str))
                        rets.append(cvar_r)
                    ic = daily_spearman_ic(np.concatenate(dates), np.concatenate(preds), np.concatenate(rets))
                if ic > best_ic:
                    best_ic = ic
                    best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            assert best_sd is not None
            best_states.append(best_sd)
            val_ics.append(best_ic)
        results[name] = {"states": best_states, "val_ic": float(np.mean(val_ics))}
        phase("g35_nn_done", rank=rank, model_name=name, val_ic=round(float(np.mean(val_ics)), 4))
    return results


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, paths, u_dict, full_heads, nn_specs, lgbm_cvar_path, lgbm_nw_path, lgbm_cols, sym_oos, oos_batches = args
    import lightgbm as lgb

    device, ext = load_quad_models(rank, paths)
    try:
        u_m = torch.as_tensor(u_dict["m"], device=device, dtype=torch.float32)
        u_b = torch.as_tensor(u_dict["b"], device=device, dtype=torch.float32)
        u_p = torch.as_tensor(u_dict["p"], device=device, dtype=torch.float32)
        u_c = torch.as_tensor(u_dict["c"], device=device, dtype=torch.float32)
        heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in full_heads.items()}

        nn_loaded: dict[str, list[nn.Module]] = {}
        for name, states in nn_specs.items():
            mods = []
            for sd in states:
                m = DualRewardTailShieldMLP(IN_DIM).to(device)
                m.load_state_dict(sd)
                m.eval()
                mods.append(m)
            nn_loaded[name] = mods

        b_cvar = lgb.Booster(model_file=lgbm_cvar_path)
        b_nw = lgb.Booster(model_file=lgbm_nw_path)

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

            df_d = pd.DataFrame({
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
                "asof_date": d_str,
                "p_g35_ridge_cvar10": raw_s["quad_cvar10"],
                "p_g35_ridge_floor10": raw_s["quad_floor10"],
                "p_g35_ridge_nowhipsaw": raw_s["quad_nowhipsaw"],
                "p_g35_ridge_calmar10": raw_s["quad_calmar10"],
                "p_g35_ridge_crashentry": raw_s["quad_crashentry_cvar"],
                "p_g35_s1s2_cvar10": raw_s["stats_cvar10"],
            })
            for name, mods in nn_loaded.items():
                mus, fls = [], []
                for m in mods:
                    mu, fl = m(xt)
                    mus.append(mu.cpu().numpy())
                    fls.append(fl.cpu().numpy())
                mu_m = standardize_1d(np.mean(mus, axis=0)).astype(np.float32)
                fl_m = standardize_1d(np.mean(fls, axis=0)).astype(np.float32)
                df_d[f"p_{name}_mu"] = mu_m
                df_d[f"p_{name}_floor"] = fl_m
                df_d[f"p_{name}_shield35"] = (mu_m + 0.35 * fl_m).astype(np.float32)
                df_d[f"p_{name}_shield70"] = (mu_m + 0.70 * fl_m).astype(np.float32)

            df_d["p_g35_lgbm_cvar10"] = b_cvar.predict(x_feat[:, lgbm_cols], num_threads=1).astype(np.float32)
            df_d["p_g35_lgbm_nowhipsaw"] = b_nw.predict(x_feat[:, lgbm_cols], num_threads=1).astype(np.float32)
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
    phase("g35_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    paths = make_c2_paths_dict(INPUT)

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G35, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G35]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    # Pass 0: PCA-48 per backbone on 40 evenly spaced dates <= 2026-04-30
    step = max(1, len(tr_batches) // 40)
    pca_pts = run_gpu_workers_failfast(
        pca_quad_worker, [(r, n_gpus, paths, sym_tr, tr_batches[::step][:40], "g35") for r in range(n_gpus)], tag="g35_pca"
    )
    u_dict = top_evecs_from_pts(pca_pts, K_PCA)
    phase("g35_pca_ready", elapsed_sec=round(time.time() - t0, 1))

    # Pass 1: Extract stats & dump Gram/RHS per scheme to disk
    main_pts = run_gpu_workers_failfast(
        _main_extract_worker,
        [(r, n_gpus, paths, u_dict, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        tag="g35_main",
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
    val_sel_recs.sort(key=lambda r: r["date"])

    x_sel = np.concatenate([np.load(r["x_full_path"]).astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["r10"]), r["date"]) for r in val_sel_recs])
    sel_targets = {
        "multi": np.concatenate([r["r10"].astype(np.float64) for r in val_sel_recs]),
        "floor10": np.concatenate([r["floor10_raw"] for r in val_sel_recs]),
        "cvar10": np.concatenate([r["cvar_raw"] for r in val_sel_recs]),
        "nowhipsaw": np.concatenate([r["nowhipsaw_raw"] for r in val_sel_recs]),
        "calmar10": np.concatenate([r["calmar_raw"] for r in val_sel_recs]),
    }

    ridge_specs = [
        ("mini_cvar", SL_MINI, "std", "cvar10", ALPHAS_SINGLE),
        ("base_cvar", SL_BASE, "std", "cvar10", ALPHAS_SINGLE),
        ("c2_cvar", SL_C2, "std", "cvar10", ALPHAS_SINGLE),
        ("stats_cvar10", SL_STATS, "std", "cvar10", ALPHAS_STATS),
        ("quad_multi", SL_ALL, "std", "multi", ALPHAS_QUAD),
        ("quad_floor10", SL_ALL, "std", "floor10", ALPHAS_QUAD),
        ("quad_cvar10", SL_ALL, "std", "cvar10", ALPHAS_QUAD),
        ("quad_nowhipsaw", SL_ALL, "regime", "nowhipsaw", ALPHAS_QUAD),
        ("quad_calmar10", SL_ALL, "std", "calmar10", ALPHAS_QUAD),
        ("quad_crashentry_cvar", SL_ALL, "crashentry", "cvar10", ALPHAS_QUAD),
    ]
    selection, full_heads, tr_heads, fold_opp_heads = solve_ridge_specs_from_disk(
        fold_meta, SCHEMES, TARGETS, ridge_specs, ANCHOR_KEYS, x_sel, d_sel, sel_targets
    )
    del fold_meta, x_sel, d_sel, sel_targets
    gc.collect()
    phase("g35_ridge_ready", elapsed_sec=round(time.time() - t0, 1))

    # Pass 2: 2-fold OOF anchor projection
    oof_pts = run_gpu_workers_failfast(
        _project_oof_worker, [(r, n_gpus, paths, fold_opp_heads[r], sym_tr, tr_batches) for r in range(n_gpus)], tag="g35_oof"
    )
    oof_map: dict[str, np.ndarray] = {}
    for p in oof_pts:
        oof_map.update(torch.load(p, map_location="cpu", weights_only=False))
        Path(p).unlink(missing_ok=True)
    del sym_tr, tr_batches, sym_val, val_batches
    gc.collect()

    train_days = []
    for rec in tr_recs:
        xf = np.concatenate([oof_map[rec["date"]], rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        train_days.append((xf, rec["y_cvar"], rec["y_floor"], rec["y_nowhipsaw"], rec["y_multi"], rec["weight_crash"]))
    tr_recs.clear()
    oof_map.clear()

    val_days = []
    for rec in val_sel_recs:
        xf_full = np.load(rec["x_full_path"]).astype(np.float32)
        Path(rec["x_full_path"]).unlink(missing_ok=True)
        anc = np.column_stack([standardize_1d(xf_full[:, tr_heads[k][0]] @ tr_heads[k][1]).astype(np.float32) for k in ANCHOR_KEYS])
        xf = np.concatenate([anc, rec["z_pca"].astype(np.float32), rec["ctx"]], axis=1).astype(np.float32)
        val_days.append((rec["date"], xf, rec["cvar_raw"], rec["nowhipsaw_raw"]))
    val_sel_recs.clear()

    train_pt, val_pt = str(TMP_DIR / "train_days.pt"), str(TMP_DIR / "val_days.pt")
    torch.save(train_days, train_pt)
    torch.save(val_days, val_pt)

    jobs_g0 = [("g35_tailshield_cvar", "cvar_shield", 18, 1.4e-3, 2e-2)]
    jobs_g1 = [("g35_tailshield_nowhipsaw", "nowhipsaw", 18, 1.4e-3, 2e-2)]
    nn_jobs = [(0, jobs_g0, train_pt, val_pt), (1 % n_gpus, jobs_g1, train_pt, val_pt)] if n_gpus >= 2 else [(0, jobs_g0 + jobs_g1, train_pt, val_pt)]
    nn_parts = run_gpu_workers_failfast(_train_dual_nn_worker, nn_jobs, tag="g35_nn")
    nn_specs = {}
    for part in nn_parts:
        for name, res in part.items():
            nn_specs[name] = res["states"]
            selection[name] = {"val_ic_cvar": res["val_ic"]}
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

    y_tr_cvar = np.concatenate([d[1] for d in train_days], axis=0)
    y_va_cvar_z = np.concatenate([standardize_1d(rank_pct_centered(d[2])) for d in val_days], axis=0)
    r_va_cvar = np.concatenate([d[2] for d in val_days])
    lgbm_cvar_path = str(OUTPUT / "g35_lgbm_cvar.txt")
    selection["g35_lgbm_cvar10"] = fit_lgbm_regressor_isolated(
        x_tr_lgb, y_tr_cvar, x_va_lgb, y_va_cvar_z, d_va_lgb, r_va_cvar, lgbm_cvar_path, seed=20261127
    )

    y_tr_nw = np.concatenate([d[3] for d in train_days], axis=0)
    y_va_nw_z = np.concatenate([standardize_1d(rank_pct_centered(d[3])) for d in val_days], axis=0)
    r_va_nw = np.concatenate([d[3] for d in val_days])
    lgbm_nw_path = str(OUTPUT / "g35_lgbm_nowhipsaw.txt")
    selection["g35_lgbm_nowhipsaw"] = fit_lgbm_regressor_isolated(
        x_tr_lgb, y_tr_nw, x_va_lgb, y_va_nw_z, d_va_lgb, r_va_nw, lgbm_nw_path, seed=20261128
    )
    del train_days, val_days, x_tr_lgb, x_va_lgb
    gc.collect()

    np.savez(OUTPUT / "g35_heads.npz", u_mini=u_dict["m"], u_base=u_dict["b"], u_pts=u_dict["p"], u_c2=u_dict["c"], **{k: v[1] for k, v in full_heads.items()})
    torch.save({"nn_specs": nn_specs, "lgbm_cols": lgbm_cols}, OUTPUT / "g35_models.pt")
    (OUTPUT / "g35_selection_partial.json").write_text(
        json.dumps({"n_train_rows": n_tr_rows, "selection": selection}, indent=2, ensure_ascii=False)
    )

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [(r, n_gpus, paths, u_dict, full_heads, nn_specs, lgbm_cvar_path, lgbm_nw_path, lgbm_cols, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g35_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g35_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g35_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for d in ("hf_kronos_mini", "hf_tok2k", "hf_kronos_base", "hf_kronos_small", "hf_tok_base"):
        shutil.rmtree(Path("/kaggle/working") / d, ignore_errors=True)
    phase("g35_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
