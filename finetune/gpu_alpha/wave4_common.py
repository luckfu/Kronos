"""Shared RAM-safe Wave 4 helpers for Quad-Family extraction, PCA, 2-fold OOF anchors, and Ridge/NN fitting."""
from __future__ import annotations

import gc
import multiprocessing as mp
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from g1_multiscale_extract_and_fit import (
    daily_spearman_ic,
    phase,
    rank_pct_centered,
    solve_ridge_subblock,
    standardize_1d,
    zscore_cross_section_torch,
)
from g2_rank_finetune import cosine_ic_loss_vec
from g4_base_kaggle_entry import _load_kronos_base
from g16_pretrained_small_entry import _load_pretrained_small
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from g31_mini_mastery_c2pool_ft_entry import build_s1_token_prototypes
from g34_quad_dep_s1s2_logits_entry import _s1_s2_stats
from short_probe_common import find_c2_weights, norm_window

D_MINI = 256
D_BASE = 832
D_SMALL = 512

# Compact high-SNR Quad-Family + dep_s2 + s1/s2 analytical logit stats (FEAT_DIM = 8744):
# Mini  (1280): [m30_last, m45_l1, m45_last, m45_m5, m45_dep] (5 * 256)
# Base  (3328): [b45_l5, b45_l8, b45_last, b45_dep] (4 * 832)
# PTS   (2048): [p45_l5, p45_l6, p45_last, p45_dep] (4 * 512)
# C2    (2048): [c45_l5, c45_l6, c45_last, c45_dep] (4 * 512)
# Stats   (40): [st_mini(10), st_base(10), st_pts(10), st_c2(10)]
MINI_DIM = D_MINI * 5    # 1280
BASE_DIM = D_BASE * 4    # 3328
PTS_DIM = D_SMALL * 4    # 2048
C2_DIM = D_SMALL * 4     # 2048
STATS_DIM = 40
FEAT_DIM = MINI_DIM + BASE_DIM + PTS_DIM + C2_DIM + STATS_DIM  # 8744

SL_MINI = slice(0, MINI_DIM)
SL_BASE = slice(MINI_DIM, MINI_DIM + BASE_DIM)
SL_PTS = slice(MINI_DIM + BASE_DIM, MINI_DIM + BASE_DIM + PTS_DIM)
SL_C2 = slice(MINI_DIM + BASE_DIM + PTS_DIM, MINI_DIM + BASE_DIM + PTS_DIM + C2_DIM)
SL_STATS = slice(MINI_DIM + BASE_DIM + PTS_DIM + C2_DIM, FEAT_DIM)
SL_ALL = slice(0, FEAT_DIM)
SL_HIDDEN = slice(0, FEAT_DIM - STATS_DIM)

K_PCA = 48
PCA_DIM = K_PCA * 4  # 192
CTX_DIM = 14


class QuadRichL45Extractor:
    """Single L=45 pass (plus tiny Mini L=30) across Mini-2k, Base, PT-Small, and C2 with dep_s2 + s1/s2 stats."""

    def __init__(self, mini_m, tok2k, base_m, pts_m, c2_m, tok_base, device: torch.device):
        self.mini = mini_m
        self.tok2k = tok2k
        self.base = base_m
        self.pts = pts_m
        self.c2 = c2_m
        self.tok_base = tok_base
        self.device = device
        self.proto_2k = build_s1_token_prototypes(tok2k, device)
        self.proto_base = build_s1_token_prototypes(tok_base, device)
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            mini_m.transformer[1].register_forward_hook(self._mk("m1")),
            base_m.transformer[5].register_forward_hook(self._mk("b5")),
            base_m.transformer[8].register_forward_hook(self._mk("b8")),
            pts_m.transformer[5].register_forward_hook(self._mk("p5")),
            pts_m.transformer[6].register_forward_hook(self._mk("p6")),
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

        x45_norm = norm_window(raw_t[:, -45:])
        cur_c = x45_norm[:, -1, 3]

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            t30 = self.tok2k.encode(norm_window(raw_t[:, -30:]), half=True)
            _, m30 = self.mini.decode_s1(t30[0], t30[1], st_t[:, -30:])
            m30_last = m30[:, -1, :].float()

            t45 = self.tok2k.encode(x45_norm, half=True)
            s1_m, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])
            m1_45 = self._cap["m1"]
            dep_m, st_m = _s1_s2_stats(self.mini, s1_m, m45, cur_c, self.proto_2k)

            tb45 = self.tok_base.encode(x45_norm, half=True)
            s1_b, c_b = self.base.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_b, st_b_out = _s1_s2_stats(self.base, s1_b, c_b, cur_c, self.proto_base)

            s1_p, c_p = self.pts.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_p, st_p = _s1_s2_stats(self.pts, s1_p, c_p, cur_c, self.proto_base)

            s1_c, c_c = self.c2.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            dep_c, st_c = _s1_s2_stats(self.c2, s1_c, c_c, cur_c, self.proto_base)

        m45_f = m45.float()
        return torch.cat([
            m30_last, m1_45, m45_f[:, -1, :], m45_f[:, -5:, :].mean(dim=1), dep_m,
            self._cap["b5"], self._cap["b8"], c_b[:, -1, :].float(), dep_b,
            self._cap["p5"], self._cap["p6"], c_p[:, -1, :].float(), dep_p,
            self._cap["c5"], self._cap["c6"], c_c[:, -1, :].float(), dep_c,
            st_m, st_b_out, st_p, st_c,
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def load_quad_models(rank: int, paths: dict[str, str]):
    """Load Mini-2k, Base, PT-Small, C2, and tokenizers strictly inside worker process."""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    base = _load_kronos_base(paths["base"], device).eval()
    pts = _load_pretrained_small(paths["pts"], device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"],
        num_sectors=86,
        num_size_buckets=0,
        context_layer=6,
        use_size_percentile=True,
        size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    return device, QuadRichL45Extractor(mini, tok2k, base, pts, c2, tok_base, device)


def fwd_date_batched(ext, sym_arr, s_idx, t_idx, sec_np, pct_np, batch_size: int = 256) -> torch.Tensor:
    """Run extractor over a single date cross-section and z-score columns."""
    n = len(s_idx)
    chunks = []
    for off in range(0, n, batch_size):
        sl = slice(off, min(off + batch_size, n))
        raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
        chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl]))
    return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)


def compute_quad_context_14d(
    sym_arr,
    s_idx: np.ndarray,
    t_idx: np.ndarray,
    pct_np: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Strictly causal [N, 14] context (scale-invariant liquidity + regime features) + daily stats."""
    n = len(s_idx)
    rev5 = np.empty(n, dtype=np.float64)
    rev10 = np.empty(n, dtype=np.float64)
    rev20 = np.empty(n, dtype=np.float64)
    vol20 = np.empty(n, dtype=np.float64)
    vrat = np.empty(n, dtype=np.float64)
    med_amt20 = np.empty(n, dtype=np.float64)
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
        med_amt20[i] = float(np.median(amt))
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
    amt_pct = rank_pct_centered(med_amt20) + 0.5  # strictly in (0, 1], scale-invariant!
    is_liq65 = (amt_pct >= 0.35).astype(np.float64)

    ctx = np.column_stack([
        standardize_1d(rank_pct_centered(pct_np.astype(np.float64))),
        standardize_1d(rank_pct_centered(rev5)),
        standardize_1d(rank_pct_centered(rev10)),
        standardize_1d(rank_pct_centered(rev20)),
        standardize_1d(rank_pct_centered(vol20)),
        standardize_1d(rank_pct_centered(vrat)),
        standardize_1d(rank_pct_centered(np.log1p(med_amt20))),
        standardize_1d(rank_pct_centered(pos20)),
        standardize_1d(amt_pct - 0.5),
        is_liq65 - 0.65,
        np.full(n, b20 - 0.40, dtype=np.float64),
        np.full(n, b5 - 0.50, dtype=np.float64),
        np.full(n, mkt_disp * 10.0, dtype=np.float64),
        np.full(n, mkt_vol * 20.0, dtype=np.float64),
    ]).astype(np.float32)

    stats = {
        "b20": b20,
        "b5": b5,
        "amt_pct": amt_pct.astype(np.float32),
        "is_liq65": (amt_pct >= 0.35),
        "vol20": np.maximum(vol20, 0.005),
    }
    return ctx, stats


def compute_wave4_date_weights(
    d_str: str,
    n_stocks: int,
    breadth: float,
    mean_r10: float = 0.0,
    mean_r5: float = 0.0,
) -> dict[str, float]:
    """Strictly causal/training-only scalar date weights across std / decay365 / regime / crashentry."""
    dt = datetime.strptime(d_str, "%Y-%m-%d")
    dt_ref = datetime.strptime("2026-07-02", "%Y-%m-%d")
    age = max(0.0, float((dt_ref - dt).days))
    eq_day = 4500.0 / float(max(n_stocks, 1))
    is_crash_entry = breadth >= 0.30 and (mean_r10 <= -0.035 or (mean_r10 - mean_r5) <= -0.030)
    return {
        "std": 1.0,
        "decay365": float(0.5 ** (age / 365.0)),
        "regime": float(eq_day * (3.0 if breadth < 0.35 else 1.0)),
        "crashentry": float(eq_day * (4.0 if is_crash_entry else (1.5 if breadth < 0.35 else 1.0))),
    }


def extract_10d_path_targets(sym_arr, s_idx: np.ndarray, t_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Extract strictly [ti .. ti+10] horizon & path-risk returns (zero access > ti+10)."""
    n = len(s_idx)
    r3 = np.empty(n, dtype=np.float64)
    r5 = np.empty(n, dtype=np.float64)
    r10 = np.empty(n, dtype=np.float64)
    r_fwd5_10 = np.empty(n, dtype=np.float64)
    floor10 = np.empty(n, dtype=np.float64)
    maxdd10 = np.empty(n, dtype=np.float64)

    for i in range(n):
        vals = sym_arr[int(s_idx[i])][0]
        ti = int(t_idx[i])
        path = np.maximum(np.nan_to_num(vals[ti : ti + 11, 3].astype(np.float64), nan=1.0), 1e-8)
        c0 = path[0]
        r3[i] = path[3] / c0 - 1.0
        r5[i] = path[5] / c0 - 1.0
        r10[i] = path[10] / c0 - 1.0
        r_fwd5_10[i] = path[10] / path[5] - 1.0
        rel = path[1:] / c0 - 1.0
        floor10[i] = float(np.min(rel))
        peaks = np.maximum.accumulate(path)
        dds = 1.0 - path / peaks
        maxdd10[i] = float(np.max(dds[1:]))

    for arr in (r3, r5, r10, r_fwd5_10, floor10, maxdd10):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "r3": r3,
        "r5": r5,
        "r10": r10,
        "r_fwd5_10": r_fwd5_10,
        "floor10": floor10,
        "maxdd10": maxdd10,
    }


@torch.inference_mode()
def pca_quad_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_tr, sub_batches, tag = args
    device, ext = load_quad_models(rank, paths)
    try:
        gm = torch.zeros((MINI_DIM, MINI_DIM), dtype=torch.float64, device=device)
        gb = torch.zeros((BASE_DIM, BASE_DIM), dtype=torch.float64, device=device)
        gp = torch.zeros((PTS_DIM, PTS_DIM), dtype=torch.float64, device=device)
        gc2 = torch.zeros((C2_DIM, C2_DIM), dtype=torch.float64, device=device)
        for b in [b for i, b in enumerate(sub_batches) if i % n_gpus == rank]:
            _, s_idx, t_idx, sec_np, pct_np, _, _, _ = b
            x64 = fwd_date_batched(ext, sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            gm.addmm_(x64[:, SL_MINI].T, x64[:, SL_MINI])
            gb.addmm_(x64[:, SL_BASE].T, x64[:, SL_BASE])
            gp.addmm_(x64[:, SL_PTS].T, x64[:, SL_PTS])
            gc2.addmm_(x64[:, SL_C2].T, x64[:, SL_C2])
        out = f"/kaggle/working/_tmp_{tag}_pca_r{rank}.pt"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gm": gm.cpu().numpy(), "gb": gb.cpu().numpy(), "gp": gp.cpu().numpy(), "gc2": gc2.cpu().numpy()}, out)
        return out
    finally:
        ext.close()


def top_evecs_from_pts(pca_pts: list[str], k: int = K_PCA) -> dict[str, np.ndarray]:
    gm = sum(torch.load(p, map_location="cpu", weights_only=False)["gm"] for p in pca_pts)
    gb = sum(torch.load(p, map_location="cpu", weights_only=False)["gb"] for p in pca_pts)
    gp = sum(torch.load(p, map_location="cpu", weights_only=False)["gp"] for p in pca_pts)
    gc2 = sum(torch.load(p, map_location="cpu", weights_only=False)["gc2"] for p in pca_pts)
    for p in pca_pts:
        Path(p).unlink(missing_ok=True)

    def _top(mat: np.ndarray) -> np.ndarray:
        _, v = np.linalg.eigh(mat)
        return v[:, -k:][:, ::-1].astype(np.float32).copy()

    return {"m": _top(gm), "b": _top(gb), "p": _top(gp), "c": _top(gc2)}


def project_pca_4(x32: torch.Tensor, u_m: torch.Tensor, u_b: torch.Tensor, u_p: torch.Tensor, u_c: torch.Tensor) -> torch.Tensor:
    z = torch.cat([
        x32[:, SL_MINI] @ u_m,
        x32[:, SL_BASE] @ u_b,
        x32[:, SL_PTS] @ u_p,
        x32[:, SL_C2] @ u_c,
    ], dim=1)
    return zscore_cross_section_torch(z, clip_val=5.0)


def solve_ridge_specs_from_disk(
    fold_meta: list[dict],
    schemes: tuple[str, ...],
    targets: tuple[str, ...],
    ridge_specs: list[tuple[str, slice, str, str, tuple[float, ...]]],
    anchor_keys: list[str],
    x_sel: np.ndarray,
    d_sel: np.ndarray,
    sel_targets: dict[str, np.ndarray],
) -> tuple[dict, dict, dict, list[dict]]:
    """Solve all Ridge heads one weighting scheme at a time from disk, then build opposite-fold OOF heads."""
    n_gpus = len(fold_meta)
    selection: dict[str, dict] = {}
    full_heads: dict[str, tuple[slice, np.ndarray]] = {}
    tr_heads: dict[str, tuple[slice, np.ndarray]] = {}
    sel_alphas: dict[str, float] = {}

    for sch in schemes:
        sch_specs = [sp for sp in ridge_specs if sp[2] == sch]
        if not sch_specs:
            continue
        g_tr_s = sum(np.load(d["tr_g_paths"][sch]) for d in fold_meta)
        g_all_s = g_tr_s + sum(np.load(d["va_g_paths"][sch]) for d in fold_meta)
        for name, sl, _s, tkey, alphas in sch_specs:
            r_tr_s = sum(np.load(d["tr_r_paths"][sch][tkey]) for d in fold_meta)
            r_all_s = r_tr_s + sum(np.load(d["va_r_paths"][sch][tkey]) for d in fold_meta)
            xs = np.asarray(x_sel[:, sl], dtype=np.float64)
            y_eval = sel_targets[tkey]
            best_a, best_ic, grid = None, -1e9, []
            for a in alphas:
                w = solve_ridge_subblock(g_tr_s, r_tr_s, sl, a)
                ic = daily_spearman_ic(d_sel, xs @ w, y_eval)
                grid.append({"alpha": a, "val_ic": ic})
                if ic > best_ic:
                    best_ic, best_a = ic, a
            assert best_a is not None
            sel_alphas[name] = best_a
            tr_heads[name] = (sl, solve_ridge_subblock(g_tr_s, r_tr_s, sl, best_a).astype(np.float32))
            full_heads[name] = (sl, solve_ridge_subblock(g_all_s, r_all_s, sl, best_a).astype(np.float32))
            selection[f"ridge_{name}"] = {"alpha": best_a, "val_ic": float(best_ic), "grid": grid}
        del g_tr_s, g_all_s
        for d in fold_meta:
            Path(d["va_g_paths"][sch]).unlink(missing_ok=True)
            for t in targets:
                if t in d["va_r_paths"][sch]:
                    Path(d["va_r_paths"][sch][t]).unlink(missing_ok=True)
        gc.collect()

    fold_opp_heads: list[dict] = [{} for _ in range(n_gpus)]
    for sch in schemes:
        sch_anchors = [k for k in anchor_keys if next(sp for sp in ridge_specs if sp[0] == k)[2] == sch]
        if not sch_anchors:
            for d in fold_meta:
                Path(d["tr_g_paths"][sch]).unlink(missing_ok=True)
                for t in targets:
                    if t in d["tr_r_paths"][sch]:
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
                    a_half = sel_alphas[k] * (0.5 if n_gpus > 1 else 1.0)
                    w_opp = solve_ridge_subblock(g_opp, r_opp, sl, a_half).astype(np.float32)
                    fold_opp_heads[r][k] = (sl, w_opp)
            del g_opp
        for d in fold_meta:
            Path(d["tr_g_paths"][sch]).unlink(missing_ok=True)
            for t in targets:
                if t in d["tr_r_paths"][sch]:
                    Path(d["tr_r_paths"][sch][t]).unlink(missing_ok=True)
        gc.collect()

    return selection, full_heads, tr_heads, fold_opp_heads


def train_lgbm_regressor(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va_eval: np.ndarray,
    d_va: np.ndarray,
    r_va_eval: np.ndarray,
    *,
    seed: int,
    n_estimators: int = 550,
    min_child_samples: int = 800,
    reg_lambda: float = 250.0,
):
    """Fit regularized LightGBM regressor with early stopping on VAL_SEL."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=min_child_samples,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=reg_lambda,
        reg_alpha=10.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va_eval)],
        callbacks=[lgb.early_stopping(45, verbose=False)],
    )
    p_va = model.predict(x_va)
    return model, float(daily_spearman_ic(d_va, p_va, r_va_eval))


def _isolated_target(q_path: str, fn, args: tuple) -> None:
    res = fn(*args)
    with open(q_path, "wb") as fh:
        pickle.dump(res, fh)


def run_in_isolated_process(fn, *args):
    """Run CPU/OpenMP heavy function (e.g. LightGBM) in an isolated child process.

    Prevents initializing libgomp/OpenMP thread pools inside the parent Python process,
    eliminating Linux fork-after-OpenMP deadlocks in subsequent multi-GPU workers.
    """
    ctx = mp.get_context("fork")
    out_path = f"/tmp/_iso_{time.time_ns()}.pkl"
    proc = ctx.Process(target=_isolated_target, args=(out_path, fn, args))
    proc.start()
    try:
        while proc.is_alive():
            time.sleep(1.0)
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Isolated process {getattr(fn, '__name__', 'fn')} exited with code {proc.exitcode}")
        with open(out_path, "rb") as fh:
            return pickle.load(fh)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        Path(out_path).unlink(missing_ok=True)


def _fit_lgbm_child(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va_eval: np.ndarray,
    d_va: np.ndarray,
    r_va_eval: np.ndarray,
    out_model_path: str,
    seed: int,
    n_estimators: int,
    min_child_samples: int,
    reg_lambda: float,
) -> dict:
    model, val_ic = train_lgbm_regressor(
        x_tr,
        y_tr,
        x_va,
        y_va_eval,
        d_va,
        r_va_eval,
        seed=seed,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        reg_lambda=reg_lambda,
    )
    model.booster_.save_model(out_model_path)
    return {"val_ic": float(val_ic), "best_iteration": int(model.best_iteration_ or 0)}


def fit_lgbm_regressor_isolated(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va_eval: np.ndarray,
    d_va: np.ndarray,
    r_va_eval: np.ndarray,
    out_model_path: str,
    *,
    seed: int,
    n_estimators: int = 550,
    min_child_samples: int = 800,
    reg_lambda: float = 250.0,
) -> dict:
    """Fit LightGBM regressor in an isolated child process and persist booster to `out_model_path`."""
    return run_in_isolated_process(
        _fit_lgbm_child,
        x_tr,
        y_tr,
        x_va,
        y_va_eval,
        d_va,
        r_va_eval,
        out_model_path,
        seed,
        n_estimators,
        min_child_samples,
        reg_lambda,
    )


def resolve_hf_quad_dirs(work_dir: Path | str = "/kaggle/working") -> dict[str, str]:
    """Download/locate all HuggingFace Kronos backbones once in parent process."""
    from huggingface_hub import snapshot_download

    w = Path(work_dir)
    return {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir=str(w / "hf_kronos_mini")),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir=str(w / "hf_tok2k")),
        "base": snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir=str(w / "hf_kronos_base")),
        "pts": snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir=str(w / "hf_kronos_small")),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(w / "hf_tok_base")),
    }


def make_c2_paths_dict(input_root: Path, work_dir: Path | str = "/kaggle/working") -> dict[str, str]:
    paths = resolve_hf_quad_dirs(work_dir)
    paths["c2"] = str(find_c2_weights(input_root).parent)
    return paths
