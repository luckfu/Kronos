"""Shared weighted/regime/multi-horizon streaming ridge helpers for Wave 2 (G24-G26)."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np
import torch


def compute_date_weights(
    d_str: str,
    n_stocks: int,
    breadth_ma20: float,
    ref_end: str = "2026-07-02",
    half_life_days: float = 730.0,
    weak_breadth_thresh: float = 0.35,
    weak_breadth_mult: float = 3.0,
    ref_cross_section: float = 4500.0,
) -> tuple[float, float]:
    """Return strictly causal (w_decay, w_regime) scalar weights for one trading date."""
    dt = datetime.strptime(d_str, "%Y-%m-%d")
    dt_ref = datetime.strptime(ref_end, "%Y-%m-%d")
    age_days = max(0.0, float((dt_ref - dt).days))
    w_decay = float(0.5 ** (age_days / half_life_days))
    eq_day = ref_cross_section / float(max(n_stocks, 1))
    w_regime = float(eq_day * (weak_breadth_mult if breadth_ma20 < weak_breadth_thresh else 1.0))
    return w_decay, w_regime


def extract_r5_and_breadth(sym_arr, s_idx: np.ndarray, t_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute 5-day forward return vector and strictly causal MA20 market breadth for a date batch."""
    n = len(s_idx)
    c0 = np.empty(n, dtype=np.float64)
    c5 = np.empty(n, dtype=np.float64)
    above_ma20 = 0
    for i in range(n):
        vals = sym_arr[int(s_idx[i])][0]
        ti = int(t_idx[i])
        cur = float(vals[ti, 3])
        c0[i] = cur
        c5[i] = float(vals[ti + 5, 3])
        ma20 = float(vals[ti - 19 : ti + 1, 3].mean())
        if cur > ma20:
            above_ma20 += 1
    r5 = np.nan_to_num(c5 / np.maximum(c0, 1e-8) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
    breadth = float(above_ma20) / float(max(n, 1))
    return r5, breadth


def run_worker_stats_weighted(
    *,
    tag: str,
    rank: int,
    n_gpus: int,
    feat_dim: int,
    batch_size: int,
    extract_fn: Callable,
    sym_tr,
    tr_batches: list,
    sym_val,
    val_batches: list,
    device: torch.device,
    schemes: tuple[str, ...] = ("std", "decay", "regime"),
) -> str:
    """Stream train/val batches on one GPU, accumulating Gram/RHS for multiple causal weighting schemes."""
    from g1_multiscale_extract_and_fit import phase, rank_pct_centered, standardize_1d, zscore_cross_section_torch

    grams = {s: torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device) for s in schemes}
    rhs = {
        s: {
            "raw": torch.zeros(feat_dim, dtype=torch.float64, device=device),
            "multi": torch.zeros(feat_dim, dtype=torch.float64, device=device),
            "ret5": torch.zeros(feat_dim, dtype=torch.float64, device=device),
        }
        for s in schemes
    }
    n_rows = 0
    n_weak_days = 0

    def _fwd_date(sym_arr, s_idx, t_idx, sec_np, pct_np) -> torch.Tensor:
        n = len(s_idx)
        chunks = []
        for off in range(0, n, batch_size):
            sl = slice(off, min(off + batch_size, n))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            chunks.append(extract_fn(raw_b, st_b, sec_np[sl], pct_np[sl]))
        return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
    for bi, b in enumerate(my_tr):
        d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
        r5, breadth = extract_r5_and_breadth(sym_tr, s_idx, t_idx)
        if breadth < 0.35:
            n_weak_days += 1
        w_decay, w_regime = compute_date_weights(str(d_str), len(s_idx), breadth)
        w_map = {"std": 1.0, "decay": w_decay, "regime": w_regime}

        x64 = _fwd_date(sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
        yr = torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64)
        ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
        y5 = torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64)

        xx = x64.T @ x64
        xr = x64.T @ yr
        xm = x64.T @ ym
        x5 = x64.T @ y5
        for s in schemes:
            w = w_map[s]
            grams[s].add_(xx, alpha=w)
            rhs[s]["raw"].add_(xr, alpha=w)
            rhs[s]["multi"].add_(xm, alpha=w)
            rhs[s]["ret5"].add_(x5, alpha=w)
        n_rows += int(x64.shape[0])
        if (bi + 1) % 80 == 0 or bi + 1 == len(my_tr):
            phase(f"{tag}_train_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows, weak_days=n_weak_days)

    my_val = [b for i, b in enumerate(val_batches) if i % n_gpus == rank]
    val_recs = []
    for b in my_val:
        d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
        r5, breadth = extract_r5_and_breadth(sym_val, s_idx, t_idx)
        x_d = _fwd_date(sym_val, s_idx, t_idx, sec_np, pct_np)
        val_recs.append({
            "date": str(d_str),
            "n": int(len(s_idx)),
            "breadth": float(breadth),
            "x": x_d.half().cpu().numpy(),
            "y_raw": standardize_1d(rank_pct_centered(r10)).astype(np.float32),
            "y_multi": y_multi.astype(np.float32),
            "y_ret5": standardize_1d(rank_pct_centered(r5)).astype(np.float32),
            "ret10": r10.astype(np.float32),
            "ret5": r5.astype(np.float32),
        })

    out_pt = f"/tmp/{tag}_stats_r{rank}.pt"
    payload = {
        "n_rows": n_rows,
        "n_weak_days": n_weak_days,
        "grams": {s: grams[s].cpu().numpy() for s in schemes},
        "rhs": {s: {k: v.cpu().numpy() for k, v in rhs[s].items()} for s in schemes},
        "val_recs": val_recs,
    }
    torch.save(payload, out_pt)
    return out_pt


def chunked_val_accumulate_weighted(
    pt_paths: list[str],
    feat_dim: int,
    val_sel_start: str,
    val_sel_end: str,
    schemes: tuple[str, ...] = ("std", "decay", "regime"),
) -> dict:
    """Merge worker train stats and accumulate val stats into per-scheme Gram/RHS cleanly in RAM."""
    grams_tr = {s: np.zeros((feat_dim, feat_dim), dtype=np.float64) for s in schemes}
    rhs_tr = {s: {k: np.zeros(feat_dim, dtype=np.float64) for k in ("raw", "multi", "ret5")} for s in schemes}
    n_tr_rows = 0
    n_weak_days = 0
    val_recs: list[dict] = []

    for p in pt_paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        n_tr_rows += int(d["n_rows"])
        n_weak_days += int(d.get("n_weak_days", 0))
        for s in schemes:
            grams_tr[s] += d["grams"][s]
            for k in ("raw", "multi", "ret5"):
                rhs_tr[s][k] += d["rhs"][s][k]
        val_recs.extend(d["val_recs"])
    val_recs.sort(key=lambda r: r["date"])

    grams_all = {s: grams_tr[s].copy() for s in schemes}
    rhs_all = {s: {k: v.copy() for k, v in rhs_tr[s].items()} for s in schemes}
    sel_x, sel_d, sel_r10, sel_r5 = [], [], [], []
    for rec in val_recs:
        d_str = str(rec["date"])
        w_decay, w_regime = compute_date_weights(d_str, int(rec["n"]), float(rec["breadth"]))
        w_map = {"std": 1.0, "decay": w_decay, "regime": w_regime}
        x64 = rec["x"].astype(np.float64)
        xx = x64.T @ x64
        xr = x64.T @ rec["y_raw"].astype(np.float64)
        xm = x64.T @ rec["y_multi"].astype(np.float64)
        x5 = x64.T @ rec["y_ret5"].astype(np.float64)
        for s in schemes:
            w = w_map[s]
            grams_all[s] += w * xx
            rhs_all[s]["raw"] += w * xr
            rhs_all[s]["multi"] += w * xm
            rhs_all[s]["ret5"] += w * x5
        if val_sel_start <= d_str <= val_sel_end:
            sel_x.append(rec["x"].astype(np.float32))
            sel_d.append(np.full(len(rec["ret10"]), d_str))
            sel_r10.append(rec["ret10"].astype(np.float64))
            sel_r5.append(rec["ret5"].astype(np.float64))
        rec.clear()
    val_recs.clear()

    if not sel_x:
        raise RuntimeError(f"No validation records inside {val_sel_start}..{val_sel_end}")
    return {
        "n_tr_rows": n_tr_rows,
        "n_weak_days": n_weak_days,
        "grams_tr": grams_tr,
        "rhs_tr": rhs_tr,
        "grams_all": grams_all,
        "rhs_all": rhs_all,
        "x_sel": np.concatenate(sel_x, axis=0),
        "d_sel": np.concatenate(sel_d),
        "r10_sel": np.concatenate(sel_r10),
        "r5_sel": np.concatenate(sel_r5),
    }


def select_weighted_heads(
    specs: list[tuple[str, slice, str, str, tuple[float, ...]]],
    stats: dict,
) -> tuple[dict[str, dict], dict[str, tuple[slice, np.ndarray]]]:
    """Grid-search scalar L2 penalty on val for each (name, slice, scheme, target_key, alphas)."""
    from g1_multiscale_extract_and_fit import daily_spearman_ic, solve_ridge_subblock

    selection: dict[str, dict] = {}
    heads_np: dict[str, tuple[slice, np.ndarray]] = {}
    x_sel = stats["x_sel"]
    d_sel = stats["d_sel"]
    r10_sel = stats["r10_sel"]
    r5_sel = stats["r5_sel"]

    for name, sl, scheme, target_key, alphas in specs:
        gram_tr = stats["grams_tr"][scheme]
        rhs_tr = stats["rhs_tr"][scheme][target_key]
        gram_all = stats["grams_all"][scheme]
        rhs_all = stats["rhs_all"][scheme][target_key]
        xs = np.asarray(x_sel[:, sl], dtype=np.float64)
        best_a, best_ic, grid = None, -1e9, []
        for a in alphas:
            w = solve_ridge_subblock(gram_tr, rhs_tr, sl, a)
            pred = xs @ w
            ic10 = daily_spearman_ic(d_sel, pred, r10_sel)
            ic5 = daily_spearman_ic(d_sel, pred, r5_sel)
            sc = ic5 if target_key == "ret5" else ic10
            grid.append({"alpha": a, "val_ic_r10": ic10, "val_ic_r5": ic5})
            if sc > best_ic:
                best_ic, best_a = sc, a
        assert best_a is not None
        w_full = solve_ridge_subblock(gram_all, rhs_all, sl, best_a).astype(np.float32)
        heads_np[name] = (sl, w_full)
        selection[name] = {
            "scheme": scheme,
            "target": target_key,
            "dim": int(w_full.shape[0]),
            "selected_alpha": best_a,
            "best_val_ic": best_ic,
            "grid": grid,
        }
    return selection, heads_np
