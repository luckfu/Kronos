"""Shared RAM-safe helpers and Kaggle bundler for G19-G22 short-window ridge probes."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "finetune" / "base_clean_bootstrap_relay") not in sys.path:
    sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))


def detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


def norm_window(raw_t: torch.Tensor) -> torch.Tensor:
    mean = raw_t.mean(dim=1, keepdim=True)
    std = raw_t.std(dim=1, unbiased=False, keepdim=True)
    return torch.clamp((raw_t - mean) / (std + 1e-5), -5.0, 5.0)


def find_c2_weights(input_root: Path) -> Path:
    matches = sorted(
        p for p in input_root.rglob("model.safetensors")
        if any(tag in p.as_posix() for tag in ("c2-best-weights", "c2-best-seg179", "c2-best-checkpoint", "c2-best"))
    )
    if not matches:
        raise RuntimeError(f"Could not find C2 model.safetensors under {input_root}")
    return matches[0]


def chunked_val_accumulate_and_sel(
    pt_paths: list[str],
    feat_dim: int,
    val_sel_start: str,
    val_sel_end: str,
    gram_tr: np.ndarray,
    rhs_tr: dict[str, np.ndarray],
) -> tuple[
    int,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Merge worker train statistics and accumulate val gram without a giant float64 concat."""
    n_tr_rows = 0
    val_recs: list[dict] = []
    for p in pt_paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        n_tr_rows += int(d["n_rows"])
        gram_tr += d["gram"]
        for k in rhs_tr:
            rhs_tr[k] += d[f"b_{k}"]
        val_recs.extend(d["val_recs"])
    val_recs.sort(key=lambda r: r["date"])

    gram_all = gram_tr.copy()
    rhs_all = {k: v.copy() for k, v in rhs_tr.items()}
    sel_x, sel_d, sel_r, sel_yo = [], [], [], []
    for rec in val_recs:
        x64 = rec["x"].astype(np.float64)
        gram_all += x64.T @ x64
        rhs_all["raw"] += x64.T @ rec["y_raw"].astype(np.float64)
        rhs_all["orth"] += x64.T @ rec["y_orth"].astype(np.float64)
        rhs_all["multi"] += x64.T @ rec["y_multi"].astype(np.float64)
        d_str = str(rec["date"])
        if val_sel_start <= d_str <= val_sel_end:
            sel_x.append(rec["x"].astype(np.float32))
            sel_d.append(np.full(len(rec["ret10"]), d_str))
            sel_r.append(rec["ret10"].astype(np.float64))
            sel_yo.append(rec["y_orth"].astype(np.float64))
        rec.clear()
    val_recs.clear()

    if not sel_x:
        raise RuntimeError(f"No validation records inside {val_sel_start}..{val_sel_end}")
    x_sel = np.concatenate(sel_x, axis=0)
    d_sel = np.concatenate(sel_d)
    r_sel = np.concatenate(sel_r)
    yo_sel = np.concatenate(sel_yo)
    if gram_all.shape != (feat_dim, feat_dim):
        raise RuntimeError(f"Expected gram shape {(feat_dim, feat_dim)}, got {gram_all.shape}")
    return n_tr_rows, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel


def solve_block_penalties(
    gram: np.ndarray,
    rhs: np.ndarray,
    sub_slice: slice,
    penalties: list[tuple[slice, float]],
) -> np.ndarray:
    """Solve ridge regression on `sub_slice` with per-slice L2 diagonal penalties."""
    g_sub = gram[sub_slice, sub_slice]
    b_sub = rhs[sub_slice]
    start = sub_slice.start or 0
    stop = sub_slice.stop if sub_slice.stop is not None else gram.shape[0]
    pen = np.zeros(stop - start, dtype=np.float64)
    for blk_sl, alpha in penalties:
        lo = max(0, (blk_sl.start or 0) - start)
        hi = min(stop - start, (blk_sl.stop if blk_sl.stop is not None else gram.shape[0]) - start)
        if hi > lo:
            pen[lo:hi] = float(alpha)
    return np.linalg.solve(g_sub + np.diag(pen), b_sub)


def select_isotropic_heads(
    specs: list[tuple[str, slice, str]],
    alphas: tuple[float, ...],
    gram_tr: np.ndarray,
    rhs_tr: dict[str, np.ndarray],
    gram_all: np.ndarray,
    rhs_all: dict[str, np.ndarray],
    x_sel: np.ndarray,
    d_sel: np.ndarray,
    r_sel: np.ndarray,
    yo_sel: np.ndarray,
) -> tuple[dict[str, dict], dict[str, tuple[slice, np.ndarray]]]:
    """Grid-search scalar L2 penalty per head on selection val and refit through TRAIN_FIT_END."""
    from g1_multiscale_extract_and_fit import daily_spearman_ic, solve_ridge_subblock

    selection: dict[str, dict] = {}
    heads_np: dict[str, tuple[slice, np.ndarray]] = {}
    for name, sl, key in specs:
        best_a, best_ic, grid = None, -1e9, []
        xs = np.asarray(x_sel[:, sl], dtype=np.float64)
        for a in alphas:
            w = solve_ridge_subblock(gram_tr, rhs_tr[key], sl, a)
            pred = xs @ w
            ic_r = daily_spearman_ic(d_sel, pred, r_sel)
            ic_o = daily_spearman_ic(d_sel, pred, yo_sel)
            sc = ic_o if key == "orth" else ic_r
            grid.append({"alpha": a, "val_ic_raw": ic_r, "val_ic_orth": ic_o})
            if sc > best_ic:
                best_ic, best_a = sc, a
        heads_np[name] = (sl, solve_ridge_subblock(gram_all, rhs_all[key], sl, best_a).astype(np.float32))
        selection[name] = {"selected_alpha": best_a, "best_val_ic": best_ic, "grid": grid}
    return selection, heads_np


def select_anisotropic_head(
    name: str,
    sub_slice: slice,
    target_key: str,
    block_slices: list[slice],
    alpha_combos: list[tuple[float, ...]],
    gram_tr: np.ndarray,
    rhs_tr: dict[str, np.ndarray],
    gram_all: np.ndarray,
    rhs_all: dict[str, np.ndarray],
    x_sel: np.ndarray,
    d_sel: np.ndarray,
    r_sel: np.ndarray,
    yo_sel: np.ndarray,
) -> tuple[dict, tuple[slice, np.ndarray]]:
    """Grid-search per-block L2 penalty tuple on selection val and refit full train."""
    from g1_multiscale_extract_and_fit import daily_spearman_ic

    xs = np.asarray(x_sel[:, sub_slice], dtype=np.float64)
    best_combo: tuple[float, ...] | None = None
    best_ic, grid = -1e9, []
    for combo in alpha_combos:
        penalties = list(zip(block_slices, combo))
        w = solve_block_penalties(gram_tr, rhs_tr[target_key], sub_slice, penalties)
        pred = xs @ w
        ic_r = daily_spearman_ic(d_sel, pred, r_sel)
        ic_o = daily_spearman_ic(d_sel, pred, yo_sel)
        sc = ic_o if target_key == "orth" else ic_r
        grid.append({"alphas": list(combo), "val_ic_raw": ic_r, "val_ic_orth": ic_o})
        if sc > best_ic:
            best_ic, best_combo = sc, combo
    assert best_combo is not None
    w_full = solve_block_penalties(
        gram_all, rhs_all[target_key], sub_slice, list(zip(block_slices, best_combo)),
    ).astype(np.float32)
    return {"selected_alphas": list(best_combo), "best_val_ic": best_ic, "grid": grid}, (sub_slice, w_full)


def run_worker_stats(
    *,
    tag: str,
    rank: int,
    n_gpus: int,
    feat_dim: int,
    batch_size: int,
    extract_fn,
    sym_tr,
    tr_batches: list,
    sym_val,
    val_batches: list,
    device: torch.device,
) -> str:
    """Stream train and validation batches on one GPU worker and serialize gram/rhs."""
    from g1_multiscale_extract_and_fit import phase, rank_pct_centered, standardize_1d, zscore_cross_section_torch

    gram_tr = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
    b_raw = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    b_orth = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    b_multi = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    n_rows = 0

    def _fwd_date(sym_arr, s_idx, t_idx, sec_np, pct_np) -> torch.Tensor:
        n = len(s_idx)
        chunks = []
        for off in range(0, n, batch_size):
            sl = slice(off, min(off + batch_size, n))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            chunks.append(extract_fn(raw_b, st_b, sec_np[sl], pct_np[sl]))
        return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
    for bi, b in enumerate(my_tr):
        _, s_idx, t_idx, sec_np, pct_np, y_multi, y_orth, r10 = b
        x64 = _fwd_date(sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
        yr = torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64)
        yo = torch.as_tensor(y_orth, device=device, dtype=torch.float64)
        ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
        gram_tr.addmm_(x64.T, x64)
        b_raw.addmv_(x64.T, yr)
        b_orth.addmv_(x64.T, yo)
        b_multi.addmv_(x64.T, ym)
        n_rows += int(x64.shape[0])
        if (bi + 1) % 80 == 0 or bi + 1 == len(my_tr):
            phase(f"{tag}_train_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

    my_val = [b for i, b in enumerate(val_batches) if i % n_gpus == rank]
    val_recs = []
    for b in my_val:
        d_str, s_idx, t_idx, sec_np, pct_np, y_multi, y_orth, r10 = b
        x_d = _fwd_date(sym_val, s_idx, t_idx, sec_np, pct_np)
        val_recs.append({
            "date": d_str,
            "x": x_d.half().cpu().numpy(),
            "y_raw": standardize_1d(rank_pct_centered(r10)).astype(np.float32),
            "y_multi": y_multi.astype(np.float32),
            "y_orth": y_orth.astype(np.float32),
            "ret10": r10.astype(np.float32),
        })

    out_pt = f"/tmp/{tag}_stats_r{rank}.pt"
    torch.save({
        "n_rows": n_rows,
        "gram": gram_tr.cpu().numpy(),
        "b_raw": b_raw.cpu().numpy(),
        "b_orth": b_orth.cpu().numpy(),
        "b_multi": b_multi.cpu().numpy(),
        "val_recs": val_recs,
    }, out_pt)
    return out_pt


def run_worker_oos(
    *,
    tag: str,
    rank: int,
    n_gpus: int,
    batch_size: int,
    extract_fn,
    sym_oos,
    oos_batches: list,
    heads_np: dict[str, tuple[slice, np.ndarray]],
    device: torch.device,
) -> str:
    """Score OOS dates for one GPU worker and write compressed CSV."""
    import pandas as pd
    from g1_multiscale_extract_and_fit import zscore_cross_section_torch

    heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_np.items()}
    frames = []
    my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]
    for d_str, recs in my_oos:
        chunks = []
        for off in range(0, len(recs), batch_size):
            sl = recs[off:off + batch_size]
            raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
            st_b = np.stack([sym_oos[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
            sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
            chunks.append(extract_fn(raw_b, st_b, sec_b, pct_b))
        x_d = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        df_d = pd.DataFrame({"identity": [r[4] for r in recs], "symbol": [r[5] for r in recs], "asof_date": d_str})
        for name, (sl_w, wt) in heads_t.items():
            df_d[f"p_{name}"] = (x_d[:, sl_w] @ wt).cpu().numpy().astype(np.float32)
        frames.append(df_d)
    out_csv = f"/tmp/{tag}_oos_r{rank}.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
    return out_csv


def source_bundle(entry_filename: str, extra_gpu_files: list[str] | None = None) -> bytes:
    entries: list[tuple[Path, Path]] = []
    for path in sorted((ROOT / "model").glob("*.py")):
        entries.append((Path("model") / path.name, path))
    entries.append((
        Path("finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"),
        ROOT / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py",
    ))
    base_files = [
        "g1_multiscale_extract_and_fit.py",
        "g1_kaggle_entry.py",
        "g2_rank_finetune.py",
        "short_probe_common.py",
    ]
    for name in base_files + (extra_gpu_files or []) + [entry_filename]:
        entries.append((Path("finetune/gpu_alpha") / name, HERE / name))
    seen: set[str] = set()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, src in entries:
            key = arc.as_posix()
            if key in seen:
                continue
            seen.add(key)
            info = zipfile.ZipInfo(key, (2026, 11, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, src.read_bytes())
    return buf.getvalue()


def push_kernel(
    *,
    owner: str,
    slug: str,
    title: str,
    datasets: list[str],
    entry_filename: str,
    work_tag: str,
    extra_gpu_files: list[str] | None = None,
    push: bool = False,
) -> dict:
    import relay_auth

    payload = source_bundle(entry_filename, extra_gpu_files)
    digest = hashlib.sha256(payload).hexdigest()
    code = f'''import base64, hashlib, io, os, runpy, shutil, sys, zipfile
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/{work_tag}_src")
if repo.exists():
    shutil.rmtree(repo)
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "finetune/gpu_alpha"))
runpy.run_path(str(repo / "finetune/gpu_alpha/{entry_filename}"), run_name="__main__")
'''
    build = HERE / "build" / slug
    build.mkdir(parents=True, exist_ok=True)
    (build / "script.py").write_text(code)
    compile(code, str(build / "script.py"), "exec")
    meta = {
        "slug": f"{owner}/{slug}",
        "title": title,
        "datasets": datasets,
        "sha256": digest,
        "bundle_bytes": len(payload),
    }
    if not push:
        return meta
    session = relay_auth.session()
    resp = session.post(
        f"{relay_auth.API_ROOT}/v1/kernels.KernelsApiService/SaveKernel",
        json={
            "slug": f"{owner}/{slug}",
            "newTitle": title,
            "text": code,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "datasetDataSources": datasets,
            "kernelDataSources": [],
            "machineShape": "NvidiaTeslaT4",
        },
        timeout=180,
    )
    meta["http_status"] = resp.status_code
    meta["response"] = resp.text[:500]
    return meta
