"""G14: Kronos-base 2018-2026 Dense Streaming Ridge around L=60 [L45, L60, L75] with wider alpha up to 1e7."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
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
from g2_rank_finetune import build_g2_date_batches
from g4_base_kaggle_entry import _load_kronos_base

OUTPUT = Path(os.environ.get("G14_OUTPUT", "/kaggle/working/g14_out"))
INPUT = Path(os.environ.get("G14_INPUT", "/kaggle/input"))
D_BASE = 832
WINDOWS = (45, 60, 75)
FEAT_DIM = D_BASE * len(WINDOWS)  # 2496
TRAIN_START_G14 = "2018-01-01"
TRAIN_STRIDE_G14 = 2
ALPHAS_G14 = (3e4, 1e5, 3e5, 1e6, 3e6, 1e7)


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


def _norm_w(raw_t: torch.Tensor) -> torch.Tensor:
    mean = raw_t.mean(dim=1, keepdim=True)
    std = raw_t.std(dim=1, unbiased=False, keepdim=True)
    return torch.clamp((raw_t - mean) / (std + 1e-5), -5.0, 5.0)


@torch.inference_mode()
def extract_band_batch(
    base_model,
    tokenizer,
    raw_b: np.ndarray,
    st_b: np.ndarray,
    sec_b: np.ndarray,
    pct_b: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
    st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
    sec_t = torch.as_tensor(sec_b, device=device, dtype=torch.long)
    pct_t = torch.as_tensor(pct_b, device=device, dtype=torch.float32)
    parts = []
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for w in WINDOWS:
            toks = tokenizer.encode(_norm_w(raw_t[:, -w:]), half=True)
            _, ctx = base_model.decode_s1(toks[0], toks[1], st_t[:, -w:], sector_id=sec_t, size_percentile=pct_t)
            parts.append(ctx[:, -1, :].float())
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, base_hf_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    base = _load_kronos_base(base_hf_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()

    gram_tr = torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device)
    b_raw = torch.zeros(FEAT_DIM, dtype=torch.float64, device=device)
    b_orth = torch.zeros(FEAT_DIM, dtype=torch.float64, device=device)
    b_multi = torch.zeros(FEAT_DIM, dtype=torch.float64, device=device)
    n_rows = 0

    def _fwd_date(sym_arr, s_idx, t_idx, sec_np, pct_np) -> torch.Tensor:
        n = len(s_idx)
        chunks = []
        for off in range(0, n, 384):
            sl = slice(off, min(off + 384, n))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            chunks.append(extract_band_batch(base, tok, raw_b, st_b, sec_np[sl], pct_np[sl], device))
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
        if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
            phase("g14_train_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

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

    out_pt = f"/tmp/g14_stats_r{rank}.pt"
    torch.save({
        "n_rows": n_rows,
        "gram": gram_tr.cpu().numpy(),
        "b_raw": b_raw.cpu().numpy(),
        "b_orth": b_orth.cpu().numpy(),
        "b_multi": b_multi.cpu().numpy(),
        "val_recs": val_recs,
    }, out_pt)
    return out_pt


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, base_hf_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    base = _load_kronos_base(base_hf_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_np.items()}

    frames = []
    my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]
    for d_str, recs in my_oos:
        chunks = []
        for off in range(0, len(recs), 384):
            sl = recs[off:off + 384]
            raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
            st_b = np.stack([sym_oos[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
            sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
            chunks.append(extract_band_batch(base, tok, raw_b, st_b, sec_b, pct_b, device))
        x_d = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        df_d = pd.DataFrame({"identity": [r[4] for r in recs], "symbol": [r[5] for r in recs], "asof_date": d_str})
        for name, (sl_w, wt) in heads_t.items():
            df_d[f"p_{name}"] = (x_d[:, sl_w] @ wt).cpu().numpy().astype(np.float32)
        frames.append(df_d)
    out_csv = f"/tmp/g14_oos_r{rank}.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
    return out_csv


def main() -> None:
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = _detect_gpu_count()
    phase("g14_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    base_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G14, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G14]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(_extract_worker, [(r, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)])
    del sym_tr, tr_batches, sym_val, val_batches

    gram_tr = np.zeros((FEAT_DIM, FEAT_DIM), dtype=np.float64)
    rhs_tr = {k: np.zeros(FEAT_DIM, dtype=np.float64) for k in ("raw", "orth", "multi")}
    n_tr_rows = 0
    val_recs = []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        n_tr_rows += int(d["n_rows"])
        gram_tr += d["gram"]
        for k in rhs_tr:
            rhs_tr[k] += d[f"b_{k}"]
        val_recs.extend(d["val_recs"])
    val_recs.sort(key=lambda r: r["date"])

    x_val = np.concatenate([r["x"].astype(np.float64) for r in val_recs], axis=0)
    d_val = np.concatenate([[r["date"]] * len(r["ret10"]) for r in val_recs])
    yr_val = np.concatenate([r["y_raw"].astype(np.float64) for r in val_recs])
    yo_val = np.concatenate([r["y_orth"].astype(np.float64) for r in val_recs])
    ym_val = np.concatenate([r["y_multi"].astype(np.float64) for r in val_recs])
    r10_val = np.concatenate([r["ret10"].astype(np.float64) for r in val_recs])

    gram_all = gram_tr + x_val.T @ x_val
    rhs_all = {
        "raw": rhs_tr["raw"] + x_val.T @ yr_val,
        "orth": rhs_tr["orth"] + x_val.T @ yo_val,
        "multi": rhs_tr["multi"] + x_val.T @ ym_val,
    }

    sel = (d_val >= VAL_SEL_START) & (d_val <= VAL_SEL_END)
    x_sel, d_sel, r_sel, yo_sel = x_val[sel], d_val[sel], r10_val[sel], yo_val[sel]

    specs = [
        ("g14_base_l45_832", slice(0, D_BASE), "raw"),
        ("g14_base_l60_832", slice(D_BASE, D_BASE * 2), "raw"),
        ("g14_base_l60_orth", slice(D_BASE, D_BASE * 2), "orth"),
        ("g14_base_l60_multi", slice(D_BASE, D_BASE * 2), "multi"),
        ("g14_base_l75_832", slice(D_BASE * 2, FEAT_DIM), "raw"),
        ("g14_base_band_raw", slice(0, FEAT_DIM), "raw"),
        ("g14_base_band_orth", slice(0, FEAT_DIM), "orth"),
        ("g14_base_band_multi", slice(0, FEAT_DIM), "multi"),
    ]
    selection, heads_np = {}, {}
    for name, sl, key in specs:
        best_a, best_ic, grid = None, -1e9, []
        for a in ALPHAS_G14:
            w = solve_ridge_subblock(gram_tr, rhs_tr[key], sl, a)
            pred = x_sel[:, sl] @ w
            ic_r = daily_spearman_ic(d_sel, pred, r_sel)
            ic_o = daily_spearman_ic(d_sel, pred, yo_sel)
            sc = ic_o if key == "orth" else ic_r
            grid.append({"alpha": a, "val_ic_raw": ic_r, "val_ic_orth": ic_o})
            if sc > best_ic:
                best_ic, best_a = sc, a
        heads_np[name] = (sl, solve_ridge_subblock(gram_all, rhs_all[key], sl, best_a).astype(np.float32))
        selection[name] = {"selected_alpha": best_a, "best_val_ic": best_ic, "grid": grid}

    np.savez(OUTPUT / "g14_heads.npz", **{k: v[1] for k, v in heads_np.items()})
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(_score_oos_worker, [(r, n_gpus, base_dir, tok_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)])
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g14_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g14_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("g14_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
