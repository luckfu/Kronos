"""G22: Fast Pretrained Kronos-small Micro-Window Sweep [L16, L24, L32, L40] (2021-2026 stride=3)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from g1_kaggle_entry import build_oos_batches, find_dataset_file
from g1_multiscale_extract_and_fit import TRAIN_FIT_END, VAL_SEL_END, VAL_SEL_START, build_fast_metadata_lookup, phase
from g2_rank_finetune import build_g2_date_batches
from g16_pretrained_small_entry import _load_pretrained_small
from short_probe_common import (
    chunked_val_accumulate_and_sel,
    detect_gpu_count,
    norm_window,
    run_worker_oos,
    run_worker_stats,
    select_isotropic_heads,
)

OUTPUT = Path(os.environ.get("G22_OUTPUT", "/kaggle/working/g22_out"))
INPUT = Path(os.environ.get("G22_INPUT", "/kaggle/input"))
D_SMALL = 512
WINDOWS = (16, 24, 32, 40)
FEAT_DIM = D_SMALL * len(WINDOWS)  # 2048
TRAIN_START_G22 = "2021-01-01"
TRAIN_STRIDE_G22 = 3
ALPHAS_G22 = (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8)


@torch.inference_mode()
def extract_microband_batch(
    pts_model,
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
            toks = tokenizer.encode(norm_window(raw_t[:, -w:]), half=True)
            _, ctx = pts_model.decode_s1(toks[0], toks[1], st_t[:, -w:], sector_id=sec_t, size_percentile=pct_t)
            parts.append(ctx[:, -1, :].float())
    return torch.cat(parts, dim=1)


def _load_worker_models(rank: int, small_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    pts = _load_pretrained_small(small_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, pts, tok


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, small_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, pts, tok = _load_worker_models(rank, small_dir, tok_dir)
    return run_worker_stats(
        tag="g22", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=512,
        extract_fn=lambda r, s, sc, p: extract_microband_batch(pts, tok, r, s, sc, p, device),
        sym_tr=sym_tr, tr_batches=tr_batches, sym_val=sym_val, val_batches=val_batches, device=device,
    )


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, small_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    device, pts, tok = _load_worker_models(rank, small_dir, tok_dir)
    return run_worker_oos(
        tag="g22", rank=rank, n_gpus=n_gpus, batch_size=512,
        extract_fn=lambda r, s, sc, p: extract_microband_batch(pts, tok, r, s, sc, p, device),
        sym_oos=sym_oos, oos_batches=oos_batches, heads_np=heads_np, device=device,
    )


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g22_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    small_dir = snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G22, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G22]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(_extract_worker, [(r, n_gpus, small_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)])
    del sym_tr, tr_batches, sym_val, val_batches

    gram_tr = np.zeros((FEAT_DIM, FEAT_DIM), dtype=np.float64)
    rhs_tr = {k: np.zeros(FEAT_DIM, dtype=np.float64) for k in ("raw", "orth", "multi")}
    n_tr_rows, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel = chunked_val_accumulate_and_sel(
        pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END, gram_tr, rhs_tr,
    )

    specs = [
        ("g22_pts_l16", slice(0, D_SMALL), "raw"),
        ("g22_pts_l24", slice(D_SMALL, D_SMALL * 2), "raw"),
        ("g22_pts_l32", slice(D_SMALL * 2, D_SMALL * 3), "raw"),
        ("g22_pts_l40", slice(D_SMALL * 3, FEAT_DIM), "raw"),
        ("g22_pts_micro_raw", slice(0, FEAT_DIM), "raw"),
        ("g22_pts_micro_multi", slice(0, FEAT_DIM), "multi"),
        ("g22_pts_micro_orth", slice(0, FEAT_DIM), "orth"),
    ]
    selection, heads_np = select_isotropic_heads(
        specs=specs, alphas=ALPHAS_G22,
        gram_tr=gram_tr, rhs_tr=rhs_tr, gram_all=gram_all, rhs_all=rhs_all,
        x_sel=x_sel, d_sel=d_sel, r_sel=r_sel, yo_sel=yo_sel,
    )

    np.savez(OUTPUT / "g22_heads.npz", **{k: v[1] for k, v in heads_np.items()})
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(_score_oos_worker, [(r, n_gpus, small_dir, tok_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)])
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g22_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g22_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in (small_dir, tok_dir):
        shutil.rmtree(d, ignore_errors=True)
    phase("g22_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
