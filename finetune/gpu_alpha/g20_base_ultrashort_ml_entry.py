"""G20: Kronos-base 2018-2026 Ultrashort + L=45 Multilayer Band [L20, L35, L45(L5,L8,L10), L52] Ridge."""
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
from g4_base_kaggle_entry import _load_kronos_base
from short_probe_common import (
    chunked_val_accumulate_and_sel,
    detect_gpu_count,
    norm_window,
    run_worker_oos,
    run_worker_stats,
    select_isotropic_heads,
)

OUTPUT = Path(os.environ.get("G20_OUTPUT", "/kaggle/working/g20_out"))
INPUT = Path(os.environ.get("G20_INPUT", "/kaggle/input"))
D_BASE = 832
FEAT_DIM = D_BASE * 6  # [L20_L10, L35_L10, L45_L5, L45_L8, L45_L10, L52_L10] = 4992
TRAIN_START_G20 = "2018-01-01"
TRAIN_STRIDE_G20 = 2
ALPHAS_G20 = (3e5, 1e6, 3e6, 1e7, 3e7, 1e8)


class BaseUltrashortMLExtractor:
    """Extract [L20_L10, L35_L10, L45_L5, L45_L8, L45_L10, L52_L10] from Kronos-base."""

    def __init__(self, base_model, tokenizer, device: torch.device):
        self.base = base_model
        self.tok = tokenizer
        self.device = device
        self._l5: torch.Tensor | None = None
        self._l8: torch.Tensor | None = None
        self._hooks = [
            base_model.transformer[5].register_forward_hook(self._save_l5),
            base_model.transformer[8].register_forward_hook(self._save_l8),
        ]

    def _save_l5(self, _mod, _inp, out):
        self._l5 = out[:, -1, :].detach().float()

    def _save_l8(self, _mod, _inp, out):
        self._l8 = out[:, -1, :].detach().float()

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            tok20 = self.tok.encode(norm_window(raw_t[:, -20:]), half=True)
            _, ctx20 = self.base.decode_s1(tok20[0], tok20[1], st_t[:, -20:], sector_id=sec_t, size_percentile=pct_t)
            tok35 = self.tok.encode(norm_window(raw_t[:, -35:]), half=True)
            _, ctx35 = self.base.decode_s1(tok35[0], tok35[1], st_t[:, -35:], sector_id=sec_t, size_percentile=pct_t)
            tok45 = self.tok.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx45 = self.base.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            l45_l5, l45_l8 = self._l5, self._l8
            tok52 = self.tok.encode(norm_window(raw_t[:, -52:]), half=True)
            _, ctx52 = self.base.decode_s1(tok52[0], tok52[1], st_t[:, -52:], sector_id=sec_t, size_percentile=pct_t)
        return torch.cat([
            ctx20[:, -1, :].float(),
            ctx35[:, -1, :].float(),
            l45_l5,
            l45_l8,
            ctx45[:, -1, :].float(),
            ctx52[:, -1, :].float(),
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_worker_ext(rank: int, base_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    base = _load_kronos_base(base_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, BaseUltrashortMLExtractor(base, tok, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_worker_ext(rank, base_dir, tok_dir)
    try:
        return run_worker_stats(
            tag="g20", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=384,
            extract_fn=ext.extract_batch, sym_tr=sym_tr, tr_batches=tr_batches,
            sym_val=sym_val, val_batches=val_batches, device=device,
        )
    finally:
        ext.close()


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, base_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    device, ext = _load_worker_ext(rank, base_dir, tok_dir)
    try:
        return run_worker_oos(
            tag="g20", rank=rank, n_gpus=n_gpus, batch_size=384,
            extract_fn=ext.extract_batch, sym_oos=sym_oos, oos_batches=oos_batches,
            heads_np=heads_np, device=device,
        )
    finally:
        ext.close()


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g20_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    base_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G20, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G20]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(_extract_worker, [(r, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)])
    del sym_tr, tr_batches, sym_val, val_batches

    gram_tr = np.zeros((FEAT_DIM, FEAT_DIM), dtype=np.float64)
    rhs_tr = {k: np.zeros(FEAT_DIM, dtype=np.float64) for k in ("raw", "orth", "multi")}
    n_tr_rows, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel = chunked_val_accumulate_and_sel(
        pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END, gram_tr, rhs_tr,
    )

    specs = [
        ("g20_base_l20", slice(0, D_BASE), "raw"),
        ("g20_base_l35", slice(D_BASE, D_BASE * 2), "raw"),
        ("g20_base_l45_832", slice(D_BASE * 4, D_BASE * 5), "raw"),
        ("g20_base_l52", slice(D_BASE * 5, FEAT_DIM), "raw"),
        ("g20_base_l45_ml_raw", slice(D_BASE * 2, D_BASE * 5), "raw"),
        ("g20_base_l45_ml_multi", slice(D_BASE * 2, D_BASE * 5), "multi"),
        ("g20_base_band20_52_raw", slice(0, FEAT_DIM), "raw"),
        ("g20_base_band20_52_multi", slice(0, FEAT_DIM), "multi"),
        ("g20_base_band20_52_orth", slice(0, FEAT_DIM), "orth"),
    ]
    selection, heads_np = select_isotropic_heads(
        specs=specs, alphas=ALPHAS_G20,
        gram_tr=gram_tr, rhs_tr=rhs_tr, gram_all=gram_all, rhs_all=rhs_all,
        x_sel=x_sel, d_sel=d_sel, r_sel=r_sel, yo_sel=yo_sel,
    )

    np.savez(OUTPUT / "g20_heads.npz", **{k: v[1] for k, v in heads_np.items()})
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(_score_oos_worker, [(r, n_gpus, base_dir, tok_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)])
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g20_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g20_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in (base_dir, tok_dir):
        shutil.rmtree(d, ignore_errors=True)
    phase("g20_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
