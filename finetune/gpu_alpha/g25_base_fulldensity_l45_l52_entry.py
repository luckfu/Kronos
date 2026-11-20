"""G25: Full-density (2021-2026 stride=1) Kronos-base L=45 + L=52 Ultra-Reg + Pooling + Decay/Regime/5d Probe."""
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
from g24_common import chunked_val_accumulate_weighted, run_worker_stats_weighted, select_weighted_heads
from short_probe_common import detect_gpu_count, norm_window, run_worker_oos

OUTPUT = Path(os.environ.get("G25_OUTPUT", "/kaggle/working/g25_out"))
INPUT = Path(os.environ.get("G25_INPUT", "/kaggle/input"))
D_BASE = 832
# [L45_L5, L45_L8, L45_L10_last, L45_L10_m5, L45_L10_diff5, L52_L10_last, L52_L10_m5] = 7 * 832 = 5824
FEAT_DIM = D_BASE * 7
SL_45_L5 = slice(0, D_BASE)
SL_45_L8 = slice(D_BASE, D_BASE * 2)
SL_45_L10 = slice(D_BASE * 2, D_BASE * 3)
SL_45_ML = slice(0, D_BASE * 3)
SL_45_POOL = slice(D_BASE * 2, D_BASE * 5)
SL_45_ALL = slice(0, D_BASE * 5)
SL_52_L10 = slice(D_BASE * 5, D_BASE * 6)
SL_52_POOL = slice(D_BASE * 5, D_BASE * 7)
SL_ALL = slice(0, FEAT_DIM)

TRAIN_START_G25 = "2021-01-01"
TRAIN_STRIDE_G25 = 1
ALPHAS_ULTRA = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)


class BaseApexL45L52Extractor:
    """Extract L=45 [L5, L8, L10, L10_m5, L10_diff5] and L=52 [L10, L10_m5] from Kronos-base."""

    def __init__(self, base_model, tokenizer, device: torch.device):
        self.base = base_model
        self.tok = tokenizer
        self.device = device
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            base_model.transformer[5].register_forward_hook(self._make_hook("b5")),
            base_model.transformer[8].register_forward_hook(self._make_hook("b8")),
        ]

    def _make_hook(self, key: str):
        def _hook(_mod, _inp, out):
            self._cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            tok45 = self.tok.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx45 = self.base.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            l45_l5 = self._cap["b5"]
            l45_l8 = self._cap["b8"]
            tok52 = self.tok.encode(norm_window(raw_t[:, -52:]), half=True)
            _, ctx52 = self.base.decode_s1(tok52[0], tok52[1], st_t[:, -52:], sector_id=sec_t, size_percentile=pct_t)
        l45_last = ctx45[:, -1, :].float()
        l45_m5 = ctx45[:, -5:, :].float().mean(dim=1)
        l45_diff5 = l45_last - l45_m5
        l52_last = ctx52[:, -1, :].float()
        l52_m5 = ctx52[:, -5:, :].float().mean(dim=1)
        return torch.cat([l45_l5, l45_l8, l45_last, l45_m5, l45_diff5, l52_last, l52_m5], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, base_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    base = _load_kronos_base(base_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, BaseApexL45L52Extractor(base, tok, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, base_dir, tok_dir)
    try:
        return run_worker_stats_weighted(
            tag="g25", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=384,
            extract_fn=ext.extract_batch, sym_tr=sym_tr, tr_batches=tr_batches,
            sym_val=sym_val, val_batches=val_batches, device=device,
        )
    finally:
        ext.close()


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, base_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    device, ext = _load_models(rank, base_dir, tok_dir)
    try:
        return run_worker_oos(
            tag="g25", rank=rank, n_gpus=n_gpus, batch_size=384,
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
    phase("g25_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    base_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G25, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G25]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(
            _extract_worker,
            [(r, n_gpus, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        )
    del sym_tr, tr_batches, sym_val, val_batches

    stats = chunked_val_accumulate_weighted(pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END)
    for p in pts:
        Path(p).unlink(missing_ok=True)

    specs = [
        ("g25_base_l45_ultra", SL_45_L10, "std", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_decay", SL_45_L10, "decay", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_regime", SL_45_L10, "regime", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_ret5", SL_45_L10, "std", "ret5", ALPHAS_ULTRA),
        ("g25_base_l45_pool", SL_45_POOL, "std", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_ml_ultra", SL_45_ML, "std", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_ml_multi", SL_45_ML, "std", "multi", ALPHAS_ULTRA),
        ("g25_base_l45_ml_decay", SL_45_ML, "decay", "multi", ALPHAS_ULTRA),
        ("g25_base_l45_ml_regime", SL_45_ML, "regime", "multi", ALPHAS_ULTRA),
        ("g25_base_l52_full", SL_52_L10, "std", "raw", ALPHAS_ULTRA),
        ("g25_base_l52_pool", SL_52_POOL, "std", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_52_raw", SL_ALL, "std", "raw", ALPHAS_ULTRA),
        ("g25_base_l45_52_multi", SL_ALL, "std", "multi", ALPHAS_ULTRA),
    ]
    selection, heads_np = select_weighted_heads(specs, stats)
    np.savez(OUTPUT / "g25_heads.npz", **{k: v[1] for k, v in heads_np.items()})

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [(r, n_gpus, base_dir, tok_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)],
        )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g25_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_train_rows": stats["n_tr_rows"],
        "n_weak_days": stats["n_weak_days"],
        "selection": selection,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g25_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in (base_dir, tok_dir):
        shutil.rmtree(d, ignore_errors=True)
    phase("g25_finished", n_train_rows=stats["n_tr_rows"], elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
