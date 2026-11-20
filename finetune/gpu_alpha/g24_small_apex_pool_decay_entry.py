"""G24: Full-density (2021-2026 stride=1) Ultra-Reg + Temporal Pooling + Decay/Regime/5d Probe for C2 L120 & PT-Small L45."""
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
from g24_common import chunked_val_accumulate_weighted, run_worker_stats_weighted, select_weighted_heads
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window, run_worker_oos

OUTPUT = Path(os.environ.get("G24_OUTPUT", "/kaggle/working/g24_out"))
INPUT = Path(os.environ.get("G24_INPUT", "/kaggle/input"))
D_SMALL = 512
# [C2_last, C2_m5, C2_diff5, PTS_L3, PTS_L5, PTS_L6_last, PTS_L6_m5, PTS_L6_diff5] = 8 * 512 = 4096
FEAT_DIM = D_SMALL * 8
SL_C2_LAST = slice(0, D_SMALL)
SL_C2_POOL = slice(0, D_SMALL * 3)
SL_PTS_L3 = slice(D_SMALL * 3, D_SMALL * 4)
SL_PTS_L5 = slice(D_SMALL * 4, D_SMALL * 5)
SL_PTS_L6 = slice(D_SMALL * 5, D_SMALL * 6)
SL_PTS_ML356 = slice(D_SMALL * 3, D_SMALL * 6)
SL_PTS_L6_POOL = slice(D_SMALL * 5, D_SMALL * 8)
SL_PTS_ALL = slice(D_SMALL * 3, D_SMALL * 8)

TRAIN_START_G24 = "2021-01-01"
TRAIN_STRIDE_G24 = 1
ALPHAS_ULTRA = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_C2 = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)


class SmallApexPoolExtractor:
    """Single-pass extraction of pooled C2 L=120 and multi-layer pooled PT-Small L=45."""

    def __init__(self, c2_model, pt_small, tokenizer, device: torch.device):
        self.c2 = c2_model
        self.pts = pt_small
        self.tok = tokenizer
        self.device = device
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            pt_small.transformer[3].register_forward_hook(self._make_hook("l3")),
            pt_small.transformer[5].register_forward_hook(self._make_hook("l5")),
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
            tok120 = self.tok.encode(norm_window(raw_t), half=True)
            _, ctx_c2 = self.c2.decode_s1(tok120[0], tok120[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            tok45 = self.tok.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx_pts = self.pts.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
        c2_last = ctx_c2[:, -1, :].float()
        c2_m5 = ctx_c2[:, -5:, :].float().mean(dim=1)
        c2_diff5 = c2_last - c2_m5
        p6_last = ctx_pts[:, -1, :].float()
        p6_m5 = ctx_pts[:, -5:, :].float().mean(dim=1)
        p6_diff5 = p6_last - p6_m5
        return torch.cat([c2_last, c2_m5, c2_diff5, self._cap["l3"], self._cap["l5"], p6_last, p6_m5, p6_diff5], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, c2_dir: str, pts_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    c2 = Kronos.from_pretrained(
        c2_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    pts = _load_pretrained_small(pts_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, SmallApexPoolExtractor(c2, pts, tok, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, c2_dir, pts_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, c2_dir, pts_dir, tok_dir)
    try:
        return run_worker_stats_weighted(
            tag="g24", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=512,
            extract_fn=ext.extract_batch, sym_tr=sym_tr, tr_batches=tr_batches,
            sym_val=sym_val, val_batches=val_batches, device=device,
        )
    finally:
        ext.close()


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, c2_dir, pts_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    device, ext = _load_models(rank, c2_dir, pts_dir, tok_dir)
    try:
        return run_worker_oos(
            tag="g24", rank=rank, n_gpus=n_gpus, batch_size=512,
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
    phase("g24_started", gpus=n_gpus)

    c2_w = find_c2_weights(INPUT)
    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    pts_dir = snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G24, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G24]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(
            _extract_worker,
            [(r, n_gpus, str(c2_w.parent), pts_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        )
    del sym_tr, tr_batches, sym_val, val_batches

    stats = chunked_val_accumulate_weighted(pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END)
    for p in pts:
        Path(p).unlink(missing_ok=True)

    specs = [
        ("g24_c2_512_decay", SL_C2_LAST, "decay", "raw", ALPHAS_C2),
        ("g24_c2_512_multi_decay", SL_C2_LAST, "decay", "multi", ALPHAS_C2),
        ("g24_c2_512_regime", SL_C2_LAST, "regime", "raw", ALPHAS_C2),
        ("g24_c2_512_ret5", SL_C2_LAST, "std", "ret5", ALPHAS_C2),
        ("g24_c2_pool1536", SL_C2_POOL, "std", "raw", ALPHAS_C2),
        ("g24_pts_l45_l6_ultra", SL_PTS_L6, "std", "raw", ALPHAS_ULTRA),
        ("g24_pts_l45_l6_pool", SL_PTS_L6_POOL, "std", "raw", ALPHAS_ULTRA),
        ("g24_pts_l45_ml356_raw", SL_PTS_ML356, "std", "raw", ALPHAS_ULTRA),
        ("g24_pts_l45_ml356_multi", SL_PTS_ML356, "std", "multi", ALPHAS_ULTRA),
        ("g24_pts_l45_ml356_decay", SL_PTS_ML356, "decay", "multi", ALPHAS_ULTRA),
        ("g24_pts_l45_ml356_regime", SL_PTS_ML356, "regime", "multi", ALPHAS_ULTRA),
        ("g24_pts_l45_ml356_ret5", SL_PTS_ML356, "std", "ret5", ALPHAS_ULTRA),
        ("g24_pts_l45_full2560_raw", SL_PTS_ALL, "std", "raw", ALPHAS_ULTRA),
        ("g24_pts_l45_full2560_multi", SL_PTS_ALL, "std", "multi", ALPHAS_ULTRA),
    ]
    selection, heads_np = select_weighted_heads(specs, stats)
    np.savez(OUTPUT / "g24_heads.npz", **{k: v[1] for k, v in heads_np.items()})

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [(r, n_gpus, str(c2_w.parent), pts_dir, tok_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)],
        )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g24_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_train_rows": stats["n_tr_rows"],
        "n_weak_days": stats["n_weak_days"],
        "selection": selection,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g24_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in (pts_dir, tok_dir):
        shutil.rmtree(d, ignore_errors=True)
    phase("g24_finished", n_train_rows=stats["n_tr_rows"], elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
