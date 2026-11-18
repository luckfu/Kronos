"""G19: Full-density (stride=1, 2021-2026) L=45 Tri-Arch Probe [C2 L120 + PT-Small L45[L4,L6] + Base L45[L5,L8,L10]]."""
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
from g16_pretrained_small_entry import _load_pretrained_small
from short_probe_common import (
    chunked_val_accumulate_and_sel,
    detect_gpu_count,
    find_c2_weights,
    norm_window,
    run_worker_oos,
    run_worker_stats,
    select_anisotropic_head,
    select_isotropic_heads,
)

OUTPUT = Path(os.environ.get("G19_OUTPUT", "/kaggle/working/g19_out"))
INPUT = Path(os.environ.get("G19_INPUT", "/kaggle/input"))
D_SMALL = 512
D_BASE = 832
FEAT_DIM = D_SMALL * 3 + D_BASE * 3  # 1536 + 2496 = 4032
SL_C2 = slice(0, D_SMALL)
SL_PTS_L4 = slice(D_SMALL, D_SMALL * 2)
SL_PTS_L6 = slice(D_SMALL * 2, D_SMALL * 3)
SL_PTS_DUAL = slice(D_SMALL, D_SMALL * 3)
SL_BASE_L5 = slice(D_SMALL * 3, D_SMALL * 3 + D_BASE)
SL_BASE_L8 = slice(D_SMALL * 3 + D_BASE, D_SMALL * 3 + D_BASE * 2)
SL_BASE_L10 = slice(D_SMALL * 3 + D_BASE * 2, FEAT_DIM)
SL_BASE_ML = slice(D_SMALL * 3, FEAT_DIM)
SL_ZS_ALL = slice(D_SMALL, FEAT_DIM)
SL_ALL = slice(0, FEAT_DIM)

TRAIN_START_G19 = "2021-01-01"
TRAIN_STRIDE_G19 = 1
ALPHAS_HIGH = (3e5, 1e6, 3e6, 1e7, 3e7, 1e8)
ALPHAS_C2 = (3e3, 1e4, 3e4, 1e5)


class TriArchL45Extractor:
    """Extract [C2_L120, PTS_L45_L4, PTS_L45_L6, Base_L45_L5, Base_L45_L8, Base_L45_L10] in one step."""

    def __init__(self, c2_model, pt_small, base_model, tokenizer, device: torch.device):
        self.c2 = c2_model
        self.pts = pt_small
        self.base = base_model
        self.tok = tokenizer
        self.device = device
        self._cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            pt_small.transformer[4].register_forward_hook(self._make_hook("pts4")),
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
            tok120 = self.tok.encode(norm_window(raw_t), half=True)
            _, ctx_c2 = self.c2.decode_s1(tok120[0], tok120[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            tok45 = self.tok.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx_pts = self.pts.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            _, ctx_base = self.base.decode_s1(tok45[0], tok45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
        return torch.cat([
            ctx_c2[:, -1, :].float(),
            self._cap["pts4"],
            ctx_pts[:, -1, :].float(),
            self._cap["b5"],
            self._cap["b8"],
            ctx_base[:, -1, :].float(),
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, c2_dir: str, pts_dir: str, base_dir: str, tok_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    c2 = Kronos.from_pretrained(
        c2_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    pts = _load_pretrained_small(pts_dir, device).eval()
    base = _load_kronos_base(base_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    return device, TriArchL45Extractor(c2, pts, base, tok, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, c2_dir, pts_dir, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, c2_dir, pts_dir, base_dir, tok_dir)
    try:
        return run_worker_stats(
            tag="g19", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=384,
            extract_fn=ext.extract_batch, sym_tr=sym_tr, tr_batches=tr_batches,
            sym_val=sym_val, val_batches=val_batches, device=device,
        )
    finally:
        ext.close()


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, c2_dir, pts_dir, base_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    device, ext = _load_models(rank, c2_dir, pts_dir, base_dir, tok_dir)
    try:
        return run_worker_oos(
            tag="g19", rank=rank, n_gpus=n_gpus, batch_size=384,
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
    phase("g19_started", gpus=n_gpus)

    c2_w = find_c2_weights(INPUT)
    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer"),
    ]
    pts_dir, base_dir, tok_dir = hf_dirs
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G19, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G19]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(
            _extract_worker,
            [(r, n_gpus, str(c2_w.parent), pts_dir, base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        )
    del sym_tr, tr_batches, sym_val, val_batches

    gram_tr = np.zeros((FEAT_DIM, FEAT_DIM), dtype=np.float64)
    rhs_tr = {k: np.zeros(FEAT_DIM, dtype=np.float64) for k in ("raw", "orth", "multi")}
    n_tr_rows, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel = chunked_val_accumulate_and_sel(
        pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END, gram_tr, rhs_tr,
    )

    selection, heads_np = select_isotropic_heads(
        specs=[
            ("g19_base_l45_full", SL_BASE_L10, "raw"),
            ("g19_base_l45_ml", SL_BASE_ML, "raw"),
            ("g19_pt_small_l45_full", SL_PTS_L6, "raw"),
            ("g19_pt_small_l45_dual", SL_PTS_DUAL, "raw"),
        ],
        alphas=ALPHAS_HIGH, gram_tr=gram_tr, rhs_tr=rhs_tr, gram_all=gram_all, rhs_all=rhs_all,
        x_sel=x_sel, d_sel=d_sel, r_sel=r_sel, yo_sel=yo_sel,
    )
    sel_c2, heads_c2 = select_isotropic_heads(
        specs=[("g19_c2_l120_anchor", SL_C2, "raw")],
        alphas=ALPHAS_C2, gram_tr=gram_tr, rhs_tr=rhs_tr, gram_all=gram_all, rhs_all=rhs_all,
        x_sel=x_sel, d_sel=d_sel, r_sel=r_sel, yo_sel=yo_sel,
    )
    selection.update(sel_c2)
    heads_np.update(heads_c2)

    zs_combos = [(a_pt, a_base) for a_pt in (1e6, 3e6, 1e7, 3e7) for a_base in (3e6, 1e7, 3e7, 1e8)]
    for name, key in (("g19_aniso_zs_l45_raw", "raw"), ("g19_aniso_zs_l45_multi", "multi")):
        sel_m, head_m = select_anisotropic_head(
            name, SL_ZS_ALL, key, [SL_PTS_DUAL, SL_BASE_ML], zs_combos,
            gram_tr, rhs_tr, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel,
        )
        selection[name] = sel_m
        heads_np[name] = head_m

    tri_combos = [(a_c2, a_zs) for a_c2 in (3e3, 1e4, 3e4) for a_zs in (3e6, 1e7, 3e7, 1e8)]
    for name, key in (
        ("g19_aniso_tri_l45_raw", "raw"),
        ("g19_aniso_tri_l45_multi", "multi"),
        ("g19_aniso_tri_l45_orth", "orth"),
    ):
        sel_m, head_m = select_anisotropic_head(
            name, SL_ALL, key, [SL_C2, SL_ZS_ALL], tri_combos,
            gram_tr, rhs_tr, gram_all, rhs_all, x_sel, d_sel, r_sel, yo_sel,
        )
        selection[name] = sel_m
        heads_np[name] = head_m

    np.savez(OUTPUT / "g19_heads.npz", **{k: v[1] for k, v in heads_np.items()})
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [(r, n_gpus, str(c2_w.parent), pts_dir, base_dir, tok_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)],
        )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g19_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g19_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g19_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
