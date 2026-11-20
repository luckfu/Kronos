"""G29: Full-density (2021-2026 stride=1) Kronos-mini (Tok-2k) All-5-Layer + Temporal Pooling + Regime/Ret5
+ Cross-Family Mini+Base and Tri-Family (Mini+Base+PT-Small) Joint Ridge at L=45.

Strictly leak-free: train <= 2026-04-30, val selection 2026-05-18..2026-07-02, OOS 2026-07-17..2026-10-23.
"""
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
from g24_common import chunked_val_accumulate_weighted, run_worker_stats_weighted, select_weighted_heads
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from short_probe_common import detect_gpu_count, norm_window, run_worker_oos

OUTPUT = Path(os.environ.get("G29_OUTPUT", "/kaggle/working/g29_out"))
INPUT = Path(os.environ.get("G29_INPUT", "/kaggle/input"))
D_MINI = 256
D_BASE = 832
D_SMALL = 512

# Contiguous block layout (FEAT_DIM = 2048 + 2496 + 2048 = 6592):
# Mini (2048): [m45_l0, m45_l2, m45_l3_pre, m45_l1, m45_l4_last, m45_l4_m5, m45_l4_diff5, m30_l4_last]
# Base (2496): [b45_l5, b45_l8, b45_l10_last]
# PTS  (2048): [p45_l3, p45_l5, p45_l7_last, p45_l7_m5]
MINI_DIM = D_MINI * 8
BASE_DIM = D_BASE * 3
PTS_DIM = D_SMALL * 4
FEAT_DIM = MINI_DIM + BASE_DIM + PTS_DIM  # 6592

SL_MINI_45_ALL5 = slice(0, D_MINI * 5)
SL_MINI_45_POOL = slice(D_MINI * 3, D_MINI * 7)
SL_MINI_45_FULL = slice(0, D_MINI * 7)
SL_MINI_ALL = slice(0, MINI_DIM)
SL_MINI_BASE = slice(0, MINI_DIM + BASE_DIM)
SL_TRIFAM = slice(0, FEAT_DIM)

TRAIN_START_G29 = "2021-01-01"
TRAIN_STRIDE_G29 = 1
ALPHAS_MINI_POOL = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_JOINT = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)


class MiniPoolTriFamilyExtractor:
    """Single L=45 pass (plus tiny Mini L=30) across Mini-2k, Base, and PT-Small."""

    def __init__(self, mini_model, tok2k, base_model, pts_model, tok_base, device: torch.device):
        self.mini = mini_model
        self.tok2k = tok2k
        self.base = base_model
        self.pts = pts_model
        self.tok_base = tok_base
        self.device = device
        self._m_cap: dict[int, torch.Tensor] = {}
        self._b_cap: dict[str, torch.Tensor] = {}
        self._p_cap: dict[str, torch.Tensor] = {}
        self._hooks = [
            mini_model.transformer[i].register_forward_hook(self._make_m_hook(i)) for i in range(4)
        ] + [
            base_model.transformer[5].register_forward_hook(self._make_b_hook("b5")),
            base_model.transformer[8].register_forward_hook(self._make_b_hook("b8")),
            pts_model.transformer[3].register_forward_hook(self._make_p_hook("p3")),
            pts_model.transformer[5].register_forward_hook(self._make_p_hook("p5")),
        ]

    def _make_m_hook(self, idx: int):
        def _hook(_m, _i, out):
            self._m_cap[idx] = out[:, -1, :].detach().float()
        return _hook

    def _make_b_hook(self, key: str):
        def _hook(_m, _i, out):
            self._b_cap[key] = out[:, -1, :].detach().float()
        return _hook

    def _make_p_hook(self, key: str):
        def _hook(_m, _i, out):
            self._p_cap[key] = out[:, -1, :].detach().float()
        return _hook

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b, device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b, device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            t30 = self.tok2k.encode(norm_window(raw_t[:, -30:]), half=True)
            _, m30 = self.mini.decode_s1(t30[0], t30[1], st_t[:, -30:])
            m30_last = m30[:, -1, :].float()

            t45 = self.tok2k.encode(norm_window(raw_t[:, -45:]), half=True)
            _, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])

            tb45 = self.tok_base.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx_base = self.base.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)
            _, ctx_pts = self.pts.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)

        m45_last = m45[:, -1, :].float()
        m45_m5 = m45[:, -5:, :].float().mean(dim=1)
        m45_diff5 = m45_last - m45_m5

        p7_last = ctx_pts[:, -1, :].float()
        p7_m5 = ctx_pts[:, -5:, :].float().mean(dim=1)

        return torch.cat([
            self._m_cap[0], self._m_cap[2], self._m_cap[3], self._m_cap[1], m45_last, m45_m5, m45_diff5, m30_last,
            self._b_cap["b5"], self._b_cap["b8"], ctx_base[:, -1, :].float(),
            self._p_cap["p3"], self._p_cap["p5"], p7_last, p7_m5,
        ], dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, mini_dir: str, tok2k_dir: str, base_dir: str, pts_dir: str, tok_base_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(mini_dir, device).eval()
    tok2k = KronosTokenizer.from_pretrained(tok2k_dir).to(device).eval()
    base = _load_kronos_base(base_dir, device).eval()
    pts = _load_pretrained_small(pts_dir, device).eval()
    tok_base = KronosTokenizer.from_pretrained(tok_base_dir).to(device).eval()
    return device, MiniPoolTriFamilyExtractor(mini, tok2k, base, pts, tok_base, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir)
    try:
        return run_worker_stats_weighted(
            tag="g29", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=384,
            extract_fn=ext.extract_batch, sym_tr=sym_tr, tr_batches=tr_batches,
            sym_val=sym_val, val_batches=val_batches, device=device,
        )
    finally:
        ext.close()


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir, sym_oos, oos_batches, heads_np = args
    device, ext = _load_models(rank, mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir)
    try:
        return run_worker_oos(
            tag="g29", rank=rank, n_gpus=n_gpus, batch_size=384,
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
    phase("g29_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir = hf_dirs
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G29, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G29]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(
            _extract_worker,
            [(r, n_gpus, mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir, sym_tr, tr_batches, sym_val, val_batches)
             for r in range(n_gpus)],
        )
    del sym_tr, tr_batches, sym_val, val_batches

    stats = chunked_val_accumulate_weighted(pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END)
    for p in pts:
        Path(p).unlink(missing_ok=True)

    specs = [
        ("g29_mini_l45_all5", SL_MINI_45_ALL5, "std", "raw", ALPHAS_MINI_POOL),
        ("g29_mini_l45_pool_raw", SL_MINI_45_POOL, "std", "raw", ALPHAS_MINI_POOL),
        ("g29_mini_l45_pool_multi", SL_MINI_45_POOL, "std", "multi", ALPHAS_MINI_POOL),
        ("g29_mini_l45_pool_decay", SL_MINI_45_POOL, "decay", "multi", ALPHAS_MINI_POOL),
        ("g29_mini_l45_pool_regime", SL_MINI_45_POOL, "regime", "multi", ALPHAS_MINI_POOL),
        ("g29_mini_l45_pool_ret5", SL_MINI_45_POOL, "std", "ret5", ALPHAS_MINI_POOL),
        ("g29_mini_l45_full", SL_MINI_45_FULL, "std", "raw", ALPHAS_MINI_POOL),
        ("g29_mini_l30_45_raw", SL_MINI_ALL, "std", "raw", ALPHAS_MINI_POOL),
        ("g29_mini_l30_45_regime", SL_MINI_ALL, "regime", "multi", ALPHAS_MINI_POOL),
        ("g29_mini_base_raw", SL_MINI_BASE, "std", "raw", ALPHAS_JOINT),
        ("g29_mini_base_regime", SL_MINI_BASE, "regime", "multi", ALPHAS_JOINT),
        ("g29_trifam_l45_raw", SL_TRIFAM, "std", "raw", ALPHAS_JOINT),
        ("g29_trifam_l45_multi", SL_TRIFAM, "std", "multi", ALPHAS_JOINT),
        ("g29_trifam_l45_decay", SL_TRIFAM, "decay", "multi", ALPHAS_JOINT),
        ("g29_trifam_l45_regime", SL_TRIFAM, "regime", "multi", ALPHAS_JOINT),
        ("g29_trifam_l45_ret5", SL_TRIFAM, "std", "ret5", ALPHAS_JOINT),
    ]
    selection, heads_np = select_weighted_heads(specs, stats)
    np.savez(OUTPUT / "g29_heads.npz", **{k: v[1] for k, v in heads_np.items()})

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [(r, n_gpus, mini_dir, tok2k_dir, base_dir, pts_dir, tok_base_dir, sym_oos, oos_batches, heads_np)
             for r in range(n_gpus)],
        )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g29_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_train_rows": stats["n_tr_rows"],
        "n_weak_days": stats["n_weak_days"],
        "selection": selection,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g29_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g29_finished", n_train_rows=stats["n_tr_rows"], elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
