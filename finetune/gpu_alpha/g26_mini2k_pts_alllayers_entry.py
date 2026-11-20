"""G26: Full-density (2021-2026 stride=1) Kronos-mini (4.1M, Tokenizer-2k) + PT-Small L=45 All-8-Layer Spectrum."""
from __future__ import annotations

import inspect
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
from short_probe_common import detect_gpu_count, norm_window, run_worker_oos

OUTPUT = Path(os.environ.get("G26_OUTPUT", "/kaggle/working/g26_out"))
INPUT = Path(os.environ.get("G26_INPUT", "/kaggle/input"))
D_MINI = 256
D_SMALL = 512
# Mini: [L30_L3, L45_L1, L45_L3, L60_L3, L120_L3] = 5 * 256 = 1280
# PTS L=45: [L0, L1, L2, L3, L4, L5, L6, L7(norm)] = 8 * 512 = 4096
MINI_DIM = D_MINI * 5
FEAT_DIM = MINI_DIM + D_SMALL * 8  # 5376

SL_MINI_L30 = slice(0, D_MINI)
SL_MINI_L45_L1 = slice(D_MINI, D_MINI * 2)
SL_MINI_L45_L3 = slice(D_MINI * 2, D_MINI * 3)
SL_MINI_L45_ML = slice(D_MINI, D_MINI * 3)
SL_MINI_L60 = slice(D_MINI * 3, D_MINI * 4)
SL_MINI_L120 = slice(D_MINI * 4, MINI_DIM)
SL_MINI_BAND = slice(0, MINI_DIM)

SL_PTS_LAYERS = [slice(MINI_DIM + i * D_SMALL, MINI_DIM + (i + 1) * D_SMALL) for i in range(8)]
SL_PTS_MIDDEEP = slice(MINI_DIM + 3 * D_SMALL, FEAT_DIM)  # L3..L7 = 2560
SL_PTS_ALL8 = slice(MINI_DIM, FEAT_DIM)                   # L0..L7 = 4096
SL_ALL = slice(0, FEAT_DIM)

TRAIN_START_G26 = "2021-01-01"
TRAIN_STRIDE_G26 = 1
ALPHAS_MINI = (1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9)
ALPHAS_ULTRA = (3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
SCHEMES_G26 = ("std", "decay")


def _load_pretrained_mini(hf_dir: str, device: torch.device):
    from safetensors.torch import load_file as load_safetensors
    from model import Kronos

    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    valid = set(inspect.signature(Kronos.__init__).parameters.keys()) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in valid}
    model = Kronos(**cfg)
    model.load_state_dict(load_safetensors(str(Path(hf_dir) / "model.safetensors")), strict=False)
    return model.to(device)


class MiniAndPTSAllLayerExtractor:
    """Extract Kronos-mini (Tokenizer-2k) multi-lookback + PT-Small L=45 all 8 layers."""

    def __init__(self, mini_model, tok2k, pts_model, tok_base, device: torch.device):
        self.mini = mini_model
        self.tok2k = tok2k
        self.pts = pts_model
        self.tok_base = tok_base
        self.device = device
        self._mini_l1: torch.Tensor | None = None
        self._pts_layers: dict[int, torch.Tensor] = {}
        self._hooks = [mini_model.transformer[1].register_forward_hook(self._save_mini_l1)]
        for idx in range(7):
            self._hooks.append(pts_model.transformer[idx].register_forward_hook(self._make_pts_hook(idx)))

    def _save_mini_l1(self, _mod, _inp, out):
        self._mini_l1 = out[:, -1, :].detach().float()

    def _make_pts_hook(self, idx: int):
        def _hook(_mod, _inp, out):
            self._pts_layers[idx] = out[:, -1, :].detach().float()
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
            t45 = self.tok2k.encode(norm_window(raw_t[:, -45:]), half=True)
            _, m45 = self.mini.decode_s1(t45[0], t45[1], st_t[:, -45:])
            m45_l1 = self._mini_l1
            t60 = self.tok2k.encode(norm_window(raw_t[:, -60:]), half=True)
            _, m60 = self.mini.decode_s1(t60[0], t60[1], st_t[:, -60:])
            t120 = self.tok2k.encode(norm_window(raw_t), half=True)
            _, m120 = self.mini.decode_s1(t120[0], t120[1], st_t)

            tb45 = self.tok_base.encode(norm_window(raw_t[:, -45:]), half=True)
            _, ctx_pts = self.pts.decode_s1(tb45[0], tb45[1], st_t[:, -45:], sector_id=sec_t, size_percentile=pct_t)

        parts = [
            m30[:, -1, :].float(),
            m45_l1,
            m45[:, -1, :].float(),
            m60[:, -1, :].float(),
            m120[:, -1, :].float(),
        ] + [self._pts_layers[i] for i in range(7)] + [ctx_pts[:, -1, :].float()]
        return torch.cat(parts, dim=1)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()


def _load_models(rank: int, mini_dir: str, tok2k_dir: str, pts_dir: str, tok_base_dir: str):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(mini_dir, device).eval()
    tok2k = KronosTokenizer.from_pretrained(tok2k_dir).to(device).eval()
    pts = _load_pretrained_small(pts_dir, device).eval()
    tok_base = KronosTokenizer.from_pretrained(tok_base_dir).to(device).eval()
    return device, MiniAndPTSAllLayerExtractor(mini, tok2k, pts, tok_base, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, mini_dir, tok2k_dir, pts_dir, tok_base_dir, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, mini_dir, tok2k_dir, pts_dir, tok_base_dir)
    try:
        return run_worker_stats_weighted(
            tag="g26", rank=rank, n_gpus=n_gpus, feat_dim=FEAT_DIM, batch_size=512,
            extract_fn=ext.extract_batch, sym_tr=sym_tr, tr_batches=tr_batches,
            sym_val=sym_val, val_batches=val_batches, device=device, schemes=SCHEMES_G26,
        )
    finally:
        ext.close()


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, mini_dir, tok2k_dir, pts_dir, tok_base_dir, sym_oos, oos_batches, heads_np = args
    device, ext = _load_models(rank, mini_dir, tok2k_dir, pts_dir, tok_base_dir)
    try:
        return run_worker_oos(
            tag="g26", rank=rank, n_gpus=n_gpus, batch_size=512,
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
    phase("g26_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    mini_dir, tok2k_dir, pts_dir, tok_base_dir = hf_dirs
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G26, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G26]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        pts = pool.map(
            _extract_worker,
            [(r, n_gpus, mini_dir, tok2k_dir, pts_dir, tok_base_dir, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        )
    del sym_tr, tr_batches, sym_val, val_batches

    stats = chunked_val_accumulate_weighted(pts, FEAT_DIM, VAL_SEL_START, VAL_SEL_END, schemes=SCHEMES_G26)
    for p in pts:
        Path(p).unlink(missing_ok=True)

    specs = [
        ("g26_mini_l30", SL_MINI_L30, "std", "raw", ALPHAS_MINI),
        ("g26_mini_l45_l3", SL_MINI_L45_L3, "std", "raw", ALPHAS_MINI),
        ("g26_mini_l45_ml", SL_MINI_L45_ML, "std", "raw", ALPHAS_MINI),
        ("g26_mini_l60", SL_MINI_L60, "std", "raw", ALPHAS_MINI),
        ("g26_mini_l120", SL_MINI_L120, "std", "raw", ALPHAS_MINI),
        ("g26_mini_band_raw", SL_MINI_BAND, "std", "raw", ALPHAS_MINI),
        ("g26_mini_band_multi", SL_MINI_BAND, "std", "multi", ALPHAS_MINI),
        ("g26_mini_band_decay", SL_MINI_BAND, "decay", "multi", ALPHAS_MINI),
        ("g26_pts_l45_l0", SL_PTS_LAYERS[0], "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_l1", SL_PTS_LAYERS[1], "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_l2", SL_PTS_LAYERS[2], "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_l4", SL_PTS_LAYERS[4], "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_l6_pre", SL_PTS_LAYERS[6], "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_middeep_raw", SL_PTS_MIDDEEP, "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_middeep_multi", SL_PTS_MIDDEEP, "std", "multi", ALPHAS_ULTRA),
        ("g26_pts_l45_all8_raw", SL_PTS_ALL8, "std", "raw", ALPHAS_ULTRA),
        ("g26_pts_l45_all8_multi", SL_PTS_ALL8, "std", "multi", ALPHAS_ULTRA),
        ("g26_mini_pts_joint_raw", SL_ALL, "std", "raw", ALPHAS_ULTRA),
        ("g26_mini_pts_joint_multi", SL_ALL, "std", "multi", ALPHAS_ULTRA),
    ]
    selection, heads_np = select_weighted_heads(specs, stats)
    np.savez(OUTPUT / "g26_heads.npz", **{k: v[1] for k, v in heads_np.items()})

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [(r, n_gpus, mini_dir, tok2k_dir, pts_dir, tok_base_dir, sym_oos, oos_batches, heads_np) for r in range(n_gpus)],
        )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g26_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_train_rows": stats["n_tr_rows"],
        "n_weak_days": stats["n_weak_days"],
        "selection": selection,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g26_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g26_finished", n_train_rows=stats["n_tr_rows"], elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
