"""G10: Residual CrossStockPeerRanker with frozen closed-form G1_UNIV_LAST linear skip (alpha=1e4)."""
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
import torch.nn as nn

from g1_kaggle_entry import build_oos_batches, find_dataset_file
from g1_multiscale_extract_and_fit import (
    TRAIN_FIT_END,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    daily_spearman_ic,
    phase,
    rank_pct_centered,
    standardize_1d,
)
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec
from g3_cross_stock_context import CrossStockPeerRanker
from g3_kaggle_entry import _extract_h120_last_worker

OUTPUT = Path(os.environ.get("G10_OUTPUT", "/kaggle/working/g10_out"))
INPUT = Path(os.environ.get("G10_INPUT", "/kaggle/input"))
TRAIN_START_G10 = "2022-01-01"
TRAIN_STRIDE_G10 = 2


class FrozenSkipPeerRanker(nn.Module):
    """Wrap CrossStockPeerRanker with a frozen closed-form linear skip and scaled residual gate."""

    def __init__(self, w_skip: np.ndarray, proj_dim: int = 96, dropout: float = 0.45, resid_scale: float = 0.25):
        super().__init__()
        self.core = CrossStockPeerRanker(in_dim=len(w_skip), proj_dim=proj_dim, num_sectors=86, num_heads=4, dropout=dropout)
        with torch.no_grad():
            self.core.skip.weight.copy_(torch.as_tensor(w_skip, dtype=torch.float32).unsqueeze(0))
            self.core.skip.bias.zero_()
        for p in self.core.skip.parameters():
            p.requires_grad_(False)
        self.resid_scale = float(resid_scale)

    def forward(self, x: torch.Tensor, sector_id: torch.Tensor) -> torch.Tensor:
        skip_out = self.core.skip(x).squeeze(-1)
        full_out = self.core(x, sector_id)
        return skip_out + self.resid_scale * (full_out - skip_out)


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


def fit_raw_ridge_512(days: list[dict], alpha: float = 1e4) -> np.ndarray:
    gram = np.zeros((512, 512), dtype=np.float64)
    rhs = np.zeros(512, dtype=np.float64)
    for d in days:
        x = d["x"].astype(np.float64)
        y = standardize_1d(rank_pct_centered(d["ret10"])).astype(np.float64)
        gram += x.T @ x
        rhs += x.T @ y
    return np.linalg.solve(gram + alpha * np.eye(512), rhs).astype(np.float32)


def train_frozen_skip_peer(
    train_days: list[dict],
    val_days: list[dict] | None,
    w_skip: np.ndarray,
    mode: str,
    resid_scale: float,
    wd: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[FrozenSkipPeerRanker, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FrozenSkipPeerRanker(w_skip=w_skip, proj_dim=96, dropout=0.45, resid_scale=resid_scale).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * len(train_days), 1), eta_min=1e-5)

    val_curve: list[float] = []
    for _ in range(epochs):
        model.train()
        for di in rng.permutation(len(train_days)):
            d = train_days[di]
            x = torch.as_tensor(d["x"], device=device, dtype=torch.float32)
            sec = torch.as_tensor(d["sec"], device=device, dtype=torch.long)
            yr = torch.as_tensor(standardize_1d(rank_pct_centered(d["ret10"])), device=device, dtype=torch.float32)
            yo = torch.as_tensor(d["y_orth"], device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(x, sec)
            loss = cosine_ic_loss_vec(pred, yr) if mode == "raw" else 0.7 * cosine_ic_loss_vec(pred, yr) + 0.3 * cosine_ic_loss_vec(pred, yo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
        if val_days is not None:
            model.eval()
            dates_v, preds_v, r10_v, yo_v = [], [], [], []
            with torch.inference_mode():
                for d in val_days:
                    x = torch.as_tensor(d["x"], device=device, dtype=torch.float32)
                    sec = torch.as_tensor(d["sec"], device=device, dtype=torch.long)
                    p = model(x, sec).cpu().numpy()
                    preds_v.append(p)
                    dates_v.extend([d["date"]] * len(p))
                    r10_v.append(d["ret10"])
                    yo_v.append(d["y_orth"])
            d_arr = np.asarray(dates_v)
            p_arr = np.concatenate(preds_v)
            ic_r = daily_spearman_ic(d_arr, p_arr, np.concatenate(r10_v))
            ic_o = daily_spearman_ic(d_arr, p_arr, np.concatenate(yo_v))
            val_curve.append(ic_r if mode == "raw" else 0.5 * ic_r + 0.5 * ic_o)
    return model, val_curve


def main() -> None:
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = _detect_gpu_count()
    phase("g10_started", gpus=n_gpus)

    weights = find_dataset_file(INPUT, "kronos-small-cosine-c2-best-weights", "model.safetensors")
    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G10, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G10]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)

    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool:
        tr_pts = pool.map(_extract_h120_last_worker, [(r, "tr", str(weights.parent), tok_dir, sym_tr, tr_batches, True) for r in range(2)])
        val_pts = pool.map(_extract_h120_last_worker, [(r, "val", str(weights.parent), tok_dir, sym_val, val_batches, True) for r in range(2)])
        oos_pts = pool.map(_extract_h120_last_worker, [(r, "oos", str(weights.parent), tok_dir, sym_oos, oos_batches, False) for r in range(2)])
    del sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches

    train_days = sorted([rec for p in tr_pts for rec in torch.load(p, map_location="cpu", weights_only=False)], key=lambda d: d["date"])
    val_all = sorted([rec for p in val_pts for rec in torch.load(p, map_location="cpu", weights_only=False)], key=lambda d: d["date"])
    oos_days = sorted([rec for p in oos_pts for rec in torch.load(p, map_location="cpu", weights_only=False)], key=lambda d: d["date"])
    val_sel = [d for d in val_all if VAL_SEL_START <= d["date"] <= VAL_SEL_END]
    all_fit_days = train_days + val_all

    w_tr = fit_raw_ridge_512(train_days, alpha=1e4)
    w_all = fit_raw_ridge_512(all_fit_days, alpha=1e4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Quick val selection over residual scale & epochs on frozen skip
    configs = [(0.15, 2e-2), (0.30, 2e-2), (0.50, 3e-2)]
    results, models_oos = {}, {}
    for mode in ("raw", "mix"):
        best_score, best_cfg = -1e9, None
        trials = []
        for scale, wd in configs:
            _, curve = train_frozen_skip_peer(train_days, val_sel, w_tr, mode, scale, wd, epochs=12, seed=20261117, device=device)
            ep = int(np.argmax(curve)) + 1
            sc = float(curve[ep - 1])
            trials.append({"scale": scale, "wd": wd, "best_epoch": ep, "best_val": sc, "curve": curve})
            if sc > best_score:
                best_score, best_cfg = sc, (scale, wd, ep)
        results[mode] = {"selected": {"scale": best_cfg[0], "wd": best_cfg[1], "epochs": best_cfg[2], "val": best_score}, "trials": trials}
        m_final, _ = train_frozen_skip_peer(all_fit_days, None, w_all, mode, best_cfg[0], best_cfg[1], epochs=best_cfg[2], seed=20261117, device=device)
        models_oos[f"g10_peer_{mode}"] = m_final.eval()

    frames = []
    with torch.inference_mode():
        for d in oos_days:
            x = torch.as_tensor(d["x"], device=device, dtype=torch.float32)
            sec = torch.as_tensor(d["sec"], device=device, dtype=torch.long)
            df_d = pd.DataFrame({"identity": d["identity"], "symbol": d["symbol"], "asof_date": d["date"]})
            df_d["p_g10_linear_g1ref"] = (d["x"].astype(np.float32) @ w_all).astype(np.float32)
            for name, m in models_oos.items():
                df_d[f"p_{name}"] = m(x, sec).cpu().numpy().astype(np.float32)
            r1 = standardize_1d(df_d["p_g10_peer_raw"].to_numpy())
            r2 = standardize_1d(df_d["p_g10_peer_mix"].to_numpy())
            df_d["p_g10_peer_blend"] = (0.5 * r1 + 0.5 * r2).astype(np.float32)
            frames.append(df_d)

    oos_df = pd.concat(frames, ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g10_scores.csv.gz", index=False, compression="gzip")
    summary = {"selection": results, "n_train_days": len(train_days), "n_fit_days": len(all_fit_days), "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g10_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("g10_finished", elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
