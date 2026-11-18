"""Kaggle GPU entry point for Experiment G3: CrossStockPeerRanker on 2x T4."""
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
    OOS_END,
    OOS_START,
    TRAIN_FIT_END,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    daily_spearman_ic,
    phase,
    zscore_cross_section_torch,
)
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec
from g3_cross_stock_context import CrossStockPeerRanker

OUTPUT = Path(os.environ.get("G3_OUTPUT", "/kaggle/working/g3_out"))
INPUT = Path(os.environ.get("G3_INPUT", "/kaggle/input"))
TRAIN_START_G3 = "2020-07-01"
TRAIN_STRIDE_G3 = 2


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


@torch.inference_mode()
def _extract_h120_last_worker(args: tuple) -> str:
    rank, tag, weights_dir, tokenizer_dir, sym_arrays, batches, has_labels = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = Kronos.from_pretrained(
        weights_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()

    my_batches = [b for i, b in enumerate(batches) if i % 2 == rank]
    records_out = []
    for b in my_batches:
        if has_labels:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, y_orth, r10 = b
            n = len(s_idx)
            chunks = []
            for off in range(0, n, 512):
                sl = slice(off, min(off + 512, n))
                raw_b = np.stack([sym_arrays[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_arrays[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
                mean = raw_t.mean(dim=1, keepdim=True)
                std = raw_t.std(dim=1, unbiased=False, keepdim=True)
                x_norm = torch.clamp((raw_t - mean) / (std + 1e-5), -5.0, 5.0)
                st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
                sec_t = torch.as_tensor(sec_np[sl], device=device, dtype=torch.long)
                pct_t = torch.as_tensor(pct_np[sl], device=device, dtype=torch.float32)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    toks = tokenizer.encode(x_norm, half=True)
                    _, ctx = model.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
                chunks.append(ctx[:, -1, :].float())
            xz = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0).half().cpu().numpy()
            records_out.append({
                "date": d_str,
                "x": xz,
                "sec": sec_np.astype(np.int16),
                "y_multi": y_multi.astype(np.float32),
                "y_orth": y_orth.astype(np.float32),
                "ret10": r10.astype(np.float32),
            })
        else:
            d_str, recs = b
            n = len(recs)
            chunks = []
            for off in range(0, n, 512):
                sl = recs[off:off + 512]
                raw_b = np.stack([sym_arrays[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
                st_b = np.stack([sym_arrays[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
                sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
                pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
                raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
                mean = raw_t.mean(dim=1, keepdim=True)
                std = raw_t.std(dim=1, unbiased=False, keepdim=True)
                x_norm = torch.clamp((raw_t - mean) / (std + 1e-5), -5.0, 5.0)
                st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
                sec_t = torch.as_tensor(sec_b, device=device, dtype=torch.long)
                pct_t = torch.as_tensor(pct_b, device=device, dtype=torch.float32)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    toks = tokenizer.encode(x_norm, half=True)
                    _, ctx = model.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
                chunks.append(ctx[:, -1, :].float())
            xz = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0).half().cpu().numpy()
            records_out.append({
                "date": d_str,
                "x": xz,
                "sec": np.asarray([r[2] for r in recs], dtype=np.int16),
                "identity": [r[4] for r in recs],
                "symbol": [r[5] for r in recs],
            })
    out_pt = f"/tmp/g3_cache_{tag}_r{rank}.pt"
    torch.save(records_out, out_pt)
    return out_pt


def fit_ridge_skip(train_days: list[dict], alpha: float = 1e5) -> np.ndarray:
    gram = np.zeros((512, 512), dtype=np.float64)
    rhs = np.zeros(512, dtype=np.float64)
    for d in train_days:
        x = d["x"].astype(np.float64)
        y = 0.7 * d["y_multi"].astype(np.float64) + 0.3 * d["y_orth"].astype(np.float64)
        gram += x.T @ x
        rhs += x.T @ y
    return np.linalg.solve(gram + alpha * np.eye(512), rhs).astype(np.float32)


def train_one_peer_run(
    train_days: list[dict],
    val_days: list[dict] | None,
    w_skip: np.ndarray,
    proj_dim: int,
    dropout: float,
    wd: float,
    epochs: int,
    seed: int,
    device,
) -> tuple[CrossStockPeerRanker, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CrossStockPeerRanker(in_dim=512, proj_dim=proj_dim, num_sectors=86, num_heads=4, dropout=dropout).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(torch.as_tensor(w_skip, device=device, dtype=torch.float32).unsqueeze(0))
        model.skip.bias.zero_()
    opt = torch.optim.AdamW([
        {"params": model.skip.parameters(), "lr": 1e-4, "weight_decay": wd},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("skip.")], "lr": 5e-4, "weight_decay": wd},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * len(train_days), 1), eta_min=1e-5)

    val_curve: list[float] = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_days))
        for di in order:
            d = train_days[di]
            x = torch.as_tensor(d["x"], device=device, dtype=torch.float32)
            sec = torch.as_tensor(d["sec"], device=device, dtype=torch.long)
            ym = torch.as_tensor(d["y_multi"], device=device, dtype=torch.float32)
            yo = torch.as_tensor(d["y_orth"], device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(x, sec)
            loss = 0.7 * cosine_ic_loss_vec(pred, ym) + 0.3 * cosine_ic_loss_vec(pred, yo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            val_curve.append(0.5 * ic_r + 0.5 * ic_o)
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
    phase("g3_started", gpus=n_gpus)

    weights = find_dataset_file(INPUT, "kronos-small-cosine-c2-best-weights", "model.safetensors")
    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    tokenizer_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G3, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G3]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    phase("g3_panels_loaded", tr_dates=len(tr_batches), val_dates=len(val_batches), oos_dates=len(oos_batches))

    ctx = mp.get_context("fork")
    world = min(2, n_gpus)
    if world < 2:
        raise RuntimeError("Expected 2 GPUs on Kaggle T4x2")
    with ctx.Pool(2) as pool:
        tr_pts = pool.map(_extract_h120_last_worker, [(r, "tr", str(weights.parent), tokenizer_dir, sym_tr, tr_batches, True) for r in range(2)])
        val_pts = pool.map(_extract_h120_last_worker, [(r, "val", str(weights.parent), tokenizer_dir, sym_val, val_batches, True) for r in range(2)])
        oos_pts = pool.map(_extract_h120_last_worker, [(r, "oos", str(weights.parent), tokenizer_dir, sym_oos, oos_batches, False) for r in range(2)])

    del sym_tr, sym_val, sym_oos
    train_days = sorted([rec for p in tr_pts for rec in torch.load(p, weights_only=False)], key=lambda r: r["date"])
    val_all = sorted([rec for p in val_pts for rec in torch.load(p, weights_only=False)], key=lambda r: r["date"])
    oos_days = sorted([rec for p in oos_pts for rec in torch.load(p, weights_only=False)], key=lambda r: r["date"])

    val_sel = [d for d in val_all if VAL_SEL_START <= d["date"] <= VAL_SEL_END]
    val_sub = [d for d in val_all if d["date"] <= TRAIN_FIT_END]
    device = torch.device("cuda:0")
    w_skip = fit_ridge_skip(train_days, alpha=1e5)

    grid = [(96, 0.4, 2e-2), (128, 0.4, 2e-2), (128, 0.5, 5e-2), (160, 0.5, 5e-2)]
    candidates = []
    best_cfg = None
    for pd_dim, drop, wd in grid:
        curves = []
        for s in (20261101, 20261102, 20261103):
            _, curve = train_one_peer_run(train_days, val_sel, w_skip, pd_dim, drop, wd, 15, s, device)
            curves.append(curve)
        mean_c = np.mean(curves, axis=0)
        ep_idx = int(np.argmax(mean_c))
        rec = {"proj_dim": pd_dim, "dropout": drop, "wd": wd, "best_epoch": ep_idx + 1, "val_comp": float(mean_c[ep_idx])}
        candidates.append(rec)
        phase("g3_cfg_done", **rec)
        if best_cfg is None or rec["val_comp"] > best_cfg["val_comp"]:
            best_cfg = rec

    # Refit cleanly on train + all val <= 2026-07-02 with 3-seed ensemble
    refit_days = train_days + val_all
    w_skip_full = fit_ridge_skip(refit_days, alpha=1e5)
    models = []
    for s in (20261101, 20261102, 20261103):
        m, _ = train_one_peer_run(
            refit_days, None, w_skip_full, best_cfg["proj_dim"], best_cfg["dropout"], best_cfg["wd"], best_cfg["best_epoch"], s, device
        )
        m.eval()
        models.append(m)

    oos_frames = []
    with torch.inference_mode():
        for d in oos_days:
            x = torch.as_tensor(d["x"], device=device, dtype=torch.float32)
            sec = torch.as_tensor(d["sec"], device=device, dtype=torch.long)
            p_peer = np.mean([m(x, sec).cpu().numpy() for m in models], axis=0)
            p_skip = (d["x"].astype(np.float32) @ w_skip_full)
            oos_frames.append(pd.DataFrame({
                "identity": d["identity"],
                "symbol": d["symbol"],
                "asof_date": d["date"],
                "p_g3_peer": p_peer.astype(np.float32),
                "p_g3_skip": p_skip.astype(np.float32),
            }))
    oos_df = pd.concat(oos_frames, ignore_index=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g3_scores.csv.gz", index=False, compression="gzip")
    summary = {"best_cfg": best_cfg, "candidates": candidates, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("g3_finished", elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
