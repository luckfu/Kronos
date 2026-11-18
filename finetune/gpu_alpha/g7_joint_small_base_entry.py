"""G7: Joint cross-architecture Ridge/CosineIC over [C2_512d, Kronos-base_1664d] = 2176d."""
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
    ALPHAS,
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
from g2_rank_finetune import build_g2_date_batches, cosine_ic_loss_vec
from g4_base_kaggle_entry import _load_kronos_base

OUTPUT = Path(os.environ.get("G7_OUTPUT", "/kaggle/working/g7_out"))
INPUT = Path(os.environ.get("G7_INPUT", "/kaggle/input"))
JOINT_DIM = 512 + 1664  # 2176
TRAIN_START_G7 = "2021-01-01"
TRAIN_STRIDE_G7 = 3


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
def _extract_joint_worker(args: tuple) -> str:
    rank, c2_dir, base_hf_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}")
    c2 = Kronos.from_pretrained(
        c2_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    base = _load_kronos_base(base_hf_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()

    gram_tr = torch.zeros((JOINT_DIM, JOINT_DIM), dtype=torch.float64, device=device)
    b_raw = torch.zeros(JOINT_DIM, dtype=torch.float64, device=device)
    b_orth = torch.zeros(JOINT_DIM, dtype=torch.float64, device=device)
    b_multi = torch.zeros(JOINT_DIM, dtype=torch.float64, device=device)

    def _fwd_date(sym_arr, s_idx, t_idx, sec_np, pct_np):
        n = len(s_idx)
        chunks = []
        for off in range(0, n, 384):
            sl = slice(off, min(off + 384, n))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
            mean = raw_t.mean(dim=1, keepdim=True)
            std = raw_t.std(dim=1, unbiased=False, keepdim=True)
            x_norm = torch.clamp((raw_t - mean) / (std + 1e-5), -5.0, 5.0)
            st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
            sec_t = torch.as_tensor(sec_np[sl], device=device, dtype=torch.long)
            pct_t = torch.as_tensor(pct_np[sl], device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                toks = tok.encode(x_norm, half=True)
                _, ctx_c2 = c2.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
                _, ctx_base = base.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            feat = torch.cat([
                ctx_c2[:, -1, :].float(),
                ctx_base[:, -1, :].float(),
                ctx_base[:, -5:, :].float().mean(dim=1),
            ], dim=1)
            chunks.append(feat)
        return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    my_tr = [b for i, b in enumerate(tr_batches) if i % 2 == rank]
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
        if (bi + 1) % 40 == 0 or bi + 1 == len(my_tr):
            phase("g7_train_progress", rank=rank, done=bi + 1, total=len(my_tr))

    my_val = [b for i, b in enumerate(val_batches) if i % 2 == rank]
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

    out_pt = f"/tmp/g7_stats_r{rank}.pt"
    torch.save({
        "gram": gram_tr.cpu().numpy(),
        "b_raw": b_raw.cpu().numpy(),
        "b_orth": b_orth.cpu().numpy(),
        "b_multi": b_multi.cpu().numpy(),
        "val_recs": val_recs,
    }, out_pt)
    return out_pt


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, c2_dir, base_hf_dir, tok_dir, sym_oos, oos_batches, heads_np = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}")
    c2 = Kronos.from_pretrained(
        c2_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    base = _load_kronos_base(base_hf_dir, device).eval()
    tok = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    heads_t = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in heads_np.items()}

    frames = []
    my_oos = [b for i, b in enumerate(oos_batches) if i % 2 == rank]
    for d_str, recs in my_oos:
        chunks = []
        for off in range(0, len(recs), 384):
            sl = recs[off:off + 384]
            raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
            st_b = np.stack([sym_oos[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
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
                toks = tok.encode(x_norm, half=True)
                _, ctx_c2 = c2.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
                _, ctx_base = base.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            feat = torch.cat([ctx_c2[:, -1, :].float(), ctx_base[:, -1, :].float(), ctx_base[:, -5:, :].float().mean(dim=1)], dim=1)
            chunks.append(feat)
        x_d = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        df_d = pd.DataFrame({"identity": [r[4] for r in recs], "symbol": [r[5] for r in recs], "asof_date": d_str})
        for name, wt in heads_t.items():
            df_d[f"p_{name}"] = (x_d @ wt).cpu().numpy().astype(np.float32)
        frames.append(df_d)
    out_csv = f"/tmp/g7_oos_r{rank}.csv.gz"
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
    phase("g7_started", gpus=n_gpus)

    c2_w = find_dataset_file(INPUT, "kronos-small-cosine-c2-best-weights", "model.safetensors")
    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")

    base_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G7, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G7]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool:
        pts = pool.map(_extract_joint_worker, [
            (r, str(c2_w.parent), base_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches)
            for r in range(2)
        ])
    del sym_tr, tr_batches, sym_val, val_batches

    gram_tr = np.zeros((JOINT_DIM, JOINT_DIM), dtype=np.float64)
    rhs_tr = {k: np.zeros(JOINT_DIM, dtype=np.float64) for k in ("raw", "orth", "multi")}
    val_recs = []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        gram_tr += d["gram"]
        rhs_tr["raw"] += d["b_raw"]
        rhs_tr["orth"] += d["b_orth"]
        rhs_tr["multi"] += d["b_multi"]
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

    selection, heads_np = {}, {}
    for arm, key in [("g7_joint_raw", "raw"), ("g7_joint_orth", "orth"), ("g7_joint_multi", "multi")]:
        best_a, best_ic = None, -1e9
        grid = []
        for a in ALPHAS:
            w = np.linalg.solve(gram_tr + a * np.eye(JOINT_DIM), rhs_tr[key])
            pred = x_sel @ w
            ic_r = daily_spearman_ic(d_sel, pred, r_sel)
            ic_o = daily_spearman_ic(d_sel, pred, yo_sel)
            sc = ic_o if key == "orth" else ic_r
            grid.append({"alpha": a, "val_ic_raw": ic_r, "val_ic_orth": ic_o})
            if sc > best_ic:
                best_ic, best_a = sc, a
        heads_np[arm] = np.linalg.solve(gram_all + best_a * np.eye(JOINT_DIM), rhs_all[key]).astype(np.float32)
        selection[arm] = {"selected_alpha": best_a, "best_val_ic": best_ic, "grid": grid}

    # Also build exact 2-ridge balanced blend head: normalized w_raw + w_orth
    selection["g7_joint_orth_blend"] = {"note": "50/50 L2-normalized joint_raw + joint_orth"}
    u_raw = heads_np["g7_joint_raw"] / max(float(np.linalg.norm(heads_np["g7_joint_raw"])), 1e-6)
    u_orth = heads_np["g7_joint_orth"] / max(float(np.linalg.norm(heads_np["g7_joint_orth"])), 1e-6)
    heads_np["g7_joint_mix"] = (0.5 * u_raw + 0.5 * u_orth).astype(np.float32)

    np.savez(OUTPUT / "g7_heads.npz", **heads_np)
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    with ctx.Pool(2) as pool:
        oos_csvs = pool.map(_score_oos_worker, [
            (r, str(c2_w.parent), base_dir, tok_dir, sym_oos, oos_batches, heads_np)
            for r in range(2)
        ])
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g7_scores.csv.gz", index=False, compression="gzip")
    summary = {"selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g7_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("g7_finished", elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
