"""G32: Pure Zero-Shot L=45 Multi-Path Autoregressive (AR) Trajectory Frontier across 3 Backbones
(Kronos-mini 4.1M Tok-2k AR-16 + C2 24.7M AR-16 + PT-Small 24.7M AR-8) on full 65d OOS (335,312 rows).

Zero label fitting, zero training leakage: evaluates generative multi-step trajectory alpha directly on
2026-07-17 .. 2026-10-23 using production decode parameters (T=0.65, top_p=0.8, pred_len=10).
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
from g1_multiscale_extract_and_fit import phase
from g16_pretrained_small_entry import _load_pretrained_small
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from short_probe_common import detect_gpu_count, find_c2_weights, run_gpu_workers_failfast

OUTPUT = Path(os.environ.get("G32_OUTPUT", "/kaggle/working/g32_out"))
INPUT = Path(os.environ.get("G32_INPUT", "/kaggle/input"))

LB = 45
PRED_LEN = 10
DECODE_T = 0.65
DECODE_TOP_P = 0.80
SEED = 20261121


def _summarize_paths(path_rets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute [B] (d10_med, d10_mean, d10_sharpe, d10_probup, d5_med) from [B, P, 10] path returns."""
    d10 = path_rets[:, :, 9]
    d5 = path_rets[:, :, 4]
    d10_med = np.median(d10, axis=1)
    d10_mean = np.mean(d10, axis=1)
    q75, q25 = np.percentile(d10, [75, 25], axis=1)
    iqr = np.maximum(q75 - q25, 0.0)
    d10_sharpe = d10_med / (iqr + 0.02)
    d10_probup = np.mean(d10 > 0.0, axis=1)
    d5_med = np.median(d5, axis=1)
    return (
        d10_med.astype(np.float32),
        d10_mean.astype(np.float32),
        d10_sharpe.astype(np.float32),
        d10_probup.astype(np.float32),
        d5_med.astype(np.float32),
    )


@torch.inference_mode()
def _run_ar_for_model(
    *,
    tag: str,
    model,
    tokenizer,
    use_cond: bool,
    n_paths: int,
    batch_size: int,
    sym_oos: list,
    my_oos: list,
    device: torch.device,
    rank: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    from model.kronos import auto_regressive_inference

    results: dict[str, tuple] = {}
    t_start = time.time()
    for di, (d_str, recs) in enumerate(my_oos):
        n = len(recs)
        d10_med = np.empty(n, dtype=np.float32)
        d10_mean = np.empty(n, dtype=np.float32)
        d10_sh = np.empty(n, dtype=np.float32)
        d10_up = np.empty(n, dtype=np.float32)
        d5_med = np.empty(n, dtype=np.float32)

        for off in range(0, n, batch_size):
            sl = recs[off : off + batch_size]
            b = len(sl)
            raw_b = np.stack([sym_oos[r[0]][0][r[1] - LB + 1 : r[1] + 1] for r in sl])  # [B, 45, 6]
            x_st = np.stack([sym_oos[r[0]][1][r[1] - LB + 1 : r[1] + 1] for r in sl])   # [B, 45, 5]
            y_st = np.stack([sym_oos[r[0]][1][r[1] + 1 : r[1] + 1 + PRED_LEN] for r in sl])  # [B, 10, 5]

            mean = raw_b.mean(axis=1, keepdims=True)
            std = raw_b.std(axis=1, keepdims=True)
            x_norm = np.clip((raw_b - mean) / (std + 1e-5), -5.0, 5.0).astype(np.float32)
            last_close = np.maximum(raw_b[:, -1, 3].astype(np.float64), 1e-8)
            mean_c = mean[:, 0, 3].astype(np.float64)
            std_c = std[:, 0, 3].astype(np.float64)

            xt = torch.as_tensor(x_norm, device=device, dtype=torch.float32)
            xst = torch.as_tensor(x_st, device=device, dtype=torch.float32)
            yst = torch.as_tensor(y_st, device=device, dtype=torch.float32)
            sec_t = torch.as_tensor([r[2] for r in sl], device=device, dtype=torch.long) if use_cond else None
            pct_t = torch.as_tensor([r[3] for r in sl], device=device, dtype=torch.float32) if use_cond else None

            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                fc = auto_regressive_inference(
                    tokenizer, model, xt, xst, yst,
                    max_context=512, pred_len=PRED_LEN, clip=5, T=DECODE_T, top_k=0, top_p=DECODE_TOP_P,
                    sample_count=n_paths, verbose=False, sector_id=sec_t, size_percentile=pct_t,
                    return_samples=True,
                )
            # fc shape: [B, n_paths, pred_len, 6] (or [B, n_paths, seq_len, 6], slice last PRED_LEN)
            pred_close = fc[:, :, -PRED_LEN:, 3].astype(np.float64) * (std_c[:, None, None] + 1e-5) + mean_c[:, None, None]
            path_rets = np.nan_to_num(pred_close / last_close[:, None, None] - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
            m10, mu10, sh10, up10, m5 = _summarize_paths(path_rets)
            d10_med[off : off + b] = m10
            d10_mean[off : off + b] = mu10
            d10_sh[off : off + b] = sh10
            d10_up[off : off + b] = up10
            d5_med[off : off + b] = m5

        results[d_str] = (d10_med, d10_mean, d10_sh, d10_up, d5_med)
        if (di + 1) % 8 == 0 or di + 1 == len(my_oos):
            phase("g32_ar_progress", tag=tag, rank=rank, done=di + 1, total=len(my_oos), sec=round(time.time() - t_start, 1))
    return results


def _ar_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)
    my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]

    # 1. Mini-2k L=45 16-path AR
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    res_mini = _run_ar_for_model(
        tag="mini_l45_ar16", model=mini, tokenizer=tok2k, use_cond=False,
        n_paths=16, batch_size=64, sym_oos=sym_oos, my_oos=my_oos, device=device, rank=rank,
    )
    del mini, tok2k
    torch.cuda.empty_cache()

    # 2. C2 L=45 16-path AR
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    res_c2 = _run_ar_for_model(
        tag="c2_l45_ar16", model=c2, tokenizer=tok_base, use_cond=True,
        n_paths=16, batch_size=32, sym_oos=sym_oos, my_oos=my_oos, device=device, rank=rank,
    )
    del c2
    torch.cuda.empty_cache()

    # 3. PT-Small L=45 8-path AR
    pts = _load_pretrained_small(paths["pts"], device).eval()
    res_pts = _run_ar_for_model(
        tag="pts_l45_ar8", model=pts, tokenizer=tok_base, use_cond=True,
        n_paths=8, batch_size=64, sym_oos=sym_oos, my_oos=my_oos, device=device, rank=rank,
    )
    del pts, tok_base
    torch.cuda.empty_cache()

    frames = []
    for d_str, recs in my_oos:
        mm, mmu, msh, mup, mm5 = res_mini[d_str]
        cm, cmu, csh, cup, cm5 = res_c2[d_str]
        pm, pmu, psh, _pup, pm5 = res_pts[d_str]
        frames.append(pd.DataFrame({
            "identity": [r[4] for r in recs],
            "symbol": [r[5] for r in recs],
            "asof_date": d_str,
            "p_g32_mini_l45_ar16_d10_med": mm,
            "p_g32_mini_l45_ar16_d10_mean": mmu,
            "p_g32_mini_l45_ar16_d10_sharpe": msh,
            "p_g32_mini_l45_ar16_d10_probup": mup,
            "p_g32_mini_l45_ar16_d5_med": mm5,
            "p_g32_c2_l45_ar16_d10_med": cm,
            "p_g32_c2_l45_ar16_d10_mean": cmu,
            "p_g32_c2_l45_ar16_d10_sharpe": csh,
            "p_g32_c2_l45_ar16_d10_probup": cup,
            "p_g32_c2_l45_ar16_d5_med": cm5,
            "p_g32_pts_l45_ar8_d10_med": pm,
            "p_g32_pts_l45_ar8_d10_mean": pmu,
            "p_g32_pts_l45_ar8_d10_sharpe": psh,
            "p_g32_pts_l45_ar8_d5_med": pm5,
        }))

    out_csv = f"/tmp/g32_oos_r{rank}.csv.gz"
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
    n_gpus = detect_gpu_count()
    phase("g32_started", gpus=n_gpus)

    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)
    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    paths = {"mini": hf_dirs[0], "tok2k": hf_dirs[1], "pts": hf_dirs[2], "tok_base": hf_dirs[3], "c2": c2_dir}

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    csvs = run_gpu_workers_failfast(_ar_worker, [(r, n_gpus, paths, sym_oos, oos_batches) for r in range(n_gpus)], tag="g32_ar")

    oos_df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g32_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "n_oos_rows": int(len(oos_df)),
        "score_columns": [c for c in oos_df.columns if c.startswith("p_")],
        "lb": LB,
        "pred_len": PRED_LEN,
        "temperature": DECODE_T,
        "top_p": DECODE_TOP_P,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g32_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g32_finished", n_oos_rows=len(oos_df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
