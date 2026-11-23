"""Kaggle 2x T4 Sealed Acceptance Kernel B (2026-10-26..2026-11-20, 86,195 rows):
Pure Zero-Fit G32 C2 L=45 16-Path Autoregressive (AR) Trajectory Scorer.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from g1_multiscale_extract_and_fit import phase
from g32_l45_ar16_triarch_entry import _run_ar_for_model
from sealed_common import build_sealed_batches
from short_probe_common import detect_gpu_count, find_c2_weights, run_gpu_workers_failfast

INPUT = Path(os.environ.get("SEALED_INPUT", "/kaggle/input"))
WORK = Path("/kaggle/working")
OUTPUT = WORK / "sealed_ar_out"


@torch.inference_mode()
def _sealed_ar_worker(args: tuple) -> str:
    rank, n_gpus, c2_dir, tok_dir, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    my_oos = [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]

    c2 = Kronos.from_pretrained(
        str(c2_dir), num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok = KronosTokenizer.from_pretrained(str(tok_dir)).to(device).eval()

    res = _run_ar_for_model(
        tag="c2_l45_ar16",
        model=c2,
        tokenizer=tok,
        use_cond=True,
        n_paths=16,
        batch_size=24,
        sym_oos=sym_oos,
        my_oos=my_oos,
        device=device,
        rank=rank,
    )

    frames = []
    for d_str, recs in my_oos:
        d10_med, d10_mean, d10_sh, d10_up, d5_med = res[d_str]
        frames.append(pd.DataFrame({
            "identity": [r[4] for r in recs],
            "symbol": [r[5] for r in recs],
            "asof_date": d_str,
            "p_g32_c2_l45_ar16_d10_med": d10_med,
            "p_g32_c2_l45_ar16_d10_mean": d10_mean,
            "p_g32_c2_l45_ar16_d10_sharpe": d10_sh,
            "p_g32_c2_l45_ar16_d10_probup": d10_up,
            "p_g32_c2_l45_ar16_d5_med": d5_med,
        }))

    out = f"/tmp/sealed_ar_r{rank}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    return out


def main() -> None:
    from huggingface_hub import snapshot_download

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("sealed_ar_started", gpus=n_gpus)

    c2_dir = str(find_c2_weights(INPUT).parent)
    art_dir = next(INPUT.rglob("sealed_kline_150d.parquet")).parent
    kline_pq = art_dir / "sealed_kline_150d.parquet"
    ranks_pq = art_dir / "sealed_shadow_rankings.parquet"
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tokenizer"))

    sym_oos, oos_batches = build_sealed_batches(kline_pq, ranks_pq)
    n_expected = sum(len(b[1]) for b in oos_batches)
    phase("sealed_ar_batches_ready", n_dates=len(oos_batches), n_rows=n_expected, elapsed_sec=round(time.time() - t0, 1))

    parts = run_gpu_workers_failfast(
        _sealed_ar_worker,
        [(r, n_gpus, c2_dir, tok_dir, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="sealed_ar",
    )
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("identity").reset_index(drop=True)
    for p in parts:
        Path(p).unlink(missing_ok=True)
    if len(df) != n_expected:
        raise RuntimeError(f"Expected {n_expected} rows, got {len(df)}")

    df.to_parquet(OUTPUT / "sealed_g32_ar_scores.parquet", index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_dates": int(len(oos_batches)),
        "score_columns": [c for c in df.columns if c.startswith("p_")],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "sealed_ar_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(tok_dir, ignore_errors=True)
    phase("sealed_ar_finished", n_rows=len(df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
