"""Score already-trained G30c Quad-Family NonLinear + Cross-Stock Attention + 4-Signal FiLM + Dual-LGBM
models on the 65-day OOS set (2026-07-17..2026-10-23, 335,312 rows) on Kaggle 2x T4 (~6 min).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune" / "gpu_alpha"))
sys.path.insert(0, str(ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"))

import evaluate_base_sealed as ev  # noqa: E402
from g1_kaggle_entry import build_oos_batches  # noqa: E402
from g1_multiscale_extract_and_fit import phase  # noqa: E402
from g30_quadfam_nonlin_xstock_entry import (  # noqa: E402
    SL_ALL,
    SL_BASE,
    SL_C2,
    SL_MINI,
    SL_PTS,
    _score_oos_worker,
)
from short_probe_common import detect_gpu_count, find_c2_weights, run_gpu_workers_failfast  # noqa: E402

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "g30_out"


def main() -> None:
    t0 = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    n_gpus = detect_gpu_count()
    phase("g30_score_started", gpus=n_gpus)

    oos_raw = next(p for p in INPUT_ROOT.rglob("combined_raw.csv") if "oos" in p.as_posix().lower())
    art_dir = next(INPUT_ROOT.rglob("g30_heads.npz")).parent
    c2_dir = str(find_c2_weights(INPUT_ROOT).parent)
    phase("g30_inputs_found", oos_dir=str(oos_raw.parent), art_dir=str(art_dir), c2_dir=c2_dir)

    from huggingface_hub import snapshot_download

    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir=str(WORK / "hf_kronos_mini")),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir=str(WORK / "hf_tok2k")),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir=str(WORK / "hf_kronos_base")),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir=str(WORK / "hf_kronos_small")),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tok_base")),
    ]
    paths = {
        "mini": hf_dirs[0], "tok2k": hf_dirs[1], "base": hf_dirs[2],
        "pts": hf_dirs[3], "tok_base": hf_dirs[4], "c2": c2_dir,
    }

    npz = np.load(art_dir / "g30_heads.npz")
    u_dict = {
        "m": npz["u_mini"].astype(np.float32),
        "b": npz["u_base"].astype(np.float32),
        "p": npz["u_pts"].astype(np.float32),
        "c": npz["u_c2"].astype(np.float32),
    }
    sl_map = {
        "mini_multi": SL_MINI,
        "mini_regime": SL_MINI,
        "base_raw": SL_BASE,
        "pts_multi": SL_PTS,
        "c2_multi": SL_C2,
        "quad_multi": SL_ALL,
        "quad_decay": SL_ALL,
        "quad_regime": SL_ALL,
        "quad_ret5": SL_ALL,
    }
    full_heads = {k: (sl, npz[k].astype(np.float32)) for k, sl in sl_map.items()}

    ckpt = torch.load(art_dir / "g30_models.pt", map_location="cpu", weights_only=False)
    nn_specs = ckpt["nn_specs"]
    lgbm_cols = ckpt["lgbm_cols"]
    lgbm_multi_path = str(art_dir / "g30_lgbm_multi.txt")
    lgbm_ret5_path = str(art_dir / "g30_lgbm_ret5.txt")

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [
            (r, n_gpus, paths, u_dict, full_heads, nn_specs, lgbm_multi_path, lgbm_ret5_path, lgbm_cols, sym_oos, oos_batches)
            for r in range(n_gpus)
        ],
        tag="g30_score",
    )

    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")

    oos_df.to_csv(OUTPUT / "oos_g30_scores.csv.gz", index=False, compression="gzip")
    summary = {
        "n_oos_rows": int(len(oos_df)),
        "score_columns": [c for c in oos_df.columns if c.startswith("p_")],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g30_score_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g30_score_finished", n_oos_rows=len(oos_df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
