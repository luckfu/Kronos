"""Score already-trained G40 FT-Mini + Frozen C2 models on 65d OOS with ZERO Base/PTS weights."""
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
from g1_kaggle_entry import build_oos_batches, find_dataset_file  # noqa: E402
from g1_multiscale_extract_and_fit import phase  # noqa: E402
from g40_mini_tailaware_gentle_ft_entry import (  # noqa: E402
    SL_ALL,
    SL_FTMINI,
    TMP_DIR,
    _score_oos_worker,
)
from short_probe_common import detect_gpu_count, find_c2_weights, run_gpu_workers_failfast  # noqa: E402

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "g40_out"


def main() -> None:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    n_gpus = detect_gpu_count()
    phase("g40_score_started", gpus=n_gpus)

    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    art_dir = next(INPUT.rglob("g40_heads.npz")).parent
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir=str(WORK / "hf_kronos_mini")),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir=str(WORK / "hf_tok2k")),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tok_base")),
    }
    paths = {**hf_dirs, "c2": c2_dir}

    npz = np.load(art_dir / "g40_heads.npz")
    u_ftm = npz["u_ftm"].astype(np.float32)
    u_c2 = npz["u_c2"].astype(np.float32)
    sl_map = {
        "ftmini_cvar": SL_FTMINI,
        "ftmini_fwd5_10": SL_FTMINI,
        "ftmini_nowhipsaw": SL_FTMINI,
        "joint_cvar": SL_ALL,
        "joint_nowhipsaw": SL_ALL,
        "joint_fwd5_10": SL_ALL,
        "joint_crashentry": SL_ALL,
        "joint_liqcvar": SL_ALL,
    }
    full_heads = {k: (sl, npz[k].astype(np.float32)) for k, sl in sl_map.items()}
    ckpt = torch.load(art_dir / "g40_models.pt", map_location="cpu", weights_only=False)
    lgbm_cvar_path = str(art_dir / "g40_lgbm_cvar10.txt")
    lgbm_fwd_path = str(art_dir / "g40_lgbm_fwd5_10.txt")

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [
            (
                r, n_gpus, paths, ckpt["top30"], ckpt["top45"], u_ftm, u_c2,
                full_heads, ckpt["nn_specs"], lgbm_cvar_path, lgbm_fwd_path, sym_oos, oos_batches,
            )
            for r in range(n_gpus)
        ],
        tag="g40_score",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")

    oos_df.to_csv(OUTPUT / "oos_g40_scores.csv.gz", index=False, compression="gzip")
    summary = {
        "n_oos_rows": int(len(oos_df)),
        "score_columns": [c for c in oos_df.columns if c.startswith("p_")],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g40_score_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    phase("g40_score_finished", n_oos_rows=len(oos_df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
