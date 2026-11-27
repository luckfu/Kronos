"""Score already-trained G36 LambdaRank/Focal-ListNet models on 65-day OOS (335,312 rows) in ~8 min."""
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
from g36_lambdarank_topk_liquid_entry import TMP_DIR, _score_oos_worker  # noqa: E402
from short_probe_common import detect_gpu_count, run_gpu_workers_failfast  # noqa: E402
from wave4_common import SL_ALL, SL_BASE, SL_C2, SL_MINI, SL_PTS, make_c2_paths_dict  # noqa: E402

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "g36_out"


def main() -> None:
    t0 = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    n_gpus = detect_gpu_count()
    phase("g36_score_started", gpus=n_gpus)

    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    art_dir = next(INPUT.rglob("g36_heads.npz")).parent
    paths = make_c2_paths_dict(INPUT, WORK)
    phase("g36_score_inputs_ready", art_dir=str(art_dir))

    npz = np.load(art_dir / "g36_heads.npz")
    u_dict = {
        "m": npz["u_mini"].astype(np.float32),
        "b": npz["u_base"].astype(np.float32),
        "p": npz["u_pts"].astype(np.float32),
        "c": npz["u_c2"].astype(np.float32),
    }
    sl_map = {
        "mini_regime": SL_MINI,
        "base_raw": SL_BASE,
        "pts_multi": SL_PTS,
        "c2_multi": SL_C2,
        "quad_multi": SL_ALL,
        "quad_regime": SL_ALL,
        "quad_ret5": SL_ALL,
        "quad_liq_multi": SL_ALL,
    }
    full_heads = {k: (sl, npz[k].astype(np.float32)) for k, sl in sl_map.items()}
    ckpt = torch.load(art_dir / "g36_models.pt", map_location="cpu", weights_only=False)
    nn_specs = ckpt["nn_specs"]
    lgbm_cols = ckpt["lgbm_cols"]
    r10_path = str(art_dir / "g36_lambdarank_r10.txt")
    rmulti_path = str(art_dir / "g36_lambdarank_multi.txt")

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    oos_csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [
            (r, n_gpus, paths, u_dict, full_heads, nn_specs, r10_path, rmulti_path, lgbm_cols, sym_oos, oos_batches)
            for r in range(n_gpus)
        ],
        tag="g36_score",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")

    oos_df.to_csv(OUTPUT / "oos_g36_scores.csv.gz", index=False, compression="gzip")
    summary = {
        "n_oos_rows": int(len(oos_df)),
        "score_columns": [c for c in oos_df.columns if c.startswith("p_")],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g36_score_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in ("hf_kronos_mini", "hf_tok2k", "hf_kronos_base", "hf_kronos_small", "hf_tok_base"):
        shutil.rmtree(WORK / d, ignore_errors=True)
    phase("g36_score_finished", n_oos_rows=len(oos_df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
