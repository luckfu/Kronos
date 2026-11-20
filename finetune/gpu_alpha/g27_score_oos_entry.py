"""Score already-trained G27 models on the 65-day OOS set (2026-07-17..2026-10-23) on Kaggle 2xT4."""
from __future__ import annotations

import json
import multiprocessing as mp
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
from g27_nonlinear_moe_lgbm_entry import (  # noqa: E402
    SL_ALL,
    SL_BASE,
    SL_PTS,
    _score_oos_worker,
)
from short_probe_common import detect_gpu_count  # noqa: E402

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "g27_out"


def main() -> None:
    t0 = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    n_gpus = detect_gpu_count()
    phase("g27_score_started", gpus=n_gpus)

    oos_raw = next(p for p in INPUT_ROOT.rglob("combined_raw.csv") if "oos" in p.as_posix().lower())
    art_dir = next(INPUT_ROOT.rglob("g27_heads.npz")).parent
    phase("g27_inputs_found", oos_dir=str(oos_raw.parent), art_dir=str(art_dir))

    from huggingface_hub import snapshot_download

    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir=str(WORK / "hf_kronos_small")),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir=str(WORK / "hf_kronos_base")),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tokenizer")),
    ]
    pts_dir, base_dir, tok_dir = hf_dirs

    npz = np.load(art_dir / "g27_heads.npz")
    u_pts = npz["u_pts"].astype(np.float32)
    u_base = npz["u_base"].astype(np.float32)
    sl_map = {
        "pts_all": SL_PTS,
        "pts_bull": SL_PTS,
        "pts_bear": SL_PTS,
        "base_all": SL_BASE,
        "base_bull": SL_BASE,
        "base_bear": SL_BASE,
        "joint_all": SL_ALL,
        "joint_bull": SL_ALL,
        "joint_bear": SL_ALL,
    }
    full_heads = {k: (sl, npz[k].astype(np.float32)) for k, sl in sl_map.items()}

    ckpt = torch.load(art_dir / "g27_models.pt", map_location="cpu", weights_only=False)
    nn_specs = ckpt["nn_specs"]
    moe_gates = ckpt["moe_gates"]
    lgbm_cols = ckpt["lgbm_cols"]
    lgbm_path = str(art_dir / "g27_lgbm.txt")

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_gpus) as pool:
        oos_csvs = pool.map(
            _score_oos_worker,
            [
                (
                    r,
                    n_gpus,
                    pts_dir,
                    base_dir,
                    tok_dir,
                    u_pts,
                    u_base,
                    full_heads,
                    moe_gates,
                    nn_specs,
                    lgbm_path,
                    lgbm_cols,
                    sym_oos,
                    oos_batches,
                )
                for r in range(n_gpus)
            ],
        )

    oos_df = pd.concat([pd.read_csv(p) for p in oos_csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in oos_csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")

    out_csv = OUTPUT / "oos_g27_scores.csv.gz"
    oos_df.to_csv(out_csv, index=False, compression="gzip")

    summary = {
        "n_oos_rows": int(len(oos_df)),
        "score_columns": [c for c in oos_df.columns if c.startswith("p_")],
        "moe_gates": {k: list(v) for k, v in moe_gates.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g27_score_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g27_score_finished", n_oos_rows=len(oos_df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
