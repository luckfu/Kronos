"""Score already-trained G38 Term-Structure & Crash-Hazard models on 65-day OOS (335,312 rows) in ~8 min."""
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
from g38_termstruct_hazard_net_entry import (  # noqa: E402
    SL_ALL,
    SL_BASE,
    SL_C2,
    SL_MINI,
    TMP_DIR,
    _score_oos_worker,
)
from short_probe_common import detect_gpu_count, find_c2_weights, run_gpu_workers_failfast  # noqa: E402

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "g38_out"


def main() -> None:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    n_gpus = detect_gpu_count()
    phase("g38_score_started", gpus=n_gpus)

    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    art_dir = next(INPUT.rglob("g38_heads.npz")).parent
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir=str(WORK / "hf_kronos_mini")),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir=str(WORK / "hf_tok2k")),
        "base": snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir=str(WORK / "hf_kronos_base")),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tok_base")),
    }
    paths = {**hf_dirs, "c2": c2_dir}

    npz = np.load(art_dir / "g38_heads.npz")
    u_dict = {
        "m": npz["u_mini"].astype(np.float32),
        "c": npz["u_c2"].astype(np.float32),
        "b": npz["u_base"].astype(np.float32),
    }
    sl_map = {
        "mini_r5": SL_MINI,
        "c2_r5": SL_C2,
        "c2_fwd5_10": SL_C2,
        "base_r10": SL_BASE,
        "tri_r3": SL_ALL,
        "tri_r5": SL_ALL,
        "tri_fwd5_10": SL_ALL,
        "tri_r10": SL_ALL,
        "tri_regime": SL_ALL,
    }
    full_heads = {k: (sl, npz[k].astype(np.float32)) for k, sl in sl_map.items()}
    ckpt = torch.load(art_dir / "g38_models.pt", map_location="cpu", weights_only=False)
    mlp_states = ckpt["mlp_states"]
    lgbm_cols = ckpt["lgbm_cols"]
    hazard_meta = ckpt["hazard"]
    lgb_cont_path = str(art_dir / "g38_lgbm_cont.txt")
    lgb_r10_path = str(art_dir / "g38_lgbm_r10.txt")

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    res_pairs = run_gpu_workers_failfast(
        _score_oos_worker,
        [
            (r, n_gpus, paths, u_dict, full_heads, mlp_states, lgb_cont_path, lgb_r10_path, lgbm_cols, sym_oos, oos_batches)
            for r in range(n_gpus)
        ],
        tag="g38_score",
    )
    csv_paths, mac_paths = zip(*(p.split("||") for p in res_pairs))
    oos_df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in csv_paths:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g38_scores.csv.gz", index=False, compression="gzip")

    macro_oos = []
    for mp_p in mac_paths:
        macro_oos.extend(json.loads(Path(mp_p).read_text()))
        Path(mp_p).unlink(missing_ok=True)
    macro_oos.sort(key=lambda r: r["asof_date"])

    sc_mean = np.asarray(hazard_meta["scaler_mean"], dtype=np.float64)
    sc_scale = np.maximum(np.asarray(hazard_meta["scaler_scale"], dtype=np.float64), 1e-8)
    w_clf = np.asarray(hazard_meta["coef"], dtype=np.float64)
    b_clf = float(hazard_meta["intercept"])
    w_rdg = np.asarray(hazard_meta["ridge_coef"], dtype=np.float64)
    b_rdg = float(hazard_meta["ridge_intercept"])
    haz_rows = []
    for r in macro_oos:
        f = np.asarray(r["feat"], dtype=np.float64)
        fz = np.clip((f - sc_mean) / sc_scale, -4.0, 4.0)
        logit = float(fz @ w_clf + b_clf)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
        pred_ret = float(fz @ w_rdg + b_rdg)
        haz_rows.append({
            "asof_date": r["asof_date"],
            "p_crash_hazard": prob,
            "pred_univ_ret10": pred_ret,
            "breadth_ma20": float(f[0]),
            "breadth_ma5": float(f[1]),
            "delta_b20_3d": float(f[3]),
            "delta_b20_5d": float(f[4]),
            "mean_s1_ent_mini": float(f[11]),
            "hidden_cos_disp_c2": float(f[15]),
        })
    pd.DataFrame(haz_rows).to_csv(OUTPUT / "oos_g38_regime_hazard.csv", index=False)

    summary = {
        "n_oos_rows": int(len(oos_df)),
        "score_columns": [c for c in oos_df.columns if c.startswith("p_")],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g38_score_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    phase("g38_score_finished", n_oos_rows=len(oos_df), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
