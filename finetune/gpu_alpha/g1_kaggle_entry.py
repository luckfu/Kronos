"""Kaggle GPU entry point for Experiment G1 (2x T4 parallel workers)."""
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

from g1_multiscale_extract_and_fit import (
    ALPHAS,
    BLOCK_SLICES,
    FEATURES,
    OOS_END,
    OOS_START,
    TOTAL_DIM,
    TRAIN_DATE_STRIDE,
    TRAIN_FIT_END,
    TRAIN_START,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    build_panel_date_batches,
    daily_spearman_ic,
    orthogonalize_target,
    phase,
    run_gpu_extract_stats,
    score_oos_on_gpu,
    solve_ridge_subblock,
)

OUTPUT = Path(os.environ.get("G1_OUTPUT", "/kaggle/working/g1_out"))
INPUT = Path(os.environ.get("G1_INPUT", "/kaggle/input"))
BATCH_SIZE = int(os.environ.get("G1_BATCH", "512") or "512")


def _worker_extract_save(args: tuple) -> str:
    rank = args[0]
    res = run_gpu_extract_stats(args)
    tmp_path = f"/tmp/g1_stats_r{rank}.npz"
    payload = {
        "gram_tr": res["gram_tr"],
        "rhs_raw_tr": res["rhs_raw_tr"],
        "rhs_orth_tr": res["rhs_orth_tr"],
        "n_tr_rows": np.asarray([res["n_tr_rows"]], dtype=np.int64),
    }
    if res["val"] is not None:
        for k, v in res["val"].items():
            payload[f"val_{k}"] = v
    np.savez(tmp_path, **payload)
    return tmp_path


def _worker_score_oos(args: tuple) -> str:
    rank = args[0]
    df = score_oos_on_gpu(args)
    tmp_path = f"/tmp/g1_oos_r{rank}.csv.gz"
    df.to_csv(tmp_path, index=False, compression="gzip")
    return tmp_path


def build_oos_batches(oos_root: Path, sector_map: dict[str, int]) -> tuple[list[tuple], list[tuple]]:
    raw = pd.read_csv(oos_root / "combined_raw.csv")
    raw["date"] = raw["date"].astype(str)
    raw = raw[raw["volume"] > 0].sort_values(["symbol", "date"])
    samples = pd.read_json(oos_root / "evaluation_samples.jsonl", lines=True)
    samples["asof_date"] = samples["asof_date"].astype(str)
    samples["target_date"] = samples["target_date"].astype(str)
    if samples["asof_date"].max() > OOS_END or samples["asof_date"].min() < OOS_START:
        raise RuntimeError(f"OOS date bounds violated: {samples['asof_date'].min()}..{samples['asof_date'].max()}")
    if len(samples) != 335312 or samples["asof_date"].nunique() != 65:
        raise RuntimeError(f"Unexpected OOS sample count: {len(samples)} / {samples['asof_date'].nunique()}")

    raw["size_percentile"] = raw.groupby("date")["market_cap"].rank(pct=True, method="average").fillna(0.5)
    unk = sector_map.get("unknown", 85)
    sec_by_key = {(r.symbol, r.asof_date): sector_map.get(str(r.sector), unk) for r in samples.itertuples()}

    sym_arrays: list[tuple] = []
    sym_to_idx: dict[str, tuple[int, dict[str, int]]] = {}
    for sym, grp in raw.groupby("symbol", sort=False):
        idx = pd.DatetimeIndex(pd.to_datetime(grp["date"]))
        raw_vals = grp[FEATURES].to_numpy(dtype=np.float32)
        stamps = np.column_stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month]).astype(np.float32)
        dates_list = grp["date"].tolist()
        date_pos = {d: i for i, d in enumerate(dates_list)}
        pcts = grp["size_percentile"].to_numpy(dtype=np.float32)
        sym_arrays.append((raw_vals, stamps, pcts))
        sym_to_idx[str(sym)] = (len(sym_arrays) - 1, date_pos)

    date_to_recs: dict[str, list[tuple]] = {}
    for r in samples.itertuples():
        s_i, dpos = sym_to_idx[str(r.symbol)]
        t = dpos[r.asof_date]
        if t < 119:
            raise RuntimeError(f"Insufficient history for {r.symbol} on {r.asof_date}: t={t}")
        sec_id = sec_by_key[(r.symbol, r.asof_date)]
        pct = float(sym_arrays[s_i][2][t])
        identity = "|".join(str(getattr(r, k)) for k in ("symbol", "start_index", "asof_date", "target_date"))
        date_to_recs.setdefault(r.asof_date, []).append((s_i, t, sec_id, pct, identity, str(r.symbol)))

    oos_batches = [(d, date_to_recs[d]) for d in sorted(date_to_recs.keys())]
    return sym_arrays, oos_batches


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


def find_dataset_file(root: Path, parent_substr: str, filename: str) -> Path:
    matches = sorted(p for p in root.rglob(filename) if parent_substr in p.as_posix())
    if len(matches) != 1:
        all_hits = sorted(p.as_posix() for p in root.rglob(filename))
        raise RuntimeError(f"Expected 1 {parent_substr}/{filename}, got {matches} (all: {all_hits})")
    return matches[0]


def main() -> None:
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = _detect_gpu_count()
    phase("started", gpus=n_gpus, train_start=TRAIN_START, train_stride=TRAIN_DATE_STRIDE)

    weights = [find_dataset_file(INPUT, "kronos-small-cosine-c2-best-weights", "model.safetensors")]
    train_pkl = [find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")]
    val_pkl = [find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")]
    meta_csv = [find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")]
    oos_raw = [find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")]

    tokenizer_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    phase("tokenizer_ready", dir=tokenizer_dir)

    lookup = build_fast_metadata_lookup(meta_csv[0], ev.SECTOR_MAP)
    phase("metadata_indexed", symbols=len(lookup), elapsed=round(time.time() - t0, 1))

    sym_tr, tr_batches = build_panel_date_batches(
        train_pkl[0], lookup, TRAIN_START, TRAIN_FIT_END, date_stride=TRAIN_DATE_STRIDE
    )
    sym_val, val_batches = build_panel_date_batches(
        val_pkl[0], lookup, "2025-07-01", VAL_SEL_END, date_stride=1
    )
    assert tr_batches[-1][0] <= TRAIN_FIT_END
    assert val_batches[-1][0] <= VAL_SEL_END
    phase("panels_prepared", train_dates=len(tr_batches), train_rows=sum(len(b[1]) for b in tr_batches),
          val_dates=len(val_batches), val_rows=sum(len(b[1]) for b in val_batches), elapsed=round(time.time() - t0, 1))

    ctx = mp.get_context("fork")
    world = min(2, n_gpus)
    worker_args = [
        (r, str(weights[0].parent), tokenizer_dir, sym_tr, tr_batches, sym_val, val_batches, None, BATCH_SIZE)
        for r in range(world)
    ]
    if world == 1:
        # Handle single-GPU fallback cleanly by adjusting modulo or running single
        raise RuntimeError("Expected 2 GPUs on Kaggle T4x2 (or update modulo)")
    with ctx.Pool(world) as pool:
        stat_paths = pool.map(_worker_extract_save, worker_args)

    # Free heavy train panel before OOS scoring
    del sym_tr, tr_batches, sym_val, val_batches
    gram_tr = np.zeros((TOTAL_DIM, TOTAL_DIM), dtype=np.float64)
    rhs_raw_tr = np.zeros(TOTAL_DIM, dtype=np.float64)
    rhs_orth_tr = np.zeros(TOTAL_DIM, dtype=np.float64)
    n_tr_rows = 0
    vx_list, vd_list, vr_list, vyr_list, vyo_list = [], [], [], [], []
    for p in stat_paths:
        d = np.load(p, allow_pickle=False)
        gram_tr += d["gram_tr"]
        rhs_raw_tr += d["rhs_raw_tr"]
        rhs_orth_tr += d["rhs_orth_tr"]
        n_tr_rows += int(d["n_tr_rows"][0])
        if "val_x" in d:
            vx_list.append(d["val_x"])
            vd_list.append(d["val_dates"])
            vr_list.append(d["val_ret10"])
            vyr_list.append(d["val_y_raw"])
            vyo_list.append(d["val_y_orth"])

    x_val = np.concatenate(vx_list, axis=0)
    dates_val = np.concatenate(vd_list).astype(str)
    ret10_val = np.concatenate(vr_list).astype(np.float64)
    y_raw_val = np.concatenate(vyr_list).astype(np.float64)
    y_orth_val = np.concatenate(vyo_list).astype(np.float64)

    sel_mask = (dates_val >= VAL_SEL_START) & (dates_val <= VAL_SEL_END)
    val_sub_mask = dates_val <= TRAIN_FIT_END
    x_sel = x_val[sel_mask].astype(np.float64)
    d_sel = dates_val[sel_mask]
    r_sel = ret10_val[sel_mask]
    yo_sel = y_orth_val[sel_mask]

    arms_spec = {
        "g1_univ_last": (slice(0, 512), "raw"),
        "g1_l60_only": (slice(2048, 3072), "raw"),
        "g1_multi_all": (slice(0, 3072), "raw"),
        "g1_multi_orth": (slice(0, 3072), "orth"),
    }

    # Precompute val full sufficient statistics for refit
    x_val64 = x_val.astype(np.float64)
    gram_all = gram_tr + x_val64.T @ x_val64
    rhs_raw_all = rhs_raw_tr + x_val64.T @ y_raw_val
    rhs_orth_all = rhs_orth_tr + x_val64.T @ y_orth_val

    selection_summary = {}
    solved_heads = {}
    for arm_name, (sl, mode) in arms_spec.items():
        rhs_tr = rhs_raw_tr if mode == "raw" else rhs_orth_tr
        rhs_all = rhs_raw_all if mode == "raw" else rhs_orth_all
        target_sel = r_sel if mode == "raw" else yo_sel
        grid = []
        best_alpha, best_ic = None, -1e9
        for alpha in ALPHAS:
            w = solve_ridge_subblock(gram_tr, rhs_tr, sl, alpha)
            pred_sel = x_sel[:, sl] @ w
            ic_raw = daily_spearman_ic(d_sel, pred_sel, r_sel)
            ic_orth = daily_spearman_ic(d_sel, pred_sel, yo_sel)
            score_val = ic_raw if mode == "raw" else ic_orth
            grid.append({"alpha": alpha, "val_ic_raw": ic_raw, "val_ic_orth": ic_orth})
            if score_val > best_ic:
                best_ic, best_alpha = score_val, alpha
        w_refit = solve_ridge_subblock(gram_all, rhs_all, sl, best_alpha)
        solved_heads[arm_name] = (sl, w_refit.astype(np.float32))
        selection_summary[arm_name] = {"selected_alpha": best_alpha, "best_val_ic": best_ic, "grid": grid}

    # Also evaluate val-only baseline on val_sub_mask -> sel_mask for sanity comparison
    xv_sub = x_val[val_sub_mask].astype(np.float64)[:, :512]
    yv_sub = y_raw_val[val_sub_mask]
    gv_sub = xv_sub.T @ xv_sub
    rv_sub = xv_sub.T @ yv_sub
    val_only_grid = []
    for alpha in (1e4, 1e5, 3e5):
        w0 = np.linalg.solve(gv_sub + alpha * np.eye(512), rv_sub)
        ic0 = daily_spearman_ic(d_sel, x_sel[:, :512] @ w0, r_sel)
        val_only_grid.append({"alpha": alpha, "val_ic_raw": ic0})
    selection_summary["val_only_baseline_512"] = val_only_grid
    phase("selection_done", summary=selection_summary)

    # Save weights
    np.savez(OUTPUT / "g1_heads.npz", **{k: v[1] for k, v in solved_heads.items()})

    # Score OOS
    sym_oos, oos_batches = build_oos_batches(oos_raw[0].parent, ev.SECTOR_MAP)
    oos_args = [
        (r, str(weights[0].parent), tokenizer_dir, sym_oos, oos_batches, solved_heads, BATCH_SIZE)
        for r in range(world)
    ]
    with ctx.Pool(world) as pool:
        oos_paths = pool.map(_worker_score_oos, oos_args)

    oos_df = pd.concat([pd.read_csv(p) for p in oos_paths], ignore_index=True).sort_values(["asof_date", "symbol"])
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    out_csv = OUTPUT / "oos_g1_scores.csv.gz"
    oos_df.to_csv(out_csv, index=False, compression="gzip")

    summary = {
        "n_train_rows": n_tr_rows,
        "n_val_rows": int(len(dates_val)),
        "n_val_sel_rows": int(sel_mask.sum()),
        "n_oos_rows": int(len(oos_df)),
        "train_stride": TRAIN_DATE_STRIDE,
        "train_start": TRAIN_START,
        "selection": selection_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("finished", elapsed_sec=summary["elapsed_sec"], out_bytes=out_csv.stat().st_size)


if __name__ == "__main__":
    main()
