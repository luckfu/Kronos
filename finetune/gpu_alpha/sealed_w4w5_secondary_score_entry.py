"""Kaggle 2x T4 December Secondary Sealed Track Inference Kernel (Wave 4 + Wave 5 + W2/W3 Refs).

Pure frozen forward inference (zero training, zero tuning, zero hyperparameter selection):
- Reuses exact _score_val1e4_parity_worker parity check + G25..G40 OOS workers with frozen <= 2026-07-02
  weights/heads/models on sealed kline & shadow ranking snapshots.
- Exports sealed_w4w5_secondary_scores.parquet and sealed_g38_regime_hazard.csv.
"""
from __future__ import annotations

import gc
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
from sealed_common import build_sealed_batches
from sealed_w2w3_hidden_score_entry import _score_val1e4_parity_worker
from short_probe_common import detect_gpu_count, find_c2_weights, run_gpu_workers_failfast
from wave4_common import SL_ALL, SL_BASE, SL_C2, SL_MINI, SL_PTS, SL_STATS

INPUT = Path(os.environ.get("SEALED_INPUT", "/kaggle/input"))
WORK = Path("/kaggle/working")
OUTPUT = WORK / "sealed_w4w5_out"


def _find_optional(filename: str) -> Path | None:
    hits = sorted(INPUT.rglob(filename))
    return hits[0] if hits else None


def _merge_csvs(csvs: list[str], base_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True).sort_values("identity").reset_index(drop=True)
    for p in csvs:
        Path(p).unlink(missing_ok=True)
    score_cols = [c for c in df.columns if c.startswith("p_") and c not in base_df.columns]
    if not score_cols:
        return base_df
    return base_df.merge(df[["identity"] + score_cols], on="identity", how="inner", validate="one_to_one")


def main() -> None:
    from huggingface_hub import snapshot_download

    import g35_crash_shield_tailrisk_entry as g35
    import g36_lambdarank_topk_liquid_entry as g36
    import g37_quad_to_light_distill_entry as g37
    import g38_termstruct_hazard_net_entry as g38
    import g39_tailshield_fwd_light_distill_entry as g39
    import g40_mini_tailaware_gentle_ft_entry as g40

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("sealed_w4w5_started", gpus=n_gpus)

    c2_dir = str(find_c2_weights(INPUT).parent)
    kline_pq = next(INPUT.rglob("sealed_kline_150d.parquet"))
    ranks_pq = next(INPUT.rglob("sealed_shadow_rankings.parquet"))
    val1e4_npy = next(INPUT.rglob("head_val_alpha1e4.npy"))

    hf_dirs = {
        "mini": snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir=str(WORK / "hf_kronos_mini")),
        "tok2k": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir=str(WORK / "hf_tok2k")),
        "base": snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir=str(WORK / "hf_kronos_base")),
        "pts": snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir=str(WORK / "hf_kronos_small")),
        "tok_base": snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tokenizer")),
    }
    paths = {**hf_dirs, "c2": c2_dir}
    light_paths = {k: hf_dirs[k] for k in ("mini", "tok2k", "tok_base")}
    light_paths["c2"] = c2_dir

    sym_oos, oos_batches = build_sealed_batches(kline_pq, ranks_pq)
    n_expected = sum(len(b[1]) for b in oos_batches)

    # 1. Production Parity Gate: head_val_alpha1e4.npy
    head_val = np.load(val1e4_npy).astype(np.float32)
    csvs = run_gpu_workers_failfast(
        _score_val1e4_parity_worker,
        [(r, n_gpus, c2_dir, paths["tok_base"], head_val, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="sec_val1e4",
    )
    merged = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True).sort_values("identity").reset_index(drop=True)
    for p in csvs:
        Path(p).unlink(missing_ok=True)
    phase("sec_val1e4_done", rows=len(merged), elapsed_sec=round(time.time() - t0, 1))

    # Merge or score Wave 2/3 frozen references (if pre-scored parquet present, merge first)
    w2w3_pq = _find_optional("sealed_w2w3_hidden_scores.parquet")
    if w2w3_pq is not None:
        ref_df = pd.read_parquet(w2w3_pq)
        ref_cols = [c for c in ref_df.columns if c.startswith("p_") and c not in merged.columns]
        if ref_cols:
            merged = merged.merge(ref_df[["identity"] + ref_cols], on="identity", how="inner", validate="one_to_one")
            phase("sec_w2w3_merged", cols=len(ref_cols))

    # 2. G35 Crash-Shield
    g35_hp = _find_optional("g35_heads.npz")
    if g35_hp is not None:
        g35_dir = g35_hp.parent
        z35 = np.load(g35_hp)
        u35 = {"m": z35["u_mini"].astype(np.float32), "b": z35["u_base"].astype(np.float32), "p": z35["u_pts"].astype(np.float32), "c": z35["u_c2"].astype(np.float32)}
        sl35 = {
            "mini_cvar": SL_MINI, "base_cvar": SL_BASE, "c2_cvar": SL_C2, "stats_cvar10": SL_STATS,
            "quad_multi": SL_ALL, "quad_floor10": SL_ALL, "quad_cvar10": SL_ALL, "quad_nowhipsaw": SL_ALL,
            "quad_calmar10": SL_ALL, "quad_crashentry_cvar": SL_ALL,
        }
        h35 = {k: (sl, z35[k].astype(np.float32)) for k, sl in sl35.items()}
        c35 = torch.load(g35_dir / "g35_models.pt", map_location="cpu", weights_only=False)
        g35.TMP_DIR.mkdir(parents=True, exist_ok=True)
        csvs = run_gpu_workers_failfast(
            g35._score_oos_worker,
            [(r, n_gpus, paths, u35, h35, c35["nn_specs"], str(g35_dir / "g35_lgbm_cvar.txt"), str(g35_dir / "g35_lgbm_nowhipsaw.txt"), c35["lgbm_cols"], sym_oos, oos_batches) for r in range(n_gpus)],
            tag="sec_g35",
        )
        merged = _merge_csvs(csvs, merged)
        del c35
        gc.collect()
        phase("sec_g35_done", elapsed_sec=round(time.time() - t0, 1))

    # 3. G36 Positive Control (Quad Liq Ridge; ranking-loss arms excluded from candidate adjudication)
    g36_hp = _find_optional("g36_heads.npz")
    if g36_hp is not None:
        g36_dir = g36_hp.parent
        z36 = np.load(g36_hp)
        u36 = {"m": z36["u_mini"].astype(np.float32), "b": z36["u_base"].astype(np.float32), "p": z36["u_pts"].astype(np.float32), "c": z36["u_c2"].astype(np.float32)}
        sl36 = {
            "mini_regime": SL_MINI, "base_raw": SL_BASE, "pts_multi": SL_PTS, "c2_multi": SL_C2,
            "quad_multi": SL_ALL, "quad_regime": SL_ALL, "quad_ret5": SL_ALL, "quad_liq_multi": SL_ALL,
        }
        h36 = {k: (sl, z36[k].astype(np.float32)) for k, sl in sl36.items()}
        c36 = torch.load(g36_dir / "g36_models.pt", map_location="cpu", weights_only=False)
        g36.TMP_DIR.mkdir(parents=True, exist_ok=True)
        csvs = run_gpu_workers_failfast(
            g36._score_oos_worker,
            [(r, n_gpus, paths, u36, h36, c36["nn_specs"], str(g36_dir / "g36_lambdarank_r10.txt"), str(g36_dir / "g36_lambdarank_multi.txt"), c36["lgbm_cols"], sym_oos, oos_batches) for r in range(n_gpus)],
            tag="sec_g36",
        )
        merged = _merge_csvs(csvs, merged)
        del c36
        gc.collect()
        phase("sec_g36_done", elapsed_sec=round(time.time() - t0, 1))

    # 4. G37 Distilled Light Students (zero Base/PTS path)
    g37_hp = _find_optional("g37_heads.npz")
    if g37_hp is not None:
        g37_dir = g37_hp.parent
        z37 = np.load(g37_hp)
        sl37 = {
            "c2_std_multi": g37.SL_C2, "c2_regime": g37.SL_C2, "c2_ret5": g37.SL_C2,
            "c2_l45_regime": slice(g37.D_SMALL * 1, g37.D_SMALL * 8),
            "mini_regime": g37.SL_MINI, "mc2_regime": g37.SL_MINIC2, "mc2_ret5": g37.SL_MINIC2,
        }
        h37 = {k: (sl, z37[k].astype(np.float32)) for k, sl in sl37.items()}
        c37 = torch.load(g37_dir / "g37_models.pt", map_location="cpu", weights_only=False)
        g37.TMP_DIR.mkdir(parents=True, exist_ok=True)
        csvs = run_gpu_workers_failfast(
            g37._score_oos_worker,
            [(r, n_gpus, light_paths, z37["u_c2"].astype(np.float32), z37["u_mini"].astype(np.float32), h37, c37["nn_specs"], str(g37_dir / "g37_lgbm_c2only.txt"), str(g37_dir / "g37_lgbm_minic2.txt"), c37["c2_anc_keys"], c37["mc2_anc_keys"], sym_oos, oos_batches) for r in range(n_gpus)],
            tag="sec_g37",
        )
        merged = _merge_csvs(csvs, merged)
        del c37
        gc.collect()
        phase("sec_g37_done", elapsed_sec=round(time.time() - t0, 1))

    # 5. G38 Term-Structure & Causal Crash Hazard
    g38_hp = _find_optional("g38_heads.npz")
    if g38_hp is not None:
        g38_dir = g38_hp.parent
        z38 = np.load(g38_hp)
        u38 = {"m": z38["u_mini"].astype(np.float32), "c": z38["u_c2"].astype(np.float32), "b": z38["u_base"].astype(np.float32)}
        sl38 = {
            "mini_r5": g38.SL_MINI, "c2_r5": g38.SL_C2, "c2_fwd5_10": g38.SL_C2, "base_r10": g38.SL_BASE,
            "tri_r3": g38.SL_ALL, "tri_r5": g38.SL_ALL, "tri_fwd5_10": g38.SL_ALL, "tri_r10": g38.SL_ALL, "tri_regime": g38.SL_ALL,
        }
        h38 = {k: (sl, z38[k].astype(np.float32)) for k, sl in sl38.items()}
        c38 = torch.load(g38_dir / "g38_models.pt", map_location="cpu", weights_only=False)
        g38.TMP_DIR.mkdir(parents=True, exist_ok=True)
        res_pairs = run_gpu_workers_failfast(
            g38._score_oos_worker,
            [(r, n_gpus, paths, u38, h38, c38["mlp_states"], str(g38_dir / "g38_lgbm_cont.txt"), str(g38_dir / "g38_lgbm_r10.txt"), c38["lgbm_cols"], sym_oos, oos_batches) for r in range(n_gpus)],
            tag="sec_g38",
        )
        csv38, mac38 = zip(*(p.split("||") for p in res_pairs))
        merged = _merge_csvs(list(csv38), merged)
        mac_all = []
        for mp_p in mac38:
            mac_all.extend(json.loads(Path(mp_p).read_text()))
            Path(mp_p).unlink(missing_ok=True)
        mac_all.sort(key=lambda r: r["asof_date"])
        hm = c38["hazard"]
        sc_m = np.asarray(hm["scaler_mean"], dtype=np.float64)
        sc_s = np.maximum(np.asarray(hm["scaler_scale"], dtype=np.float64), 1e-8)
        wc, bc = np.asarray(hm["coef"], dtype=np.float64), float(hm["intercept"])
        wr, br = np.asarray(hm["ridge_coef"], dtype=np.float64), float(hm["ridge_intercept"])
        haz_rows = []
        for r in mac_all:
            f = np.asarray(r["feat"], dtype=np.float64)
            fz = np.clip((f - sc_m) / sc_s, -4.0, 4.0)
            prob = 1.0 / (1.0 + np.exp(-np.clip(float(fz @ wc + bc), -20.0, 20.0)))
            haz_rows.append({
                "asof_date": r["asof_date"],
                "p_crash_hazard": prob,
                "pred_univ_ret10": float(fz @ wr + br),
                "breadth_ma20": float(f[0]),
                "breadth_ma5": float(f[1]),
                "delta_b20_3d": float(f[3]),
                "delta_b20_5d": float(f[4]),
            })
        pd.DataFrame(haz_rows).to_csv(OUTPUT / "sealed_g38_regime_hazard.csv", index=False)
        del c38
        gc.collect()
        phase("sec_g38_done", elapsed_sec=round(time.time() - t0, 1))

    # Release heavy teacher backbones before running Wave 5 zero-Base/PTS models
    for k in ("base", "pts"):
        shutil.rmtree(hf_dirs[k], ignore_errors=True)

    # 6. G39 3-Target Distilled Tail-Shield & Fwd5-10
    g39_hp = _find_optional("g39_heads.npz")
    if g39_hp is not None:
        g39_dir = g39_hp.parent
        z39 = np.load(g39_hp)
        sl39 = {
            "c2_cvar": g39.SL_C2, "c2_nowhipsaw": g39.SL_C2, "c2_fwd5_10": g39.SL_C2, "c2_crashentry": g39.SL_C2,
            "mini_cvar": g39.SL_MINI, "mc2_cvar": g39.SL_MINIC2, "mc2_fwd5_10": g39.SL_MINIC2, "mc2_liqcvar": g39.SL_MINIC2,
        }
        h39 = {k: (sl, z39[k].astype(np.float32)) for k, sl in sl39.items()}
        c39 = torch.load(g39_dir / "g39_models.pt", map_location="cpu", weights_only=False)
        lg39 = {
            "c2_cvar": str(g39_dir / "g39_lgbm_c2only_cvar.txt"),
            "c2_fwd": str(g39_dir / "g39_lgbm_c2only_fwd.txt"),
            "mc2_cvar": str(g39_dir / "g39_lgbm_minic2_cvar.txt"),
            "mc2_fwd": str(g39_dir / "g39_lgbm_minic2_fwd.txt"),
        }
        g39.TMP_DIR.mkdir(parents=True, exist_ok=True)
        csvs = run_gpu_workers_failfast(
            g39._score_oos_worker,
            [(r, n_gpus, light_paths, z39["u_c2"].astype(np.float32), z39["u_mini"].astype(np.float32), h39, c39["nn_specs"], lg39, c39["c2_anc_keys"], c39["mc2_anc_keys"], sym_oos, oos_batches) for r in range(n_gpus)],
            tag="sec_g39",
        )
        merged = _merge_csvs(csvs, merged)
        del c39
        gc.collect()
        phase("sec_g39_done", elapsed_sec=round(time.time() - t0, 1))

    # 7. G40 STE dep_layer Tail-Aware FT-Mini + C2
    g40_hp = _find_optional("g40_heads.npz")
    if g40_hp is not None:
        g40_dir = g40_hp.parent
        z40 = np.load(g40_hp)
        sl40 = {
            "ftmini_cvar": g40.SL_FTMINI, "ftmini_fwd5_10": g40.SL_FTMINI, "ftmini_nowhipsaw": g40.SL_FTMINI,
            "joint_cvar": g40.SL_ALL, "joint_nowhipsaw": g40.SL_ALL, "joint_fwd5_10": g40.SL_ALL,
            "joint_crashentry": g40.SL_ALL, "joint_liqcvar": g40.SL_ALL,
        }
        h40 = {k: (sl, z40[k].astype(np.float32)) for k, sl in sl40.items()}
        c40 = torch.load(g40_dir / "g40_models.pt", map_location="cpu", weights_only=False)
        g40.TMP_DIR.mkdir(parents=True, exist_ok=True)
        csvs = run_gpu_workers_failfast(
            g40._score_oos_worker,
            [(r, n_gpus, light_paths, c40["top30"], c40["top45"], z40["u_ftm"].astype(np.float32), z40["u_c2"].astype(np.float32), h40, c40["nn_specs"], str(g40_dir / "g40_lgbm_cvar10.txt"), str(g40_dir / "g40_lgbm_fwd5_10.txt"), sym_oos, oos_batches) for r in range(n_gpus)],
            tag="sec_g40",
        )
        merged = _merge_csvs(csvs, merged)
        del c40
        gc.collect()
        phase("sec_g40_done", elapsed_sec=round(time.time() - t0, 1))

    if len(merged) != n_expected:
        raise RuntimeError(f"Row count mismatch: {len(merged)} vs {n_expected}")

    ranks_df = pd.read_parquet(ranks_pq)
    ranks_df["asof"] = ranks_df["asof"].astype(str)
    ranks_df["code"] = ranks_df["code"].astype(str)
    ranks_df["identity"] = ranks_df["asof"] + "|" + ranks_df["code"]
    parity_info: dict = {}
    if "score_hidden" in ranks_df.columns:
        last_asof = str(ranks_df["asof"].max())
        dp = ranks_df[ranks_df["asof"] == last_asof].merge(merged[["identity", "p_val_1e4"]], on="identity", how="inner")
        dp["r_c2"] = dp["predicted_return_d10"].rank(method="average", pct=True)
        dp["r_val"] = dp["p_val_1e4"].rank(method="average", pct=True)
        dp["f_val"] = 0.5 * dp["r_c2"] + 0.5 * dp["r_val"]
        parity_info = {
            "asof": last_asof,
            "n": int(len(dp)),
            "score_hidden_corr": float(dp["score_hidden"].corr(dp["p_val_1e4"])),
            "score_fused_corr": float(dp["score_fused"].corr(dp["f_val"])),
        }

    out_pq = OUTPUT / "sealed_w4w5_secondary_scores.parquet"
    merged.to_parquet(out_pq, index=False)
    summary = {
        "n_rows": int(len(merged)),
        "n_dates": int(len(oos_batches)),
        "score_columns": [c for c in merged.columns if c.startswith("p_")],
        "parity_latest": parity_info,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "sealed_w4w5_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in light_paths.values():
        if Path(d).as_posix().startswith("/kaggle/working/hf_"):
            shutil.rmtree(d, ignore_errors=True)
    phase("sealed_w4w5_finished", n_rows=len(merged), n_cols=len(summary["score_columns"]), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
