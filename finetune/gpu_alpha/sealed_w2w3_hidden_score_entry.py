"""Kaggle 2x T4 Sealed Acceptance Kernel A (2026-10-26..2026-11-20, 86,195 rows):
Pure frozen forward inference across ALL pre-registered Wave 2 & Wave 3 non-AR models
(Parity head_val_alpha1e4, G25, G29, G31, G33, G34, G27, G30).

Zero fitting, zero tuning: reuses the exact unmodified _score_oos_worker / _part1_oos_worker
functions from each experiment with frozen <= 2026-07-02 heads, PCA bases, NNs, and LGBMs.
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

from g1_multiscale_extract_and_fit import phase, zscore_cross_section_torch
from sealed_common import build_sealed_batches
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window, run_gpu_workers_failfast

INPUT = Path(os.environ.get("SEALED_INPUT", "/kaggle/input"))
WORK = Path("/kaggle/working")
OUTPUT = WORK / "sealed_hidden_out"


@torch.inference_mode()
def _score_val1e4_parity_worker(args: tuple) -> str:
    """Exact production parity scorer for head_val_alpha1e4.npy on C2 L=120 float32."""
    rank, n_gpus, c2_dir, tok_dir, head_val, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    c2 = Kronos.from_pretrained(
        str(c2_dir), num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tok = KronosTokenizer.from_pretrained(str(tok_dir)).to(device).eval()
    w_t = torch.as_tensor(head_val, device=device, dtype=torch.float32).reshape(-1)

    frames = []
    for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
        chunks = []
        for off in range(0, len(recs), 256):
            sl = recs[off : off + 256]
            raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119 : r[1] + 1] for r in sl])
            st_b = np.stack([sym_oos[r[0]][1][r[1] - 119 : r[1] + 1] for r in sl])
            sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
            raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
            st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
            sec_t = torch.as_tensor(sec_b, device=device, dtype=torch.long)
            pct_t = torch.as_tensor(pct_b, device=device, dtype=torch.float32)
            tok120 = tok.encode(norm_window(raw_t), half=True)
            _, ctx_c2 = c2.decode_s1(tok120[0], tok120[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            chunks.append(ctx_c2[:, -1, :].float())
        x_d = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        s = (x_d @ w_t).cpu().numpy().astype(np.float32)
        frames.append(pd.DataFrame({
            "identity": [r[4] for r in recs],
            "symbol": [r[5] for r in recs],
            "asof_date": d_str,
            "p_val_1e4": s,
        }))
    out = f"/tmp/sealed_val1e4_r{rank}.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, compression="gzip")
    return out


def _merge_csvs(csvs: list[str], base_df: pd.DataFrame | None) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True).sort_values("identity").reset_index(drop=True)
    for p in csvs:
        Path(p).unlink(missing_ok=True)
    if base_df is None:
        return df
    score_cols = [c for c in df.columns if c.startswith("p_")]
    return base_df.merge(df[["identity"] + score_cols], on="identity", how="inner", validate="one_to_one")


def main() -> None:
    from huggingface_hub import snapshot_download

    import g25_base_fulldensity_l45_l52_entry as g25
    import g27_nonlinear_moe_lgbm_entry as g27
    import g29_mini_pool_trifamily_entry as g29
    import g30_quadfam_nonlin_xstock_entry as g30
    import g31_mini_mastery_c2pool_ft_entry as g31
    import g33_base_dep_pool_gentle_ft_entry as g33
    import g34_quad_dep_s1s2_logits_entry as g34

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("sealed_hidden_started", gpus=n_gpus)

    c2_dir = str(find_c2_weights(INPUT).parent)
    art_dir = next(INPUT.rglob("sealed_kline_150d.parquet")).parent
    kline_pq = art_dir / "sealed_kline_150d.parquet"
    ranks_pq = art_dir / "sealed_shadow_rankings.parquet"
    phase("sealed_inputs_found", c2_dir=c2_dir, art_dir=str(art_dir))

    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir=str(WORK / "hf_kronos_mini")),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir=str(WORK / "hf_tok_2k")),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir=str(WORK / "hf_kronos_base")),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir=str(WORK / "hf_kronos_small")),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir=str(WORK / "hf_tokenizer")),
    ]
    paths = {
        "mini": hf_dirs[0],
        "tok2k": hf_dirs[1],
        "base": hf_dirs[2],
        "pts": hf_dirs[3],
        "tok_base": hf_dirs[4],
        "c2": c2_dir,
    }
    sym_oos, oos_batches = build_sealed_batches(kline_pq, ranks_pq)
    n_expected = sum(len(b[1]) for b in oos_batches)
    phase("sealed_batches_ready", n_dates=len(oos_batches), n_rows=n_expected, elapsed_sec=round(time.time() - t0, 1))

    # 1. Production parity check: head_val_alpha1e4.npy
    head_val = np.load(art_dir / "head_val_alpha1e4.npy").astype(np.float32)
    csvs = run_gpu_workers_failfast(
        _score_val1e4_parity_worker,
        [(r, n_gpus, c2_dir, paths["tok_base"], head_val, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="sealed_val1e4",
    )
    merged = _merge_csvs(csvs, None)
    phase("stage_val1e4_done", rows=len(merged), elapsed_sec=round(time.time() - t0, 1))

    # 2. G25: Base L=52 pool
    z25 = np.load(art_dir / "g25_heads.npz")
    heads_25 = {
        "g25_base_l52_pool": (g25.SL_52_POOL, z25["g25_base_l52_pool"].astype(np.float32)),
        "g25_base_l45_52_multi": (g25.SL_ALL, z25["g25_base_l45_52_multi"].astype(np.float32)),
    }
    csvs = run_gpu_workers_failfast(
        g25._score_oos_worker,
        [(r, n_gpus, paths["base"], paths["tok_base"], sym_oos, oos_batches, heads_25) for r in range(n_gpus)],
        tag="sealed_g25",
    )
    merged = _merge_csvs(csvs, merged)
    phase("stage_g25_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    # 3. G29: Mini pool + TriFam
    z29 = np.load(art_dir / "g29_heads.npz")
    heads_29 = {
        "g29_mini_l45_pool_regime": (g29.SL_MINI_45_POOL, z29["g29_mini_l45_pool_regime"].astype(np.float32)),
        "g29_mini_l30_45_raw": (g29.SL_MINI_ALL, z29["g29_mini_l30_45_raw"].astype(np.float32)),
        "g29_trifam_l45_decay": (g29.SL_TRIFAM, z29["g29_trifam_l45_decay"].astype(np.float32)),
    }
    csvs = run_gpu_workers_failfast(
        g29._score_oos_worker,
        [(r, n_gpus, paths["mini"], paths["tok2k"], paths["base"], paths["pts"], paths["tok_base"], sym_oos, oos_batches, heads_29) for r in range(n_gpus)],
        tag="sealed_g29",
    )
    merged = _merge_csvs(csvs, merged)
    phase("stage_g29_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    # 4. G31: Mini mastery (s1_stats, dep_s2, volnorm) + C2 L=45 Rich Regime + Mini+C2 Joint
    z31 = np.load(art_dir / "g31_heads.npz")
    m_heads = {
        "g31_mini_s1_stats": (g31.SL_M_S1_STATS, z31["g31_mini_s1_stats"].astype(np.float32)),
        "g31_mini_l45_dep_s2": (g31.SL_M_45_DEP, z31["g31_mini_l45_dep_s2"].astype(np.float32)),
        "g31_mini_l30_45_rich_volnorm": (g31.SL_M_30_45_RICH, z31["g31_mini_l30_45_rich_volnorm"].astype(np.float32)),
    }
    c_heads = {
        "g31_c2_l45_rich_regime": (g31.SL_C_45_RICH, z31["g31_c2_l45_rich_regime"].astype(np.float32)),
    }
    j_heads = {
        "g31_minic2_joint_ret5": (g31.SL_J_ALL, z31["g31_minic2_joint_ret5"].astype(np.float32)),
        "g31_minic2_joint_regime": (g31.SL_J_ALL, z31["g31_minic2_joint_regime"].astype(np.float32)),
        "g31_minic2_joint_decay365": (g31.SL_J_ALL, z31["g31_minic2_joint_decay365"].astype(np.float32)),
    }
    csvs = run_gpu_workers_failfast(
        g31._part1_oos_worker,
        [(r, n_gpus, paths, m_heads, c_heads, j_heads, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="sealed_g31",
    )
    merged = _merge_csvs(csvs, merged)
    phase("stage_g31_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    # 5. G33: Base L=45 dep+regime & Base L=52 ret20/regime
    z33 = np.load(art_dir / "g33_heads.npz")
    heads_33_45 = {
        "g33_base_l45_dep_s2": (g33.SL_B45_DEP, z33["g33_base_l45_dep_s2"].astype(np.float32)),
        "g33_base_l45_all_dep_regime": (g33.SL_B45_ALL, z33["g33_base_l45_all_dep_regime"].astype(np.float32)),
        "g33_base_l45_all_dep_ret20": (g33.SL_B45_ALL, z33["g33_base_l45_all_dep_ret20"].astype(np.float32)),
    }
    heads_33_52 = {
        "g33_base_l52_rich_ret20": (g33.SL_B52_ALL, z33["g33_base_l52_rich_ret20"].astype(np.float32)),
        "g33_base_l52_rich_regime": (g33.SL_B52_ALL, z33["g33_base_l52_rich_regime"].astype(np.float32)),
    }
    csvs = run_gpu_workers_failfast(
        g33._part1_oos_worker,
        [(r, n_gpus, paths["base"], paths["tok_base"], heads_33_45, heads_33_52, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="sealed_g33",
    )
    merged = _merge_csvs(csvs, merged)
    phase("stage_g33_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    # 6. G34: Quad dep_layer(s2) + s1/s2 analytical logit stats
    z34 = np.load(art_dir / "g34_heads.npz")
    heads_34 = {
        "g34_dep_c2_raw": (g34.SL_DEP_C2, z34["g34_dep_c2_raw"].astype(np.float32)),
        "g34_dep_quad_raw": (g34.SL_DEP_QUAD, z34["g34_dep_quad_raw"].astype(np.float32)),
        "g34_dep_quad_decay365": (g34.SL_DEP_QUAD, z34["g34_dep_quad_decay365"].astype(np.float32)),
        "g34_dep_quad_regime": (g34.SL_DEP_QUAD, z34["g34_dep_quad_regime"].astype(np.float32)),
        "g34_dep_plus_stats_regime": (g34.SL_ALL, z34["g34_dep_plus_stats_regime"].astype(np.float32)),
        "g34_s1s2_stats_quad40": (g34.SL_ST_QUAD, z34["g34_s1s2_stats_quad40"].astype(np.float32)),
    }
    csvs = run_gpu_workers_failfast(
        g34._score_oos_worker,
        [(r, n_gpus, paths, heads_34, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="sealed_g34",
    )
    merged = _merge_csvs(csvs, merged)
    phase("stage_g34_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    # 7. G27: Dual-family FiLM / Resid-CosIC / LGBM (LightGBM only runs inside forked child!)
    z27 = np.load(art_dir / "g27_heads.npz")
    sl27 = {
        "pts_all": g27.SL_PTS, "pts_bull": g27.SL_PTS, "pts_bear": g27.SL_PTS,
        "base_all": g27.SL_BASE, "base_bull": g27.SL_BASE, "base_bear": g27.SL_BASE,
        "joint_all": g27.SL_ALL, "joint_bull": g27.SL_ALL, "joint_bear": g27.SL_ALL,
    }
    heads_27 = {k: (sl, z27[k].astype(np.float32)) for k, sl in sl27.items()}
    ckpt27 = torch.load(art_dir / "g27_models.pt", map_location="cpu", weights_only=False)
    csvs = run_gpu_workers_failfast(
        g27._score_oos_worker,
        [
            (
                r, n_gpus, paths["pts"], paths["base"], paths["tok_base"],
                z27["u_pts"].astype(np.float32), z27["u_base"].astype(np.float32),
                heads_27, ckpt27["moe_gates"], ckpt27["nn_specs"],
                str(art_dir / "g27_lgbm.txt"), ckpt27["lgbm_cols"],
                sym_oos, oos_batches,
            )
            for r in range(n_gpus)
        ],
        tag="sealed_g27",
    )
    merged = _merge_csvs(csvs, merged)
    del ckpt27
    gc.collect()
    phase("stage_g27_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    # 8. G30: Quad-family Non-Linear (Resid, FiLM, CrossStock-Attn, LGBM Multi & Ret5)
    z30 = np.load(art_dir / "g30_heads.npz")
    u30 = {
        "m": z30["u_mini"].astype(np.float32),
        "b": z30["u_base"].astype(np.float32),
        "p": z30["u_pts"].astype(np.float32),
        "c": z30["u_c2"].astype(np.float32),
    }
    sl30 = {
        "mini_multi": g30.SL_MINI, "base_raw": g30.SL_BASE, "pts_multi": g30.SL_PTS, "c2_multi": g30.SL_C2,
        "quad_multi": g30.SL_ALL, "quad_ret5": g30.SL_ALL, "quad_decay": g30.SL_ALL,
        "mini_regime": g30.SL_MINI, "quad_regime": g30.SL_ALL,
    }
    heads_30 = {k: (sl, z30[k].astype(np.float32)) for k, sl in sl30.items()}
    ckpt30 = torch.load(art_dir / "g30_models.pt", map_location="cpu", weights_only=False)
    csvs = run_gpu_workers_failfast(
        g30._score_oos_worker,
        [
            (
                r, n_gpus, paths, u30, heads_30, ckpt30["nn_specs"],
                str(art_dir / "g30_lgbm_multi.txt"), str(art_dir / "g30_lgbm_ret5.txt"),
                ckpt30["lgbm_cols"], sym_oos, oos_batches,
            )
            for r in range(n_gpus)
        ],
        tag="sealed_g30",
    )
    merged = _merge_csvs(csvs, merged)
    del ckpt30
    gc.collect()
    phase("stage_g30_done", cols=len(merged.columns), elapsed_sec=round(time.time() - t0, 1))

    if len(merged) != n_expected:
        raise RuntimeError(f"Row count mismatch: {len(merged)} vs {n_expected}")

    # Live 2026-11-20 parity verification directly against shadow rankings!
    ranks_df = pd.read_parquet(ranks_pq)
    ranks_df["asof"] = ranks_df["asof"].astype(str)
    ranks_df["code"] = ranks_df["code"].astype(str)
    ranks_df["identity"] = ranks_df["asof"] + "|" + ranks_df["code"]
    parity_info: dict = {}
    if "score_hidden" in ranks_df.columns:
        d20 = ranks_df[ranks_df["asof"] == "2026-11-20"].merge(
            merged[["identity", "p_val_1e4"]], on="identity", how="inner"
        )
        d20["r_c2"] = d20["predicted_return_d10"].rank(method="average", pct=True)
        d20["r_val"] = d20["p_val_1e4"].rank(method="average", pct=True)
        d20["f_val"] = 0.5 * d20["r_c2"] + 0.5 * d20["r_val"]
        parity_info = {
            "asof": "2026-11-20",
            "n": int(len(d20)),
            "score_hidden_corr": float(d20["score_hidden"].corr(d20["p_val_1e4"])),
            "score_fused_corr": float(d20["score_fused"].corr(d20["f_val"])),
            "score_fused_max_abs_diff": float((d20["score_fused"] - d20["f_val"]).abs().max()),
        }
        phase("sealed_parity_check", **parity_info)

    out_pq = OUTPUT / "sealed_w2w3_hidden_scores.parquet"
    merged.to_parquet(out_pq, index=False)
    summary = {
        "n_rows": int(len(merged)),
        "n_dates": int(len(oos_batches)),
        "score_columns": [c for c in merged.columns if c.startswith("p_")],
        "parity_2026_11_20": parity_info,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "sealed_hidden_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("sealed_hidden_finished", n_rows=len(merged), n_cols=len(summary["score_columns"]), elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
