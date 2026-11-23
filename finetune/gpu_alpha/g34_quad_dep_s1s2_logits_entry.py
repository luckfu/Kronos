"""G34: Quad-Family (Mini-2k + PT-Small + C2 + Base) dep_layer (s2) Hidden States
+ Deterministic 1-Step Analytical s1/s2 Generative Logit Distribution Stats (40d) at L=45.

Designed for ultra-compact RAM (<3 GB) and fast runtime (~45 min on 2x T4, stride=3 = ~1.95M train rows)
to cleanly leverage smmt315's remaining ~1.33h GPU quota.

Strictly leak-free: train <= 2026-04-30, val_sel 2026-05-18..2026-07-02, OOS 2026-07-17..2026-10-23.
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
import torch.nn.functional as F

from g1_kaggle_entry import build_oos_batches, find_dataset_file
from g1_multiscale_extract_and_fit import (
    TRAIN_FIT_END,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    daily_spearman_ic,
    phase,
    rank_pct_centered,
    solve_ridge_subblock,
    standardize_1d,
    zscore_cross_section_torch,
)
from g2_rank_finetune import build_g2_date_batches
from g4_base_kaggle_entry import _load_kronos_base
from g16_pretrained_small_entry import _load_pretrained_small
from g26_mini2k_pts_alllayers_entry import _load_pretrained_mini
from g31_mini_mastery_c2pool_ft_entry import (
    build_s1_token_prototypes,
    compute_date_weights_g31,
    extract_r5_vol_breadth,
)
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window, run_gpu_workers_failfast

OUTPUT = Path(os.environ.get("G34_OUTPUT", "/kaggle/working/g34_out"))
INPUT = Path(os.environ.get("G34_INPUT", "/kaggle/input"))

# Layout (FEAT_DIM = 2112 + 40 = 2152):
#   0.. 256: dep_mini  (256)
# 256.. 768: dep_pts   (512)
# 768..1280: dep_c2    (512)
#1280..2112: dep_base  (832)
#2112..2122: stats_mini (10)
#2122..2132: stats_pts  (10)
#2132..2142: stats_c2   (10)
#2142..2152: stats_base (10)
SL_DEP_MINI = slice(0, 256)
SL_DEP_PTS = slice(256, 768)
SL_DEP_C2 = slice(768, 1280)
SL_DEP_BASE = slice(1280, 2112)
SL_DEP_QUAD = slice(0, 2112)

SL_ST_MINI = slice(2112, 2122)
SL_ST_PTS = slice(2122, 2132)
SL_ST_C2 = slice(2132, 2142)
SL_ST_BASE = slice(2142, 2152)
SL_ST_QUAD = slice(2112, 2152)
SL_ALL = slice(0, 2152)
FEAT_DIM = 2152

SCHEMES = ("std", "decay365", "regime")
TARGETS = ("raw", "multi", "ret5", "volnorm10")
TRAIN_START_G34 = "2021-01-01"
TRAIN_STRIDE_G34 = 3
ALPHAS_DEP = (1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
ALPHAS_STAT = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)


def _s1_s2_stats(
    model,
    s1_logits: torch.Tensor,
    ctx: torch.Tensor,
    cur_c_norm: torch.Tensor,
    protos: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dep_last [B, D], stats10 [B, 10]) from s1_logits, ctx, and dep_layer/s2 head."""
    p_close, p_oc, p_hl = protos
    pred_s1 = s1_logits.argmax(dim=-1)
    sib = model.embedding.emb_s1(pred_s1)
    dep_seq = model.dep_layer(ctx, sib)
    dep_last = dep_seq[:, -1, :].float()
    s2_logits = model.head.cond_forward(dep_seq[:, -1:, :]).squeeze(1).float()

    l1 = s1_logits[:, -1, :].float()
    p1 = F.softmax(l1, dim=-1)
    lp1 = F.log_softmax(l1, dim=-1)
    exp_c = p1 @ p_close
    std_c = (p1 @ (p_close ** 2) - exp_c ** 2).clamp_min(1e-6).sqrt()
    exp_oc = p1 @ p_oc
    exp_hl = p1 @ p_hl
    exp_ret = exp_c - cur_c_norm
    prob_up = (p1 * (p_close.unsqueeze(0) > cur_c_norm.unsqueeze(1)).float()).sum(dim=-1)
    ent1 = -(p1 * lp1).sum(dim=-1)
    max_p1 = p1.max(dim=-1).values
    top5_p1 = p1.topk(5, dim=-1).values.sum(dim=-1)

    p2 = F.softmax(s2_logits, dim=-1)
    lp2 = F.log_softmax(s2_logits, dim=-1)
    ent2 = -(p2 * lp2).sum(dim=-1)
    max_p2 = p2.max(dim=-1).values

    stats10 = torch.stack([exp_ret, exp_oc, std_c, exp_hl, prob_up, ent1, max_p1, top5_p1, ent2, max_p2], dim=1)
    return dep_last, stats10


class QuadDepLogitExtractor:
    """Single L=45 pass across Mini-2k, PT-Small, C2, and Base extracting dep_layer(s2) + s1/s2 logit stats."""

    def __init__(self, mini_m, tok2k, pts_m, c2_m, base_m, tok_base, device: torch.device):
        self.mini = mini_m
        self.tok2k = tok2k
        self.pts = pts_m
        self.c2 = c2_m
        self.base = base_m
        self.tok_base = tok_base
        self.device = device
        self.proto_2k = build_s1_token_prototypes(tok2k, device)
        self.proto_base = build_s1_token_prototypes(tok_base, device)

    @torch.inference_mode()
    def extract_batch(self, raw_b: np.ndarray, st_b: np.ndarray, sec_b: np.ndarray, pct_b: np.ndarray) -> torch.Tensor:
        raw_t = torch.as_tensor(raw_b[:, -45:], device=self.device, dtype=torch.float32)
        st_t = torch.as_tensor(st_b[:, -45:], device=self.device, dtype=torch.float32)
        sec_t = torch.as_tensor(sec_b, device=self.device, dtype=torch.long)
        pct_t = torch.as_tensor(pct_b, device=self.device, dtype=torch.float32)
        x_norm = norm_window(raw_t)
        cur_c = x_norm[:, -1, 3]

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            t2k = self.tok2k.encode(x_norm, half=True)
            s1_m, c_m = self.mini.decode_s1(t2k[0], t2k[1], st_t)
            dep_m, st_m = _s1_s2_stats(self.mini, s1_m, c_m, cur_c, self.proto_2k)

            tb = self.tok_base.encode(x_norm, half=True)
            s1_p, c_p = self.pts.decode_s1(tb[0], tb[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            dep_p, st_p = _s1_s2_stats(self.pts, s1_p, c_p, cur_c, self.proto_base)

            s1_c, c_c = self.c2.decode_s1(tb[0], tb[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            dep_c, st_c = _s1_s2_stats(self.c2, s1_c, c_c, cur_c, self.proto_base)

            s1_b, c_b = self.base.decode_s1(tb[0], tb[1], st_t, sector_id=sec_t, size_percentile=pct_t)
            dep_b, st_b = _s1_s2_stats(self.base, s1_b, c_b, cur_c, self.proto_base)

        return torch.cat([dep_m, dep_p, dep_c, dep_b, st_m, st_p, st_c, st_b], dim=1)


def _load_models(rank: int, paths: dict[str, str]):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mini = _load_pretrained_mini(paths["mini"], device).eval()
    tok2k = KronosTokenizer.from_pretrained(paths["tok2k"]).to(device).eval()
    pts = _load_pretrained_small(paths["pts"], device).eval()
    c2 = Kronos.from_pretrained(
        paths["c2"], num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    base = _load_kronos_base(paths["base"], device).eval()
    tok_base = KronosTokenizer.from_pretrained(paths["tok_base"]).to(device).eval()
    return device, QuadDepLogitExtractor(mini, tok2k, pts, c2, base, tok_base, device)


@torch.inference_mode()
def _extract_worker(args: tuple) -> str:
    rank, n_gpus, paths, sym_tr, tr_batches, sym_val, val_batches = args
    device, ext = _load_models(rank, paths)

    g_tr = {s: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for s in SCHEMES}
    r_tr = {s: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in TARGETS} for s in SCHEMES}
    g_va = {s: torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device) for s in SCHEMES}
    r_va = {s: {t: torch.zeros(FEAT_DIM, dtype=torch.float64, device=device) for t in TARGETS} for s in SCHEMES}

    def _fwd(sym_arr, s_idx, t_idx, sec_np, pct_np, bs: int = 320) -> torch.Tensor:
        chunks = []
        for off in range(0, len(s_idx), bs):
            sl = slice(off, min(off + bs, len(s_idx)))
            raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
            chunks.append(ext.extract_batch(raw_b, st_b, sec_np[sl], pct_np[sl]))
        return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    n_rows = 0
    my_tr = [b for i, b in enumerate(tr_batches) if i % n_gpus == rank]
    for bi, b in enumerate(my_tr):
        d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
        r5, vol20, breadth = extract_r5_vol_breadth(sym_tr, s_idx, t_idx)
        w_map = compute_date_weights_g31(str(d_str), len(s_idx), breadth)
        x64 = _fwd(sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
        y_map = {
            "raw": torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64),
            "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
            "ret5": torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64),
            "volnorm10": torch.as_tensor(standardize_1d(rank_pct_centered(r10 / vol20)), device=device, dtype=torch.float64),
        }
        xx = x64.T @ x64
        xys = {t: x64.T @ y_map[t] for t in TARGETS}
        for s in SCHEMES:
            w = w_map[s]
            g_tr[s].add_(xx, alpha=w)
            for t in TARGETS:
                r_tr[s][t].add_(xys[t], alpha=w)
        n_rows += int(x64.shape[0])
        if (bi + 1) % 60 == 0 or bi + 1 == len(my_tr):
            phase("g34_train_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_rows)

    val_sel_recs = []
    for b in [b for i, b in enumerate(val_batches) if i % n_gpus == rank]:
        d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _y_orth, r10 = b
        r5, vol20, breadth = extract_r5_vol_breadth(sym_val, s_idx, t_idx)
        w_map = compute_date_weights_g31(str(d_str), len(s_idx), breadth)
        x32 = _fwd(sym_val, s_idx, t_idx, sec_np, pct_np)
        x64 = x32.to(torch.float64)
        y_map = {
            "raw": torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64),
            "multi": torch.as_tensor(y_multi, device=device, dtype=torch.float64),
            "ret5": torch.as_tensor(standardize_1d(rank_pct_centered(r5)), device=device, dtype=torch.float64),
            "volnorm10": torch.as_tensor(standardize_1d(rank_pct_centered(r10 / vol20)), device=device, dtype=torch.float64),
        }
        xx = x64.T @ x64
        xys = {t: x64.T @ y_map[t] for t in TARGETS}
        for s in SCHEMES:
            w = w_map[s]
            g_va[s].add_(xx, alpha=w)
            for t in TARGETS:
                r_va[s][t].add_(xys[t], alpha=w)
        if VAL_SEL_START <= str(d_str) <= VAL_SEL_END:
            val_sel_recs.append({
                "date": str(d_str),
                "x": x32.half().cpu().numpy(),
                "ret10": r10.astype(np.float32),
                "ret5": r5.astype(np.float32),
            })

    out = f"/tmp/g34_stats_r{rank}.pt"
    torch.save({
        "n_rows": n_rows,
        "g_tr": {s: g_tr[s].cpu().numpy() for s in SCHEMES},
        "r_tr": {s: {t: r_tr[s][t].cpu().numpy() for t in TARGETS} for s in SCHEMES},
        "g_va": {s: g_va[s].cpu().numpy() for s in SCHEMES},
        "r_va": {s: {t: r_va[s][t].cpu().numpy() for t in TARGETS} for s in SCHEMES},
        "val_sel_recs": val_sel_recs,
    }, out)
    phase("g34_extract_saved", rank=rank, n_rows=n_rows)
    return out


@torch.inference_mode()
def _score_oos_worker(args: tuple) -> str:
    rank, n_gpus, paths, heads_np, sym_oos, oos_batches = args
    device, ext = _load_models(rank, paths)
    heads_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in heads_np.items()}
    frames = []
    for d_str, recs in [b for i, b in enumerate(oos_batches) if i % n_gpus == rank]:
        chunks = []
        for off in range(0, len(recs), 320):
            sl = recs[off : off + 320]
            raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119 : r[1] + 1] for r in sl])
            st_b = np.stack([sym_oos[r[0]][1][r[1] - 119 : r[1] + 1] for r in sl])
            sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
            chunks.append(ext.extract_batch(raw_b, st_b, sec_b, pct_b))
        x_d = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        df_d = pd.DataFrame({
            "identity": [r[4] for r in recs],
            "symbol": [r[5] for r in recs],
            "asof_date": d_str,
        })
        for k, (sl, wt) in heads_t.items():
            df_d[f"p_{k}"] = (x_d[:, sl] @ wt).cpu().numpy().astype(np.float32)
        frames.append(df_d)
    out = f"/tmp/g34_oos_r{rank}.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, compression="gzip")
    return out


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g34_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    c2_dir = str(find_c2_weights(INPUT).parent)

    hf_dirs = [
        snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="/kaggle/working/hf_kronos_mini"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="/kaggle/working/hf_tok2k"),
        snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small"),
        snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base"),
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tok_base"),
    ]
    paths = {"mini": hf_dirs[0], "tok2k": hf_dirs[1], "pts": hf_dirs[2], "base": hf_dirs[3], "tok_base": hf_dirs[4], "c2": c2_dir}

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G34, TRAIN_FIT_END, min_cross_section=200)
    tr_batches = tr_batches[::TRAIN_STRIDE_G34]
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)

    pts = run_gpu_workers_failfast(
        _extract_worker,
        [(r, n_gpus, paths, sym_tr, tr_batches, sym_val, val_batches) for r in range(n_gpus)],
        tag="g34_ext",
    )
    del sym_tr, tr_batches, sym_val, val_batches
    gc.collect()

    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in pts]
    for p in pts:
        Path(p).unlink(missing_ok=True)

    n_tr_rows = sum(d["n_rows"] for d in parts)
    g_tr = {s: sum(d["g_tr"][s] for d in parts) for s in SCHEMES}
    r_tr = {s: {t: sum(d["r_tr"][s][t] for d in parts) for t in TARGETS} for s in SCHEMES}
    g_all = {s: g_tr[s] + sum(d["g_va"][s] for d in parts) for s in SCHEMES}
    r_all = {s: {t: r_tr[s][t] + sum(d["r_va"][s][t] for d in parts) for t in TARGETS} for s in SCHEMES}

    val_sel_recs = []
    for d in parts:
        val_sel_recs.extend(d["val_sel_recs"])
    val_sel_recs.sort(key=lambda r: r["date"])
    del parts
    gc.collect()

    x_sel = np.concatenate([r["x"].astype(np.float32) for r in val_sel_recs], axis=0)
    d_sel = np.concatenate([np.full(len(r["ret10"]), r["date"]) for r in val_sel_recs])
    r10_sel = np.concatenate([r["ret10"].astype(np.float64) for r in val_sel_recs])
    r5_sel = np.concatenate([r["ret5"].astype(np.float64) for r in val_sel_recs])
    val_sel_recs.clear()

    specs = [
        ("g34_dep_mini_raw", SL_DEP_MINI, "std", "raw", ALPHAS_DEP),
        ("g34_dep_pts_raw", SL_DEP_PTS, "std", "raw", ALPHAS_DEP),
        ("g34_dep_c2_raw", SL_DEP_C2, "std", "raw", ALPHAS_DEP),
        ("g34_dep_base_raw", SL_DEP_BASE, "std", "raw", ALPHAS_DEP),
        ("g34_dep_quad_raw", SL_DEP_QUAD, "std", "raw", ALPHAS_DEP),
        ("g34_dep_quad_multi", SL_DEP_QUAD, "std", "multi", ALPHAS_DEP),
        ("g34_dep_quad_decay365", SL_DEP_QUAD, "decay365", "multi", ALPHAS_DEP),
        ("g34_dep_quad_regime", SL_DEP_QUAD, "regime", "multi", ALPHAS_DEP),
        ("g34_dep_quad_ret5", SL_DEP_QUAD, "std", "ret5", ALPHAS_DEP),
        ("g34_dep_quad_volnorm", SL_DEP_QUAD, "std", "volnorm10", ALPHAS_DEP),
        ("g34_s1s2_stats_mini", SL_ST_MINI, "std", "raw", ALPHAS_STAT),
        ("g34_s1s2_stats_pts", SL_ST_PTS, "std", "raw", ALPHAS_STAT),
        ("g34_s1s2_stats_c2", SL_ST_C2, "std", "raw", ALPHAS_STAT),
        ("g34_s1s2_stats_base", SL_ST_BASE, "std", "raw", ALPHAS_STAT),
        ("g34_s1s2_stats_quad40", SL_ST_QUAD, "std", "multi", ALPHAS_STAT),
        ("g34_s1s2_stats_quad40_regime", SL_ST_QUAD, "regime", "multi", ALPHAS_STAT),
        ("g34_dep_plus_stats_multi", SL_ALL, "std", "multi", ALPHAS_DEP),
        ("g34_dep_plus_stats_regime", SL_ALL, "regime", "multi", ALPHAS_DEP),
    ]
    selection, heads_np = {}, {}
    for name, sl, sch, tkey, alphas in specs:
        xs = np.asarray(x_sel[:, sl], dtype=np.float64)
        best_a, best_ic = None, -1e9
        for a in alphas:
            w = solve_ridge_subblock(g_tr[sch], r_tr[sch][tkey], sl, a)
            pred = xs @ w
            sc = daily_spearman_ic(d_sel, pred, r5_sel if tkey == "ret5" else r10_sel)
            if sc > best_ic:
                best_ic, best_a = sc, a
        assert best_a is not None
        wf = solve_ridge_subblock(g_all[sch], r_all[sch][tkey], sl, best_a).astype(np.float32)
        heads_np[name] = (sl, wf)
        selection[name] = {"scheme": sch, "target": tkey, "dim": int(wf.shape[0]), "alpha": best_a, "val_ic": float(best_ic)}

    np.savez(OUTPUT / "g34_heads.npz", **{k: v[1] for k, v in heads_np.items()})
    del g_tr, r_tr, g_all, r_all, x_sel
    gc.collect()
    phase("g34_heads_fitted", n_heads=len(heads_np), elapsed_sec=round(time.time() - t0, 1))

    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    csvs = run_gpu_workers_failfast(
        _score_oos_worker,
        [(r, n_gpus, paths, heads_np, sym_oos, oos_batches) for r in range(n_gpus)],
        tag="g34_oos",
    )
    oos_df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True).sort_values(["asof_date", "symbol"])
    for p in csvs:
        Path(p).unlink(missing_ok=True)
    if len(oos_df) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(oos_df)}")
    oos_df.to_csv(OUTPUT / "oos_g34_scores.csv.gz", index=False, compression="gzip")

    summary = {"n_train_rows": n_tr_rows, "selection": selection, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g34_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in hf_dirs:
        shutil.rmtree(d, ignore_errors=True)
    phase("g34_finished", n_train_rows=n_tr_rows, elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
