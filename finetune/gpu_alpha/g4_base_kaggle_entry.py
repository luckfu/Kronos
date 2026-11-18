"""Kaggle GPU entry point for Experiment G4: Kronos-base (832d) Frozen Ridge + Top-Block RankFT."""
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
import torch

from g1_kaggle_entry import build_oos_batches, find_dataset_file
from g1_multiscale_extract_and_fit import (
    FEATURES,
    OOS_END,
    OOS_START,
    TRAIN_FIT_END,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    daily_spearman_ic,
    orthogonalize_target,
    phase,
    solve_ridge_subblock,
    zscore_cross_section_torch,
)
from g2_rank_finetune import (
    TRAIN_START_G2,
    TopBlockRankAdapter,
    build_g2_date_batches,
    cosine_ic_loss_vec,
    kl_logits_anchor,
)

OUTPUT = Path(os.environ.get("G4_OUTPUT", "/kaggle/working/g4_out"))
INPUT = Path(os.environ.get("G4_INPUT", "/kaggle/input"))
ALPHAS_G4 = (1e4, 3e4, 1e5, 3e5, 1e6, 3e6)


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


def _load_kronos_base(hf_model_dir: str, device):
    import inspect
    from safetensors.torch import load_file as load_safetensors
    from model import Kronos
    cfg_path = Path(hf_model_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(num_sectors=86, num_size_buckets=0, context_layer=10, use_size_percentile=True, size_mlp_hidden_dim=64)
    valid = set(inspect.signature(Kronos.__init__).parameters.keys()) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in valid}
    model = Kronos(**cfg)
    model.load_state_dict(load_safetensors(str(Path(hf_model_dir) / "model.safetensors")), strict=False)
    return model.to(device)


@torch.inference_mode()
def _extract_base_feat(model, tokenizer, raw_t: torch.Tensor, st_t: torch.Tensor, sec_t: torch.Tensor, pct_t: torch.Tensor) -> torch.Tensor:
    device = raw_t.device
    mean = raw_t.mean(dim=1, keepdim=True)
    std = raw_t.std(dim=1, unbiased=False, keepdim=True)
    x_norm = torch.clamp((raw_t - mean) / (std + 1e-5), -5.0, 5.0)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        toks = tokenizer.encode(x_norm, half=True)
        _, ctx = model.decode_s1(toks[0], toks[1], st_t, sector_id=sec_t, size_percentile=pct_t)
    h_last = ctx[:, -1, :].float()
    h_m5 = ctx[:, -5:, :].float().mean(dim=1)
    return torch.cat([h_last, h_m5], dim=1)  # [B, 1664]


def gpu0_base_ridge_worker(args: tuple) -> str:
    """GPU 0: Full-universe frozen Kronos-base 832d / 1664d Ridge extraction, val selection, refit & OOS scoring."""
    hf_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer

    device = torch.device("cuda:0")
    model = _load_kronos_base(hf_dir, device).eval()
    tokenizer = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    d_model = int(model.d_model)
    feat_dim = d_model * 2

    gram_tr = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
    rhs_raw_tr = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    rhs_orth_tr = torch.zeros(feat_dim, dtype=torch.float64, device=device)

    def _run_batch(sym_arr, recs_tuple, is_oos=False):
        if is_oos:
            chunks = []
            for off in range(0, len(recs_tuple), 384):
                sl = recs_tuple[off:off + 384]
                raw_b = np.stack([sym_arr[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
                st_b = np.stack([sym_arr[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
                sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
                pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
                feats = _extract_base_feat(
                    model, tokenizer,
                    torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                    torch.as_tensor(st_b, device=device, dtype=torch.float32),
                    torch.as_tensor(sec_b, device=device, dtype=torch.long),
                    torch.as_tensor(pct_b, device=device, dtype=torch.float32),
                )
                chunks.append(feats)
            return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        else:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, y_orth, r10 = recs_tuple
            n = len(s_idx)
            chunks = []
            for off in range(0, n, 384):
                sl = slice(off, min(off + 384, n))
                raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                feats = _extract_base_feat(
                    model, tokenizer,
                    torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                    torch.as_tensor(st_b, device=device, dtype=torch.float32),
                    torch.as_tensor(sec_np[sl], device=device, dtype=torch.long),
                    torch.as_tensor(pct_np[sl], device=device, dtype=torch.float32),
                )
                chunks.append(feats)
            return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

    # Stride=4 train batches for Base (faster + covers ~370 full cross-sections ~1.6M rows!)
    tr_sub = tr_batches[::4]
    for bi, b in enumerate(tr_sub):
        x_d = _run_batch(sym_tr, b, is_oos=False).to(torch.float64)
        _, _, _, _, _, y_multi, y_orth, r10 = b
        from g1_multiscale_extract_and_fit import rank_pct_centered, standardize_1d
        yr = torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64)
        yo = torch.as_tensor(y_orth, device=device, dtype=torch.float64)
        gram_tr.addmm_(x_d.T, x_d)
        rhs_raw_tr.addmv_(x_d.T, yr)
        rhs_orth_tr.addmv_(x_d.T, yo)
        if (bi + 1) % 60 == 0 or bi + 1 == len(tr_sub):
            phase("g4_ridge_train", done=bi + 1, total=len(tr_sub))

    vx_l, vd_l, vr_l, vyr_l, vyo_l = [], [], [], [], []
    for b in val_batches:
        d_str, _, _, _, _, _, y_orth, r10 = b
        x_d = _run_batch(sym_val, b, is_oos=False)
        vx_l.append(x_d.cpu().numpy())
        vd_l.extend([d_str] * len(r10))
        vr_l.append(r10)
        from g1_multiscale_extract_and_fit import rank_pct_centered, standardize_1d
        vyr_l.append(standardize_1d(rank_pct_centered(r10)).astype(np.float32))
        vyo_l.append(y_orth)

    x_val = np.concatenate(vx_l, axis=0).astype(np.float64)
    dates_val = np.asarray(vd_l)
    ret10_val = np.concatenate(vr_l).astype(np.float64)
    y_raw_val = np.concatenate(vyr_l).astype(np.float64)
    y_orth_val = np.concatenate(vyo_l).astype(np.float64)

    g_tr_np = gram_tr.cpu().numpy()
    rr_tr_np = rhs_raw_tr.cpu().numpy()
    ro_tr_np = rhs_orth_tr.cpu().numpy()
    g_all = g_tr_np + x_val.T @ x_val
    rr_all = rr_tr_np + x_val.T @ y_raw_val
    ro_all = ro_tr_np + x_val.T @ y_orth_val

    sel_mask = (dates_val >= VAL_SEL_START) & (dates_val <= VAL_SEL_END)
    x_sel, d_sel, r_sel, yo_sel = x_val[sel_mask], dates_val[sel_mask], ret10_val[sel_mask], y_orth_val[sel_mask]

    arms = {
        "g4_base_ridge_832": (slice(0, d_model), "raw"),
        "g4_base_ridge_m5": (slice(0, feat_dim), "raw"),
        "g4_base_orth_832": (slice(0, d_model), "orth"),
    }
    sel_summary, heads = {}, {}
    for name, (sl, mode) in arms.items():
        rhs_tr = rr_tr_np if mode == "raw" else ro_tr_np
        rhs_all = rr_all if mode == "raw" else ro_all
        best_a, best_ic = None, -1e9
        grid = []
        for a in ALPHAS_G4:
            w = solve_ridge_subblock(g_tr_np, rhs_tr, sl, a)
            pred = x_sel[:, sl] @ w
            ic_r = daily_spearman_ic(d_sel, pred, r_sel)
            ic_o = daily_spearman_ic(d_sel, pred, yo_sel)
            sc = ic_r if mode == "raw" else ic_o
            grid.append({"alpha": a, "val_ic_raw": ic_r, "val_ic_orth": ic_o})
            if sc > best_ic:
                best_ic, best_a = sc, a
        heads[name] = (sl, torch.as_tensor(solve_ridge_subblock(g_all, rhs_all, sl, best_a), device=device, dtype=torch.float32))
        sel_summary[name] = {"selected_alpha": best_a, "best_val_ic": best_ic, "grid": grid}

    oos_frames = []
    for d_str, recs in oos_batches:
        x_d = _run_batch(sym_oos, recs, is_oos=True)
        df_d = pd.DataFrame({"identity": [r[4] for r in recs], "symbol": [r[5] for r in recs], "asof_date": d_str})
        for name, (sl, wt) in heads.items():
            df_d[f"p_{name}"] = (x_d[:, sl] @ wt).cpu().numpy().astype(np.float32)
        oos_frames.append(df_d)

    out_path = "/tmp/g4_gpu0_ridge.pt"
    torch.save({
        "selection": sel_summary,
        "oos_df": pd.concat(oos_frames, ignore_index=True),
        "heads": {k: v[1].cpu().numpy() for k, v in heads.items()},
    }, out_path)
    phase("g4_ridge_done", summary={k: v["best_val_ic"] for k, v in sel_summary.items()})
    return out_path


def gpu1_base_rankft_worker(args: tuple) -> str:
    """GPU 1: Freeze Kronos-base layers 0..8, fine-tune layers 9..11 + norm + linear rank head."""
    hf_dir, tok_dir, sym_tr, tr_batches, sym_val, val_sel_batches, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import KronosTokenizer
    from g2_kaggle_entry import evaluate_adapter_on_batches, warm_start_rank_head

    device = torch.device("cuda:1")
    model = _load_kronos_base(hf_dir, device)
    tokenizer = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    adapter = TopBlockRankAdapter(model, freeze_until=9).to(device)
    warm_start_rank_head(adapter, tokenizer, sym_tr, tr_batches, device)
    init_vm, _ = evaluate_adapter_on_batches(adapter, tokenizer, sym_val, val_sel_batches, device, (120,), eval_batch=256)
    phase("g4_ft_init_val", **init_vm)

    top_params = [p for n, p in adapter.model.named_parameters() if p.requires_grad]
    head_params = list(adapter.rank_norm.parameters()) + list(adapter.rank_head.parameters())
    opt = torch.optim.AdamW([
        {"params": top_params, "lr": 1.5e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 1.5e-4, "weight_decay": 1e-2},
    ])
    epochs = 3
    total_steps = epochs * len(tr_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=1.5e-6)
    rng = np.random.default_rng(20261116)
    best_comp = init_vm["val_composite"]
    best_state = adapter.extract_top_state_dict()
    history = [{"step": 0, **init_vm}]

    step = 0
    val_every = max(1, len(tr_batches) // 2)
    for _ in range(epochs):
        for bi in rng.permutation(len(tr_batches)):
            _, s_idx, t_idx, sec_np, pct_np, y_multi, y_orth, _ = tr_batches[bi]
            n = len(s_idx)
            take = rng.choice(n, size=min(192, n), replace=False) if n > 192 else np.arange(n)
            raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in take])
            st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in take])
            h_low = adapter.encode_frozen_lower(
                tokenizer,
                torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
            )
            adapter.train()
            adapter.model.embedding.eval()
            adapter.model.time_emb.eval()
            for idx in range(adapter.freeze_until):
                adapter.model.transformer[idx].eval()

            opt.zero_grad(set_to_none=True)
            score, s1_ft, s1_c2 = adapter.forward_top(
                h_low,
                torch.as_tensor(sec_np[take], device=device, dtype=torch.long),
                torch.as_tensor(pct_np[take], device=device, dtype=torch.float32),
                with_teacher=True,
            )
            ym = torch.as_tensor(y_multi[take], device=device, dtype=torch.float32)
            yo = torch.as_tensor(y_orth[take], device=device, dtype=torch.float32)
            loss = (
                0.70 * cosine_ic_loss_vec(score, ym)
                + 0.30 * cosine_ic_loss_vec(score, yo)
                + 0.05 * kl_logits_anchor(s1_c2, s1_ft, temperature=1.5)
                + 1000.0 * adapter.weight_l2_anchor()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(top_params + head_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % val_every == 0 or step == total_steps:
                vm, _ = evaluate_adapter_on_batches(adapter, tokenizer, sym_val, val_sel_batches, device, (120,), eval_batch=256)
                history.append({"step": step, **vm})
                phase("g4_ft_val", step=step, **vm)
                if vm["val_composite"] > best_comp:
                    best_comp = vm["val_composite"]
                    best_state = adapter.extract_top_state_dict()

    # Load best checkpoint and score OOS
    model_sd = adapter.model.state_dict()
    for k, v in best_state.items():
        if k.startswith("model."):
            mkey = k[6:]
            model_sd[mkey] = v.to(dtype=model_sd[mkey].dtype, device=device)
    adapter.model.load_state_dict(model_sd, strict=True)
    adapter.rank_norm.load_state_dict({k[10:]: v.to(device) for k, v in best_state.items() if k.startswith("rank_norm.")})
    adapter.rank_head.load_state_dict({k[10:]: v.to(device) for k, v in best_state.items() if k.startswith("rank_head.")})
    adapter.eval()

    oos_frames = []
    with torch.inference_mode():
        for d_str, recs in oos_batches:
            n = len(recs)
            s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
            t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
            sec_np = np.asarray([r[2] for r in recs], dtype=np.int64)
            pct_np = np.asarray([r[3] for r in recs], dtype=np.float32)
            preds = []
            for off in range(0, n, 256):
                sl = slice(off, min(off + 256, n))
                raw_b = np.stack([sym_oos[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_oos[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                h_low = adapter.encode_frozen_lower(
                    tokenizer,
                    torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                    torch.as_tensor(st_b, device=device, dtype=torch.float32),
                )
                sc, _, _ = adapter.forward_top(
                    h_low,
                    torch.as_tensor(sec_np[sl], device=device, dtype=torch.long),
                    torch.as_tensor(pct_np[sl], device=device, dtype=torch.float32),
                    with_teacher=False,
                )
                preds.append(sc.cpu().numpy())
            vec = np.concatenate(preds)
            vec = (vec - vec.mean()) / max(float(vec.std()), 1e-6)
            oos_frames.append(pd.DataFrame({
                "identity": [r[4] for r in recs],
                "p_g4_base_ft": vec.astype(np.float32),
            }))

    out_path = "/tmp/g4_gpu1_ft.pt"
    torch.save({
        "best_composite": best_comp,
        "history": history,
        "oos_df": pd.concat(oos_frames, ignore_index=True),
    }, out_path)
    phase("g4_ft_done", best_composite=best_comp)
    return out_path


def main() -> None:
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = _detect_gpu_count()
    phase("g4_started", gpus=n_gpus)

    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")

    hf_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="/kaggle/working/hf_kronos_base")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")

    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G2, TRAIN_FIT_END, min_cross_section=200)
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)
    val_sel_batches = [b for b in val_batches if VAL_SEL_START <= b[0] <= VAL_SEL_END]
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)
    phase("g4_data_ready", tr_dates=len(tr_batches), val_dates=len(val_batches), elapsed=round(time.time() - t0, 1))

    ctx = mp.get_context("fork")
    p0 = ctx.Process(target=gpu0_base_ridge_worker, args=((hf_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),))
    p1 = ctx.Process(target=gpu1_base_rankft_worker, args=((hf_dir, tok_dir, sym_tr, tr_batches, sym_val, val_sel_batches, sym_oos, oos_batches),))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"Worker failure: p0={p0.exitcode}, p1={p1.exitcode}")

    r0 = torch.load("/tmp/g4_gpu0_ridge.pt", map_location="cpu", weights_only=False)
    r1 = torch.load("/tmp/g4_gpu1_ft.pt", map_location="cpu", weights_only=False)
    merged = r0["oos_df"].merge(r1["oos_df"], on="identity", how="inner", validate="one_to_one")
    if len(merged) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(merged)}")
    merged.to_csv(OUTPUT / "oos_g4_scores.csv.gz", index=False, compression="gzip")
    np.savez(OUTPUT / "g4_base_heads.npz", **r0["heads"])
    summary = {
        "ridge_selection": r0["selection"],
        "ft_best_composite": r1["best_composite"],
        "ft_history": r1["history"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT / "g4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("g4_finished", elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
