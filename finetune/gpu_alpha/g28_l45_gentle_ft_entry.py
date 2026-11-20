"""G28: Parallel 2x T4 Ultra-Gentle L=45 Top-Block Cosine-IC Fine-Tuning + Post-FT Hidden Ridge.

Arm 0 (GPU 0): NeoQuasar/Kronos-small fine-tuned specifically at L=45 (100% y_multi, strong KL+L2-SP anchor).
Arm 1 (GPU 1): C2 (Kronos-small-0.1-Cosine-C2-Best) fine-tuned specifically at L=45.
After FT, each arm scores both direct E2E rank head AND closed-form Ridge on fine-tuned L=45 [L3,L5,L6_pre,L7,L7_m5].
Strictly leak-free: train <= 2026-04-30, val selection 2026-05-18..2026-07-02, OOS 2026-07-17..2026-10-23.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
from g2_kaggle_entry import evaluate_adapter_on_batches
from g2_rank_finetune import TopBlockRankAdapter, build_g2_date_batches, cosine_ic_loss_vec, kl_logits_anchor
from g16_pretrained_small_entry import _load_pretrained_small
from short_probe_common import detect_gpu_count, find_c2_weights, norm_window

OUTPUT = Path(os.environ.get("G28_OUTPUT", "/kaggle/working/g28_out"))
INPUT = Path(os.environ.get("G28_INPUT", "/kaggle/input"))
LB = 45
D_SMALL = 512
FEAT_DIM = D_SMALL * 5  # [L3, L5, L6_pre, L7_last, L7_m5] = 2560
SL_L7 = slice(D_SMALL * 3, D_SMALL * 4)
SL_ML = slice(0, D_SMALL * 4)
SL_ALL = slice(0, FEAT_DIM)

TRAIN_START_G28 = "2021-01-01"
FT_STRIDE = 2
RIDGE_STRIDE = 3
BATCH_CAP = 384
EPOCHS = 4
ALPHAS_FT_RIDGE = (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8)


@torch.no_grad()
def warm_start_l45(adapter: TopBlockRankAdapter, tokenizer, sym_tr, tr_batches, device) -> None:
    """Warm-start rank_head at L=45 using 60 evenly spaced train dates."""
    adapter.eval()
    xs, ys = [], []
    step = max(1, len(tr_batches) // 60)
    rng = np.random.default_rng(20261119)
    for b in tr_batches[::step][:60]:
        _, s_idx, t_idx, sec_np, pct_np, y_multi, _, _ = b
        n = len(s_idx)
        take = rng.choice(n, size=min(128, n), replace=False)
        raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - LB + 1 : t_idx[i] + 1] for i in take])
        st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - LB + 1 : t_idx[i] + 1] for i in take])
        h_low = adapter.encode_frozen_lower(
            tokenizer,
            torch.as_tensor(raw_b, device=device, dtype=torch.float32),
            torch.as_tensor(st_b, device=device, dtype=torch.float32),
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = h_low
            sec_t = torch.as_tensor(sec_np[take], device=device, dtype=torch.long)
            pct_t = torch.as_tensor(pct_np[take], device=device, dtype=torch.float32)
            for idx in range(adapter.freeze_until, len(adapter.model.transformer)):
                if adapter.model.context_layer > 0 and idx == adapter.model.context_layer:
                    h = adapter.model._add_context(h, sector_id=sec_t, size_percentile=pct_t)
                h = adapter.model.transformer[idx](h, key_padding_mask=None)
            h_out = adapter.model.norm(h).float()
        feat = torch.cat([h_out[:, -1, :], h_out[:, -5:, :].mean(dim=1)], dim=-1)
        feat = adapter.rank_norm(feat).cpu().numpy().astype(np.float64)
        xs.append(feat)
        ys.append(y_multi[take].astype(np.float64))
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    w = np.linalg.solve(X.T @ X + 1e5 * np.eye(X.shape[1]), X.T @ Y)
    pred = X @ w
    scale = 1.0 / max(float(pred.std()), 1e-6)
    adapter.rank_head.weight.copy_(torch.as_tensor(w * scale, dtype=torch.float32, device=device).unsqueeze(0))
    adapter.rank_head.bias.zero_()


def _apply_top_state(adapter: TopBlockRankAdapter, top_sd: dict[str, torch.Tensor]) -> None:
    model_sd = adapter.model.state_dict()
    rn_sd = adapter.rank_norm.state_dict()
    rh_sd = adapter.rank_head.state_dict()
    for k, v in top_sd.items():
        if k.startswith("model."):
            mk = k[6:]
            model_sd[mk] = v.to(model_sd[mk].device, dtype=model_sd[mk].dtype)
        elif k.startswith("rank_norm."):
            rk = k[10:]
            rn_sd[rk] = v.to(rn_sd[rk].device, dtype=rn_sd[rk].dtype)
        elif k.startswith("rank_head."):
            rk = k[10:]
            rh_sd[rk] = v.to(rh_sd[rk].device, dtype=rh_sd[rk].dtype)
    adapter.model.load_state_dict(model_sd)
    adapter.rank_norm.load_state_dict(rn_sd)
    adapter.rank_head.load_state_dict(rh_sd)


@torch.inference_mode()
def _extract_ft_hidden(model, tokenizer, raw_b, st_b, sec_b, pct_b, cap, device) -> torch.Tensor:
    raw_t = torch.as_tensor(raw_b, device=device, dtype=torch.float32)
    st_t = torch.as_tensor(st_b, device=device, dtype=torch.float32)
    sec_t = torch.as_tensor(sec_b, device=device, dtype=torch.long)
    pct_t = torch.as_tensor(pct_b, device=device, dtype=torch.float32)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        tok45 = tokenizer.encode(norm_window(raw_t[:, -LB:]), half=True)
        _, ctx = model.decode_s1(tok45[0], tok45[1], st_t[:, -LB:], sector_id=sec_t, size_percentile=pct_t)
    h7_last = ctx[:, -1, :].float()
    h7_m5 = ctx[:, -5:, :].float().mean(dim=1)
    return torch.cat([cap["l3"], cap["l5"], cap["l6"], h7_last, h7_m5], dim=1)


def run_arm_worker(args: tuple) -> str:
    (
        rank, arm_name, is_pretrained_small, weights_dir, tok_dir,
        sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches,
    ) = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if is_pretrained_small:
        model = _load_pretrained_small(weights_dir, device)
    else:
        model = Kronos.from_pretrained(
            weights_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
            use_size_percentile=True, size_mlp_hidden_dim=64,
        ).to(device)
    tokenizer = KronosTokenizer.from_pretrained(tok_dir).to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    val_sel_batches = [b for b in val_batches if VAL_SEL_START <= b[0] <= VAL_SEL_END]
    ft_batches = tr_batches[::FT_STRIDE]

    adapter = TopBlockRankAdapter(model, freeze_until=5).to(device)
    warm_start_l45(adapter, tokenizer, sym_tr, ft_batches, device)
    init_m, _ = evaluate_adapter_on_batches(adapter, tokenizer, sym_val, val_sel_batches, device, (LB,))
    phase("g28_init_val", arm=arm_name, rank=rank, **init_m)

    top_params = [p for _, p in adapter.model.named_parameters() if p.requires_grad]
    head_params = list(adapter.rank_norm.parameters()) + list(adapter.rank_head.parameters())
    opt = torch.optim.AdamW([
        {"params": top_params, "lr": 1.5e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 1.5e-4, "weight_decay": 1e-2},
    ])
    total_steps = EPOCHS * len(ft_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=1.5e-6)

    rng = np.random.default_rng(20261119 + rank)
    best_ic = init_m["val_ic_raw"]
    best_state = adapter.extract_top_state_dict()
    history = [{"step": 0, **init_m}]

    step = 0
    val_every = max(1, len(ft_batches) // 2)
    for ep in range(EPOCHS):
        order = rng.permutation(len(ft_batches))
        loss_acc, n_acc = 0.0, 0
        for bi in order:
            _, s_idx, t_idx, sec_np, pct_np, y_multi, _, _ = ft_batches[bi]
            n = len(s_idx)
            take = rng.choice(n, size=min(BATCH_CAP, n), replace=False) if n > BATCH_CAP else np.arange(n)
            raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - LB + 1 : t_idx[i] + 1] for i in take])
            st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - LB + 1 : t_idx[i] + 1] for i in take])
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
            score, s1_ft, s1_teacher = adapter.forward_top(
                h_low,
                torch.as_tensor(sec_np[take], device=device, dtype=torch.long),
                torch.as_tensor(pct_np[take], device=device, dtype=torch.float32),
                with_teacher=True,
            )
            ym = torch.as_tensor(y_multi[take], device=device, dtype=torch.float32)
            loss = (
                cosine_ic_loss_vec(score, ym)
                + 0.08 * kl_logits_anchor(s1_teacher, s1_ft, temperature=1.5)
                + 2000.0 * adapter.weight_l2_anchor()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(top_params + head_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            loss_acc += float(loss.detach().item())
            n_acc += 1

            if step % val_every == 0 or step == total_steps:
                vm, _ = evaluate_adapter_on_batches(adapter, tokenizer, sym_val, val_sel_batches, device, (LB,))
                rec = {"epoch": round(step / len(ft_batches), 2), "step": step,
                       "train_loss": round(loss_acc / max(n_acc, 1), 4), **vm}
                history.append(rec)
                phase("g28_val_ckpt", arm=arm_name, **rec)
                loss_acc, n_acc = 0.0, 0
                if vm["val_ic_raw"] > best_ic:
                    best_ic = vm["val_ic_raw"]
                    best_state = adapter.extract_top_state_dict()

    # Load best checkpoint for post-FT Ridge + OOS scoring
    _apply_top_state(adapter, best_state)
    adapter.eval()
    model.eval()
    cap: dict[str, torch.Tensor] = {}
    hooks = [
        model.transformer[3].register_forward_hook(lambda _m, _i, o: cap.__setitem__("l3", o[:, -1, :].detach().float())),
        model.transformer[5].register_forward_hook(lambda _m, _i, o: cap.__setitem__("l5", o[:, -1, :].detach().float())),
        model.transformer[6].register_forward_hook(lambda _m, _i, o: cap.__setitem__("l6", o[:, -1, :].detach().float())),
    ]

    try:
        def _fwd_date(sym_arr, s_idx, t_idx, sec_np, pct_np) -> torch.Tensor:
            chunks = []
            for off in range(0, len(s_idx), 512):
                sl = slice(off, min(off + 512, len(s_idx)))
                raw_b = np.stack([sym_arr[s_idx[i]][0][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_arr[s_idx[i]][1][t_idx[i] - 119 : t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                chunks.append(_extract_ft_hidden(model, tokenizer, raw_b, st_b, sec_np[sl], pct_np[sl], cap, device))
            return zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)

        gram_tr = torch.zeros((FEAT_DIM, FEAT_DIM), dtype=torch.float64, device=device)
        b_raw = torch.zeros(FEAT_DIM, dtype=torch.float64, device=device)
        b_multi = torch.zeros(FEAT_DIM, dtype=torch.float64, device=device)
        for b in tr_batches[::RIDGE_STRIDE]:
            _, s_idx, t_idx, sec_np, pct_np, y_multi, _, r10 = b
            x64 = _fwd_date(sym_tr, s_idx, t_idx, sec_np, pct_np).to(torch.float64)
            yr = torch.as_tensor(standardize_1d(rank_pct_centered(r10)), device=device, dtype=torch.float64)
            ym = torch.as_tensor(y_multi, device=device, dtype=torch.float64)
            gram_tr.addmm_(x64.T, x64)
            b_raw.addmv_(x64.T, yr)
            b_multi.addmv_(x64.T, ym)

        g_tr_np = gram_tr.cpu().numpy()
        br_tr_np = b_raw.cpu().numpy()
        bm_tr_np = b_multi.cpu().numpy()
        g_all_np = g_tr_np.copy()
        br_all_np = br_tr_np.copy()
        bm_all_np = bm_tr_np.copy()
        sel_x, sel_d, sel_r = [], [], []
        for b in val_batches:
            d_str, s_idx, t_idx, sec_np, pct_np, y_multi, _, r10 = b
            x32 = _fwd_date(sym_val, s_idx, t_idx, sec_np, pct_np)
            x64 = x32.cpu().numpy().astype(np.float64)
            g_all_np += x64.T @ x64
            br_all_np += x64.T @ standardize_1d(rank_pct_centered(r10))
            bm_all_np += x64.T @ y_multi.astype(np.float64)
            if VAL_SEL_START <= d_str <= VAL_SEL_END:
                sel_x.append(x32.cpu().numpy().astype(np.float32))
                sel_d.append(np.full(len(r10), d_str))
                sel_r.append(r10.astype(np.float64))
        x_sel = np.concatenate(sel_x, axis=0)
        d_sel = np.concatenate(sel_d)
        r_sel = np.concatenate(sel_r)

        ridge_sel = {}
        ridge_heads = {}
        for hname, sl, tkey in (
            (f"{arm_name}_ridge_l7", SL_L7, "raw"),
            (f"{arm_name}_ridge_ml", SL_ML, "multi"),
            (f"{arm_name}_ridge_pool", SL_ALL, "multi"),
        ):
            r_tr = br_tr_np if tkey == "raw" else bm_tr_np
            r_all = br_all_np if tkey == "raw" else bm_all_np
            xs = x_sel[:, sl].astype(np.float64)
            ba, bic = None, -1e9
            for a in ALPHAS_FT_RIDGE:
                w = solve_ridge_subblock(g_tr_np, r_tr, sl, a)
                ic = daily_spearman_ic(d_sel, xs @ w, r_sel)
                if ic > bic:
                    bic, ba = ic, a
            assert ba is not None
            ridge_heads[hname] = (sl, solve_ridge_subblock(g_all_np, r_all, sl, ba).astype(np.float32))
            ridge_sel[hname] = {"selected_alpha": ba, "val_ic": bic}

        # Score OOS (both E2E rank head + post-FT Ridge heads)
        rh_t = {k: (sl, torch.as_tensor(w, device=device, dtype=torch.float32)) for k, (sl, w) in ridge_heads.items()}
        frames = []
        with torch.no_grad():
            for d_str, recs in oos_batches:
                n = len(recs)
                e2e_chunks, hid_chunks = [], []
                for off in range(0, n, 384):
                    sl = recs[off : off + 384]
                    raw_b = np.stack([sym_oos[r[0]][0][r[1] - 119 : r[1] + 1] for r in sl])
                    st_b = np.stack([sym_oos[r[0]][1][r[1] - 119 : r[1] + 1] for r in sl])
                    sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
                    pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)

                    h_low = adapter.encode_frozen_lower(
                        tokenizer,
                        torch.as_tensor(raw_b[:, -LB:], device=device, dtype=torch.float32),
                        torch.as_tensor(st_b[:, -LB:], device=device, dtype=torch.float32),
                    )
                    sc, _, _ = adapter.forward_top(
                        h_low,
                        torch.as_tensor(sec_b, device=device, dtype=torch.long),
                        torch.as_tensor(pct_b, device=device, dtype=torch.float32),
                        with_teacher=False,
                    )
                    e2e_chunks.append(sc.cpu().numpy())
                    hid_chunks.append(_extract_ft_hidden(model, tokenizer, raw_b, st_b, sec_b, pct_b, cap, device))

                e2e_v = np.concatenate(e2e_chunks)
                e2e_v = (e2e_v - e2e_v.mean()) / max(float(e2e_v.std()), 1e-6)
                x_d = zscore_cross_section_torch(torch.cat(hid_chunks, dim=0), clip_val=5.0)
                df_d = pd.DataFrame({
                    "identity": [r[4] for r in recs],
                    "symbol": [r[5] for r in recs],
                    "asof_date": d_str,
                    f"p_{arm_name}_e2e": e2e_v.astype(np.float32),
                })
                for hname, (sl_w, wt) in rh_t.items():
                    df_d[f"p_{hname}"] = (x_d[:, sl_w] @ wt).cpu().numpy().astype(np.float32)
                frames.append(df_d)

        out_csv = f"/tmp/g28_oos_{arm_name}.csv.gz"
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
        out_meta = f"/tmp/g28_meta_{arm_name}.pt"
        torch.save({
            "arm": arm_name,
            "best_e2e_val_ic": best_ic,
            "history": history,
            "ridge_selection": ridge_sel,
            "ridge_heads": {k: v[1] for k, v in ridge_heads.items()},
            "top_state": best_state,
            "oos_csv": out_csv,
        }, out_meta)
        return out_meta
    finally:
        for h in hooks:
            h.remove()


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = detect_gpu_count()
    phase("g28_started", gpus=n_gpus)

    c2_w = find_c2_weights(INPUT)
    train_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")
    val_pkl = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")
    meta_csv = find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")
    oos_raw = find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")
    pts_dir = snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="/kaggle/working/hf_kronos_small")
    tok_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv, ev.SECTOR_MAP)

    sym_tr, tr_batches = build_g2_date_batches(train_pkl, lookup, TRAIN_START_G28, TRAIN_FIT_END, min_cross_section=200)
    sym_val, val_batches = build_g2_date_batches(val_pkl, lookup, "2025-07-01", VAL_SEL_END, min_cross_section=100)
    sym_oos, oos_batches = build_oos_batches(oos_raw.parent, ev.SECTOR_MAP)

    arm_jobs = [
        (0, "g28_pts_l45_ft", True, pts_dir, tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),
        (1 % n_gpus, "g28_c2_l45_ft", False, str(c2_w.parent), tok_dir, sym_tr, tr_batches, sym_val, val_batches, sym_oos, oos_batches),
    ]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(2, n_gpus)) as pool:
        meta_paths = pool.map(run_arm_worker, arm_jobs)

    metas = [torch.load(p, map_location="cpu", weights_only=False) for p in meta_paths]
    for p in meta_paths:
        Path(p).unlink(missing_ok=True)

    dfs = [pd.read_csv(m["oos_csv"]).sort_values(["asof_date", "symbol"]).reset_index(drop=True) for m in metas]
    for m in metas:
        Path(m["oos_csv"]).unlink(missing_ok=True)
    merged = dfs[0].merge(dfs[1].drop(columns=["symbol", "asof_date"]), on="identity", how="inner", validate="one_to_one")
    if len(merged) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(merged)}")
    merged.to_csv(OUTPUT / "oos_g28_scores.csv.gz", index=False, compression="gzip")

    all_heads = {}
    summary_arms = {}
    for m in metas:
        arm = m["arm"]
        all_heads.update(m["ridge_heads"])
        torch.save(m["top_state"], OUTPUT / f"{arm}_top.pt")
        summary_arms[arm] = {
            "best_e2e_val_ic": m["best_e2e_val_ic"],
            "history": m["history"],
            "ridge_selection": m["ridge_selection"],
        }
    np.savez(OUTPUT / "g28_heads.npz", **all_heads)
    summary = {"arms": summary_arms, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g28_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for d in (pts_dir, tok_dir):
        shutil.rmtree(d, ignore_errors=True)
    phase("g28_finished", elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
