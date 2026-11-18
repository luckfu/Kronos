"""Kaggle GPU entry point for Experiment G2: parallel 2x T4 top-block fine-tuning."""
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

from g1_multiscale_extract_and_fit import (
    OOS_END,
    OOS_START,
    TRAIN_FIT_END,
    VAL_SEL_END,
    VAL_SEL_START,
    build_fast_metadata_lookup,
    daily_spearman_ic,
    phase,
)
from g1_kaggle_entry import build_oos_batches, find_dataset_file
from g2_rank_finetune import (
    TRAIN_START_G2,
    TopBlockRankAdapter,
    build_g2_date_batches,
    cosine_ic_loss_vec,
    kl_logits_anchor,
)

OUTPUT = Path(os.environ.get("G2_OUTPUT", "/kaggle/working/g2_out"))
INPUT = Path(os.environ.get("G2_INPUT", "/kaggle/input"))
BATCH_CAP = int(os.environ.get("G2_BATCH_CAP", "256") or "256")
EPOCHS = int(os.environ.get("G2_EPOCHS", "4") or "4")


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


@torch.no_grad()
def warm_start_rank_head(adapter: TopBlockRankAdapter, tokenizer, sym_tr, train_batches, device) -> None:
    """Solve closed-form Ridge on first 60 dates to initialize rank_head without cold-start shock."""
    adapter.eval()
    xs, ys = [], []
    step = max(1, len(train_batches) // 60)
    for b in train_batches[::step][:60]:
        _, s_idx, t_idx, sec_np, pct_np, y_multi, _, _ = b
        n = len(s_idx)
        take = np.random.default_rng(20261115).choice(n, size=min(128, n), replace=False)
        raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - 119:t_idx[i] + 1] for i in take])
        st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - 119:t_idx[i] + 1] for i in take])
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
    w = np.linalg.solve(X.T @ X + 1e4 * np.eye(X.shape[1]), X.T @ Y)
    pred = X @ w
    scale = 1.0 / max(float(pred.std()), 1e-6)
    adapter.rank_head.weight.copy_(torch.as_tensor(w * scale, dtype=torch.float32, device=device).unsqueeze(0))
    adapter.rank_head.bias.zero_()


@torch.no_grad()
def evaluate_adapter_on_batches(
    adapter: TopBlockRankAdapter,
    tokenizer,
    sym_arrays,
    batches,
    device,
    lookbacks: tuple[int, ...],
    eval_batch: int = 384,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Score batches (e.g., val or oos) cleanly in eval mode."""
    adapter.eval()
    all_dates, all_ret10, all_y_orth = [], [], []
    scores_by_lb: dict[int, list[np.ndarray]] = {lb: [] for lb in lookbacks}

    for b in batches:
        d_str, s_idx, t_idx, sec_np, pct_np, _, y_orth_np, r10_np = b
        n = len(s_idx)
        for lb in lookbacks:
            preds = []
            for off in range(0, n, eval_batch):
                sl = slice(off, min(off + eval_batch, n))
                raw_b = np.stack([sym_arrays[s_idx[i]][0][t_idx[i] - lb + 1:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_arrays[s_idx[i]][1][t_idx[i] - lb + 1:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
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
            scores_by_lb[lb].append(vec)
        all_dates.extend([d_str] * n)
        all_ret10.append(r10_np)
        all_y_orth.append(y_orth_np)

    dates_np = np.asarray(all_dates)
    ret10_np = np.concatenate(all_ret10)
    orth_np = np.concatenate(all_y_orth)
    df = pd.DataFrame({"asof_date": dates_np})
    for lb in lookbacks:
        df[f"p_l{lb}"] = np.concatenate(scores_by_lb[lb])
    if len(lookbacks) > 1:
        df["p_ens"] = sum(df[f"p_l{lb}"] for lb in lookbacks) / float(len(lookbacks))
        primary_col = "p_ens"
    else:
        primary_col = f"p_l{lookbacks[0]}"

    ic_raw = daily_spearman_ic(dates_np, df[primary_col].to_numpy(), ret10_np)
    ic_orth = daily_spearman_ic(dates_np, df[primary_col].to_numpy(), orth_np)
    metrics = {
        "val_ic_raw": ic_raw,
        "val_ic_orth": ic_orth,
        "val_composite": 0.5 * ic_raw + 0.5 * ic_orth,
    }
    return metrics, df


def train_arm_worker(args: tuple) -> str:
    rank, arm_name, mode, weights_dir, tokenizer_dir, sym_tr, tr_batches, sym_val, val_sel_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = Kronos.from_pretrained(
        weights_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    adapter = TopBlockRankAdapter(model, freeze_until=5).to(device)
    warm_start_rank_head(adapter, tokenizer, sym_tr, tr_batches, device)

    eval_lbs = (120,) if mode == "l120" else (60, 120)
    init_metrics, _ = evaluate_adapter_on_batches(adapter, tokenizer, sym_val, val_sel_batches, device, eval_lbs)
    phase("g2_init_val", arm=arm_name, rank=rank, **init_metrics)

    top_params = [p for n, p in adapter.model.named_parameters() if p.requires_grad]
    head_params = list(adapter.rank_norm.parameters()) + list(adapter.rank_head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": top_params, "lr": 2e-5, "weight_decay": 1e-2},
            {"params": head_params, "lr": 2e-4, "weight_decay": 1e-2},
        ]
    )
    total_steps = EPOCHS * len(tr_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=2e-6)

    rng = np.random.default_rng(20261115 + rank)
    best_comp = init_metrics["val_composite"]
    best_state = adapter.extract_top_state_dict()
    history = [{"step": 0, **init_metrics}]

    step = 0
    val_every = max(1, len(tr_batches) // 2)
    for ep in range(EPOCHS):
        order = rng.permutation(len(tr_batches))
        loss_acc = 0.0
        n_acc = 0
        for bi in order:
            _, s_idx, t_idx, sec_np, pct_np, y_multi, y_orth, _ = tr_batches[bi]
            n = len(s_idx)
            take = rng.choice(n, size=min(BATCH_CAP, n), replace=False) if n > BATCH_CAP else np.arange(n)
            lb = 120 if mode == "l120" else int(rng.choice([60, 90, 120]))
            raw_b = np.stack([sym_tr[s_idx[i]][0][t_idx[i] - lb + 1:t_idx[i] + 1] for i in take])
            st_b = np.stack([sym_tr[s_idx[i]][1][t_idx[i] - lb + 1:t_idx[i] + 1] for i in take])

            h_low = adapter.encode_frozen_lower(
                tokenizer,
                torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
            )
            adapter.train()
            # Keep frozen modules deterministic
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
            loss_acc += float(loss.detach().item())
            n_acc += 1

            if step % val_every == 0 or step == total_steps:
                vm, _ = evaluate_adapter_on_batches(adapter, tokenizer, sym_val, val_sel_batches, device, eval_lbs)
                rec = {"epoch": round(step / len(tr_batches), 2), "step": step,
                       "train_loss": round(loss_acc / max(n_acc, 1), 4), **vm}
                history.append(rec)
                phase("g2_val_ckpt", arm=arm_name, **rec)
                loss_acc, n_acc = 0.0, 0
                if vm["val_composite"] > best_comp:
                    best_comp = vm["val_composite"]
                    best_state = adapter.extract_top_state_dict()

    out_path = f"/tmp/g2_{arm_name}.pt"
    torch.save({
        "arm": arm_name,
        "mode": mode,
        "best_composite": best_comp,
        "history": history,
        "state": best_state,
    }, out_path)
    return out_path


@torch.no_grad()
def score_oos_arm(args: tuple) -> str:
    rank, ckpt_path, weights_dir, tokenizer_dir, sym_oos, oos_batches = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mode = ckpt["mode"]
    arm_name = ckpt["arm"]
    model = Kronos.from_pretrained(
        weights_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()
    adapter = TopBlockRankAdapter(model, freeze_until=5).to(device).eval()

    # Load saved best weights into adapter
    st = ckpt["state"]
    model_sd = adapter.model.state_dict()
    for k, v in st.items():
        if k.startswith("model."):
            mkey = k[6:]
            model_sd[mkey] = v.to(dtype=model_sd[mkey].dtype, device=device)
    adapter.model.load_state_dict(model_sd, strict=True)
    adapter.rank_norm.load_state_dict({k[10:]: v.to(device) for k, v in st.items() if k.startswith("rank_norm.")})
    adapter.rank_head.load_state_dict({k[10:]: v.to(device) for k, v in st.items() if k.startswith("rank_head.")})
    adapter.eval()

    lbs = (120,) if mode == "l120" else (60, 120)
    frames = []
    for d_str, recs in oos_batches:
        n = len(recs)
        df_d = pd.DataFrame({
            "identity": [r[4] for r in recs],
            "symbol": [r[5] for r in recs],
            "asof_date": d_str,
        })
        s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
        t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
        sec_np = np.asarray([r[2] for r in recs], dtype=np.int64)
        pct_np = np.asarray([r[3] for r in recs], dtype=np.float32)
        for lb in lbs:
            preds = []
            for off in range(0, n, 384):
                sl = slice(off, min(off + 384, n))
                raw_b = np.stack([sym_oos[s_idx[i]][0][t_idx[i] - lb + 1:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
                st_b = np.stack([sym_oos[s_idx[i]][1][t_idx[i] - lb + 1:t_idx[i] + 1] for i in range(sl.start, sl.stop)])
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
            df_d[f"p_{arm_name}_l{lb}"] = vec.astype(np.float32)
        if len(lbs) > 1:
            df_d[f"p_{arm_name}_ens"] = ((df_d[f"p_{arm_name}_l60"] + df_d[f"p_{arm_name}_l120"]) * 0.5).astype(np.float32)
        frames.append(df_d)

    out_csv = f"/tmp/g2_oos_{arm_name}.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
    return out_csv


def main() -> None:
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune" / "oos_eval" / "base_clean_sealed_oos"))
    import evaluate_base_sealed as ev

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_gpus = _detect_gpu_count()
    phase("g2_started", gpus=n_gpus, epochs=EPOCHS, batch_cap=BATCH_CAP)

    weights = [find_dataset_file(INPUT, "kronos-small-cosine-c2-best-weights", "model.safetensors")]
    train_pkl = [find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "train_data.pkl")]
    val_pkl = [find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "val_data.pkl")]
    meta_csv = [find_dataset_file(INPUT, "a-share-120d-temporal-symbol-holdout", "asset_metadata.csv")]
    oos_raw = [find_dataset_file(INPUT, "a-share-120d-oos-20260717-20261023-v2", "combined_raw.csv")]

    tokenizer_dir = snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="/kaggle/working/hf_tokenizer")
    lookup = build_fast_metadata_lookup(meta_csv[0], ev.SECTOR_MAP)
    sym_tr, tr_batches = build_g2_date_batches(train_pkl[0], lookup, TRAIN_START_G2, TRAIN_FIT_END, min_cross_section=200)
    sym_val, val_sel_batches = build_g2_date_batches(val_pkl[0], lookup, VAL_SEL_START, VAL_SEL_END, min_cross_section=100)
    assert tr_batches[-1][0] <= TRAIN_FIT_END
    assert val_sel_batches[-1][0] <= VAL_SEL_END
    phase("g2_data_ready", train_dates=len(tr_batches), val_dates=len(val_sel_batches), elapsed=round(time.time() - t0, 1))

    ctx = mp.get_context("fork")
    world = min(2, n_gpus)
    if world < 2:
        raise RuntimeError("Expected 2 GPUs on Kaggle T4x2 for parallel Arm A/B training")

    train_args = [
        (0, "g2_ft_l120", "l120", str(weights[0].parent), tokenizer_dir, sym_tr, tr_batches, sym_val, val_sel_batches),
        (1, "g2_ft_varlen", "varlen", str(weights[0].parent), tokenizer_dir, sym_tr, tr_batches, sym_val, val_sel_batches),
    ]
    with ctx.Pool(2) as pool:
        ckpt_paths = pool.map(train_arm_worker, train_args)

    del sym_tr, tr_batches, sym_val, val_sel_batches

    # Save lightweight top-block checkpoints (~38 MB combined)
    saved = {}
    histories = {}
    for cp in ckpt_paths:
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        saved[obj["arm"]] = {"mode": obj["mode"], "best_composite": obj["best_composite"], "state": obj["state"]}
        histories[obj["arm"]] = obj["history"]
    torch.save(saved, OUTPUT / "g2_top_weights.pt")

    # Score OOS concurrently (Arm A on GPU 0, Arm B on GPU 1)
    sym_oos, oos_batches = build_oos_batches(oos_raw[0].parent, ev.SECTOR_MAP)
    oos_args = [
        (0, ckpt_paths[0], str(weights[0].parent), tokenizer_dir, sym_oos, oos_batches),
        (1, ckpt_paths[1], str(weights[0].parent), tokenizer_dir, sym_oos, oos_batches),
    ]
    with ctx.Pool(2) as pool:
        oos_csvs = pool.map(score_oos_arm, oos_args)

    df0 = pd.read_csv(oos_csvs[0])
    df1 = pd.read_csv(oos_csvs[1])
    merged_oos = df0.merge(df1.drop(columns=["symbol", "asof_date"]), on="identity", how="inner", validate="one_to_one")
    if len(merged_oos) != 335312:
        raise RuntimeError(f"Expected 335312 OOS rows, got {len(merged_oos)}")
    merged_oos.to_csv(OUTPUT / "oos_g2_scores.csv.gz", index=False, compression="gzip")

    summary = {"histories": histories, "elapsed_sec": round(time.time() - t0, 1)}
    (OUTPUT / "g2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    phase("g2_finished", elapsed_sec=summary["elapsed_sec"])


if __name__ == "__main__":
    main()
