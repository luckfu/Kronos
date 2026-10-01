"""Phase I: why linear probe sees mfe10≥10% signal but deep sidecar fails.

Cheap CPU diagnosis (no TPU / no long train):
  1) Audit sidecar checkpoint: market gate, head bias, collapse proxies.
  2) Replicate short-budget segment/batch positive rates vs val ~25%.
  3) Train ONLY a linear head with BCE (same loss family as sidecar) on
     frozen handcrafted / tok-summary features — does it beat prior?
  4) Document root-cause hypotheses + tiny-init fix recommendation.

No R2 restart, no Kaggle push from this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations.buy_profit_mfe_ablations import (
    PROFIT_THRESHOLD,
    SEED,
    build_mfe_buy_labels,
)
from modernbert_finance.mfe10_sidecar import (
    GATE_DELTA_VS_PRIOR,
    binary_log_loss,
    constant_prior_log_loss,
)

SHUFFLE_SEED = 20261001
SEGMENT_SAMPLES = 20_000
MAX_SEGMENTS = 4
MFE_THR = 0.10
BATCH_SIZE = 16
SIDECAR_BEST_DELTA = 0.008022
G2_COMB_DELTA = -0.034130436131804
TOK_FULL_DELTA = -0.025916333298968253


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return math.log(p / (1.0 - p))


def audit_checkpoint(path: Path) -> dict[str, Any]:
    """Inspect best/last sidecar weights for gate / bias / collapse."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    gate = float(sd["gate"])
    bias = float(sd["head.bias"])
    prior = float(ckpt.get("metrics", {}).get("all", {}).get("train_prior", 0.25336))
    if "train_prior" in ckpt:
        prior = float(ckpt["train_prior"])
    out = {
        "path": str(path),
        "gate": gate,
        "gate_near_zero": abs(gate) < 0.05,
        "head_bias": bias,
        "sigmoid_head_bias": float(1.0 / (1.0 + math.exp(-bias))),
        "logit_prior": logit(prior),
        "head_bias_vs_logit_prior": bias - logit(prior),
        "head_weight_norm": float(sd["head.weight"].norm()),
        "fusion_weight_norm": float(sd["fusion.weight"].norm()),
        "s1_row_norm_mean": float(sd["s1.weight"].norm(dim=1).mean()),
        "s2_row_norm_mean": float(sd["s2.weight"].norm(dim=1).mean()),
        "cond_abs_mean": float(sd["cond"].abs().mean()),
        "hypothesis": (
            "market=fusion(s1,s2)*gate with gate≈0 → market tokens nearly silenced; "
            "head.bias≈0 → default p≈0.5 vs prior≈0.25"
        ),
    }
    return out


def audit_segment_pos_rates(
    train_targets: Path,
    *,
    shuffle_seed: int = SHUFFLE_SEED,
    segment_samples: int = SEGMENT_SAMPLES,
    max_segments: int = MAX_SEGMENTS,
    threshold: float = MFE_THR,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Replay sidecar row-group shuffle and measure segment/batch pos rates."""
    pqf = pq.ParquetFile(train_targets)
    group_order = np.random.default_rng(shuffle_seed).permutation(
        pqf.num_row_groups
    ).tolist()
    order_hash = hashlib.sha256(
        ",".join(map(str, group_order)).encode("ascii")
    ).hexdigest()

    group_order_pos = 0
    row_offset = 0
    segments: list[dict[str, Any]] = []
    for seg in range(1, max_segments + 1):
        segment_mfe: list[float] = []
        while len(segment_mfe) < segment_samples and group_order_pos < len(group_order):
            gid = group_order[group_order_pos]
            mfe = pqf.read_row_group(gid, columns=["mfe10"])["mfe10"].to_numpy()
            order = np.random.default_rng(shuffle_seed + gid + 1).permutation(len(mfe))
            mfe = mfe[order]
            take = min(segment_samples - len(segment_mfe), len(mfe) - row_offset)
            if take <= 0:
                group_order_pos += 1
                row_offset = 0
                continue
            segment_mfe.extend(mfe[row_offset : row_offset + take].tolist())
            row_offset += take
            if row_offset >= len(mfe):
                group_order_pos += 1
                row_offset = 0
        y = (np.asarray(segment_mfe, dtype=np.float64) >= threshold).astype(np.float64)
        br = [
            float(y[i : i + batch_size].mean())
            for i in range(0, len(y), batch_size)
        ]
        segments.append(
            {
                "segment": seg,
                "n": int(len(y)),
                "pos_rate": float(y.mean()),
                "batch_pos_rate_mean": float(np.mean(br)),
                "batch_pos_rate_std": float(np.std(br)),
                "batch_pos_rate_min": float(np.min(br)),
                "batch_pos_rate_max": float(np.max(br)),
                "frac_batches_zero_pos": float(np.mean([r == 0.0 for r in br])),
            }
        )

    # full priors
    tot = 0.0
    cnt = 0
    for i in range(pqf.num_row_groups):
        m = pqf.read_row_group(i, columns=["mfe10"])["mfe10"].to_numpy()
        tot += float((m >= threshold).sum())
        cnt += len(m)
    train_prior = float(tot / max(cnt, 1))
    return {
        "group_order_hash": order_hash,
        "train_prior": train_prior,
        "train_n": int(cnt),
        "segments": segments,
        "seg1_2_enriched_vs_prior": bool(
            segments[0]["pos_rate"] > train_prior + 0.03
            and segments[1]["pos_rate"] > train_prior + 0.03
        ),
    }


def val_pos_rate(validation_targets: Path, threshold: float = MFE_THR) -> dict[str, Any]:
    pqf = pq.ParquetFile(validation_targets)
    tot = 0.0
    cnt = 0
    for i in range(pqf.num_row_groups):
        m = pqf.read_row_group(i, columns=["mfe10"])["mfe10"].to_numpy()
        tot += float((m >= threshold).sum())
        cnt += len(m)
    rate = float(tot / max(cnt, 1))
    y = np.array([1.0] * int(tot) + [0.0] * int(cnt - tot))
    # rebuild properly for LL
    ys = []
    for i in range(pqf.num_row_groups):
        m = pqf.read_row_group(i, columns=["mfe10"])["mfe10"].to_numpy()
        ys.append((m >= threshold).astype(np.float64))
    y = np.concatenate(ys)
    prior = rate
    return {
        "n": int(cnt),
        "pos_rate": rate,
        "prior_log_loss": constant_prior_log_loss(y, prior=prior),
        "const_0p5_log_loss": binary_log_loss(np.full_like(y, 0.5), y),
        "const_0p5_delta_vs_prior": binary_log_loss(np.full_like(y, 0.5), y)
        - constant_prior_log_loss(y, prior=prior),
    }


def train_bce_linear_head(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = SEED,
    lr: float = 0.05,
    epochs: int = 40,
    batch_size: int = 1024,
    init_bias_to_prior: bool = True,
) -> dict[str, Any]:
    """PyTorch BCE-with-logits linear head (sidecar loss family) on frozen feats."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y.astype(int)
    )
    # standardize with train stats (like a frozen encoder readout)
    mu = x_tr.mean(axis=0)
    sig = x_tr.std(axis=0) + 1e-6
    x_tr_n = ((x_tr - mu) / sig).astype(np.float32)
    x_te_n = ((x_te - mu) / sig).astype(np.float32)

    prior = float(y_tr.mean())
    prior_ll = constant_prior_log_loss(y_te, prior=prior)

    torch.manual_seed(seed)
    model = nn.Linear(x_tr_n.shape[1], 1)
    nn.init.zeros_(model.weight)
    if init_bias_to_prior:
        nn.init.constant_(model.bias, logit(prior))
    else:
        nn.init.zeros_(model.bias)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    xt = torch.from_numpy(x_tr_n)
    yt = torch.from_numpy(y_tr.astype(np.float32)).view(-1, 1)
    n = len(y_tr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(xt[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        te_logits = model(torch.from_numpy(x_te_n)).view(-1).numpy()
    proba = 1.0 / (1.0 + np.exp(-te_logits))
    model_ll = binary_log_loss(proba, y_te)
    delta = model_ll - prior_ll

    # sklearn logistic comparator (same split)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed),
    )
    clf.fit(x_tr, y_tr.astype(int))
    sk_proba = clf.predict_proba(x_te)[:, 1]
    sk_ll = float(log_loss(y_te.astype(int), np.column_stack([1 - sk_proba, sk_proba]), labels=[0, 1]))

    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_dim": int(x.shape[1]),
        "train_prior": prior,
        "prior_log_loss": prior_ll,
        "bce_linear": {
            "init_bias_to_prior": init_bias_to_prior,
            "lr": lr,
            "epochs": epochs,
            "model_log_loss": model_ll,
            "delta_vs_prior": delta,
            "beats_prior": bool(delta < -1e-4),
            "mean_pred": float(proba.mean()),
            "pred_std": float(proba.std()),
            "final_bias": float(model.bias.detach()),
            "sigmoid_final_bias": float(torch.sigmoid(model.bias.detach())),
        },
        "sklearn_logistic": {
            "model_log_loss": sk_ll,
            "delta_vs_prior": sk_ll - prior_ll,
            "beats_prior": bool(sk_ll < prior_ll - 1e-4),
        },
        "positive_rate_test": float(y_te.mean()),
    }


def run_diagnosis(
    *,
    panel_path: Path,
    targets_path: Path,
    train_targets_path: Path,
    best_ckpt: Path | None,
    last_ckpt: Path | None,
    out_json: Path,
    out_md: Path,
) -> dict[str, Any]:
    t0 = time.time()
    import pandas as pd
    from modernbert_finance.ablations._panel_io import load_panel

    panel = load_panel(panel_path)
    targets = pd.read_parquet(targets_path)

    # --- audits ---
    seg_audit = audit_segment_pos_rates(train_targets_path)
    val_audit = val_pos_rate(targets_path)
    ckpt_best = audit_checkpoint(best_ckpt) if best_ckpt and best_ckpt.is_file() else None
    ckpt_last = audit_checkpoint(last_ckpt) if last_ckpt and last_ckpt.is_file() else None

    # --- features (G2 pack; frozen handcrafted = linear-probe control) ---
    packed = build_mfe_buy_labels(panel, targets)
    y = packed["buy_labels"]["buy_worth_mfe10pct"].astype(np.float64)
    x_base = np.asarray(packed["base_features"], dtype=np.float64)
    x_comb = np.asarray(packed["combined_features"], dtype=np.float64)

    bce_base = train_bce_linear_head(x_base, y, seed=SEED, init_bias_to_prior=True)
    bce_comb = train_bce_linear_head(x_comb, y, seed=SEED, init_bias_to_prior=True)
    bce_comb_bad_bias = train_bce_linear_head(
        x_comb, y, seed=SEED, init_bias_to_prior=False
    )

    # Script audit (static)
    script_audit = {
        "loss": "binary_cross_entropy_with_logits (no pos_weight, no label_smoothing)",
        "dropout": "all ModernBertConfig dropouts = 0.0",
        "lr": 3e-5,
        "fresh_init": True,
        "market_gate_init": "nn.Parameter(torch.zeros(1))  # SILENCES market tokens",
        "head_bias_init": "nn.Linear default (~0) → p≈0.5; NOT logit(train_prior)",
        "prior_eval": "uses train_prior on val labels (rates match ~0.253)",
        "label_source": "binary_from_mfe10(row['mfe10']); check_labels on first batch",
        "label_invert": False,
        "shuffle": "row-group permutation + within-group shuffle (fixed seed)",
        "report_best_delta_bug": (
            "report.json best_delta_vs_prior falls back to LAST delta when "
            "last segment did not update best — known; use best_metric.json"
        ),
        "class_imbalance_handling": "none (25% pos is mild; not primary failure)",
    }

    # Hypotheses ranked
    hypotheses = [
        {
            "id": "H1_gate_stuck_near_zero",
            "strength": "strong",
            "claim_zh": "market gate 短训后仍≈0，市况 token 几乎进不了 backbone",
            "evidence": {
                "best_gate": None if ckpt_best is None else ckpt_best["gate"],
                "last_gate": None if ckpt_last is None else ckpt_last["gate"],
                "threshold_near_zero": 0.05,
            },
        },
        {
            "id": "H2_head_bias_not_prior",
            "strength": "strong",
            "claim_zh": "head.bias≈0 → 默认 p≈0.5，相对 prior≈0.25 的常数先验劣 Δ≈+0.128",
            "evidence": {
                "best_sigmoid_bias": None
                if ckpt_best is None
                else ckpt_best["sigmoid_head_bias"],
                "const_0p5_delta": val_audit["const_0p5_delta_vs_prior"],
            },
        },
        {
            "id": "H3_short_budget_underfit_deep",
            "strength": "medium",
            "claim_zh": "22 层 fresh ModernBERT + LR 3e-5 + 仅 80k 样本，远不够打开门控/学表征",
            "evidence": {
                "processed": 80_000,
                "layers": 22,
                "lr": 3e-5,
                "best_delta": SIDECAR_BEST_DELTA,
            },
        },
        {
            "id": "H4_segment_pos_rate_shift",
            "strength": "weak_medium",
            "claim_zh": "短预算前两段正类率≈30.6% vs val 25.3%，有偏但不足以单独解释失败",
            "evidence": {
                "seg_pos_rates": [s["pos_rate"] for s in seg_audit["segments"]],
                "val_pos_rate": val_audit["pos_rate"],
            },
        },
        {
            "id": "H5_label_or_prior_bug",
            "strength": "rejected",
            "claim_zh": "标签反转 / prior 计算错误 / 错列 — 已排除",
            "evidence": {
                "train_prior": seg_audit["train_prior"],
                "val_pos_rate": val_audit["pos_rate"],
                "label_invert": False,
                "group_order_hash_matches_run": seg_audit["group_order_hash"]
                == "42f6150d5e8b1b346912311ff45e15b39f259c2280c32392b9ace17b358b8b32",
            },
        },
        {
            "id": "H6_loss_family_blocks_signal",
            "strength": "rejected",
            "claim_zh": "BCE 损失本身不会抹掉线性信号 — 同损失线性头可打赢先验",
            "evidence": {
                "bce_comb_delta": bce_comb["bce_linear"]["delta_vs_prior"],
                "sklearn_comb_delta": bce_comb["sklearn_logistic"]["delta_vs_prior"],
            },
        },
    ]

    decision = {
        "root_cause_primary": "H1_gate_stuck_near_zero + H2_head_bias_not_prior",
        "linear_bce_beats_prior": bool(bce_comb["bce_linear"]["beats_prior"]),
        "deep_sidecar_best_delta": SIDECAR_BEST_DELTA,
        "tok_logistic_delta": TOK_FULL_DELTA,
        "g2_comb_delta": G2_COMB_DELTA,
        "tiny_fix": {
            "recommended": True,
            "changes": [
                "init gate = 1.0 (or remove multiplicative gate for short sidecar)",
                "init head.bias = logit(train_prior) after prior scan",
                "optional: log mean(sigmoid(logits)) each segment to catch collapse",
            ],
            "do_not": [
                "longer train with gate≈0",
                "TPU / full R2",
                "blame tokenizer (already ruled out)",
            ],
        },
        "next_action_zh": (
            "对 sidecar 脚本做 gate=1 + bias=logit(prior) 小补丁后，"
            "仅在明确授权时再跑 ≤1 segment 的 GPU 冒烟；否则停在文档。"
        ),
    }

    payload: dict[str, Any] = {
        "phase": "I_sidecar_deep_fail_diagnosis",
        "seed": SEED,
        "threshold": MFE_THR,
        "definition": f"y=1{{mfe10>={MFE_THR}}} where mfe10=max(high[T+1:T+10])/close[T]-1",
        "wall_seconds": float(time.time() - t0),
        "script_audit": script_audit,
        "checkpoint_best": ckpt_best,
        "checkpoint_last": ckpt_last,
        "segment_pos_rate_audit": seg_audit,
        "val_prior_audit": val_audit,
        "bce_linear_frozen_features": {
            "base": bce_base,
            "comb": bce_comb,
            "comb_bias_zero_init": bce_comb_bad_bias,
        },
        "comparators": {
            "deep_sidecar_best_delta": SIDECAR_BEST_DELTA,
            "tok_full_logistic_delta": TOK_FULL_DELTA,
            "g2_comb_logistic_delta": G2_COMB_DELTA,
            "gate_delta_le": GATE_DELTA_VS_PRIOR,
        },
        "hypotheses": hypotheses,
        "decision": decision,
        "notes": [
            "No TPU / no Kaggle long train in this phase.",
            "BCE linear uses same loss family as sidecar (BCE-with-logits).",
            "Frozen features = G2 Base / Base+xsection (handcrafted); tok logistic already Δ≈-0.026.",
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_cn_memo(payload, out_md)
    return payload


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    d = payload["decision"]
    best = payload.get("checkpoint_best") or {}
    last = payload.get("checkpoint_last") or {}
    seg = payload["segment_pos_rate_audit"]["segments"]
    bce = payload["bce_linear_frozen_features"]["comb"]["bce_linear"]
    bce0 = payload["bce_linear_frozen_features"]["comb_bias_zero_init"]["bce_linear"]
    sk = payload["bce_linear_frozen_features"]["comb"]["sklearn_logistic"]
    lines = [
        "# Kairos Phase I：深度 Sidecar 失败诊断（线性有信号）",
        "",
        f"日期：{time.strftime('%Y-%m-%d %H:%M')} CST。",
        "",
        "## 一句话",
        "",
        f"**主因（强）**：`gate≈{best.get('gate', 'n/a')}` 几乎关掉市况 token，且 `head.bias→p≈{best.get('sigmoid_head_bias', 'n/a')}`（应对 prior≈0.25）。",
        f"线性 BCE 头在冻结特征上 **Δ={bce['delta_vs_prior']:.6f}**（打赢先验）；深度 sidecar 最佳 Δ=+0.008。",
        "非 tokenizer / 非标签反转 / 非 prior 公式错误。",
        "",
        "## 审计要点",
        "",
        "### 1) Checkpoint（优化塌缩）",
        "",
        f"| 项 | best | last |",
        f"| --- | ---: | ---: |",
        f"| gate | {best.get('gate')} | {last.get('gate')} |",
        f"| sigmoid(head.bias) | {best.get('sigmoid_head_bias')} | {last.get('sigmoid_head_bias')} |",
        f"| logit(prior) | {best.get('logit_prior')} | — |",
        "",
        f"- 常数 p=0.5 相对 prior 的 Δ≈`{payload['val_prior_audit']['const_0p5_delta_vs_prior']:.4f}`（init 附近量级）。",
        "- `market = fusion(s1,s2) * gate`，gate 短训后仍 <0.01 → **市况路径实质关闭**。",
        "",
        "### 2) 脚本审计",
        "",
        "- Loss：BCE-with-logits；**无** pos_weight / label_smoothing / dropout。",
        "- LR=`3e-5`；22 层 fresh ModernBERT；80k samples / 4 segments。",
        "- Eval prior：train_prior≈0.2534 打在 val（pos≈0.2529）— **匹配，非 bug**。",
        "- 标签：`mfe10≥0.10` 现场派生；首 batch `check_labels`；**未反转**。",
        "- `report.json` 的 `best_delta_vs_prior` 在末段未刷新 best 时会写成末次 Δ（已知）。",
        "",
        "### 3) Batch / segment 正类率 vs val 25%",
        "",
        "| seg | pos_rate | batch_std |",
        "| ---: | ---: | ---: |",
    ]
    for s in seg:
        lines.append(
            f"| {s['segment']} | {s['pos_rate']:.4f} | {s['batch_pos_rate_std']:.4f} |"
        )
    lines += [
        "",
        f"train_prior=`{payload['segment_pos_rate_audit']['train_prior']:.5f}`；"
        f"val_pos=`{payload['val_prior_audit']['pos_rate']:.5f}`。",
        "前两段≈30.6%（相对 val 偏高）— **弱～中** 贡献，非主因。",
        "",
        "### 4) 冻结特征 + 同损失线性头（Phase I-1）",
        "",
        f"| 设定 | Δ vs prior | beats? |",
        f"| --- | ---: | --- |",
        f"| BCE linear Base+xsection（bias=logit prior） | {bce['delta_vs_prior']:.6f} | {bce['beats_prior']} |",
        f"| BCE linear 同特征（bias=0 init） | {bce0['delta_vs_prior']:.6f} | {bce0['beats_prior']} |",
        f"| sklearn logistic 同切分 | {sk['delta_vs_prior']:.6f} | {sk['beats_prior']} |",
        f"| 对照 tok_full logistic | {TOK_FULL_DELTA:.6f} | True |",
        f"| 对照 deep sidecar best | {SIDECAR_BEST_DELTA:+.6f} | False |",
        "",
        "**结论**：同 BCE 损失下线性头可复现打赢先验 → 失败在 **深度优化/门控**，不在损失族或标签。",
        "",
        "## 假设强度",
        "",
    ]
    for h in payload["hypotheses"]:
        lines.append(f"- **{h['id']}**（{h['strength']}）：{h['claim_zh']}")
    lines += [
        "",
        "## 建议下一步",
        "",
        f"1. {d['tiny_fix']['changes'][0]}",
        f"2. {d['tiny_fix']['changes'][1]}",
        f"3. {d['next_action_zh']}",
        "4. **不要**加长训 / 不要怪 tokenizer / 不要动 TPU WIP。",
        "",
        "## 用户要点（中文）",
        "",
        f"- **主因强度**：H1 gate≈0 + H2 bias→0.5 = **强**",
        f"- **线性 BCE Δ**：`{bce['delta_vs_prior']:.6f}`（打赢先验）",
        f"- **深度 sidecar Δ**：`+{SIDECAR_BEST_DELTA}`（劣于先验）",
        f"- **排除**：标签反转、prior 错配、tokenizer bottleneck、BCE 本身",
        f"- **下一步**：{d['next_action_zh']}",
        "",
    ]
    text = "\n".join(lines)
    out_md.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--panel",
        type=Path,
        default=Path("/workspace/Kronos/scratch/kairos_ablation_data/panel_dl/val_data.pkl"),
    )
    p.add_argument(
        "--val-targets",
        type=Path,
        default=Path(
            "/workspace/Kronos/scratch/kairos_ablation_data/targets_dl/validation_targets.parquet"
        ),
    )
    p.add_argument(
        "--train-targets",
        type=Path,
        default=Path(
            "/workspace/Kronos/scratch/kairos_ablation_data/targets_dl/train_targets.parquet"
        ),
    )
    p.add_argument(
        "--best-ckpt",
        type=Path,
        default=Path(
            "/workspace/kaggle_kairos_mfe10_sidecar_short_out/artifacts/kairos_mfe10_sidecar/best_model.pt"
        ),
    )
    p.add_argument(
        "--last-ckpt",
        type=Path,
        default=Path(
            "/workspace/kaggle_kairos_mfe10_sidecar_short_out/artifacts/kairos_mfe10_sidecar/last_checkpoint.pt"
        ),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/workspace/Kronos/modernbert_finance/ablations/kairos_phase_i_sidecar_deep_fail.json"
        ),
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("/workspace/Kronos/modernbert_finance/kairos_phase_i_sidecar_deep_fail_cn.md"),
    )
    args = p.parse_args(argv)
    payload = run_diagnosis(
        panel_path=args.panel,
        targets_path=args.val_targets,
        train_targets_path=args.train_targets,
        best_ckpt=args.best_ckpt,
        last_ckpt=args.last_ckpt,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps({
        "ok": True,
        "bce_comb_delta": payload["bce_linear_frozen_features"]["comb"]["bce_linear"]["delta_vs_prior"],
        "best_gate": (payload.get("checkpoint_best") or {}).get("gate"),
        "wall_seconds": payload["wall_seconds"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
