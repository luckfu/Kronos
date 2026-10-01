"""Phase Q: tabular Base+xsection → shallow MLP BCE (NO tokenizer sequence).

Different angle after Phase P shallow-freeze-embeds failed (best Δ≈0, worse than K).
Still decision-only: y=1{mfe10>=0.10}; gate = Δ logloss vs constant prior ≤ -0.04.
CPU only. Not ranking. Not 22-layer. Not TPU.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

PHASE_K_DELTA = -0.002970473307763233
PHASE_P_DELTA = 1.3803802001444154e-07
G2_COMB_DELTA = -0.034130436131804
LINEAR_BCE_DELTA = -0.031446


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    return math.log(p / (1.0 - p))


def _eval_proba(y_te: np.ndarray, proba: np.ndarray, train_prior: float) -> dict[str, Any]:
    prior_ll = constant_prior_log_loss(y_te, prior=train_prior)
    model_ll = binary_log_loss(proba, y_te)
    delta = float(model_ll - prior_ll)
    return {
        "n_test": int(len(y_te)),
        "train_prior": float(train_prior),
        "prior_log_loss": float(prior_ll),
        "model_log_loss": float(model_ll),
        "delta_vs_prior": delta,
        "gate_delta": float(GATE_DELTA_VS_PRIOR),
        "gate_passed": bool(delta <= GATE_DELTA_VS_PRIOR),
        "beats_prior": bool(delta < -1e-4),
        "mean_pred": float(np.mean(proba)),
        "pred_std": float(np.std(proba)),
        "positive_rate_test": float(np.mean(y_te)),
    }


def fit_logistic(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    prior = float(y_tr.mean())
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed),
    )
    clf.fit(x_tr, y_tr.astype(int))
    proba = clf.predict_proba(x_te)[:, 1]
    out = _eval_proba(y_te, proba, prior)
    out["kind"] = "sklearn_logistic"
    return out


def fit_logistic_calibrated(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Platt-style calibration on logistic (isotonic/sigmoid via CV)."""
    prior = float(y_tr.mean())
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed),
    )
    # Use sigmoid calibration (Platt) with 3-fold on train.
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(x_tr, y_tr.astype(int))
    proba = cal.predict_proba(x_te)[:, 1]
    out = _eval_proba(y_te, proba, prior)
    out["kind"] = "logistic_platt_calibrated"
    return out


def fit_shallow_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hidden: tuple[int, ...] = (64, 32),
    max_iter: int = 80,
    alpha: float = 1e-4,
    lr_init: float = 1e-3,
) -> dict[str, Any]:
    """Shallow MLPClassifier on standardized tabular features (no tokenizer)."""
    prior = float(y_tr.mean())
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=1024,
            learning_rate_init=lr_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        ),
    )
    t0 = time.time()
    clf.fit(x_tr, y_tr.astype(int))
    elapsed = time.time() - t0
    proba = clf.predict_proba(x_te)[:, 1]
    out = _eval_proba(y_te, proba, prior)
    mlp = clf.named_steps["mlpclassifier"]
    out.update(
        {
            "kind": "shallow_mlp",
            "hidden_layer_sizes": list(hidden),
            "n_layers_including_io": int(mlp.n_layers_),
            "n_iter": int(mlp.n_iter_),
            "alpha": float(alpha),
            "learning_rate_init": float(lr_init),
            "fit_seconds": float(elapsed),
            "feature_dim": int(x_tr.shape[1]),
            "logit_prior_ref": _logit(prior),
        }
    )
    return out


def run_phase_q(
    *,
    val_panel: Path,
    val_targets: Path,
    seed: int = SEED,
    threshold: float = PROFIT_THRESHOLD,
) -> dict[str, Any]:
    import pandas as pd

    t_start = time.time()
    targets = pd.read_parquet(val_targets)
    packed = build_mfe_buy_labels(val_panel, targets, threshold=threshold)
    base = np.asarray(packed["base_features"], dtype=np.float64)
    comb = np.asarray(packed["combined_features"], dtype=np.float64)
    y = np.asarray(packed["buy_labels"]["buy_worth_mfe10pct"], dtype=np.float64)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    comb = np.nan_to_num(comb, nan=0.0, posinf=0.0, neginf=0.0)

    # Same 75/25 stratified split as Phase I linear BCE diagnosis.
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(
        idx, test_size=0.25, random_state=seed, stratify=y.astype(int)
    )

    models: dict[str, Any] = {}
    for name, x in (("base", base), ("comb", comb)):
        x_tr, x_te = x[i_tr], x[i_te]
        y_tr, y_te = y[i_tr], y[i_te]
        models[f"logistic_{name}"] = fit_logistic(x_tr, y_tr, x_te, y_te, seed=seed)
        models[f"logistic_calibrated_{name}"] = fit_logistic_calibrated(
            x_tr, y_tr, x_te, y_te, seed=seed
        )
        models[f"mlp_2x_{name}"] = fit_shallow_mlp(
            x_tr, y_tr, x_te, y_te, seed=seed, hidden=(64, 32)
        )
        models[f"mlp_1x_{name}"] = fit_shallow_mlp(
            x_tr, y_tr, x_te, y_te, seed=seed, hidden=(64,), max_iter=60
        )
        # Slightly wider shallow net — still cheap.
        models[f"mlp_3x_{name}"] = fit_shallow_mlp(
            x_tr,
            y_tr,
            x_te,
            y_te,
            seed=seed,
            hidden=(128, 64, 32),
            max_iter=100,
        )

    # Pick best by delta (most negative).
    ranked = sorted(
        ((k, v["delta_vs_prior"]) for k, v in models.items() if "delta_vs_prior" in v),
        key=lambda kv: kv[1],
    )
    best_name, best_delta = ranked[0]
    best = models[best_name]
    gate_passed = bool(best_delta <= GATE_DELTA_VS_PRIOR)

    decision = {
        "best_model": best_name,
        "best_delta_vs_prior": float(best_delta),
        "gate_passed": gate_passed,
        "clears_logistic_ref": bool(best_delta < G2_COMB_DELTA - 1e-4),
        "vs_phase_p": "tabular_path_alive" if best_delta < -0.01 else "still_weak",
        "next": (
            "consider_mild_feature_expand_or_calibrated_deploy"
            if gate_passed
            else (
                "if_near_logistic_try_kronos_score_decision_rule"
                if best_delta > G2_COMB_DELTA - 0.005
                else "tabular_shallow_beats_logistic__scale_or_calibrate"
            )
        ),
        "not_tokenizer_sequence": True,
        "not_ranking": True,
        "not_22_layer": True,
    }

    return {
        "phase": "Q_tabular_shallow_mlp_decision",
        "kind": "cpu_tabular_base_xsection_shallow_mlp_bce",
        "target": f"y=1{{mfe10>={threshold}}}",
        "mfe10_def": packed["definition"]["mfe10_formula"],
        "no_tokenizer_sequence": True,
        "status": "COMPLETE",
        "seed": int(seed),
        "split": {"test_size": 0.25, "stratify": True},
        "n_total": int(len(y)),
        "n_train": int(len(i_tr)),
        "n_test": int(len(i_te)),
        "pos_rate_all": float(y.mean()),
        "feature_dims": {
            "base": int(base.shape[1]),
            "comb": int(comb.shape[1]),
        },
        "models": models,
        "ranked_deltas": [{"model": k, "delta": float(d)} for k, d in ranked],
        "best": {
            "model": best_name,
            **{k: best[k] for k in (
                "delta_vs_prior",
                "gate_passed",
                "model_log_loss",
                "prior_log_loss",
                "train_prior",
                "kind",
            ) if k in best},
        },
        "compare": {
            "phase_k_identity_best_delta": PHASE_K_DELTA,
            "phase_p_shallow_freeze_best_delta": PHASE_P_DELTA,
            "logistic_g2_comb_delta": G2_COMB_DELTA,
            "linear_bce_frozen_features_delta": LINEAR_BCE_DELTA,
            "gate": GATE_DELTA_VS_PRIOR,
            "phase_q_best_delta": float(best_delta),
        },
        "decision": decision,
        "elapsed_seconds": float(time.time() - t_start),
    }


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    best = payload["best"]
    dec = payload["decision"]
    cmp_ = payload["compare"]
    ranked = payload["ranked_deltas"][:8]
    lines = [
        "# Kairos 决策表格浅 MLP（Phase Q）结果",
        "",
        "日期：2026-10-02（北京时间）。",
        "设定：**Base+xsection 表格特征 → 浅 MLP / Logistic / 校准**；**无** tokenizer 序列。",
        f"状态：**{payload['status']}**（本地 CPU）",
        "",
        "## 一句话",
        "",
        (
            f"**{'过闸' if dec['gate_passed'] else '未过闸'}。** "
            f"best=`{dec['best_model']}` Δ=`{dec['best_delta_vs_prior']:.6f}`；"
            f"gate_passed=`{dec['gate_passed']}`。"
            f"对照：Phase P≈0；K≈−0.003；G2 Logistic≈−0.034；闸门≤−0.04。"
        ),
        "",
        "## 目的",
        "",
        "Phase P（冻 embeds + 浅序列头）失败后换角度：不依赖 tokenizer 序列，",
        "直接用已验证有效的 Base+xsection 表格特征训浅 MLP，看能否逼近/超过 Logistic 并冲闸。",
        "",
        "## 数据与切分",
        "",
        f"| 项 | 值 |",
        f"| --- | --- |",
        f"| n_total / train / test | {payload['n_total']} / {payload['n_train']} / {payload['n_test']} |",
        f"| 正类率（全） | {payload['pos_rate_all']:.4f} |",
        f"| base / comb 维 | {payload['feature_dims']['base']} / {payload['feature_dims']['comb']} |",
        f"| 切分 | 75/25 stratify seed={payload['seed']} |",
        f"| 墙钟 | {payload['elapsed_seconds']:.1f}s |",
        "",
        "## 主结果",
        "",
        f"| 模型 | Δ vs prior | gate |",
        f"| --- | ---: | --- |",
    ]
    for row in ranked:
        m = payload["models"][row["model"]]
        lines.append(
            f"| {row['model']} | {row['delta']:.6f} | {m.get('gate_passed', False)} |"
        )
    lines += [
        "",
        "## 对照",
        "",
        f"| 跑次 | best Δ |",
        f"| --- | ---: |",
        f"| Phase P 浅冻 embeds | {cmp_['phase_p_shallow_freeze_best_delta']:.6g} |",
        f"| Phase K identity | {cmp_['phase_k_identity_best_delta']:.6f} |",
        f"| 线性 BCE 冻结特征 | {cmp_['linear_bce_frozen_features_delta']:.6f} |",
        f"| G2 Logistic comb | {cmp_['logistic_g2_comb_delta']:.6f} |",
        f"| **Phase Q best** | **{cmp_['phase_q_best_delta']:.6f}** |",
        f"| 闸门 | ≤ {cmp_['gate']} |",
        "",
        "## 判读",
        "",
        f"- best_model=`{dec['best_model']}`；clears_logistic_ref=`{dec['clears_logistic_ref']}`。",
        f"- next=`{dec['next']}`。",
        "- **不做**：tokenizer 序列短烟；Ranking 产品；22 层；TPU WIP。",
        "",
        "## 产物",
        "",
        "- JSON：`modernbert_finance/ablations/kairos_phase_q_tabular_shallow_mlp_decision_results.json`",
        "- 本备忘：`modernbert_finance/kairos_phase_q_tabular_shallow_mlp_decision_cn.md`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-panel", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "modernbert_finance/ablations/kairos_phase_q_tabular_shallow_mlp_decision_results.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_q_tabular_shallow_mlp_decision_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    payload = run_phase_q(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        seed=args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_cn_memo(payload, args.out_md)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "best": payload["best"],
                "decision": payload["decision"],
                "ranked_top5": payload["ranked_deltas"][:5],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
