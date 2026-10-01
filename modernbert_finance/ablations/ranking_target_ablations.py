"""Phase L: ranking-target ablation (CPU, val panel).

User-aligned diagnosis: alpha lives in ranking, not binary mfe10≥10%.
Cheap local probe: predict cross-section ranks / pairwise of continuous
fwd_ret_10 or mfe10 with Base+xsection; report Rank IC / topK vs chance;
gate = clear lift vs binary path (Phase G2 / Phase K).

No long 22-layer train. No TPU WIP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations.label_protocol_ablations import (
    build_protocol_labels,
)
from modernbert_finance.ablations.teacher_distill_ablations import (
    _daily_rank_ic,
    _safe_spearman,
    _topk_hit_rate,
)

SEED = 20261001
TOPK_FRAC = 0.20
# Clear ranking signal bars (same spirit as Phase H student bars)
RANK_IC_BAR = 0.05
TOPK_LIFT_BAR = 0.05
# Binary path references (val-panel logistic / Phase K sidecar)
BINARY_COMB_DELTA_G2 = -0.03413  # Logistic Base+xsection mfe≥10%
BINARY_GATE = -0.04
BINARY_PHASE_K_BEST_DELTA = -0.002970473307763233
PAIRWISE_MAX_PAIRS = 80_000


def _cs_rank(dates: pd.Series, y: np.ndarray) -> np.ndarray:
    meta = pd.DataFrame({"asof_date": dates.to_numpy(), "_y": y})
    return meta.groupby("asof_date")["_y"].rank(pct=True).to_numpy(dtype=np.float64)


def _fit_ridge_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(y) < 200 or np.std(y) < 1e-12:
        return {"skipped": True, "reason": "too_few_or_constant"}
    x_tr, x_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        x, y, np.arange(len(y)), test_size=0.25, random_state=seed
    )
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(x_tr, y_tr)
    pred = pipe.predict(x_te)
    return {
        "skipped": False,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pred": pred,
        "y_te": y_te,
        "idx_te": idx_te,
        "mask": mask,
    }


def _pairwise_accuracy(
    dates: pd.Series,
    score: np.ndarray,
    label: np.ndarray,
    *,
    seed: int,
    max_pairs: int = PAIRWISE_MAX_PAIRS,
) -> dict[str, Any]:
    """Within-day: P(score_i > score_j | label_i > label_j), chance=0.5."""
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    for _, g_idx in pd.Series(dates).groupby(dates).groups.items():
        idx = np.asarray(list(g_idx))
        if len(idx) < 30:
            continue
        # sample pairs within day
        n_need = min(200, len(idx) * 2)
        a = rng.integers(0, len(idx), size=n_need)
        b = rng.integers(0, len(idx), size=n_need)
        keep = a != b
        a, b = a[keep], b[keep]
        ia, ib = idx[a], idx[b]
        lab_diff = label[ia] - label[ib]
        usable = np.abs(lab_diff) > 1e-12
        if not np.any(usable):
            continue
        ia, ib = ia[usable], ib[usable]
        lab_diff = lab_diff[usable]
        sc_diff = score[ia] - score[ib]
        correct += int(np.sum((sc_diff > 0) == (lab_diff > 0)))
        total += int(len(lab_diff))
        if total >= max_pairs:
            break
    if total == 0:
        return {
            "n_pairs": 0,
            "accuracy": float("nan"),
            "chance": 0.5,
            "lift_vs_chance": float("nan"),
        }
    acc = correct / total
    return {
        "n_pairs": int(total),
        "accuracy": float(acc),
        "chance": 0.5,
        "lift_vs_chance": float(acc - 0.5),
    }


def _fit_pairwise_logistic(
    x: np.ndarray,
    dates: pd.Series,
    label: np.ndarray,
    *,
    seed: int,
    max_pairs: int = PAIRWISE_MAX_PAIRS,
) -> dict[str, Any]:
    """Train logistic on feature diffs of within-day pairs; eval Rank IC via scores."""
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    ys: list[int] = []
    # Collect train pairs from a date-holdout: use even/odd day hash for cheap split
    dates_arr = dates.to_numpy()
    unique_days = np.asarray(pd.unique(dates))
    rng.shuffle(unique_days)
    cut = int(0.75 * len(unique_days))
    train_days = set(unique_days[:cut])
    test_days = set(unique_days[cut:])

    def _collect(day_set: set[Any], cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dlist: list[np.ndarray] = []
        ylist: list[int] = []
        score_idx: list[int] = []  # not used for train
        n = 0
        for day in day_set:
            idx = np.where(dates_arr == day)[0]
            if len(idx) < 30:
                continue
            n_need = min(150, len(idx) * 2)
            a = rng.integers(0, len(idx), size=n_need)
            b = rng.integers(0, len(idx), size=n_need)
            keep = a != b
            a, b = idx[a[keep]], idx[b[keep]]
            lab_diff = label[a] - label[b]
            usable = np.abs(lab_diff) > 1e-12
            a, b = a[usable], b[usable]
            lab_diff = lab_diff[usable]
            # orient so y=1 means a > b in label
            flip = lab_diff < 0
            aa = np.where(flip, b, a)
            bb = np.where(flip, a, b)
            dlist.append(x[aa] - x[bb])
            ylist.extend([1] * len(aa))
            # also negative pairs
            dlist.append(x[bb] - x[aa])
            ylist.extend([0] * len(bb))
            n += 2 * len(aa)
            if n >= cap:
                break
        if not dlist:
            return (
                np.zeros((0, x.shape[1])),
                np.zeros(0, dtype=np.int64),
                np.array([]),
                np.array([]),
            )
        return np.vstack(dlist), np.asarray(ylist, dtype=np.int64), np.array([]), np.array([])

    x_tr, y_tr, _, _ = _collect(train_days, max_pairs)
    if len(y_tr) < 500:
        return {"skipped": True, "reason": "too_few_pairs"}
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, random_state=seed),
    )
    pipe.fit(x_tr, y_tr)
    # Score each row as w·x (decision_function on x vs zeros reference)
    # Equivalent: score_i = coef · scale(x_i); Rank IC uses relative scores within day.
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    x_s = scaler.transform(x)
    scores = (x_s @ clf.coef_.reshape(-1)) + float(clf.intercept_[0])

    test_mask = np.array([d in test_days for d in dates_arr])
    dates_te = pd.Series(dates_arr[test_mask]).reset_index(drop=True)
    scores_te = scores[test_mask]
    label_te = label[test_mask]
    rank_ic = _daily_rank_ic(dates_te, scores_te, label_te)
    topk = _topk_hit_rate(dates_te, scores_te, label_te, frac=TOPK_FRAC)
    pair = _pairwise_accuracy(dates_te, scores_te, label_te, seed=seed + 1)
    # train accuracy
    tr_acc = float(pipe.score(x_tr, y_tr))
    return {
        "skipped": False,
        "n_train_pairs": int(len(y_tr)),
        "n_train_days": int(len(train_days)),
        "n_test_days": int(len(test_days)),
        "train_pair_accuracy": tr_acc,
        "rank_ic": rank_ic,
        "topk": topk,
        "pairwise": pair,
        "pooled_spearman": _safe_spearman(scores_te, label_te),
    }


def _pack_ridge_ranking(
    ridge_out: dict[str, Any],
    dates: pd.Series,
    label_full: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    if ridge_out.get("skipped"):
        return {"skipped": True, "reason": ridge_out.get("reason")}
    mask = ridge_out["mask"]
    dates_m = dates[mask].reset_index(drop=True)
    label_m = label_full[mask]
    idx_te = ridge_out["idx_te"]
    pred = ridge_out["pred"]
    y_te = ridge_out["y_te"]
    dates_te = dates_m.iloc[idx_te].reset_index(drop=True)
    label_te = label_m[idx_te]
    return {
        "skipped": False,
        "n_train": ridge_out["n_train"],
        "n_test": ridge_out["n_test"],
        "pooled_spearman_pred_vs_y": _safe_spearman(pred, y_te),
        "rank_ic": _daily_rank_ic(dates_te, pred, label_te),
        "topk": _topk_hit_rate(dates_te, pred, label_te, frac=TOPK_FRAC),
        "pairwise": _pairwise_accuracy(dates_te, pred, label_te, seed=seed),
    }


def run_ranking_ablation(
    *,
    val_panel: Path,
    val_targets: Path,
    seed: int = SEED,
    phase_g2_json: Path | None = None,
    phase_k_json: Path | None = None,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    packed = build_protocol_labels(val_panel, targets)
    base = packed["base_features"]
    comb = packed["combined_features"]
    lab = packed["protocol_labels"]
    dates = packed["meta"]["asof_date"]
    fwd = np.asarray(lab["fwd_ret_10"], dtype=np.float64)
    mfe = np.asarray(lab["mfe10"], dtype=np.float64)
    fwd_cs = _cs_rank(dates, fwd)
    mfe_cs = _cs_rank(dates, mfe)

    ridge_arms: dict[str, Any] = {}
    for tag, x, y, y_for_ic in [
        ("fwd_ret_10_base", base, fwd, fwd),
        ("fwd_ret_10_comb", comb, fwd, fwd),
        ("mfe10_base", base, mfe, mfe),
        ("mfe10_comb", comb, mfe, mfe),
        ("fwd_cs_rank_comb", comb, fwd_cs, fwd),  # train on rank, IC vs continuous
        ("mfe_cs_rank_comb", comb, mfe_cs, mfe),
    ]:
        raw = _fit_ridge_predict(x, y, seed=seed)
        ridge_arms[tag] = _pack_ridge_ranking(raw, dates, y_for_ic, seed=seed)

    pairwise_arms: dict[str, Any] = {
        "fwd_ret_10_comb": _fit_pairwise_logistic(comb, dates, fwd, seed=seed),
        "mfe10_comb": _fit_pairwise_logistic(comb, dates, mfe, seed=seed + 7),
    }

    # Binary path refs
    g2_delta = BINARY_COMB_DELTA_G2
    g2_verdict = None
    if phase_g2_json and Path(phase_g2_json).is_file():
        g2 = json.loads(Path(phase_g2_json).read_text(encoding="utf-8"))
        g2_verdict = g2.get("decision", {}).get("verdict")
        g2_delta = (
            g2.get("logistic", {})
            .get("buy_worth_mfe10pct_comb", {})
            .get("delta_model_minus_prior", g2_delta)
        )
    k_delta = BINARY_PHASE_K_BEST_DELTA
    k_gate = False
    if phase_k_json and Path(phase_k_json).is_file():
        pk = json.loads(Path(phase_k_json).read_text(encoding="utf-8"))
        k_delta = pk.get("best_delta_vs_prior", k_delta)
        k_gate = bool(pk.get("gate_passed", False))

    primary = ridge_arms["fwd_ret_10_comb"]
    primary_mfe = ridge_arms["mfe10_comb"]
    pw_fwd = pairwise_arms["fwd_ret_10_comb"]
    pw_mfe = pairwise_arms["mfe10_comb"]

    def _ic(arm: dict[str, Any]) -> float:
        if arm.get("skipped"):
            return float("nan")
        return float(arm.get("rank_ic", {}).get("mean", float("nan")))

    def _topk_lift(arm: dict[str, Any]) -> float:
        if arm.get("skipped"):
            return float("nan")
        return float(arm.get("topk", {}).get("lift_vs_chance", float("nan")))

    candidates = [
        ("ridge_fwd_comb", primary),
        ("ridge_mfe_comb", primary_mfe),
        ("ridge_fwd_cs_comb", ridge_arms["fwd_cs_rank_comb"]),
        ("ridge_mfe_cs_comb", ridge_arms["mfe_cs_rank_comb"]),
        ("pairwise_fwd", pw_fwd),
        ("pairwise_mfe", pw_mfe),
    ]
    best_name, best_arm = max(
        candidates,
        key=lambda t: (
            _ic(t[1]) if np.isfinite(_ic(t[1])) else -1.0,
        ),
    )
    best_ic = _ic(best_arm)
    best_topk = _topk_lift(best_arm)

    clears_rank_bar = np.isfinite(best_ic) and best_ic >= RANK_IC_BAR
    clears_topk_bar = np.isfinite(best_topk) and best_topk >= TOPK_LIFT_BAR
    ranking_signal = bool(clears_rank_bar or clears_topk_bar)

    # Clear lift vs binary: ranking usable while binary failed absolute gate
    binary_failed_gate = (float(g2_delta) > BINARY_GATE) and (not k_gate)
    clear_lift_vs_binary = bool(ranking_signal and binary_failed_gate)

    if clear_lift_vs_binary:
        verdict = "ranking_target__clear_lift_vs_binary__prefer_rank_path"
        recommend_next = (
            "截面排序目标（fwd_ret/mfe 连续或分位）在 Base+xsection 上给出相对二分类路径"
            "更清晰的 Rank IC / TopK 抬升；下一步可做极短 listwise/pairwise 探针，"
            "不要再开 mfe≥10% 二分类长训 / 22 层 sidecar。"
        )
        train_yes_no = "yes_short_ranking_probe_only"
    elif ranking_signal:
        verdict = "ranking_target__signal_present__binary_also_partial"
        recommend_next = (
            "排序信号存在但相对二分类优势不够干净；可记录，仍优先排序目标而非二分类。"
        )
        train_yes_no = "maybe_short_ranking_only"
    else:
        verdict = "ranking_target__no_clear_lift__do_not_train"
        recommend_next = (
            "Base+xsection 排序目标未过 Rank IC/TopK 门槛；不要据此开训。"
            "二分类路径（G2/K）亦未过闸。停廉价消融或换特征/教师。"
        )
        train_yes_no = "no"

    notes = [
        f"n_samples={packed['n_samples']}; seed={seed}; TopK frac={TOPK_FRAC}.",
        f"Binary refs: G2 comb Δ={g2_delta}; Phase K best Δ={k_delta}; gate={BINARY_GATE}.",
        f"Best ranking arm={best_name}; Rank IC mean={best_ic}; TopK lift={best_topk}.",
        "Train panel not on disk; val panel only (same as Phase D–H).",
        "No 22-layer train; no TPU WIP.",
    ]

    return {
        "phase": "L_ranking_target",
        "n_samples": int(packed["n_samples"]),
        "base_dim": int(base.shape[1]),
        "xsec_dim": int(comb.shape[1] - base.shape[1]),
        "seed": seed,
        "topk_frac": TOPK_FRAC,
        "bars": {
            "rank_ic_bar": RANK_IC_BAR,
            "topk_lift_bar": TOPK_LIFT_BAR,
            "binary_gate": BINARY_GATE,
        },
        "binary_path_refs": {
            "phase_g2_comb_delta": float(g2_delta),
            "phase_g2_verdict": g2_verdict,
            "phase_k_best_delta": float(k_delta),
            "phase_k_gate_passed": bool(k_gate),
            "binary_failed_absolute_gate": bool(binary_failed_gate),
        },
        "ridge_ranking": ridge_arms,
        "pairwise_logistic": pairwise_arms,
        "decision": {
            "verdict": verdict,
            "recommend_next": recommend_next,
            "train_yes_no": train_yes_no,
            "ranking_signal": ranking_signal,
            "clear_lift_vs_binary": clear_lift_vs_binary,
            "best_arm": best_name,
            "best_rank_ic_mean": best_ic,
            "best_topk_lift": best_topk,
            "clears_rank_ic_bar": clears_rank_bar,
            "clears_topk_lift_bar": clears_topk_bar,
            "diagnostic_notes": notes,
        },
    }


def _fmt_arm(name: str, arm: dict[str, Any]) -> str:
    if arm.get("skipped"):
        return f"| {name} | skipped | — | — | — |"
    ic = arm.get("rank_ic", {})
    tk = arm.get("topk", {})
    pw = arm.get("pairwise", {})
    return (
        f"| {name} | {ic.get('mean', float('nan')):.4f} "
        f"(n={ic.get('n_days', 0)}) | "
        f"{tk.get('mean_hit', float('nan')):.4f} "
        f"(lift={tk.get('lift_vs_chance', float('nan')):+.4f}) | "
        f"{pw.get('accuracy', float('nan')):.4f} "
        f"(lift={pw.get('lift_vs_chance', float('nan')):+.4f}) |"
    )


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    dec = payload["decision"]
    ridge = payload["ridge_ranking"]
    pw = payload["pairwise_logistic"]
    bref = payload["binary_path_refs"]
    lines = [
        "# Kairos 排序目标消融（Phase L）",
        "",
        "日期：2026-10-01（北京时间）。用户对齐诊断：α 在排序不在二分类。",
        "",
        "## 一句话",
        "",
        f"**{dec['verdict']}**；train_yes_no=`{dec['train_yes_no']}`。",
        f"最佳臂=`{dec['best_arm']}`，Rank IC mean=`{dec['best_rank_ic_mean']}`，"
        f"TopK lift=`{dec['best_topk_lift']}`。",
        f"相对二分类清晰抬升=`{dec['clear_lift_vs_binary']}`。",
        "",
        f"- 建议：{dec['recommend_next']}",
        "",
        "## 设定",
        "",
        f"- n=`{payload['n_samples']}`，base_dim=`{payload['base_dim']}`，"
        f"xsec_dim=`{payload['xsec_dim']}`，seed=`{payload['seed']}`",
        "- 特征：Base OHLCVA 汇总 + 截面（同 Phase D）",
        "- 目标：连续 `fwd_ret_10` / `mfe10`；截面分位；日内 pairwise logistic",
        f"- 门槛：Rank IC≥`{payload['bars']['rank_ic_bar']}` 或 "
        f"TopK lift≥`{payload['bars']['topk_lift_bar']}`；"
        f"二分类闸 Δ≤`{payload['bars']['binary_gate']}`",
        "- **未** 22 层长训；**未** 动 TPU WIP",
        "",
        "## 二分类路径对照",
        "",
        f"| 项 | 值 |",
        f"| --- | ---: |",
        f"| Phase G2 Logistic Base+xsection Δ | {bref['phase_g2_comb_delta']} |",
        f"| Phase K identity best Δ | {bref['phase_k_best_delta']} |",
        f"| Phase K gate_passed | {bref['phase_k_gate_passed']} |",
        f"| 二分类绝对闸失败 | {bref['binary_failed_absolute_gate']} |",
        "",
        "## Ridge 排序（预测 → 日均 Rank IC / TopK / pairwise）",
        "",
        "| 臂 | Rank IC mean | TopK hit (lift) | Pairwise acc (lift) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in [
        "fwd_ret_10_base",
        "fwd_ret_10_comb",
        "mfe10_base",
        "mfe10_comb",
        "fwd_cs_rank_comb",
        "mfe_cs_rank_comb",
    ]:
        lines.append(_fmt_arm(name, ridge[name]))
    lines += [
        "",
        "## Pairwise logistic（特征差 → 分数）",
        "",
        "| 臂 | Rank IC mean | TopK hit (lift) | Pairwise acc (lift) |",
        "| --- | ---: | ---: | ---: |",
        _fmt_arm("fwd_ret_10_comb", pw["fwd_ret_10_comb"]),
        _fmt_arm("mfe10_comb", pw["mfe10_comb"]),
        "",
        "## 决策",
        "",
        f"- verdict = `{dec['verdict']}`",
        f"- ranking_signal = `{dec['ranking_signal']}`",
        f"- clear_lift_vs_binary = `{dec['clear_lift_vs_binary']}`",
        f"- best_arm = `{dec['best_arm']}`",
        f"- best Rank IC = `{dec['best_rank_ic_mean']}`",
        f"- best TopK lift = `{dec['best_topk_lift']}`",
        f"- train_yes_no = `{dec['train_yes_no']}`",
        "",
        "### 诊断笔记",
        "",
    ]
    for n in dec["diagnostic_notes"]:
        lines.append(f"- {n}")
    lines += [
        "",
        "## 明确不做",
        "",
        "1. 不再开 mfe≥10% 二分类 22 层 sidecar / 全量 R2。",
        "2. 不碰同事 TPU WIP。",
        "3. 若过闸，下一步仅极短 ranking probe（listwise/pairwise），非长训。",
        "",
    ]
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-panel", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_l_ranking_target.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_l_ranking_target_cn.md"),
    )
    parser.add_argument(
        "--phase-g2-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_g2_buy_worth_mfe10pct.json"),
    )
    parser.add_argument(
        "--phase-k-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_k_results_ranking_next.json"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    payload = run_ranking_ablation(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        seed=args.seed,
        phase_g2_json=args.phase_g2_json,
        phase_k_json=args.phase_k_json,
    )
    # Strip non-serializable numpy from nested if any (pred arrays already dropped)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_cn_memo(payload, args.out_md)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
