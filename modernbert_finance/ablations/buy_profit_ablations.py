"""Phase G: buy-profit binary — fwd_ret_10 >= +10% (CPU, val panel).

Definition (primary):
  y = 1{ close[T+10]/close[T]-1 >= 0.10 }

Optional: same threshold on industry-mean-demeaned residual (Phase F resid).
Logistic vs constant prior on Base and Base+xsection (reuse Phase E/F).
No R2 / Kaggle train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modernbert_finance.ablations.continuous_xsection_ablations import (
    _fit_eval_logistic,
)
from modernbert_finance.ablations.industry_neutral_ablations import (
    _industry_demean,
)
from modernbert_finance.ablations.label_protocol_ablations import (
    build_protocol_labels,
)
from modernbert_finance.build_dataset import HORIZON

SEED = 20261001
EPS = 1e-4
PROFIT_THRESHOLD = 0.10

# Absolute bar (same as Phase E/F logistic gate)
LOGIT_DELTA_BAR = -0.04
# "Clearly better" than Phase E/F primary binary: more negative by this margin
LIFT_DELTA_VS_PHASE_EF = -0.02
# Flag extreme imbalance (rare positive class)
RARE_POS_RATE = 0.05
IMBALANCE_WARN_RATE = 0.10


def build_buy_profit_labels(
    panel: Any,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
    threshold: float = PROFIT_THRESHOLD,
    min_industry_n: int = 2,
) -> dict[str, Any]:
    """Reuse Phase E feature pack; add buy-profit binary on fwd / ind resid."""
    packed = build_protocol_labels(
        panel, targets, signal_start=signal_start, signal_end=signal_end
    )
    meta = packed["meta"]
    lab = packed["protocol_labels"]
    fwd10 = np.asarray(lab["fwd_ret_10"], dtype=np.float64)

    y_raw = (fwd10 >= float(threshold)).astype(np.int64)

    industry_labels: dict[str, Any] = {}
    industry_stats: dict[str, Any] | None = None
    if "sector" in meta.columns:
        resid10, st10 = _industry_demean(
            meta, fwd10, how="mean", min_industry_n=min_industry_n
        )
        y_ind = (resid10 >= float(threshold)).astype(np.int64)
        industry_labels = {
            "ind_resid_fwd_10_mean": resid10,
            "buy_profit_10pct_ind_resid": y_ind,
        }
        industry_stats = {
            "fwd10_mean_demean": st10,
            "sector_field": "sector",
            "threshold_on_resid": float(threshold),
        }
    else:
        resid10 = None
        y_ind = None

    buy_labels: dict[str, Any] = {
        "fwd_ret_10": fwd10,
        "buy_profit_10pct": y_raw,
        "fwd_sign_up": (fwd10 > 0.0).astype(np.int64),
        **industry_labels,
    }

    return {
        **packed,
        "buy_labels": buy_labels,
        "industry_stats": industry_stats,
        "threshold": float(threshold),
        "definition": {
            "primary": (
                f"y = 1{{fwd_ret_{HORIZON} >= {threshold:.2f}}} where "
                f"fwd_ret_{HORIZON} = close[T+{HORIZON}]/close[T]-1"
            ),
            "optional_industry_neutral": (
                f"y_ind = 1{{ind_resid_fwd_{HORIZON}_mean >= {threshold:.2f}}} "
                f"(Phase F mean demean; fallback market when industry-day n < {min_industry_n})"
            ),
            "horizon": int(HORIZON),
            "threshold": float(threshold),
        },
        "_y_ind_available": y_ind is not None,
    }


def run_phase_g(
    *,
    val_panel: Path,
    val_targets: Path,
    phase_e_json: Path | None = None,
    phase_f_json: Path | None = None,
    seed: int = SEED,
    threshold: float = PROFIT_THRESHOLD,
    min_industry_n: int = 2,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    packed = build_buy_profit_labels(
        val_panel,
        targets,
        threshold=threshold,
        min_industry_n=min_industry_n,
    )
    base = packed["base_features"]
    comb = packed["combined_features"]
    lab = packed["buy_labels"]
    y = lab["buy_profit_10pct"]
    pos_rate = float(np.mean(y))
    n_pos = int(np.sum(y))

    logistic: dict[str, Any] = {
        "buy_profit_10pct_base": _fit_eval_logistic(base, y, seed=seed),
        "buy_profit_10pct_comb": _fit_eval_logistic(comb, y, seed=seed),
        # Echo Phase E sign for side-by-side on same split seed
        "fwd_sign_up_base": _fit_eval_logistic(base, lab["fwd_sign_up"], seed=seed),
        "fwd_sign_up_comb": _fit_eval_logistic(comb, lab["fwd_sign_up"], seed=seed),
    }

    ind_pos_rate = None
    if packed.get("_y_ind_available") and "buy_profit_10pct_ind_resid" in lab:
        y_ind = lab["buy_profit_10pct_ind_resid"]
        ind_pos_rate = float(np.mean(y_ind))
        logistic["buy_profit_10pct_ind_resid_base"] = _fit_eval_logistic(
            base, y_ind, seed=seed
        )
        logistic["buy_profit_10pct_ind_resid_comb"] = _fit_eval_logistic(
            comb, y_ind, seed=seed
        )

    # Phase E/F primary binary baselines
    phase_e_sign_delta = None
    phase_e_cs_top_delta = None
    phase_e_verdict = None
    if phase_e_json and Path(phase_e_json).is_file():
        pe = json.loads(Path(phase_e_json).read_text(encoding="utf-8"))
        phase_e_verdict = pe.get("decision", {}).get("verdict")
        a_clf = pe.get("protocol_A_forward_return", {}).get("logistic", {})
        phase_e_sign_delta = a_clf.get("fwd_sign_up_comb", {}).get(
            "delta_model_minus_prior"
        )
        phase_e_cs_top_delta = a_clf.get("fwd_cs_top20_comb", {}).get(
            "delta_model_minus_prior"
        )

    phase_f_sign_delta = None
    phase_f_cs_top_delta = None
    phase_f_verdict = None
    if phase_f_json and Path(phase_f_json).is_file():
        pf = json.loads(Path(phase_f_json).read_text(encoding="utf-8"))
        phase_f_verdict = pf.get("decision", {}).get("verdict")
        f_clf = pf.get("logistic", {})
        phase_f_sign_delta = f_clf.get("ind_resid_sign_up_mean_comb", {}).get(
            "delta_model_minus_prior"
        )
        phase_f_cs_top_delta = f_clf.get("ind_resid_cs_top20_comb", {}).get(
            "delta_model_minus_prior"
        )

    primary = logistic["buy_profit_10pct_comb"]
    primary_delta = primary.get("delta_model_minus_prior", float("nan"))
    if primary.get("skipped"):
        primary_delta = float("nan")

    ind_primary_delta = None
    if "buy_profit_10pct_ind_resid_comb" in logistic:
        ind_p = logistic["buy_profit_10pct_ind_resid_comb"]
        if not ind_p.get("skipped"):
            ind_primary_delta = ind_p.get("delta_model_minus_prior")

    beats_abs = (
        isinstance(primary_delta, float)
        and primary_delta == primary_delta
        and primary_delta <= LOGIT_DELTA_BAR
    )

    # Best Phase E/F binary Δ among primary sign + CS top20 (more negative = better)
    ef_candidates = [
        d
        for d in (
            phase_e_sign_delta,
            phase_e_cs_top_delta,
            phase_f_sign_delta,
            phase_f_cs_top_delta,
        )
        if d is not None
    ]
    best_ef_delta = min(ef_candidates) if ef_candidates else None

    clearly_better_than_ef = (
        best_ef_delta is not None
        and isinstance(primary_delta, float)
        and primary_delta == primary_delta
        and primary_delta <= float(best_ef_delta) + LIFT_DELTA_VS_PHASE_EF
    )

    # Also allow industry-neutral variant to pass the same gates
    ind_beats_abs = (
        ind_primary_delta is not None
        and isinstance(ind_primary_delta, float)
        and ind_primary_delta == ind_primary_delta
        and ind_primary_delta <= LOGIT_DELTA_BAR
    )
    ind_clearly_better = (
        best_ef_delta is not None
        and ind_primary_delta is not None
        and isinstance(ind_primary_delta, float)
        and ind_primary_delta == ind_primary_delta
        and ind_primary_delta <= float(best_ef_delta) + LIFT_DELTA_VS_PHASE_EF
    )

    lifts_enough = bool(
        beats_abs or clearly_better_than_ef or ind_beats_abs or ind_clearly_better
    )

    class_too_rare = pos_rate < RARE_POS_RATE
    class_imbalanced = pos_rate < IMBALANCE_WARN_RATE

    notes: list[str] = []
    notes.append(
        f"Primary positive rate={pos_rate:.4f} (n_pos={n_pos}/{packed['n_samples']}); "
        f"threshold={threshold:.2f} on fwd_ret_{HORIZON}."
    )
    if ind_pos_rate is not None:
        notes.append(
            f"Industry-neutral resid >= {threshold:.2f} positive rate={ind_pos_rate:.4f}."
        )
    if class_too_rare:
        notes.append(
            f"Class very rare (pos_rate < {RARE_POS_RATE}): logistic Δ hard to interpret; "
            "imbalance may dominate."
        )
    elif class_imbalanced:
        notes.append(
            f"Class imbalanced (pos_rate < {IMBALANCE_WARN_RATE}): note prior LL shrinks; "
            "compare Δ carefully vs Phase E/F rarer heads (CS top20)."
        )
    notes.append(
        "Train panel not on disk locally; ablation uses validation panel only "
        f"(n={packed['n_samples']})."
    )
    if isinstance(primary_delta, float) and primary_delta == primary_delta:
        notes.append(
            f"Primary comb Δ={primary_delta:+.6f} vs bar {LOGIT_DELTA_BAR}; "
            f"best Phase E/F binary Δ={best_ef_delta}."
        )

    if lifts_enough:
        verdict = "buy_profit_10pct__lifts_enough_for_next_step"
        recommend_next = (
            "买入盈利(+10%/10日)二分类过关：可做短 sidecar 标签重建 + 线性确认；"
            "仍不要全量 R2 八头重启 / Kaggle 长训。注意正类稀有度。"
        )
        train_yes_no = "yes_short_sidecar_only"
    else:
        verdict = "buy_profit_10pct__no_lift__do_not_train"
        recommend_next = (
            "买入盈利(+10%/10日)未过 Δ≤-0.04，也未相对 Phase E/F 二分类明显更好；"
            "不要据此开训。可结束廉价消融，或换质变标签，而非再拧阈值。"
        )
        train_yes_no = "no"

    decision = {
        "lifts_enough": lifts_enough,
        "verdict": verdict,
        "recommend_next": recommend_next,
        "train_yes_no": train_yes_no,
        "diagnostic_notes": notes,
        "thresholds": {
            "logit_delta_bar": LOGIT_DELTA_BAR,
            "lift_delta_vs_phase_ef_binary": LIFT_DELTA_VS_PHASE_EF,
            "profit_threshold": float(threshold),
            "rare_pos_rate_flag": RARE_POS_RATE,
            "imbalance_warn_rate": IMBALANCE_WARN_RATE,
            "primary_head": "buy_profit_10pct logistic (comb)",
            "optional_head": "buy_profit_10pct_ind_resid logistic (comb)",
        },
        "phase_e_fwd_sign_up_comb_delta": phase_e_sign_delta,
        "phase_e_fwd_cs_top20_comb_delta": phase_e_cs_top_delta,
        "phase_e_verdict": phase_e_verdict,
        "phase_f_ind_resid_sign_up_mean_comb_delta": phase_f_sign_delta,
        "phase_f_ind_resid_cs_top20_comb_delta": phase_f_cs_top_delta,
        "phase_f_verdict": phase_f_verdict,
        "best_phase_ef_binary_delta": best_ef_delta,
        "primary_logit_delta": float(primary_delta)
        if isinstance(primary_delta, float) and primary_delta == primary_delta
        else None,
        "ind_resid_logit_delta": float(ind_primary_delta)
        if ind_primary_delta is not None
        and isinstance(ind_primary_delta, float)
        and ind_primary_delta == ind_primary_delta
        else None,
        "beats_abs_delta_bar": beats_abs,
        "clearly_better_than_phase_ef_binary": clearly_better_than_ef,
        "ind_beats_abs_delta_bar": ind_beats_abs,
        "ind_clearly_better_than_phase_ef_binary": ind_clearly_better,
        "class_too_rare": class_too_rare,
        "class_imbalanced": class_imbalanced,
        "rationale": (
            f"Primary: y=1{{fwd_ret_{HORIZON}>={threshold}}} logistic comb. "
            f"Pass if Δ≤{LOGIT_DELTA_BAR} or clearly better than best Phase E/F binary "
            f"(Δ more negative by {abs(LIFT_DELTA_VS_PHASE_EF)}). "
            "Industry-neutral same-threshold resid is optional diagnostic/alternate."
        ),
    }

    prevalence = {
        "buy_profit_10pct_rate": pos_rate,
        "buy_profit_10pct_n_pos": n_pos,
        "buy_profit_10pct_n": int(packed["n_samples"]),
        "fwd_ret_10_mean": float(np.mean(lab["fwd_ret_10"])),
        "fwd_ret_10_std": float(np.std(lab["fwd_ret_10"])),
        "fwd_ret_10_p90": float(np.quantile(lab["fwd_ret_10"], 0.90)),
        "fwd_ret_10_p95": float(np.quantile(lab["fwd_ret_10"], 0.95)),
        "fwd_ret_10_p99": float(np.quantile(lab["fwd_ret_10"], 0.99)),
        "fwd_sign_up_rate": float(np.mean(lab["fwd_sign_up"])),
    }
    if ind_pos_rate is not None:
        prevalence["buy_profit_10pct_ind_resid_rate"] = ind_pos_rate
        prevalence["buy_profit_10pct_ind_resid_n_pos"] = int(
            np.sum(lab["buy_profit_10pct_ind_resid"])
        )
        prevalence["ind_resid_fwd_10_mean_mean"] = float(
            np.mean(lab["ind_resid_fwd_10_mean"])
        )
        prevalence["ind_resid_fwd_10_mean_std"] = float(
            np.std(lab["ind_resid_fwd_10_mean"])
        )

    return {
        "phase": "G_buy_profit_10pct",
        "n_samples": packed["n_samples"],
        "base_dim": packed["base_dim"],
        "xsec_dim": packed["xsec_dim"],
        "xsec_cols": packed["xsec_cols"],
        "seed": seed,
        "threshold": float(threshold),
        "definition": packed["definition"],
        "industry_stats": packed["industry_stats"],
        "label_prevalence": prevalence,
        "logistic": logistic,
        "decision": decision,
    }


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    dec = payload["decision"]
    prev = payload["label_prevalence"]
    clf = payload["logistic"]

    def _lrow(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | skipped | — |"
        return (
            f"| {name} | {d['prior_log_loss']:.6f} | {d['model_log_loss']:.6f} | "
            f"{d['delta_model_minus_prior']:+.6f} | {d['positive_rate_test']:.4f} |"
        )

    lines = [
        "# Kairos 买入盈利(+10%/10日)二分类消融（Phase G）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
    ]
    if dec["lifts_enough"]:
        lines.append(
            f"**买入盈利标签效应量过关。** verdict=`{dec['verdict']}`；"
            f"train_yes_no=`{dec['train_yes_no']}`。"
        )
    else:
        lines.append(
            f"**买入盈利(+10%/10日)未达开训门槛。** verdict=`{dec['verdict']}`；"
            f"train_yes_no=`{dec['train_yes_no']}`。"
        )
    lines += [
        "",
        f"- 建议：{dec['recommend_next']}",
        "",
        "## 设定 / 定义",
        "",
        f"- n_samples=`{payload['n_samples']}`，seed=`{payload['seed']}`，"
        f"base_dim=`{payload['base_dim']}`，xsec_dim=`{payload['xsec_dim']}`",
        f"- **主定义**：`{payload['definition']['primary']}`",
        f"- 可选行业中性：`{payload['definition']['optional_industry_neutral']}`",
        f"- 阈值：`{payload['threshold']}`；horizon=`{payload['definition']['horizon']}`",
        "- 特征：Phase D/E/F 同款 Base + 截面（看 comb）",
        "- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP",
        "",
        "## 门槛",
        "",
        f"- 绝对：Δ logloss ≤ `{dec['thresholds']['logit_delta_bar']}` vs 常数先验",
        f"- 相对 Phase E/F 二分类：Δ 再负 "
        f"`{abs(dec['thresholds']['lift_delta_vs_phase_ef_binary'])}` "
        f"（对照 best of E sign / E CS-top20 / F resid-sign / F resid CS-top20）",
        f"- Phase E fwd_sign_up_comb Δ=`{dec['phase_e_fwd_sign_up_comb_delta']}`，"
        f"fwd_cs_top20_comb Δ=`{dec['phase_e_fwd_cs_top20_comb_delta']}`",
        f"- Phase F ind_resid_sign_up_mean_comb Δ="
        f"`{dec['phase_f_ind_resid_sign_up_mean_comb_delta']}`，"
        f"ind_resid_cs_top20_comb Δ=`{dec['phase_f_ind_resid_cs_top20_comb_delta']}`",
        f"- best Phase E/F binary Δ=`{dec['best_phase_ef_binary_delta']}`",
        "",
        "## 标签分布",
        "",
        f"- **正类率 (buy_profit_10pct)** = `{prev['buy_profit_10pct_rate']:.4f}` "
        f"(`{prev['buy_profit_10pct_n_pos']}` / `{prev['buy_profit_10pct_n']}`)",
        f"- class_too_rare=`{dec['class_too_rare']}` "
        f"(flag < `{dec['thresholds']['rare_pos_rate_flag']}`)；"
        f"class_imbalanced=`{dec['class_imbalanced']}` "
        f"(warn < `{dec['thresholds']['imbalance_warn_rate']}`)",
        f"- fwd_ret_10 mean/std=`{prev['fwd_ret_10_mean']:.4f}`/`{prev['fwd_ret_10_std']:.4f}`",
        f"- fwd_ret_10 p90/p95/p99=`{prev['fwd_ret_10_p90']:.4f}`/"
        f"`{prev['fwd_ret_10_p95']:.4f}`/`{prev['fwd_ret_10_p99']:.4f}`",
        f"- fwd_sign_up rate=`{prev['fwd_sign_up_rate']:.4f}`",
    ]
    if "buy_profit_10pct_ind_resid_rate" in prev:
        lines.append(
            f"- ind_resid ≥ threshold 正类率=`{prev['buy_profit_10pct_ind_resid_rate']:.4f}` "
            f"(n_pos=`{prev['buy_profit_10pct_ind_resid_n_pos']}`)"
        )
    lines += [
        "",
        "## 1) Logistic（买入盈利 vs 先验）",
        "",
        "| Head | prior LL | model LL | Δ | pos_rate_test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for k, v in clf.items():
        lines.append(_lrow(k, v))

    lines += [
        "",
        "## 2) 决策",
        "",
        f"| 指标 | 值 | 过关 |",
        f"| --- | ---: | :---: |",
        f"| 主 Δ (buy_profit comb) | {dec['primary_logit_delta']} | "
        f"{'Y' if dec['beats_abs_delta_bar'] else 'N'} |",
        f"| vs Phase E/F clearly better | — | "
        f"{'Y' if dec['clearly_better_than_phase_ef_binary'] else 'N'} |",
        f"| 行业中性 Δ (optional) | {dec['ind_resid_logit_delta']} | "
        f"{'Y' if dec['ind_beats_abs_delta_bar'] else 'N'} |",
        f"| 行业中性 vs E/F | — | "
        f"{'Y' if dec['ind_clearly_better_than_phase_ef_binary'] else 'N'} |",
        f"| 正类过稀 | {dec['class_too_rare']} | warn |",
        "",
        f"- lifts_enough = `{dec['lifts_enough']}`",
        f"- verdict = `{dec['verdict']}`",
        f"- train_yes_no = `{dec['train_yes_no']}`",
        f"- 判据：{dec['rationale']}",
        "",
        "### 诊断备注",
        "",
    ]
    for note in dec.get("diagnostic_notes") or []:
        lines.append(f"- {note}")
    lines += [
        "",
        "### 建议下一步",
        "",
    ]
    if dec["lifts_enough"]:
        lines += [
            "1. **可**按买入盈利(+10%/10日)做短 sidecar 标签 + 线性确认。",
            "2. **不要**重启同配置 R2 八头 BCE 长训。",
            "3. 注意正类稀有度；截面特征继续默认带上。",
        ]
    else:
        lines += [
            "1. **不要**基于买入盈利(+10%/10日)开标签重建长训。",
            "2. 廉价标签消融路径可收束；若再试需质变，非再拧阈值。",
            "3. **不要**重启同配置 R2。",
        ]
    lines += [
        "",
        "## 证据强度",
        "",
        "| 主张 | 强度 | 依据 |",
        "| --- | --- | --- |",
        f"| 买入盈利可开下一步 | {'中' if dec['lifts_enough'] else '弱/否'} | Δ 见表 |",
        "| 应维持停 R2 | 强 | 与 Phase C–F 一致 |",
        "",
        "## 用户要点（中文）",
        "",
        f"- **定义**：未来 {payload['definition']['horizon']} 个交易日收益 "
        f"`close[T+{payload['definition']['horizon']}]/close[T]-1 ≥ "
        f"{payload['threshold']}` 记为买入盈利正类。",
        f"- **正类率**：`{prev['buy_profit_10pct_rate']:.4f}`"
        + (
            f"；行业中性残差同阈值正类率=`{prev['buy_profit_10pct_ind_resid_rate']:.4f}`"
            if "buy_profit_10pct_ind_resid_rate" in prev
            else ""
        )
        + ("（过稀）" if dec["class_too_rare"] else ("（偏稀）" if dec["class_imbalanced"] else "")),
        f"- **指标**：Base Δ=`{clf['buy_profit_10pct_base'].get('delta_model_minus_prior')}`；"
        f"Base+xsection Δ=`{dec['primary_logit_delta']}`"
        + (
            f"；行业中性 comb Δ=`{dec['ind_resid_logit_delta']}`"
            if dec["ind_resid_logit_delta"] is not None
            else ""
        ),
        f"- **过关**：`{'是' if dec['lifts_enough'] else '否'}`"
        f"（门槛 Δ≤{LOGIT_DELTA_BAR} 或明显优于 Phase E/F 二分类）",
        f"- **开训**：`{'是（仅短 sidecar）' if dec['train_yes_no'] != 'no' else '否'}`",
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
        "--phase-e-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_e_label_protocols.json"),
    )
    parser.add_argument(
        "--phase-f-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_f_industry_neutral.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_g_buy_profit_10pct.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_g_buy_profit_10pct_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=PROFIT_THRESHOLD)
    parser.add_argument("--min-industry-n", type=int, default=2)
    args = parser.parse_args(argv)

    payload = run_phase_g(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        phase_e_json=args.phase_e_json,
        phase_f_json=args.phase_f_json,
        seed=args.seed,
        threshold=args.threshold,
        min_industry_n=args.min_industry_n,
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
                "decision": payload["decision"],
                "label_prevalence": payload["label_prevalence"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
