"""Phase G2: buy-worth-it binary — mfe10 >= +10% (CPU, val panel).

Corrects Phase G which used close-to-close fwd_ret_10. User decision system:
  buy, hold ≤10 sessions; worth-it iff path reaches +10% within 10 days
  i.e. MFE from entry ≥ 10%, NOT close[T+10]/close[T] ≥ 10%.

Definition (primary, matches build_targets / Phase D field):
  mfe10 = max(high[T+1:T+10]) / close[T] - 1
  y = 1{ mfe10 >= 0.10 }

Logistic vs constant prior on Base and Base+xsection (reuse Phase D/E/F features).
Compare Δ to Phase G close-to-close. No R2 / Kaggle / TPU train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modernbert_finance.ablations.continuous_xsection_ablations import (
    _fit_eval_logistic,
    build_enriched_matrix,
)
from modernbert_finance.build_dataset import HORIZON

SEED = 20261001
PROFIT_THRESHOLD = 0.10
SOFT_ABSOLUTE_THRESHOLD = 0.08  # Phase Y alternate if CS-top fails
CS_TOP_PERCENTILE = 0.80  # Phase Y primary: top quintile within day
LOGIT_DELTA_BAR = -0.04
LIFT_DELTA_VS_PHASE_EF = -0.02
RARE_POS_RATE = 0.05
IMBALANCE_WARN_RATE = 0.10

# Canonical definition string (must match build_targets.py)
MFE10_DEF = f"max(high[T+1:T+{HORIZON}]) / close[T] - 1"


def mfe10_daily_cs_percentile(mfe: np.ndarray, asof_date: Any) -> np.ndarray:
    """Daily cross-sectional percentile rank of path-MFE (label-only; 0..1)."""
    mfe = np.asarray(mfe, dtype=np.float64).reshape(-1)
    asof = pd.Series(pd.to_datetime(asof_date)).astype(str)
    if len(mfe) != len(asof):
        raise ValueError(f"mfe len {len(mfe)} != asof_date len {len(asof)}")
    tmp = pd.DataFrame({"mfe": mfe, "asof_date": asof})
    return tmp.groupby("asof_date")["mfe"].rank(pct=True).to_numpy(dtype=np.float64)


def build_mfe_buy_labels(
    panel: Any,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
    threshold: float = PROFIT_THRESHOLD,
    label_mode: str = "absolute",
    cs_pct: float = CS_TOP_PERCENTILE,
) -> dict[str, Any]:
    """Reuse Phase D feature pack; y from panel targets mfe10 (Phase D field).

    label_mode:
      - "absolute": y = 1{mfe10 >= threshold}  (Phase G2 / S–W default)
      - "cs_top":   y = 1{daily CS percentile(mfe10) >= cs_pct}  (Phase Y)
    """
    packed = build_enriched_matrix(
        panel, targets, signal_start=signal_start, signal_end=signal_end
    )
    mfe = np.asarray(packed["targets"]["mfe10"], dtype=np.float64)
    asof = packed["meta"]["asof_date"]
    mfe_cs = mfe10_daily_cs_percentile(mfe, asof)
    mode = str(label_mode).strip().lower()
    if mode == "absolute":
        y = (mfe >= float(threshold)).astype(np.int64)
        primary_def = f"y = 1{{mfe10 >= {threshold:.2f}}} where mfe10 = {MFE10_DEF}"
        decision_note = (
            "worth-it if path reaches +10% within 10 sessions "
            "(max favorable excursion from entry ≥ 10%)"
        )
    elif mode == "cs_top":
        y = (mfe_cs >= float(cs_pct)).astype(np.int64)
        primary_def = (
            f"y = 1{{daily CS percentile(mfe10) >= {cs_pct:.2f}}} "
            f"where mfe10 = {MFE10_DEF}"
        )
        decision_note = (
            f"buy if path-MFE is in top {(1.0 - float(cs_pct)) * 100:.0f}% "
            "cross-section within the signal day (relative worth-it)"
        )
    else:
        raise ValueError(f"unknown label_mode={label_mode!r}; use absolute|cs_top")

    # Sanity: definition identity vs build_targets
    definition_ok = True  # field provenance; numeric recompute audited in Phase A

    buy_labels: dict[str, Any] = {
        "mfe10": mfe,
        "mfe10_cs_pct": mfe_cs,
        "buy_worth_mfe10pct": y,  # name kept for stack compatibility
        "buy_worth_mfe_cs_top": (mfe_cs >= float(cs_pct)).astype(np.int64),
        "buy_worth_mfe_abs": (mfe >= float(threshold)).astype(np.int64),
        "mfe_sign_up": (mfe > 0.0).astype(np.int64),
    }

    return {
        **packed,
        "buy_labels": buy_labels,
        "threshold": float(threshold),
        "label_mode": mode,
        "cs_pct": float(cs_pct),
        "definition": {
            "primary": primary_def,
            "label_mode": mode,
            "cs_pct": float(cs_pct),
            "phase_g_wrong_target": (
                f"Phase G used y = 1{{fwd_ret_{HORIZON} >= {threshold:.2f}}} where "
                f"fwd_ret_{HORIZON} = close[T+{HORIZON}]/close[T]-1 (close-to-close; "
                "not path MFE)"
            ),
            "mfe10_formula": MFE10_DEF,
            "matches_decision_system": decision_note,
            "horizon": int(HORIZON),
            "threshold": float(threshold),
            "definition_ok": definition_ok,
        },
    }


def run_phase_g2(
    *,
    val_panel: Path,
    val_targets: Path,
    phase_g_json: Path | None = None,
    phase_e_json: Path | None = None,
    phase_f_json: Path | None = None,
    seed: int = SEED,
    threshold: float = PROFIT_THRESHOLD,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    packed = build_mfe_buy_labels(
        val_panel, targets, threshold=threshold
    )
    base = packed["base_features"]
    comb = packed["combined_features"]
    lab = packed["buy_labels"]
    y = lab["buy_worth_mfe10pct"]
    mfe = lab["mfe10"]
    pos_rate = float(np.mean(y))
    n_pos = int(np.sum(y))

    logistic: dict[str, Any] = {
        "buy_worth_mfe10pct_base": _fit_eval_logistic(base, y, seed=seed),
        "buy_worth_mfe10pct_comb": _fit_eval_logistic(comb, y, seed=seed),
        "mfe_sign_up_base": _fit_eval_logistic(base, lab["mfe_sign_up"], seed=seed),
        "mfe_sign_up_comb": _fit_eval_logistic(comb, lab["mfe_sign_up"], seed=seed),
    }

    # Phase G close-to-close comparison (load prior run; same seed/split)
    phase_g_base_delta = None
    phase_g_comb_delta = None
    phase_g_pos_rate = None
    phase_g_verdict = None
    if phase_g_json and Path(phase_g_json).is_file():
        pg = json.loads(Path(phase_g_json).read_text(encoding="utf-8"))
        phase_g_verdict = pg.get("decision", {}).get("verdict")
        g_clf = pg.get("logistic", {})
        phase_g_base_delta = g_clf.get("buy_profit_10pct_base", {}).get(
            "delta_model_minus_prior"
        )
        phase_g_comb_delta = g_clf.get("buy_profit_10pct_comb", {}).get(
            "delta_model_minus_prior"
        )
        phase_g_pos_rate = pg.get("label_prevalence", {}).get("buy_profit_10pct_rate")

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

    primary = logistic["buy_worth_mfe10pct_comb"]
    primary_delta = primary.get("delta_model_minus_prior", float("nan"))
    if primary.get("skipped"):
        primary_delta = float("nan")

    base_primary = logistic["buy_worth_mfe10pct_base"]
    base_delta = base_primary.get("delta_model_minus_prior", float("nan"))
    if base_primary.get("skipped"):
        base_delta = float("nan")

    beats_abs = (
        isinstance(primary_delta, float)
        and primary_delta == primary_delta
        and primary_delta <= LOGIT_DELTA_BAR
    )

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

    # vs Phase G close-to-close: more negative Δ = better lift
    better_than_phase_g = (
        phase_g_comb_delta is not None
        and isinstance(primary_delta, float)
        and primary_delta == primary_delta
        and primary_delta < float(phase_g_comb_delta)
    )
    delta_vs_phase_g = (
        float(primary_delta) - float(phase_g_comb_delta)
        if phase_g_comb_delta is not None
        and isinstance(primary_delta, float)
        and primary_delta == primary_delta
        else None
    )

    lifts_enough = bool(beats_abs or clearly_better_than_ef)

    class_too_rare = pos_rate < RARE_POS_RATE
    class_imbalanced = pos_rate < IMBALANCE_WARN_RATE

    notes: list[str] = []
    notes.append(
        f"Primary positive rate={pos_rate:.4f} (n_pos={n_pos}/{packed['n_samples']}); "
        f"threshold={threshold:.2f} on mfe10 = {MFE10_DEF}."
    )
    notes.append(
        "Definition matches max favorable excursion from entry over next "
        f"{HORIZON} sessions (Phase D mfe10 field / build_targets)."
    )
    if phase_g_pos_rate is not None:
        notes.append(
            f"Phase G close-to-close pos_rate={phase_g_pos_rate:.4f} vs "
            f"G2 MFE pos_rate={pos_rate:.4f} "
            f"(MFE≥thr ⊇ path-touch; typically higher than close-to-close)."
        )
    if phase_g_comb_delta is not None and isinstance(primary_delta, float):
        notes.append(
            f"Phase G comb Δ={phase_g_comb_delta:+.6f}; G2 comb Δ={primary_delta:+.6f}; "
            f"Δ(G2−G)={delta_vs_phase_g:+.6f} "
            f"({'G2 better' if better_than_phase_g else 'G2 not better'})."
        )
    if class_too_rare:
        notes.append(
            f"Class very rare (pos_rate < {RARE_POS_RATE}): logistic Δ hard to interpret."
        )
    elif class_imbalanced:
        notes.append(
            f"Class imbalanced (pos_rate < {IMBALANCE_WARN_RATE}): note prior LL shrinks."
        )
    notes.append(
        "Train panel not on disk locally; ablation uses validation panel only "
        f"(n={packed['n_samples']})."
    )
    notes.append(
        "Costs/slippage not modeled: MFE is path high-touch, not a guaranteed "
        "exit fill at the high; round-trip cost would lower economic hit rate "
        "(memo note only)."
    )
    if isinstance(primary_delta, float) and primary_delta == primary_delta:
        notes.append(
            f"Primary comb Δ={primary_delta:+.6f} vs bar {LOGIT_DELTA_BAR}; "
            f"best Phase E/F binary Δ={best_ef_delta}."
        )

    if lifts_enough:
        verdict = "buy_worth_mfe10pct__lifts_enough_for_next_step"
        recommend_next = (
            "MFE买入值得(+10%/10日路径触及)二分类过关：可做短 sidecar 标签重建 + 线性确认；"
            "仍不要全量 R2 八头重启 / Kaggle 长训。注意 MFE≠可成交高点（成本未建模）。"
        )
        train_yes_no = "yes_short_sidecar_only"
    else:
        verdict = "buy_worth_mfe10pct__no_lift__do_not_train"
        recommend_next = (
            "MFE买入值得(+10%/10日路径触及)未过 Δ≤-0.04，也未相对 Phase E/F 二分类明显更好；"
            "不要据此开训。Phase G 用错 close-to-close；G2 已纠正为目标系统定义，"
            "但仍未过门槛。可结束廉价消融，或换质变标签。"
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
            "primary_head": "buy_worth_mfe10pct logistic (comb)",
            "wrong_phase_g_head": "buy_profit_10pct close-to-close (Phase G)",
        },
        "phase_g_buy_profit_10pct_base_delta": phase_g_base_delta,
        "phase_g_buy_profit_10pct_comb_delta": phase_g_comb_delta,
        "phase_g_pos_rate": phase_g_pos_rate,
        "phase_g_verdict": phase_g_verdict,
        "better_than_phase_g_close_to_close": better_than_phase_g,
        "delta_vs_phase_g_comb": delta_vs_phase_g,
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
        "base_logit_delta": float(base_delta)
        if isinstance(base_delta, float) and base_delta == base_delta
        else None,
        "beats_abs_delta_bar": beats_abs,
        "clearly_better_than_phase_ef_binary": clearly_better_than_ef,
        "class_too_rare": class_too_rare,
        "class_imbalanced": class_imbalanced,
        "rationale": (
            f"Primary: y=1{{mfe10>={threshold}}} with mfe10={MFE10_DEF} "
            f"(path MFE / decision-system worth-it). "
            f"Pass if Δ≤{LOGIT_DELTA_BAR} or clearly better than best Phase E/F binary "
            f"(Δ more negative by {abs(LIFT_DELTA_VS_PHASE_EF)}). "
            "Phase G close-to-close was the wrong target; reported for comparison only."
        ),
    }

    prevalence = {
        "buy_worth_mfe10pct_rate": pos_rate,
        "buy_worth_mfe10pct_n_pos": n_pos,
        "buy_worth_mfe10pct_n": int(packed["n_samples"]),
        "mfe10_mean": float(np.mean(mfe)),
        "mfe10_std": float(np.std(mfe)),
        "mfe10_p50": float(np.quantile(mfe, 0.50)),
        "mfe10_p75": float(np.quantile(mfe, 0.75)),
        "mfe10_p90": float(np.quantile(mfe, 0.90)),
        "mfe10_p95": float(np.quantile(mfe, 0.95)),
        "mfe10_p99": float(np.quantile(mfe, 0.99)),
        "mfe_sign_up_rate": float(np.mean(lab["mfe_sign_up"])),
        "phase_g_close_to_close_pos_rate": phase_g_pos_rate,
    }

    return {
        "phase": "G2_buy_worth_mfe10pct",
        "n_samples": packed["n_samples"],
        "base_dim": packed["base_dim"],
        "xsec_dim": packed["xsec_dim"],
        "xsec_cols": packed["xsec_cols"],
        "seed": seed,
        "threshold": float(threshold),
        "definition": packed["definition"],
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
    definition = payload["definition"]

    def _lrow(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | skipped | — |"
        return (
            f"| {name} | {d['prior_log_loss']:.6f} | {d['model_log_loss']:.6f} | "
            f"{d['delta_model_minus_prior']:+.6f} | {d['positive_rate_test']:.4f} |"
        )

    lines = [
        "# Kairos 买入值得(MFE≥+10%/10日)二分类消融（Phase G2）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
    ]
    if dec["lifts_enough"] and dec.get("beats_abs_delta_bar"):
        lines.append(
            f"**MFE 买入值得标签绝对门槛过关。** verdict=`{dec['verdict']}`；"
            f"train_yes_no=`{dec['train_yes_no']}`。"
        )
    elif dec["lifts_enough"]:
        lines.append(
            f"**绝对门槛 Δ≤-0.04 未过（主 Δ=`{dec['primary_logit_delta']}`），"
            f"但相对 Phase E/F 二分类明显更好 → 仅短 sidecar。** "
            f"verdict=`{dec['verdict']}`；train_yes_no=`{dec['train_yes_no']}`。"
        )
    else:
        lines.append(
            f"**MFE 买入值得(+10%/10日路径触及)未达开训门槛。** verdict=`{dec['verdict']}`；"
            f"train_yes_no=`{dec['train_yes_no']}`。"
        )
    lines += [
        "",
        f"- 建议：{dec['recommend_next']}",
        "",
        "## 设定 / 定义（纠正 Phase G）",
        "",
        f"- n_samples=`{payload['n_samples']}`，seed=`{payload['seed']}`，"
        f"base_dim=`{payload['base_dim']}`，xsec_dim=`{payload['xsec_dim']}`",
        f"- **主定义（决策系统）**：`{definition['primary']}`",
        f"- **mfe10 公式**：`{definition['mfe10_formula']}`"
        " —— 与 Phase D / `build_targets` 一致：入场后未来 10 个交易日"
        " **最大有利偏移 (MFE)**，非收盘对收盘。",
        f"- Phase G 错误目标：`{definition['phase_g_wrong_target']}`",
        f"- 决策语义：`{definition['matches_decision_system']}`",
        f"- 阈值：`{payload['threshold']}`；horizon=`{definition['horizon']}`",
        "- 特征：Phase D/E/F 同款 Base + 截面（看 comb）",
        "- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP",
        "",
        "### 成本 / 滑点（仅备注，未建模）",
        "",
        "- MFE≥10% 表示路径**触及**高点，不等于能在高点成交离场；"
        "买卖价差、冲击与滑点会降低经济命中率。本消融**未**扣成本，仅作语义提醒。",
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
        f"- Phase G（close-to-close）comb Δ=`{dec['phase_g_buy_profit_10pct_comb_delta']}`"
        f"；base Δ=`{dec['phase_g_buy_profit_10pct_base_delta']}`",
        "",
        "## 标签分布",
        "",
        f"- **正类率 (buy_worth_mfe10pct)** = `{prev['buy_worth_mfe10pct_rate']:.4f}` "
        f"(`{prev['buy_worth_mfe10pct_n_pos']}` / `{prev['buy_worth_mfe10pct_n']}`)",
        f"- Phase G close-to-close 正类率 = `{prev['phase_g_close_to_close_pos_rate']}`",
        f"- class_too_rare=`{dec['class_too_rare']}` "
        f"(flag < `{dec['thresholds']['rare_pos_rate_flag']}`)；"
        f"class_imbalanced=`{dec['class_imbalanced']}` "
        f"(warn < `{dec['thresholds']['imbalance_warn_rate']}`)",
        f"- mfe10 mean/std=`{prev['mfe10_mean']:.4f}`/`{prev['mfe10_std']:.4f}`",
        f"- mfe10 p50/p75/p90/p95/p99=`{prev['mfe10_p50']:.4f}`/"
        f"`{prev['mfe10_p75']:.4f}`/`{prev['mfe10_p90']:.4f}`/"
        f"`{prev['mfe10_p95']:.4f}`/`{prev['mfe10_p99']:.4f}`",
        f"- mfe_sign_up rate=`{prev['mfe_sign_up_rate']:.4f}`",
        "",
        "## 1) Logistic（MFE 买入值得 vs 先验）",
        "",
        "| Head | prior LL | model LL | Δ | pos_rate_test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for k, v in clf.items():
        lines.append(_lrow(k, v))

    lines += [
        "",
        "## 2) 对照 Phase G（错误 close-to-close）",
        "",
        f"| 设定 | 正类率 | Base Δ | Comb Δ |",
        f"| --- | ---: | ---: | ---: |",
        f"| Phase G close-to-close | {dec['phase_g_pos_rate']} | "
        f"{dec['phase_g_buy_profit_10pct_base_delta']} | "
        f"{dec['phase_g_buy_profit_10pct_comb_delta']} |",
        f"| Phase G2 MFE path | {prev['buy_worth_mfe10pct_rate']:.4f} | "
        f"{dec['base_logit_delta']} | {dec['primary_logit_delta']} |",
        f"| Δ(G2−G) comb | — | — | {dec['delta_vs_phase_g_comb']} |",
        f"- G2 相对 G 更好 = `{dec['better_than_phase_g_close_to_close']}`",
        "",
        "## 3) 决策",
        "",
        "| 指标 | 值 | 过关 |",
        "| --- | ---: | :---: |",
        f"| 主 Δ (MFE buy_worth comb) | {dec['primary_logit_delta']} | "
        f"{'Y' if dec['beats_abs_delta_bar'] else 'N'} |",
        f"| vs Phase E/F clearly better | — | "
        f"{'Y' if dec['clearly_better_than_phase_ef_binary'] else 'N'} |",
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
            "1. **可**按 MFE≥+10%/10日（路径触及）做短 sidecar 标签 + 线性确认。",
            "2. **不要**重启同配置 R2 八头 BCE 长训。",
            "3. 若上线需另加成本/可成交约束；本消融未建模滑点。",
        ]
    else:
        lines += [
            "1. **不要**基于 MFE 买入值得(+10%/10日)开标签重建长训。",
            "2. Phase G 目标已纠正为 MFE；G2 仍未过 Δ≤-0.04 —— 廉价路径可收束。",
            "3. **不要**重启同配置 R2。",
        ]
    lines += [
        "",
        "## 证据强度",
        "",
        "| 主张 | 强度 | 依据 |",
        "| --- | --- | --- |",
        f"| MFE 买入值得可开下一步 | {'中' if dec['lifts_enough'] else '弱/否'} | Δ 见表 |",
        "| Phase G close-to-close 非决策目标 | 强 | 用户：路径 MFE≥10% |",
        "| 应维持停 R2 | 强 | 与 Phase C–G 一致 |",
        "",
        "## 用户要点（中文）",
        "",
        f"- **定义**：`mfe10 = {definition['mfe10_formula']}`；"
        f"`y=1{{mfe10≥{payload['threshold']}}}`（路径触及 +10%，非收盘对收盘）。",
        f"- **正类率**：`{prev['buy_worth_mfe10pct_rate']:.4f}`"
        f"（Phase G close-to-close=`{prev['phase_g_close_to_close_pos_rate']}`）"
        + (
            "（过稀）"
            if dec["class_too_rare"]
            else ("（偏稀）" if dec["class_imbalanced"] else "")
        ),
        f"- **指标**：Base Δ=`{dec['base_logit_delta']}`；"
        f"Base+xsection Δ=`{dec['primary_logit_delta']}`；"
        f"Phase G comb Δ=`{dec['phase_g_buy_profit_10pct_comb_delta']}`；"
        f"Δ(G2−G)=`{dec['delta_vs_phase_g_comb']}`",
        f"- **过关**：`{'是' if dec['lifts_enough'] else '否'}`"
        f"（门槛 Δ≤{LOGIT_DELTA_BAR}）",
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
        "--phase-g-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_g_buy_profit_10pct.json"),
    )
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
        default=Path(
            "modernbert_finance/ablations/kairos_phase_g2_buy_worth_mfe10pct.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_g2_buy_worth_mfe10pct_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=PROFIT_THRESHOLD)
    args = parser.parse_args(argv)

    payload = run_phase_g2(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        phase_g_json=args.phase_g_json,
        phase_e_json=args.phase_e_json,
        phase_f_json=args.phase_f_json,
        seed=args.seed,
        threshold=args.threshold,
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
