"""Phase F: industry-neutral forward-return labels (CPU, val panel).

Definition (primary):
  resid_fwd_H = stock_fwd_ret_H - same-day industry mean(fwd_ret_H)
  industry id = panel `sector` (CSRC industry label; see sector_vocabulary).

Also: median demean (sensitivity), sign(resid), optional vol-scaled resid.
Ridge/Logistic on Base and Base+xsection (reuse Phase D/E). No R2 / Kaggle train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modernbert_finance.ablations.continuous_xsection_ablations import (
    _eval_regression,
    _fit_eval_logistic,
)
from modernbert_finance.ablations.label_protocol_ablations import (
    SQRT_H,
    build_protocol_labels,
)
from modernbert_finance.build_dataset import HORIZON

SEED = 20261001
EPS = 1e-4

# Same absolute bars as Phase E; relative lift vs Phase E Protocol A fwd.
DIR_R2_BAR = 0.15
LOGIT_DELTA_BAR = -0.04
LIFT_R2_VS_PHASE_E = 0.04
LIFT_DELTA_VS_PHASE_E = -0.02  # more negative than Phase E by this margin


def _industry_demean(
    meta: pd.DataFrame,
    values: np.ndarray,
    *,
    how: str = "mean",
    min_industry_n: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stock value minus same-day industry central tendency.

    When industry-day n < min_industry_n, fall back to same-day market
    central tendency (so singletons are not forced to 0).
    """
    col = "_v"
    df = meta[["asof_date", "sector"]].copy()
    df[col] = np.asarray(values, dtype=np.float64)
    if how == "mean":
        ind_center = df.groupby(["asof_date", "sector"])[col].transform("mean")
        mkt_center = df.groupby("asof_date")[col].transform("mean")
    elif how == "median":
        ind_center = df.groupby(["asof_date", "sector"])[col].transform("median")
        mkt_center = df.groupby("asof_date")[col].transform("median")
    else:
        raise ValueError(f"unknown how={how}")
    ind_n = df.groupby(["asof_date", "sector"])[col].transform("size")
    use_ind = ind_n >= int(min_industry_n)
    center = np.where(use_ind.to_numpy(), ind_center.to_numpy(), mkt_center.to_numpy())
    resid = df[col].to_numpy(dtype=np.float64) - center
    stats = {
        "how": how,
        "min_industry_n": int(min_industry_n),
        "n_rows": int(len(df)),
        "n_used_industry": int(use_ind.sum()),
        "n_fallback_market": int((~use_ind).sum()),
        "fallback_rate": float((~use_ind).mean()),
        "n_unique_sectors": int(df["sector"].nunique()),
        "n_unique_asof": int(df["asof_date"].nunique()),
        "industry_day_n_mean": float(ind_n.mean()),
        "industry_day_n_median": float(ind_n.median()),
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
    }
    return resid, stats


def build_industry_neutral_labels(
    panel: Any,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
    min_industry_n: int = 2,
) -> dict[str, Any]:
    """Reuse Phase E feature pack; add industry-neutral fwd residuals."""
    packed = build_protocol_labels(
        panel, targets, signal_start=signal_start, signal_end=signal_end
    )
    meta = packed["meta"]
    lab = packed["protocol_labels"]
    fwd5 = lab["fwd_ret_5"]
    fwd10 = lab["fwd_ret_10"]
    vol20 = lab["vol20"]

    if "sector" not in meta.columns:
        raise RuntimeError(
            "panel/meta missing `sector` industry id — cannot build industry-neutral labels"
        )

    resid5_mean, st5_mean = _industry_demean(
        meta, fwd5, how="mean", min_industry_n=min_industry_n
    )
    resid10_mean, st10_mean = _industry_demean(
        meta, fwd10, how="mean", min_industry_n=min_industry_n
    )
    resid10_med, st10_med = _industry_demean(
        meta, fwd10, how="median", min_industry_n=min_industry_n
    )

    vol_scale = np.maximum(vol20 * SQRT_H, 1e-8)
    resid10_vs = resid10_mean / vol_scale

    # Cross-section rank of residual (diagnostic label only)
    tmp = meta[["asof_date"]].copy()
    tmp["_r"] = resid10_mean
    resid_cs = tmp.groupby("asof_date")["_r"].rank(pct=True).to_numpy(dtype=np.float64)

    industry_labels: dict[str, Any] = {
        "fwd_ret_5": fwd5,
        "fwd_ret_10": fwd10,
        "ind_resid_fwd_5_mean": resid5_mean,
        "ind_resid_fwd_10_mean": resid10_mean,
        "ind_resid_fwd_10_median": resid10_med,
        "ind_resid_fwd_10_vol_scaled": resid10_vs,
        "ind_resid_sign_up_mean": (resid10_mean > 0.0).astype(np.int64),
        "ind_resid_sign_up_median": (resid10_med > 0.0).astype(np.int64),
        "ind_resid_cs_top20": (resid_cs >= 0.80).astype(np.int64),
        "ind_resid_cs_bot20": (resid_cs <= 0.20).astype(np.int64),
        "vol20": vol20,
    }

    sector_field_note = (
        "Industry id = panel column `sector` (CSRC industry string, e.g. "
        "'G56航空运输业'). No separate industry_id field in panel; "
        "sector_vocabulary.json maps the same labels. Train panel not present "
        "locally — Phase F uses validation panel only (same as Phase D/E)."
    )

    return {
        **packed,
        "industry_labels": industry_labels,
        "industry_stats": {
            "fwd5_mean_demean": st5_mean,
            "fwd10_mean_demean": st10_mean,
            "fwd10_median_demean": st10_med,
            "sector_field": "sector",
            "sector_field_note": sector_field_note,
            "horizon": int(HORIZON),
        },
        "definition": {
            "primary": (
                f"ind_resid_fwd_{HORIZON} = close[T+{HORIZON}]/close[T]-1 "
                "- same-day industry mean(fwd); fallback to market mean when "
                f"industry-day n < {min_industry_n}"
            ),
            "sensitivity_median": "same with industry/market median",
            "vol_scaled": "resid_mean / (vol20 * sqrt(H))",
            "sign": "1 if resid > 0 else 0",
        },
    }


def run_phase_f(
    *,
    val_panel: Path,
    val_targets: Path,
    phase_e_json: Path | None = None,
    seed: int = SEED,
    min_industry_n: int = 2,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    packed = build_industry_neutral_labels(
        val_panel, targets, min_industry_n=min_industry_n
    )
    base = packed["base_features"]
    comb = packed["combined_features"]
    lab = packed["industry_labels"]

    # Raw fwd (Phase E Protocol A echo) for side-by-side
    regression = {
        "fwd_ret_10_base": _eval_regression(base, lab["fwd_ret_10"], seed=seed),
        "fwd_ret_10_comb": _eval_regression(comb, lab["fwd_ret_10"], seed=seed),
        "ind_resid_fwd_10_mean_base": _eval_regression(
            base, lab["ind_resid_fwd_10_mean"], seed=seed
        ),
        "ind_resid_fwd_10_mean_comb": _eval_regression(
            comb, lab["ind_resid_fwd_10_mean"], seed=seed
        ),
        "ind_resid_fwd_10_median_base": _eval_regression(
            base, lab["ind_resid_fwd_10_median"], seed=seed
        ),
        "ind_resid_fwd_10_median_comb": _eval_regression(
            comb, lab["ind_resid_fwd_10_median"], seed=seed
        ),
        "ind_resid_fwd_5_mean_base": _eval_regression(
            base, lab["ind_resid_fwd_5_mean"], seed=seed
        ),
        "ind_resid_fwd_5_mean_comb": _eval_regression(
            comb, lab["ind_resid_fwd_5_mean"], seed=seed
        ),
        "ind_resid_fwd_10_vol_scaled_base": _eval_regression(
            base, lab["ind_resid_fwd_10_vol_scaled"], seed=seed
        ),
        "ind_resid_fwd_10_vol_scaled_comb": _eval_regression(
            comb, lab["ind_resid_fwd_10_vol_scaled"], seed=seed
        ),
    }
    logistic = {
        "fwd_sign_up_base": _fit_eval_logistic(
            base, (lab["fwd_ret_10"] > 0).astype(np.int64), seed=seed
        ),
        "fwd_sign_up_comb": _fit_eval_logistic(
            comb, (lab["fwd_ret_10"] > 0).astype(np.int64), seed=seed
        ),
        "ind_resid_sign_up_mean_base": _fit_eval_logistic(
            base, lab["ind_resid_sign_up_mean"], seed=seed
        ),
        "ind_resid_sign_up_mean_comb": _fit_eval_logistic(
            comb, lab["ind_resid_sign_up_mean"], seed=seed
        ),
        "ind_resid_sign_up_median_base": _fit_eval_logistic(
            base, lab["ind_resid_sign_up_median"], seed=seed
        ),
        "ind_resid_sign_up_median_comb": _fit_eval_logistic(
            comb, lab["ind_resid_sign_up_median"], seed=seed
        ),
        "ind_resid_cs_top20_comb": _fit_eval_logistic(
            comb, lab["ind_resid_cs_top20"], seed=seed
        ),
        "ind_resid_cs_bot20_comb": _fit_eval_logistic(
            comb, lab["ind_resid_cs_bot20"], seed=seed
        ),
    }

    # Phase E Protocol A baselines
    phase_e_fwd_r2 = None
    phase_e_fwd_delta = None
    phase_e_verdict = None
    if phase_e_json and Path(phase_e_json).is_file():
        pe = json.loads(Path(phase_e_json).read_text(encoding="utf-8"))
        phase_e_verdict = pe.get("decision", {}).get("verdict")
        a_reg = pe.get("protocol_A_forward_return", {}).get("regression", {})
        a_clf = pe.get("protocol_A_forward_return", {}).get("logistic", {})
        phase_e_fwd_r2 = a_reg.get("fwd_ret_10_comb", {}).get("ridge_r2")
        phase_e_fwd_delta = a_clf.get("fwd_sign_up_comb", {}).get(
            "delta_model_minus_prior"
        )

    primary_r2 = regression["ind_resid_fwd_10_mean_comb"].get("ridge_r2", float("nan"))
    primary_delta = logistic["ind_resid_sign_up_mean_comb"].get(
        "delta_model_minus_prior", float("nan")
    )
    median_r2 = regression["ind_resid_fwd_10_median_comb"].get("ridge_r2", float("nan"))
    median_delta = logistic["ind_resid_sign_up_median_comb"].get(
        "delta_model_minus_prior", float("nan")
    )
    vol_r2 = regression["ind_resid_fwd_10_vol_scaled_comb"].get("ridge_r2", float("nan"))
    raw_r2 = regression["fwd_ret_10_comb"].get("ridge_r2", float("nan"))
    raw_delta = logistic["fwd_sign_up_comb"].get("delta_model_minus_prior", float("nan"))

    beats_abs_r2 = (
        isinstance(primary_r2, float) and primary_r2 == primary_r2 and primary_r2 >= DIR_R2_BAR
    )
    beats_abs_delta = (
        isinstance(primary_delta, float)
        and primary_delta == primary_delta
        and primary_delta <= LOGIT_DELTA_BAR
    )
    beats_e_r2 = (
        phase_e_fwd_r2 is not None
        and isinstance(primary_r2, float)
        and primary_r2 == primary_r2
        and primary_r2 >= float(phase_e_fwd_r2) + LIFT_R2_VS_PHASE_E
    )
    beats_e_delta = (
        phase_e_fwd_delta is not None
        and isinstance(primary_delta, float)
        and primary_delta == primary_delta
        and primary_delta <= float(phase_e_fwd_delta) + LIFT_DELTA_VS_PHASE_E
    )
    # "clear lift vs Phase E fwd_ret" — either R² or Δ bar above
    clear_lift_vs_e = bool(beats_e_r2 or beats_e_delta)
    lifts_enough = bool(beats_abs_r2 or beats_abs_delta or clear_lift_vs_e)

    notes: list[str] = []
    st = packed["industry_stats"]["fwd10_mean_demean"]
    notes.append(
        f"Industry field=`sector` (CSRC); n_sectors={st['n_unique_sectors']}; "
        f"industry-day n mean/median={st['industry_day_n_mean']:.2f}/"
        f"{st['industry_day_n_median']:.1f}; market-fallback rate="
        f"{st['fallback_rate']:.4f}."
    )
    notes.append(
        "Train panel not on disk locally; ablation uses validation panel only "
        f"(n={packed['n_samples']})."
    )
    if isinstance(primary_r2, float) and isinstance(raw_r2, float):
        notes.append(
            f"Mean-demean resid R²={primary_r2:.6f} vs raw fwd R²={raw_r2:.6f} "
            f"(ΔR²={primary_r2 - raw_r2:+.6f})."
        )
    if abs(float(primary_delta)) < 0.005:
        notes.append(
            "Industry-neutral sign Δ≈0 — demeaning does not unlock linear direction."
        )

    if lifts_enough:
        verdict = "industry_neutral__lifts_enough_for_next_step"
        recommend_next = (
            "行业中性前向收益过关：可做短 sidecar 标签重建 + 线性确认；"
            "仍不要全量 R2 八头重启 / Kaggle 长训。"
        )
        train_yes_no = "yes_short_sidecar_only"
    else:
        verdict = "industry_neutral__no_lift__do_not_train"
        recommend_next = (
            "行业中性 fwd 未过 Phase E 同门槛，也未相对 Phase E 原 fwd 明显抬升；"
            "不要据此开训。可结束廉价消融，或换成本感知/更长 horizon，而非再拧 demean。"
        )
        train_yes_no = "no"

    decision = {
        "lifts_enough": lifts_enough,
        "verdict": verdict,
        "recommend_next": recommend_next,
        "train_yes_no": train_yes_no,
        "diagnostic_notes": notes,
        "thresholds": {
            "dir_r2_bar": DIR_R2_BAR,
            "logit_delta_bar": LOGIT_DELTA_BAR,
            "lift_r2_vs_phase_e_fwd": LIFT_R2_VS_PHASE_E,
            "lift_delta_vs_phase_e_fwd": LIFT_DELTA_VS_PHASE_E,
            "primary_heads": (
                "ind_resid_fwd_10_mean ridge + ind_resid_sign_up_mean logistic (comb)"
            ),
        },
        "phase_e_fwd_ret_10_comb_r2": phase_e_fwd_r2,
        "phase_e_fwd_sign_up_comb_delta": phase_e_fwd_delta,
        "phase_e_verdict": phase_e_verdict,
        "raw_fwd_ret_10_comb_r2": float(raw_r2) if raw_r2 == raw_r2 else None,
        "raw_fwd_sign_up_comb_delta": float(raw_delta) if raw_delta == raw_delta else None,
        "primary_ridge_r2": float(primary_r2) if primary_r2 == primary_r2 else None,
        "primary_logit_delta": float(primary_delta)
        if primary_delta == primary_delta
        else None,
        "median_ridge_r2": float(median_r2) if median_r2 == median_r2 else None,
        "median_logit_delta": float(median_delta)
        if median_delta == median_delta
        else None,
        "vol_scaled_ridge_r2": float(vol_r2) if vol_r2 == vol_r2 else None,
        "beats_abs_r2_bar": beats_abs_r2,
        "beats_abs_delta_bar": beats_abs_delta,
        "beats_phase_e_r2_by_lift": beats_e_r2,
        "beats_phase_e_delta_by_lift": beats_e_delta,
        "clear_lift_vs_phase_e_fwd": clear_lift_vs_e,
        "rationale": (
            "Primary: industry-mean-demeaned fwd_ret_10 (comb). Pass if R²≥0.15 or "
            "sign Δ≤-0.04, or clear lift vs Phase E Protocol A fwd "
            f"(+{LIFT_R2_VS_PHASE_E} R² or Δ more negative by {abs(LIFT_DELTA_VS_PHASE_E)}). "
            "Median demean / vol-scaled / CS top-bot are diagnostics."
        ),
    }

    prevalence = {
        "ind_resid_sign_up_mean_rate": float(np.mean(lab["ind_resid_sign_up_mean"])),
        "ind_resid_sign_up_median_rate": float(np.mean(lab["ind_resid_sign_up_median"])),
        "ind_resid_fwd_10_mean_mean": float(np.mean(lab["ind_resid_fwd_10_mean"])),
        "ind_resid_fwd_10_mean_std": float(np.std(lab["ind_resid_fwd_10_mean"])),
        "fwd_ret_10_mean": float(np.mean(lab["fwd_ret_10"])),
        "fwd_ret_10_std": float(np.std(lab["fwd_ret_10"])),
    }

    return {
        "phase": "F_industry_neutral_fwd",
        "n_samples": packed["n_samples"],
        "base_dim": packed["base_dim"],
        "xsec_dim": packed["xsec_dim"],
        "xsec_cols": packed["xsec_cols"],
        "seed": seed,
        "definition": packed["definition"],
        "industry_stats": packed["industry_stats"],
        "label_prevalence": prevalence,
        "regression": regression,
        "logistic": logistic,
        "decision": decision,
    }


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    dec = payload["decision"]
    prev = payload["label_prevalence"]
    st = payload["industry_stats"]["fwd10_mean_demean"]
    reg = payload["regression"]
    clf = payload["logistic"]

    def _rrow(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | — | skipped |"
        return (
            f"| {name} | {d['naive_mae']:.6f} | {d['ridge_mae']:.6f} | "
            f"{d['ridge_r2']:.6f} | {'Y' if d['beats_naive_mae'] else 'N'} |"
        )

    def _lrow(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | skipped |"
        return (
            f"| {name} | {d['prior_log_loss']:.6f} | {d['model_log_loss']:.6f} | "
            f"{d['delta_model_minus_prior']:+.6f} |"
        )

    lines = [
        "# Kairos 行业中性前向收益消融（Phase F）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
    ]
    if dec["lifts_enough"]:
        lines.append(
            f"**行业中性标签效应量过关。** verdict=`{dec['verdict']}`；"
            f"train_yes_no=`{dec['train_yes_no']}`。"
        )
    else:
        lines.append(
            f"**行业中性 fwd 未达开训门槛。** verdict=`{dec['verdict']}`；"
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
        f"- 敏感性：`{payload['definition']['sensitivity_median']}`；"
        f"vol_scaled=`{payload['definition']['vol_scaled']}`",
        f"- 行业字段：`{payload['industry_stats']['sector_field']}` — "
        f"{payload['industry_stats']['sector_field_note']}",
        f"- 行业日样本：n_sectors=`{st['n_unique_sectors']}`，"
        f"industry-day n mean/median=`{st['industry_day_n_mean']:.2f}`/"
        f"`{st['industry_day_n_median']:.1f}`，"
        f"market-fallback rate=`{st['fallback_rate']:.4f}`",
        "- 特征：Phase D/E 同款 Base + 截面（看 comb）",
        "- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP",
        "",
        "## 门槛",
        "",
        f"- 绝对：方向 R²≥`{dec['thresholds']['dir_r2_bar']}` 或 sign Δ≤"
        f"`{dec['thresholds']['logit_delta_bar']}`（与 Phase E 相同）",
        f"- 相对 Phase E Protocol A fwd：R²+"
        f"`{dec['thresholds']['lift_r2_vs_phase_e_fwd']}` 或 Δ 再负"
        f"`{abs(dec['thresholds']['lift_delta_vs_phase_e_fwd'])}`",
        f"- Phase E fwd_ret_10_comb R²=`{dec['phase_e_fwd_ret_10_comb_r2']}`，"
        f"fwd_sign_up_comb Δ=`{dec['phase_e_fwd_sign_up_comb_delta']}`",
        "",
        "## 标签分布",
        "",
        f"- raw fwd_ret_10 mean/std=`{prev['fwd_ret_10_mean']:.4f}`/`{prev['fwd_ret_10_std']:.4f}`",
        f"- ind_resid_mean mean/std=`{prev['ind_resid_fwd_10_mean_mean']:.4f}`/"
        f"`{prev['ind_resid_fwd_10_mean_std']:.4f}`",
        f"- sign_up rate mean/median demean=`{prev['ind_resid_sign_up_mean_rate']:.4f}`/"
        f"`{prev['ind_resid_sign_up_median_rate']:.4f}`",
        "",
        "## 1) Ridge（连续残差）",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | beats |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for k, v in reg.items():
        lines.append(_rrow(k, v))
    lines += [
        "",
        "## 2) Logistic（符号）",
        "",
        "| Head | prior LL | model LL | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in clf.items():
        lines.append(_lrow(k, v))

    lines += [
        "",
        "## 3) 决策",
        "",
        f"| 指标 | 值 | 过关 |",
        f"| --- | ---: | :---: |",
        f"| 主方向 R² (mean demean comb) | {dec['primary_ridge_r2']} | "
        f"{'Y' if dec['beats_abs_r2_bar'] else 'N'} |",
        f"| 主方向 Δ (sign mean comb) | {dec['primary_logit_delta']} | "
        f"{'Y' if dec['beats_abs_delta_bar'] else 'N'} |",
        f"| vs Phase E R² lift | — | {'Y' if dec['beats_phase_e_r2_by_lift'] else 'N'} |",
        f"| vs Phase E Δ lift | — | {'Y' if dec['beats_phase_e_delta_by_lift'] else 'N'} |",
        f"| median demean R² / Δ | {dec['median_ridge_r2']} / {dec['median_logit_delta']} | diag |",
        f"| vol-scaled resid R² | {dec['vol_scaled_ridge_r2']} | diag |",
        f"| raw fwd R² / Δ (echo) | {dec['raw_fwd_ret_10_comb_r2']} / "
        f"{dec['raw_fwd_sign_up_comb_delta']} | — |",
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
            "1. **可**按行业中性 fwd 做短 sidecar 标签 + 线性确认。",
            "2. **不要**重启同配置 R2 八头 BCE 长训。",
            "3. 截面特征继续默认带上。",
        ]
    else:
        lines += [
            "1. **不要**基于行业中性 fwd 开标签重建长训。",
            "2. 廉价标签消融路径可收束；若再试需质变（成本/更长 horizon），非 demean 变体。",
            "3. **不要**重启同配置 R2。",
        ]
    lines += [
        "",
        "## 证据强度",
        "",
        "| 主张 | 强度 | 依据 |",
        "| --- | --- | --- |",
        f"| 行业中性可开下一步 | {'中' if dec['lifts_enough'] else '弱/否'} | R²/Δ 见表 |",
        "| 应维持停 R2 | 强 | 与 Phase C–E 一致 |",
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
        "--out-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_f_industry_neutral.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_f_industry_neutral_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-industry-n", type=int, default=2)
    args = parser.parse_args(argv)

    payload = run_phase_f(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        phase_e_json=args.phase_e_json,
        seed=args.seed,
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
                "industry_stats_fwd10": payload["industry_stats"]["fwd10_mean_demean"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
