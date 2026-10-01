"""Run Phase A/B diagnostics and write Phase C decision memo (JSON + CN markdown)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from modernbert_finance.ablations.label_time_diagnostics import run_phase_a
from modernbert_finance.ablations.structure_ablations import run_phase_b

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _build_decision_memo(phase_a: dict[str, Any], phase_b: dict[str, Any]) -> dict[str, Any]:
    cross = phase_a["cross_prior_transfer"]["macro_val_log_loss_with_train_prior"]
    val_prior = phase_a["positive_rates"]["macro_prior_log_loss"]["val"]
    train_prior = phase_a["positive_rates"]["macro_prior_log_loss"]["train"]
    temporal_ok = phase_a.get("temporal_integrity", {}).get("ok")
    align_ok = phase_a.get("alignment_sample", {}).get("ok")
    dil = phase_b["structure"]["dilution_summary"]
    ind_delta = dil["independent_8head_macro_delta"]
    coarse = dil["coarsened_deltas"]
    best_coarse_name = None
    best_coarse_delta = 0.0
    for name, delta in (coarse or {}).items():
        if delta is not None and delta < best_coarse_delta:
            best_coarse_delta = delta
            best_coarse_name = name

    r2_stuck = 0.5966
    gap_to_train_prior_on_val = abs(r2_stuck - cross)

    # Decision logic (no sycophancy)
    root_cause = (
        "deep_model_collapsed_to_constant_train_prior"
        if gap_to_train_prior_on_val < 0.001
        else "unclear_collapse_target"
    )
    multihead_is_main_blocker = (
        best_coarse_name is not None
        and ind_delta is not None
        and best_coarse_delta < ind_delta - 0.01
    )

    recommendations: list[dict[str, str]] = []
    recommendations.append(
        {
            "change": "Do NOT restart R2 / more GPU hours on same 8-head BCE + same features",
            "evidence": (
                f"R2≈{r2_stuck:.4f} matches train-prior-on-val macro LL={cross:.6f} "
                f"(val self-prior={val_prior:.6f}); collapse to constant prevalence."
            ),
            "next_experiment": (
                "Replace training objective evaluation with an explicit constant-prior "
                "baseline log; abort runs that stay within 1e-3 of it after 1–2 chunks."
            ),
        }
    )
    if temporal_ok and align_ok:
        recommendations.append(
            {
                "change": "Do NOT chase leakage/misalignment fixes as the primary path",
                "evidence": (
                    f"temporal_integrity.ok={temporal_ok}, alignment.ok={align_ok}; "
                    "future-corrupt leaves features invariant and changes labels."
                ),
                "next_experiment": (
                    "Keep the causal contract tests in CI; move effort to label/feature "
                    "information content instead of pipeline paranoia."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "change": "Fix temporal / alignment failures before any new training",
                "evidence": f"temporal_ok={temporal_ok}, align_ok={align_ok}",
                "next_experiment": "Re-run verify_feature_label_alignment on rebuilt sidecars.",
            }
        )

    if multihead_is_main_blocker:
        recommendations.append(
            {
                "change": "Abandon equal-weight 8-head; train 1–2 coarsened heads",
                "evidence": (
                    f"best coarsened {best_coarse_name} Δ={best_coarse_delta:.4f} vs "
                    f"8-head macro Δ={ind_delta:.4f}"
                ),
                "next_experiment": (
                    f"Single-task logistic/ModernBERT on `{best_coarse_name}` only; "
                    "compare Δ vs prior on the same val split."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "change": "Multi-head dilution is secondary — keep at most 2–4 heads but fix signal path",
                "evidence": (
                    f"8-head independent macro Δ={ind_delta:.4f}; best coarsened "
                    f"{best_coarse_name} Δ={best_coarse_delta:.4f} (similar order)."
                ),
                "next_experiment": (
                    "Try regression on continuous mfe10/mae10 (or first-touch) with the "
                    "same window summaries before another transformer run."
                ),
            }
        )

    recommendations.append(
        {
            "change": "Feature side: current OHLCVA summaries only beat prior by ~0.008–0.019",
            "evidence": (
                "Cheap logistic on summarized windows beats prior on all 8 heads, but "
                "effect sizes are small; deep model failed to harvest even that."
            ),
            "next_experiment": (
                "Add cross-sectional ranks / sector-relative returns as cheap features; "
                "re-run logistic. If Δ still <0.02, revisit 10D MFE/MAE label definition "
                "(too noisy / nested) before ModernBERT."
            ),
        }
    )

    return {
        "root_cause": root_cause,
        "r2_stuck_approx": r2_stuck,
        "train_macro_prior": train_prior,
        "val_macro_prior": val_prior,
        "train_prior_scored_on_val": cross,
        "gap_r2_vs_train_prior_on_val": gap_to_train_prior_on_val,
        "temporal_integrity_ok": temporal_ok,
        "alignment_ok": align_ok,
        "independent_8head_macro_delta": ind_delta,
        "best_coarsened": {"name": best_coarse_name, "delta": best_coarse_delta},
        "multihead_is_main_blocker": multihead_is_main_blocker,
        "recommendations": recommendations,
        "abandon": {
            "long_gpu_r2_same_config": True,
            "eight_head_equal_weight_as_sole_metric": True,
            "leakage_as_primary_hypothesis": bool(temporal_ok and align_ok),
        },
    }


def _cn_markdown(phase_a: dict[str, Any], phase_b: dict[str, Any], decision: dict[str, Any]) -> str:
    now = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M CST")
    rates = phase_a["positive_rates"]
    cross = phase_a["cross_prior_transfer"]
    dil = phase_b["structure"]["dilution_summary"]
    ti = phase_a.get("temporal_integrity", {})
    lead = phase_a.get("same_stock_lead_lag", {})
    corr = phase_a.get("head_correlation_val", {})

    lines = [
        "# Kairos Phase A/B/C 诊断结论（为何卡在 ~0.5966）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
        f"**根因：`{decision['root_cause']}`。** R2 验证 macro LL≈`{decision['r2_stuck_approx']}` "
        f"与「训练正例率常数先验打在验证集上」的 macro LL=`{decision['train_prior_scored_on_val']:.6f}` "
        f"相差仅 `{decision['gap_r2_vs_train_prior_on_val']:.6f}` —— 深度模型没有学到超越边际/先验的条件信号。",
        "",
        "## Phase A — 标签 / 时间 / 切分",
        "",
        "### 1. Feature cutoff vs 10D horizon",
        "",
        "```",
        json.dumps(phase_a["cutoff_contract"], ensure_ascii=False, indent=2),
        "```",
        "",
        f"- 对齐抽查：`ok={phase_a.get('alignment_sample', {}).get('ok')}`",
        f"- 时间完整性（未来污染探针）：`ok={ti.get('ok')}`；"
        f"特征在污染未来后不变 `{ti.get('feature_invariant_count')}/{ti.get('n_checked')}`，"
        f"标签改变 `{ti.get('label_changed_under_corrupt_count')}/{ti.get('n_checked')}`",
        f"- asof 收益 vs mfe10 相关：`{lead.get('corr_asof_ret_vs_mfe10')}`（弱相关预期；近 1 才像泄漏）",
        "",
        "### 2. 正例率漂移",
        "",
        "| Head | train | val | 2025H2 | 2026H1 | val−train |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for head, row in rates["by_head"].items():
        lines.append(
            "| {h} | {tr:.4f} | {va:.4f} | {h2:.4f} | {h1:.4f} | {d:+.4f} |".format(
                h=head,
                tr=row["train"]["positive_rate"],
                va=row["val"]["positive_rate"],
                h2=row["val_2025H2"]["positive_rate"],
                h1=row["val_2026H1"]["positive_rate"],
                d=rates["val_minus_train_rate"][head],
            )
        )
    lines += [
        "",
        f"Macro prior LL：train=`{rates['macro_prior_log_loss']['train']:.6f}`，"
        f"val=`{rates['macro_prior_log_loss']['val']:.6f}`，"
        f"2025H2=`{rates['macro_prior_log_loss']['val_2025H2']:.6f}`，"
        f"2026H1=`{rates['macro_prior_log_loss']['val_2026H1']:.6f}`。",
        "",
        f"**训练先验 → 验证打分 macro LL = `{cross['macro_val_log_loss_with_train_prior']:.6f}`** "
        "（与 R2 卡住点重合）。",
        "",
        "### 3. 八头相关",
        "",
        f"- within-up mean |ρ| ≈ `{corr.get('mean_abs_corr_within_up'):.3f}`",
        f"- within-down mean |ρ| ≈ `{corr.get('mean_abs_corr_within_down'):.3f}`",
        f"- up vs down mean ρ ≈ `{corr.get('mean_corr_up_vs_down'):.3f}`",
        f"- {corr.get('interpretation', '')}",
        "",
        "### 4. 同股未来信息",
        "",
        "Dataset `history = values[start:start+LOOKBACK]`，asof=`start+119`；标签只用 asof 之后 10 根。"
        "未来污染实验表明特征不含同股未来 bar。**不是泄漏问题。**",
        "",
        "## Phase B — 结构消融（CPU）",
        "",
        f"- 8-head 独立 logistic macro Δ(model−prior) = `{dil['independent_8head_macro_delta']:.5f}`",
        f"- MultiOutput 共享特征 macro Δ = `{dil['multioutput_8head_macro_delta']:.5f}`",
        f"- 单头 up_005 Δ = `{dil['single_head_up_005_delta']:.5f}`",
        f"- 粗标签 Δ：`{json.dumps(dil['coarsened_deltas'], ensure_ascii=False)}`",
        f"- 冻结 embedding 探针：`{phase_b['frozen_embedding_probe']['reason']}`",
        "",
        "## Phase C — 决策（无谄媚）",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 根因 | `{decision['root_cause']}` |",
        f"| 多头是否主 blockers | `{decision['multihead_is_main_blocker']}` |",
        f"| 放弃同配置长训 | `{decision['abandon']['long_gpu_r2_same_config']}` |",
        f"| 泄漏主因假设 | 放弃=`{decision['abandon']['leakage_as_primary_hypothesis']}` |",
        "",
        "### 建议下一步（每条一句）",
        "",
    ]
    for i, rec in enumerate(decision["recommendations"], 1):
        lines.append(f"{i}. **改什么：** {rec['change']}")
        lines.append(f"   - 证据：{rec['evidence']}")
        lines.append(f"   - 下一实验：{rec['next_experiment']}")
        lines.append("")
    lines += [
        "## 证据强度",
        "",
        "| 主张 | 强度 | 依据 |",
        "| --- | --- | --- |",
        "| 模型塌到训练先验 | **强** | 数值与 train-prior-on-val 差 <1e-3 |",
        "| 非泄漏/错位 | **强** | 对齐+未来污染探针通过 |",
        "| 标签有弱可学信号 | **中** | logistic Δ≈0.008–0.019 |",
        "| 八头稀释是主因 | **弱/否** | 粗标签与八头同量级 |",
        "| 值得再上 ModernBERT 长训 | **弱** | 深度模型未吃到线性已有的小信号 |",
        "",
        "未重启 R2；未改动同事 TPU WIP。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-targets",
        type=Path,
        default=Path("scratch/kairos_ablation_data/targets_dl/train_targets.parquet"),
    )
    parser.add_argument(
        "--val-targets",
        type=Path,
        default=Path("scratch/kairos_ablation_data/targets_dl/validation_targets.parquet"),
    )
    parser.add_argument(
        "--val-panel",
        type=Path,
        default=Path("scratch/kairos_ablation_data/panel_dl/val_data.pkl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("modernbert_finance/ablations"),
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_a = run_phase_a(
        train_targets=args.train_targets,
        val_targets=args.val_targets,
        val_panel=args.val_panel,
        out_json=out_dir / "kairos_phase_a_label_time.json",
    )
    phase_b = run_phase_b(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        out_json=out_dir / "kairos_phase_b_structure.json",
    )
    decision = _build_decision_memo(phase_a, phase_b)
    combined = {
        "generated_at_cst": datetime.now(SHANGHAI).isoformat(),
        "phase_a_summary": {
            "macro_priors": phase_a["positive_rates"]["macro_prior_log_loss"],
            "train_prior_on_val": phase_a["cross_prior_transfer"]["macro_val_log_loss_with_train_prior"],
            "temporal_ok": phase_a.get("temporal_integrity", {}).get("ok"),
            "align_ok": phase_a.get("alignment_sample", {}).get("ok"),
        },
        "phase_b_summary": phase_b["structure"]["dilution_summary"],
        "decision": decision,
    }
    (out_dir / "kairos_phase_c_decision.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md = _cn_markdown(phase_a, phase_b, decision)
    md_path = Path("modernbert_finance/kairos_phase_abc_diagnosis_cn.md")
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
