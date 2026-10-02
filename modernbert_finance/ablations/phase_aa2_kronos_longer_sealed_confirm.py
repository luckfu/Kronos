#!/usr/bin/env python3
"""Phase AA2: longer sealed confirm of a-priori AA winner (no re-pick K/stop).

Protocol (anti-peek)
--------------------
- Freeze rules declared a priori from Phase AA short-OOS selection:
    1) topk50_tp10_nostop   — AA focus / winner
    2) topk50_hold_d10      — AA cheap-tweak hold variant
    3) topk20_tp10_nostop   — same exit family, K=20
- Do NOT re-sweep K / stop / threshold on confirm panels.
- Confirm panels use Z2 Kairos-val C2@179 prod scores (2025-07..2026-07):
    * z2_late_half  — chronological second half (held-out later segment)
    * z2_walkforward — 4×~60d non-overlapping folds; majority-fold gate
    * z2_late_stride10 — late half with stride=10 sleeves (Sharpe honesty)
- short_oos_18d is SELECTION reference only — never drives AA2 verdict.
- Z2 scores sit inside C2 train window → IC may be optimistic; confirm is
  longer-window stress of frozen *rule* mechanics after costs, not a clean
  OOS generalization claim for the model.

Same execution model as Phase AA: T+1 close entry, 涨停 skip ≥9.5%,
TP10 / optional stop / time-d10, 30 bps/side.

No train / no decode change / no TPU WIP / no mfe10 revive.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from modernbert_finance.ablations.phase_aa_kronos_cost_aware_topk_backtest import (
    COST_PER_SIDE_DEFAULT,
    LIMIT_UP,
    RNG_SEED,
    SCORE_COL,
    TRADING_DAYS,
    buy_hold_equal_weight,
    load_panel,
    run_rule,
    select_random_k,
    select_top_k,
)

ROOT = Path("/workspace/Kronos")
OUT_DIR = ROOT / "scratch/kairos_phase_aa2_longer_confirm_outputs"
RESULTS_JSON = (
    ROOT
    / "modernbert_finance/ablations/kairos_phase_aa2_kronos_longer_sealed_confirm_results.json"
)
MEMO_CN = ROOT / "modernbert_finance/kairos_phase_aa2_kronos_longer_sealed_confirm_cn.md"
RESULTS_MEMO_CN = (
    ROOT / "modernbert_finance/kairos_phase_aa2_kronos_longer_sealed_confirm_results_cn.md"
)

Z2_PATH = ROOT / "scratch/kairos_phase_z2_outputs/predictions_prod_t065_p80_n5.csv.gz"
SHORT_OOS_PATH = Path(
    "/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/"
    "predictions_prod_t065_p80_n5.csv.gz"
)
if not SHORT_OOS_PATH.exists():
    SHORT_OOS_PATH = (
        ROOT
        / "finetune/kaggle_c2_top20_consensus_oos/predictions_prod_t065_p80_n5.csv.gz"
    )

# A-priori frozen from Phase AA — DO NOT edit after seeing confirm numbers.
FOCUS_RULE = "topk50_tp10_nostop"
A_PRIORI_RULES: list[tuple[str, int, float | None, float | None]] = [
    ("topk50_tp10_nostop", 50, 0.10, None),
    ("topk50_hold_d10", 50, None, None),
    ("topk20_tp10_nostop", 20, 0.10, None),
]
CONTROL_RULES: list[tuple[str, int, float | None, float | None]] = [
    ("random50_tp10_nostop", 50, 0.10, None),
    ("random50_hold_d10", 50, None, None),
    ("random20_tp10_nostop", 20, 0.10, None),
]
FOLD_SIZE = 60
MIN_FOLD_DATES = 20


def filter_dates(frame: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    ds = set(dates)
    return frame.loc[frame["asof_date"].isin(ds)].reset_index(drop=True)


def stride_dates(dates: list[str], stride: int) -> list[str]:
    return list(dates[:: max(1, int(stride))])


def make_picker(label: str, k: int, rng: np.random.Generator) -> Callable:
    if label.startswith("random"):
        return lambda day, k=k, rng=rng: select_random_k(day, k, rng)
    return lambda day, k=k: select_top_k(day, k)


def eval_frozen_panel(
    frame: pd.DataFrame,
    cost_per_side: float,
    panel_name: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dates = sorted(frame["asof_date"].unique())
    meta = {
        "panel": panel_name,
        "n_rows": int(len(frame)),
        "n_dates": int(len(dates)),
        "n_symbols": int(frame["symbol"].nunique()),
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "score_col": SCORE_COL,
        "spearman_pred_vs_return10d": float(
            frame[SCORE_COL].corr(frame["return_10d"], method="spearman")
        )
        if len(frame) > 2
        else float("nan"),
        "mean_universe": float(frame.groupby("asof_date").size().mean())
        if len(frame)
        else float("nan"),
        "contaminated_vs_c2_train": True,
        "rule_selection": "a_priori_frozen_from_AA_no_repick",
    }
    if extra_meta:
        meta.update(extra_meta)

    rng = np.random.default_rng(RNG_SEED)
    rule_metrics: list[dict[str, Any]] = []
    for label, k, tp, sl in A_PRIORI_RULES + CONTROL_RULES:
        picker = make_picker(label, k, rng)
        m = run_rule(frame, picker, tp, sl, cost_per_side, label)
        rule_metrics.append(m)

    bh = buy_hold_equal_weight(frame, cost_per_side)
    by_name = {m["rule"]: m for m in rule_metrics}
    focus = by_name.get(FOCUS_RULE, {})
    rand_match = by_name.get("random50_tp10_nostop", {})
    focus_ok = (
        focus.get("status") == "ok"
        and focus.get("mean_net_return") is not None
        and focus["mean_net_return"] > 0
        and (
            rand_match.get("mean_net_return") is None
            or focus["mean_net_return"] > rand_match["mean_net_return"]
        )
        and (
            bh.get("status") != "ok"
            or focus["mean_net_return"] > bh["mean_net_return"]
        )
    )
    return {
        "meta": meta,
        "buy_hold": bh,
        "rules": rule_metrics,
        "focus_rule": FOCUS_RULE,
        "focus_metrics": focus,
        "focus_beats_controls": bool(focus_ok),
        "a_priori_variants": {
            label: by_name.get(label)
            for label, *_ in A_PRIORI_RULES
            if label != FOCUS_RULE
        },
    }


def build_z2_splits(frame: pd.DataFrame) -> dict[str, Any]:
    dates = sorted(frame["asof_date"].unique())
    n = len(dates)
    mid = n // 2
    early = dates[:mid]
    late = dates[mid:]
    folds = []
    for i, start in enumerate(range(0, n, FOLD_SIZE)):
        chunk = dates[start : start + FOLD_SIZE]
        if len(chunk) < MIN_FOLD_DATES:
            continue
        folds.append({"fold": i, "dates": chunk})
    return {
        "all_dates": dates,
        "early_dates": early,
        "late_dates": late,
        "folds": folds,
        "late_stride10_dates": stride_dates(late, 10),
    }


def hard_conclusion(results: dict[str, Any]) -> dict[str, Any]:
    late = results["panels"].get("z2_late_half", {})
    wf = results["walkforward"]
    focus_late = late.get("focus_metrics") or {}
    late_net = focus_late.get("mean_net_return")
    late_ok = bool(late.get("focus_beats_controls"))

    fold_nets = [
        f["focus_metrics"].get("mean_net_return")
        for f in wf.get("folds", [])
        if f.get("focus_metrics", {}).get("status") == "ok"
    ]
    fold_pos = [x for x in fold_nets if x is not None and x > 0]
    n_folds = len(fold_nets)
    n_pos = len(fold_pos)
    majority = n_folds > 0 and n_pos >= math.ceil(n_folds * 0.5 + 1e-9) and n_pos >= 2
    # stricter: require >=3/4 when 4 folds
    if n_folds >= 4:
        majority = n_pos >= 3

    variants_late = late.get("a_priori_variants") or {}
    variant_summary = {
        k: (v or {}).get("mean_net_return") for k, v in variants_late.items()
    }

    sel = results["panels"].get("short_oos_18d_selection_ref", {})
    sel_focus = (sel.get("focus_metrics") or {}).get("mean_net_return")

    if late_ok and majority and late_net is not None and late_net > 0:
        decision = "PASS_PROPOSE_PAPER_TRADE"
        reason = (
            f"A-priori `{FOCUS_RULE}` stays positive after 30bps/side on Z2 late-half "
            f"(mean_net={late_net:.6f}, beats random50+EW) and is net>0 on "
            f"{n_pos}/{n_folds} walk-forward folds. Propose cheap paper-trade / "
            f"live shadow next — no deep train."
        )
        reason_cn = (
            f"先验规则 `{FOCUS_RULE}` 在 Z2 后半段成本后净期望仍为正"
            f"（mean_net={late_net:.6f}，优于 random50+等权），且 walk-forward "
            f"{n_pos}/{n_folds} 折净期望>0。建议廉价纸面交易/影子盘下一步——不深训。"
        )
        next_step = [
            "Paper-trade / shadow: daily Top50 by predicted_return_10d, TP10 nostop, T+1, limit-skip, 30bps/side book",
            "Track live vs sealed metrics weekly; kill if 20d rolling mean_net < 0 and trailing MDD worse than confirm",
            "Optional cheap: stride=5/10 sleeves to cut turnover; cost stress 20/40/60 bps",
            "Do NOT deep-train / touch TPU WIP / revive mfe10",
        ]
    else:
        decision = "HARD_CONCLUDE_RETUNE_NOT_TRAIN"
        reason = (
            f"Longer confirm FAILED for a-priori `{FOCUS_RULE}`: late_ok={late_ok}, "
            f"late_mean_net={late_net}, walkforward_pos={n_pos}/{n_folds}. "
            f"Rule needs retune (K/hold/cost/stride) — not deep train / not TPU / not mfe10."
        )
        reason_cn = (
            f"更长确认失败：先验 `{FOCUS_RULE}` late_ok={late_ok}，"
            f"late_mean_net={late_net}，walk-forward 正折 {n_pos}/{n_folds}。"
            f"规则需廉价重调（K/持有/成本/stride），禁止深训 / TPU / 复活 mfe10。"
        )
        next_step = [
            "Retune only: K∈{20,50}, hold_d10 vs tp10_nostop, stride sleeves — still on existing scores",
            "Do NOT open deep train / ModernBERT long train / TPU WIP / mfe10 binary revive",
            "If retune also fails sealed confirm → HARD STOP tradable-rule axis on C2 scores",
        ]

    return {
        "decision": decision,
        "reason": reason,
        "reason_cn": reason_cn,
        "focus_rule": FOCUS_RULE,
        "a_priori_rules": [r[0] for r in A_PRIORI_RULES],
        "cost_per_side": results["cost_per_side"],
        "z2_late_half_mean_net": late_net,
        "z2_late_half_sharpe_ann": focus_late.get("sharpe_ann_net"),
        "z2_late_half_max_dd": focus_late.get("max_drawdown_net"),
        "z2_late_half_hit_rate_net": focus_late.get("hit_rate_net"),
        "z2_late_half_turnover": focus_late.get("mean_two_way_name_turnover"),
        "z2_late_half_n_trades": focus_late.get("n_trades"),
        "z2_late_half_n_dates": (late.get("meta") or {}).get("n_dates"),
        "z2_late_half_beats_controls": late_ok,
        "z2_late_half_random50_mean_net": next(
            (
                m.get("mean_net_return")
                for m in late.get("rules", [])
                if m.get("rule") == "random50_tp10_nostop"
            ),
            None,
        ),
        "z2_late_half_buyhold_mean_net": (late.get("buy_hold") or {}).get(
            "mean_net_return"
        ),
        "walkforward_fold_mean_nets": fold_nets,
        "walkforward_n_positive": n_pos,
        "walkforward_n_folds": n_folds,
        "walkforward_majority_positive": majority,
        "a_priori_variant_late_mean_nets": variant_summary,
        "short_oos_18d_selection_ref_mean_net": sel_focus,
        "contamination_caveat": (
            "Z2 Kairos-val dates overlap C2 train window; score IC may be inflated. "
            "Confirm tests frozen rule after costs on longer window — not clean model OOS."
        ),
        "selection_caveat": (
            "short_oos_18d (18d) selected the AA winner; AA2 verdict uses only Z2 "
            "late-half + walk-forward with frozen K/stop — no re-pick."
        ),
        "next_step": next_step,
    }


def write_memo_cn(results: dict[str, Any], conclusion: dict[str, Any]) -> str:
    late = results["panels"]["z2_late_half"]
    wf = results["walkforward"]
    f = late.get("focus_metrics") or {}
    bh = late.get("buy_hold") or {}
    rand = next(
        (m for m in late.get("rules", []) if m.get("rule") == "random50_tp10_nostop"),
        {},
    )
    stride = results["panels"].get("z2_late_stride10", {})
    sf = stride.get("focus_metrics") or {}
    lines = [
        "# Kairos Phase AA2：先验 AA 赢家更长密封确认（不重选 K/止损）",
        "",
        f"日期：{results['generated_at_cst']}（北京时间）。",
        "范围：冻结 Phase AA 短 OOS 选出的先验规则，在 Z2 Kairos-val "
        "（C2 Seg@179 prod T=0.65/top_p=0.8/N=5）上做更长确认；"
        "**禁止**重扫 K/止损/阈值；**禁止**训练 / 改解码 / 碰 TPU WIP / 复活 mfe10。",
        "",
        "## 一句话结论（硬结论）",
        "",
        f"**{conclusion['decision']}** — {conclusion.get('reason_cn') or conclusion['reason']}",
        "",
        f"- 焦点先验规则 = `{FOCUS_RULE}`",
        f"- Z2 后半段 mean_net = **{f.get('mean_net_return')}**；"
        f"Sharpe≈**{f.get('sharpe_ann_net')}**；MDD≈**{f.get('max_drawdown_net')}**；"
        f"命中≈**{f.get('hit_rate_net')}**；换手≈**{f.get('mean_two_way_name_turnover')}**；"
        f"n_trades={f.get('n_trades')} / n_dates={late['meta'].get('n_dates')}",
        f"- 对照 random50_tp10_nostop mean_net = **{rand.get('mean_net_return')}**",
        f"- 对照等权买持 mean_net = **{bh.get('mean_net_return')}**",
        f"- Walk-forward 正折 = **{conclusion['walkforward_n_positive']}/"
        f"{conclusion['walkforward_n_folds']}**；折净期望 = "
        f"{conclusion['walkforward_fold_mean_nets']}",
        f"- 后半 stride10 袖 mean_net = **{sf.get('mean_net_return')}** "
        f"(Sharpe≈{sf.get('sharpe_ann_net')}) — 降重叠 vintage 膨胀",
        f"- 污染警告：{conclusion['contamination_caveat']}",
        f"- 选择警告：{conclusion['selection_caveat']}",
        "",
        "## 协议（防偷看）",
        "",
        "| 项 | 设定 |",
        "| --- | --- |",
        f"| 先验规则 | {', '.join(r[0] for r in A_PRIORI_RULES)} |",
        "| 重选 K/止损 | **否** |",
        "| 主确认窗 | Z2 后半段（时间后半） |",
        "| 辅确认 | Z2 walk-forward 4×~60d；多数折净期望>0 |",
        "| 选择窗 | short_oos_18d 仅作参照，不进 AA2 判决 |",
        "| 成本 | 单边 30bps；双边往返 2× |",
        "| 执行 | T+1 收盘近似；涨停≥9.5% 跳过；TP10 / 到期 d10 |",
        "",
        "## Z2 后半段硬指标（焦点）",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| rule | {FOCUS_RULE} |",
        f"| date_range | {late['meta'].get('date_min')}..{late['meta'].get('date_max')} |",
        f"| n_dates | {late['meta'].get('n_dates')} |",
        f"| n_trades | {f.get('n_trades')} |",
        f"| mean_net_return | {f.get('mean_net_return')} |",
        f"| mean_gross_return | {f.get('mean_gross_return')} |",
        f"| hit_rate_net | {f.get('hit_rate_net')} |",
        f"| sharpe_ann_net | {f.get('sharpe_ann_net')} |",
        f"| max_drawdown_net | {f.get('max_drawdown_net')} |",
        f"| compounded_vintage_return_net | {f.get('compounded_vintage_return_net')} |",
        f"| mean_two_way_name_turnover | {f.get('mean_two_way_name_turnover')} |",
        f"| tp_hit_rate | {f.get('tp_hit_rate')} |",
        f"| beats_random_and_ew | {late.get('focus_beats_controls')} |",
        "",
        "## Walk-forward 折",
        "",
        "| fold | dates | mean_net | sharpe | MDD | hit | beats |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for fold in wf.get("folds", []):
        fm = fold.get("focus_metrics") or {}
        lines.append(
            f"| {fold['fold']} | {fold['date_min']}..{fold['date_max']} "
            f"({fold['n_dates']}d) | {fm.get('mean_net_return')} | "
            f"{fm.get('sharpe_ann_net')} | {fm.get('max_drawdown_net')} | "
            f"{fm.get('hit_rate_net')} | {fold.get('focus_beats_controls')} |"
        )
    lines += [
        "",
        "## 先验变体（后半段，不重选）",
        "",
    ]
    for label, metrics in (late.get("a_priori_variants") or {}).items():
        m = metrics or {}
        lines.append(
            f"- `{label}` mean_net={m.get('mean_net_return')}；"
            f"Sharpe≈{m.get('sharpe_ann_net')}；MDD≈{m.get('max_drawdown_net')}；"
            f"hit≈{m.get('hit_rate_net')}"
        )
    lines += [
        "",
        "## 下一步",
        "",
    ]
    for i, s in enumerate(conclusion["next_step"], 1):
        lines.append(f"{i}. {s}")
    lines += [
        "",
        "## 产物",
        "",
        "- 脚本：`modernbert_finance/ablations/phase_aa2_kronos_longer_sealed_confirm.py`",
        "- JSON：`modernbert_finance/ablations/kairos_phase_aa2_kronos_longer_sealed_confirm_results.json`",
        "- 备忘：`modernbert_finance/kairos_phase_aa2_kronos_longer_sealed_confirm_cn.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE_DEFAULT)
    parser.add_argument("--out-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--out-memo", type=Path, default=MEMO_CN)
    parser.add_argument("--out-results-memo", type=Path, default=RESULTS_MEMO_CN)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not Z2_PATH.exists():
        raise SystemExit(f"missing Z2 scores: {Z2_PATH}")

    z2 = load_panel(Z2_PATH)
    splits = build_z2_splits(z2)

    panels: dict[str, Any] = {}
    panels["z2_late_half"] = eval_frozen_panel(
        filter_dates(z2, splits["late_dates"]),
        args.cost_per_side,
        "z2_late_half",
        {
            "role": "primary_confirm_held_out_later_segment",
            "n_dates_full_z2": len(splits["all_dates"]),
        },
    )
    panels["z2_early_half_shape_only"] = eval_frozen_panel(
        filter_dates(z2, splits["early_dates"]),
        args.cost_per_side,
        "z2_early_half_shape_only",
        {"role": "shape_only_not_for_verdict"},
    )
    panels["z2_late_stride10"] = eval_frozen_panel(
        filter_dates(z2, splits["late_stride10_dates"]),
        args.cost_per_side,
        "z2_late_stride10",
        {
            "role": "turnover_honest_sleeve_confirm",
            "stride": 10,
            "note": "every 10th late signal date to cut overlapping-vintage Sharpe inflation",
        },
    )
    panels["z2_full_frozen_stress"] = eval_frozen_panel(
        z2,
        args.cost_per_side,
        "z2_full_frozen_stress",
        {
            "role": "full_window_stress_not_primary_verdict",
            "note": "AA already saw aggregate Z2; reported for completeness only",
        },
    )

    if SHORT_OOS_PATH.exists():
        oos = load_panel(SHORT_OOS_PATH)
        panels["short_oos_18d_selection_ref"] = eval_frozen_panel(
            oos,
            args.cost_per_side,
            "short_oos_18d_selection_ref",
            {
                "role": "selection_reference_only_not_for_aa2_verdict",
                "contaminated_vs_c2_train": False,
            },
        )
        panels["short_oos_18d_selection_ref"]["meta"]["contaminated_vs_c2_train"] = False

    wf_folds = []
    for fold in splits["folds"]:
        sub = filter_dates(z2, fold["dates"])
        ev = eval_frozen_panel(
            sub,
            args.cost_per_side,
            f"z2_wf_fold{fold['fold']}",
            {"role": "walkforward_fold", "fold": fold["fold"]},
        )
        wf_folds.append(
            {
                "fold": fold["fold"],
                "date_min": fold["dates"][0],
                "date_max": fold["dates"][-1],
                "n_dates": len(fold["dates"]),
                "focus_metrics": ev.get("focus_metrics"),
                "focus_beats_controls": ev.get("focus_beats_controls"),
                "buy_hold_mean_net": (ev.get("buy_hold") or {}).get("mean_net_return"),
                "a_priori_variants": {
                    k: (v or {}).get("mean_net_return")
                    for k, v in (ev.get("a_priori_variants") or {}).items()
                },
            }
        )

    results = {
        "phase": "AA2",
        "purpose": "longer_sealed_confirm_a_priori_aa_winner_no_repick",
        "not_mfe10_decision_product": True,
        "training_performed": False,
        "decode_changed": False,
        "tpu_wip_touched": False,
        "k_or_stop_repicked": False,
        "focus_rule_a_priori": FOCUS_RULE,
        "a_priori_rules": [
            {
                "rule": label,
                "k": k,
                "take_profit": tp,
                "stop_loss": sl,
            }
            for label, k, tp, sl in A_PRIORI_RULES
        ],
        "cost_per_side": args.cost_per_side,
        "limit_up_threshold": LIMIT_UP,
        "execution": {
            "entry": "T+1 close proxy",
            "path": "relative to actual_return_d1",
            "exits": ["take_profit_10pct", "stop_loss_optional", "time_d10"],
            "cost": "2 * cost_per_side per completed trade",
        },
        "score_source": {
            "checkpoint": "C2 Best Seg@179",
            "decode": "prod T=0.65 top_p=0.8 N=5",
            "z2_path": str(Z2_PATH),
            "short_oos_path": str(SHORT_OOS_PATH),
        },
        "splits": {
            "z2_n_dates": len(splits["all_dates"]),
            "z2_date_min": splits["all_dates"][0],
            "z2_date_max": splits["all_dates"][-1],
            "early_n": len(splits["early_dates"]),
            "late_n": len(splits["late_dates"]),
            "late_date_min": splits["late_dates"][0],
            "late_date_max": splits["late_dates"][-1],
            "n_walkforward_folds": len(splits["folds"]),
            "fold_size": FOLD_SIZE,
        },
        "panels": panels,
        "walkforward": {"folds": wf_folds},
        "elapsed_sec": time.time() - t0,
        "generated_at_cst": time.strftime("%Y-%m-%d %H:%M:%S CST", time.localtime()),
    }
    conclusion = hard_conclusion(results)
    results["hard_conclusion"] = conclusion

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    memo = write_memo_cn(results, conclusion)
    args.out_memo.write_text(memo)
    args.out_results_memo.write_text(memo)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "hard_conclusion": conclusion,
                "z2_late_focus": panels["z2_late_half"].get("focus_metrics"),
                "walkforward": wf_folds,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_memo}")


if __name__ == "__main__":
    main()
