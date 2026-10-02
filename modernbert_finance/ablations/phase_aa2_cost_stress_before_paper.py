#!/usr/bin/env python3
"""Phase AA2 cost stress: 20/30/40/60 bps/side before paper/shadow.

Frozen a-priori rule from AA2: topk50_tp10_nostop (no K/stop re-pick).
Confirm windows: Z2 late-half (primary) + walk-forward folds (majority gate).
Also reports a-priori variants (topk50_hold_d10, topk20_tp10_nostop) cheaply.

Cost convention (same as AA/AA2):
  - cost_per_side = ONE-WAY fees+slip approx
  - round-trip charged = 2 * cost_per_side per completed trade
  - AA/AA2 baseline = 30 bps/side (= 60 bps round-trip)

Same execution: T+1 close entry, 涨停≥9.5% skip, TP10 nostop / time-d10.
No train / no decode change / no TPU WIP / no mfe10 revive.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from modernbert_finance.ablations.phase_aa_kronos_cost_aware_topk_backtest import (
    LIMIT_UP,
    load_panel,
)
from modernbert_finance.ablations.phase_aa2_kronos_longer_sealed_confirm import (
    A_PRIORI_RULES,
    FOCUS_RULE,
    FOLD_SIZE,
    Z2_PATH,
    build_z2_splits,
    eval_frozen_panel,
    filter_dates,
)

ROOT = Path("/workspace/Kronos")
OUT_DIR = ROOT / "scratch/kairos_phase_aa2_cost_stress_outputs"
RESULTS_JSON = (
    ROOT
    / "modernbert_finance/ablations/kairos_phase_aa2_cost_stress_before_paper_results.json"
)
MEMO_CN = ROOT / "modernbert_finance/kairos_phase_aa2_cost_stress_before_paper_cn.md"
RESULTS_MEMO_CN = (
    ROOT / "modernbert_finance/kairos_phase_aa2_cost_stress_before_paper_results_cn.md"
)

# User chose: 20/40/60 + baseline 30 (AA used 30bps/side)
COST_BPS_LEVELS = (20, 30, 40, 60)


def _bps_to_side(bps: int) -> float:
    return bps / 10000.0


def _rule_row(m: dict[str, Any] | None) -> dict[str, Any]:
    m = m or {}
    return {
        "status": m.get("status"),
        "n_trades": m.get("n_trades"),
        "mean_net_return": m.get("mean_net_return"),
        "mean_gross_return": m.get("mean_gross_return"),
        "sharpe_ann_net": m.get("sharpe_ann_net"),
        "max_drawdown_net": m.get("max_drawdown_net"),
        "hit_rate_net": m.get("hit_rate_net"),
        "compounded_vintage_return_net": m.get("compounded_vintage_return_net"),
        "mean_two_way_name_turnover": m.get("mean_two_way_name_turnover"),
        "tp_hit_rate": m.get("tp_hit_rate"),
        "n_signal_dates": m.get("n_signal_dates"),
        "n_skipped_limit": m.get("n_skipped_limit"),
        "cost_per_side": m.get("cost_per_side"),
    }


def eval_at_cost(z2, splits: dict[str, Any], cost_per_side: float) -> dict[str, Any]:
    late = eval_frozen_panel(
        filter_dates(z2, splits["late_dates"]),
        cost_per_side,
        "z2_late_half",
        {"role": "primary_confirm_cost_stress"},
    )
    by_name = {m["rule"]: m for m in late.get("rules", [])}
    focus = by_name.get(FOCUS_RULE, {})
    rand = by_name.get("random50_tp10_nostop", {})
    bh = late.get("buy_hold") or {}

    wf_folds = []
    for fold in splits["folds"]:
        sub = filter_dates(z2, fold["dates"])
        ev = eval_frozen_panel(
            sub,
            cost_per_side,
            f"z2_wf_fold{fold['fold']}",
            {"role": "walkforward_fold", "fold": fold["fold"]},
        )
        fm = ev.get("focus_metrics") or {}
        wf_folds.append(
            {
                "fold": fold["fold"],
                "date_min": fold["dates"][0],
                "date_max": fold["dates"][-1],
                "n_dates": len(fold["dates"]),
                "focus_mean_net": fm.get("mean_net_return"),
                "focus_sharpe_ann": fm.get("sharpe_ann_net"),
                "focus_max_dd": fm.get("max_drawdown_net"),
                "focus_hit_rate_net": fm.get("hit_rate_net"),
                "focus_beats_controls": ev.get("focus_beats_controls"),
                "buy_hold_mean_net": (ev.get("buy_hold") or {}).get("mean_net_return"),
                "a_priori_variants_mean_net": {
                    k: (v or {}).get("mean_net_return")
                    for k, v in (ev.get("a_priori_variants") or {}).items()
                },
            }
        )

    fold_nets = [
        f["focus_mean_net"]
        for f in wf_folds
        if f.get("focus_mean_net") is not None
    ]
    n_folds = len(fold_nets)
    n_pos = sum(1 for x in fold_nets if x > 0)
    majority = n_folds > 0 and n_pos >= math.ceil(n_folds * 0.5 + 1e-9) and n_pos >= 2
    if n_folds >= 4:
        majority = n_pos >= 3

    late_ok = bool(late.get("focus_beats_controls"))
    late_net = focus.get("mean_net_return")
    pass_gate = (
        late_ok
        and majority
        and late_net is not None
        and late_net > 0
    )

    variants = {}
    for label, *_ in A_PRIORI_RULES:
        if label == FOCUS_RULE:
            continue
        variants[label] = _rule_row(by_name.get(label))

    return {
        "cost_per_side": cost_per_side,
        "cost_bps_per_side": int(round(cost_per_side * 10000)),
        "cost_bps_round_trip": int(round(cost_per_side * 10000 * 2)),
        "cost_convention": "one_way_per_side; round_trip=2*side charged per trade",
        "z2_late_half": {
            "meta": late.get("meta"),
            "focus": _rule_row(focus),
            "random50_tp10_nostop": _rule_row(rand),
            "buy_hold_ew": {
                "mean_net_return": bh.get("mean_net_return"),
                "sharpe_ann_net": bh.get("sharpe_ann_net"),
                "max_drawdown_net": bh.get("max_drawdown_net"),
                "hit_rate_net": bh.get("hit_rate_net"),
                "status": bh.get("status"),
            },
            "focus_beats_random_and_ew": late_ok,
            "a_priori_variants": variants,
        },
        "walkforward": {
            "folds": wf_folds,
            "fold_mean_nets": fold_nets,
            "n_positive": n_pos,
            "n_folds": n_folds,
            "majority_positive": majority,
        },
        "pass_late_and_wf_gate": pass_gate,
    }


def hard_conclusion(results: dict[str, Any]) -> dict[str, Any]:
    levels = results["levels"]
    still_pass = []
    still_positive_late = []
    fail_first = None
    for bps in COST_BPS_LEVELS:
        key = f"bps_{bps}"
        lv = levels[key]
        late_net = lv["z2_late_half"]["focus"]["mean_net_return"]
        beats = lv["z2_late_half"]["focus_beats_random_and_ew"]
        if late_net is not None and late_net > 0:
            still_positive_late.append(bps)
        if lv["pass_late_and_wf_gate"]:
            still_pass.append(bps)
        elif fail_first is None:
            fail_first = bps

    # Recommend paper if baseline 30 and at least 40 still pass; kill if even 20 fails
    # Conservative: paper if 30 AND 40 both pass gate; soft paper if 30 pass but 40 fails late>0;
    # kill if 30 fails gate.
    p30 = levels["bps_30"]["pass_late_and_wf_gate"]
    p40 = levels["bps_40"]["pass_late_and_wf_gate"]
    p20 = levels["bps_20"]["pass_late_and_wf_gate"]
    p60 = levels["bps_60"]["pass_late_and_wf_gate"]
    late30 = levels["bps_30"]["z2_late_half"]["focus"]["mean_net_return"]
    late40 = levels["bps_40"]["z2_late_half"]["focus"]["mean_net_return"]
    late60 = levels["bps_60"]["z2_late_half"]["focus"]["mean_net_return"]

    if not p30:
        decision = "KILL_NO_PAPER"
        reason = (
            f"Frozen `{FOCUS_RULE}` fails AA2 late+WF gate at baseline 30bps/side "
            f"(late_mean_net={late30}). Do not paper; retune or HARD STOP axis."
        )
        reason_cn = (
            f"先验 `{FOCUS_RULE}` 在基准 30bps/单边 已挂 AA2 后半+WF 门"
            f"（late_mean_net={late30}）。不纸面；重调或 HARD STOP。"
        )
        recommend = "kill"
    elif p40 and p60:
        decision = "PAPER_OK_ROBUST_TO_60BPS"
        reason = (
            f"Frozen `{FOCUS_RULE}` stays positive vs random/EW on Z2 late-half and "
            f"majority-WF at 20/30/40/60 bps/side. Recommend paper/shadow with "
            f"book cost ≥40bps/side stress in live kill criteria."
        )
        reason_cn = (
            f"先验 `{FOCUS_RULE}` 在 20/30/40/60bps/单边 均过 Z2 后半+WF 门。"
            f"建议纸面/影子；实盘杀伤标准按 ≥40bps 压力。"
        )
        recommend = "paper"
    elif p40:
        decision = "PAPER_OK_STRESS_PASS_40_FAIL_OR_FRAGILE_60"
        reason = (
            f"Frozen `{FOCUS_RULE}` passes at 20/30/40 bps/side "
            f"(60 pass={p60}, late60={late60}). Recommend paper/shadow; "
            f"treat 60bps as fragile / kill-watch."
        )
        reason_cn = (
            f"先验 `{FOCUS_RULE}` 在 20/30/40bps/单边 过门"
            f"（60 pass={p60}, late60={late60}）。建议纸面；60bps 视为脆弱/杀伤观察。"
        )
        recommend = "paper"
    elif p30 and (late40 is not None and late40 > 0):
        decision = "PAPER_SOFT_BASELINE_ONLY_40_FRAGILE"
        reason = (
            f"Passes AA2 gate at 30bps/side but fails full gate at 40 "
            f"(late40={late40}, still_net_pos={late40 is not None and late40 > 0}). "
            f"Soft paper only with tight live kill; prefer lower turnover sleeves."
        )
        reason_cn = (
            f"30bps 过门但 40bps 全门失败（late40={late40}）。"
            f"仅软纸面+紧杀伤；优先降换手袖。"
        )
        recommend = "paper_soft"
    else:
        decision = "NO_PAPER_COST_FRAGILE"
        reason = (
            f"Passes at ≤30bps but collapses by 40bps/side "
            f"(late40={late40}, p40={p40}). Do not paper under realistic A-share costs."
        )
        reason_cn = (
            f"≤30bps 过门但 40bps 崩（late40={late40}）。"
            f"现实 A 股成本下不建议纸面。"
        )
        recommend = "kill"

    table = []
    for bps in COST_BPS_LEVELS:
        lv = levels[f"bps_{bps}"]
        f = lv["z2_late_half"]["focus"]
        r = lv["z2_late_half"]["random50_tp10_nostop"]
        bh = lv["z2_late_half"]["buy_hold_ew"]
        table.append(
            {
                "bps_per_side": bps,
                "bps_round_trip": bps * 2,
                "focus_mean_net": f.get("mean_net_return"),
                "focus_sharpe_ann": f.get("sharpe_ann_net"),
                "focus_mdd": f.get("max_drawdown_net"),
                "focus_hit_rate_net": f.get("hit_rate_net"),
                "random50_mean_net": r.get("mean_net_return"),
                "ew_buyhold_mean_net": bh.get("mean_net_return"),
                "beats_random_and_ew": lv["z2_late_half"]["focus_beats_random_and_ew"],
                "wf_n_positive": lv["walkforward"]["n_positive"],
                "wf_n_folds": lv["walkforward"]["n_folds"],
                "wf_fold_mean_nets": lv["walkforward"]["fold_mean_nets"],
                "pass_late_and_wf_gate": lv["pass_late_and_wf_gate"],
                "variants_late_mean_net": {
                    k: (v or {}).get("mean_net_return")
                    for k, v in lv["z2_late_half"]["a_priori_variants"].items()
                },
            }
        )

    return {
        "decision": decision,
        "recommend": recommend,
        "reason": reason,
        "reason_cn": reason_cn,
        "focus_rule": FOCUS_RULE,
        "a_priori_rules": [r[0] for r in A_PRIORI_RULES],
        "cost_convention": (
            "bps figures are ONE-WAY per side; round-trip = 2× side "
            "(AA/AA2 baseline 30bps/side = 60bps RT)"
        ),
        "still_pass_gate_bps": still_pass,
        "still_positive_late_mean_net_bps": still_positive_late,
        "first_fail_gate_bps": fail_first,
        "baseline_30bps_pass": p30,
        "pass_20": p20,
        "pass_40": p40,
        "pass_60": p60,
        "hard_table_z2_late_half": table,
        "contamination_caveat": (
            "Z2 Kairos-val overlaps C2 train; IC may be inflated. "
            "Cost stress tests frozen rule mechanics — not clean model OOS."
        ),
        "next_step": (
            [
                "Paper/shadow: Top50 predicted_return_10d, TP10 nostop, T+1, limit-skip",
                f"Book assumed cost ≥ max(pass levels) stress; kill if 20d rolling mean_net<0",
                "Optional: stride=5/10 sleeves to cut ~1.0 two-way turnover",
                "Do NOT deep-train / touch TPU WIP / revive mfe10",
            ]
            if recommend.startswith("paper")
            else [
                "Do NOT paper under current cost fragility",
                "Cheap retune only: K/hold/stride on existing scores — or HARD STOP axis",
                "Do NOT deep-train / touch TPU WIP / revive mfe10",
            ]
        ),
    }


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "NA"
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def write_memo_cn(results: dict[str, Any], conclusion: dict[str, Any]) -> str:
    lines = [
        "# Kairos Phase AA2 成本压力：纸面/影子前 20/30/40/60 bps/单边",
        "",
        f"日期：{results['generated_at_cst']}（北京时间）。",
        f"范围：冻结 AA2 先验规则 `{FOCUS_RULE}`，在 Z2 后半段 + walk-forward 上扫成本；"
        f"**不**重选 K/止损；**不**训练 / 改解码 / 碰 TPU WIP / 复活 mfe10。",
        "",
        "## 成本约定（务必读清）",
        "",
        "| 项 | 设定 |",
        "| --- | --- |",
        "| 口径 | **单边（one-way / per side）** bps；往返 = 2× 单边 |",
        "| AA/AA2 基准 | 30 bps/单边 = 60 bps 往返 |",
        "| 本扫 | 20 / 30 / 40 / 60 bps/单边（= 40 / 60 / 80 / 120 bps 往返） |",
        "| 入账 | 每笔完成交易扣 `2 * cost_per_side` |",
        "| 执行 | T+1 收盘近似；涨停≥9.5% 跳过；TP10 nostop / 到期 d10 |",
        "",
        "## 一句话结论（硬结论）",
        "",
        f"**{conclusion['decision']}** — {conclusion['reason_cn']}",
        "",
        f"- 建议 = **{conclusion['recommend']}**",
        f"- 仍过 late+WF 门的单边 bps = **{conclusion['still_pass_gate_bps']}**",
        f"- 后半 mean_net>0 的单边 bps = **{conclusion['still_positive_late_mean_net_bps']}**",
        f"- 首次挂门单边 bps = **{conclusion.get('first_fail_gate_bps')}**",
        f"- 污染警告：{conclusion['contamination_caveat']}",
        "",
        "## 硬表：Z2 后半段焦点 `topk50_tp10_nostop` vs random/EW",
        "",
        "| bps/单边 | 往返bps | mean_net | Sharpe | MDD | hit | random50 net | EW net | beats | WF正折 | 过门 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in conclusion["hard_table_z2_late_half"]:
        lines.append(
            f"| {row['bps_per_side']} | {row['bps_round_trip']} | "
            f"{_fmt(row['focus_mean_net'])} | {_fmt(row['focus_sharpe_ann'], 3)} | "
            f"{_fmt(row['focus_mdd'])} | {_fmt(row['focus_hit_rate_net'], 4)} | "
            f"{_fmt(row['random50_mean_net'])} | {_fmt(row['ew_buyhold_mean_net'])} | "
            f"{row['beats_random_and_ew']} | "
            f"{row['wf_n_positive']}/{row['wf_n_folds']} | "
            f"{row['pass_late_and_wf_gate']} |"
        )
    lines += [
        "",
        "## Walk-forward 折净期望（焦点）",
        "",
    ]
    for bps in COST_BPS_LEVELS:
        lv = results["levels"][f"bps_{bps}"]
        nets = lv["walkforward"]["fold_mean_nets"]
        lines.append(
            f"- **{bps}bps/单边**：正折 {lv['walkforward']['n_positive']}/"
            f"{lv['walkforward']['n_folds']}；折净期望 = [{', '.join(_fmt(x) for x in nets)}]"
        )
    lines += [
        "",
        "## 先验变体（后半段，廉价附报，不重选）",
        "",
        "| bps/单边 | topk50_hold_d10 mean_net | topk20_tp10_nostop mean_net |",
        "| ---: | ---: | ---: |",
    ]
    for row in conclusion["hard_table_z2_late_half"]:
        v = row["variants_late_mean_net"]
        lines.append(
            f"| {row['bps_per_side']} | {_fmt(v.get('topk50_hold_d10'))} | "
            f"{_fmt(v.get('topk20_tp10_nostop'))} |"
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
        "- 脚本：`modernbert_finance/ablations/phase_aa2_cost_stress_before_paper.py`",
        "- JSON：`modernbert_finance/ablations/kairos_phase_aa2_cost_stress_before_paper_results.json`",
        "- 备忘：`modernbert_finance/kairos_phase_aa2_cost_stress_before_paper_cn.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
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

    levels: dict[str, Any] = {}
    for bps in COST_BPS_LEVELS:
        cost = _bps_to_side(bps)
        print(f"=== cost stress {bps} bps/side ({cost}) ===", flush=True)
        levels[f"bps_{bps}"] = eval_at_cost(z2, splits, cost)

    results = {
        "phase": "AA2_cost_stress",
        "purpose": "cost_stress_20_30_40_60_bps_before_paper_shadow",
        "not_mfe10_decision_product": True,
        "training_performed": False,
        "decode_changed": False,
        "tpu_wip_touched": False,
        "k_or_stop_repicked": False,
        "focus_rule_a_priori": FOCUS_RULE,
        "a_priori_rules": [r[0] for r in A_PRIORI_RULES],
        "cost_levels_bps_per_side": list(COST_BPS_LEVELS),
        "cost_convention": {
            "unit": "bps_per_side_one_way",
            "round_trip_multiplier": 2,
            "aa_aa2_baseline_bps_per_side": 30,
            "charge_per_trade": "2 * cost_per_side",
        },
        "limit_up_threshold": LIMIT_UP,
        "execution": {
            "entry": "T+1 close proxy",
            "path": "relative to actual_return_d1",
            "exits": ["take_profit_10pct", "time_d10"],
            "stop_loss": None,
            "cost": "2 * cost_per_side per completed trade",
        },
        "score_source": {
            "checkpoint": "C2 Best Seg@179",
            "decode": "prod T=0.65 top_p=0.8 N=5",
            "z2_path": str(Z2_PATH),
        },
        "splits": {
            "z2_n_dates": len(splits["all_dates"]),
            "late_n": len(splits["late_dates"]),
            "late_date_min": splits["late_dates"][0],
            "late_date_max": splits["late_dates"][-1],
            "n_walkforward_folds": len(splits["folds"]),
            "fold_size": FOLD_SIZE,
        },
        "levels": levels,
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
            {"hard_conclusion": conclusion, "levels_pass": {
                f"bps_{bps}": levels[f"bps_{bps}"]["pass_late_and_wf_gate"]
                for bps in COST_BPS_LEVELS
            }},
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
