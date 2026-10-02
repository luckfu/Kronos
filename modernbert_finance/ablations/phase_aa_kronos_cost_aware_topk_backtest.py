#!/usr/bin/env python3
"""Phase AA: cost-aware top-K tradable rules on existing C2 Seg@179 Kronos scores.

NOT a binary mfe10 decision product. Uses ranking/return predictions
(predicted_return_10d) → long-only buy/sell with A-share frictions.

Panels
------
- kairos_val: Phase Z2 prod scores (2025-07-03..2026-07-02). Contaminated
  (inside C2 train window) — rule shape only.
- short_oos_18d: sealed 18d confirmatory OOS (2026-08-11..2026-09-03).
  Primary hard-number panel.

Execution model (documented approximations)
------------------------------------------
- Signal at close of asof_date T.
- T+1: enter at close of session T+1 (proxy; no open prices in dump).
- Limit: skip entry if actual_return_d1 >= +9.5% (涨停买不进 proxy).
- Holding path relative to entry: r_h = (1+actual_d_h)/(1+actual_d1)-1 for h=2..10.
- Max hold through d10 (~9 sessions after entry).
- Exits (first hit on close path): take-profit +10%, stop-loss (param), else time.
- Cost: COST_PER_SIDE on entry and exit (fees+slip approx). Default 30 bps/side.

Does not train, decode, touch TPU WIP, or revive mfe10 decision head.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/workspace/Kronos")
OUT_DIR = ROOT / "scratch/kairos_phase_aa_cost_backtest_outputs"
RESULTS_JSON = (
    ROOT / "modernbert_finance/ablations/kairos_phase_aa_kronos_cost_aware_topk_results.json"
)
MEMO_CN = ROOT / "modernbert_finance/kairos_phase_aa_kronos_cost_aware_topk_cn.md"

PANELS = {
    "kairos_val": ROOT
    / "scratch/kairos_phase_z2_outputs/predictions_prod_t065_p80_n5.csv.gz",
    "short_oos_18d": Path(
        "/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/"
        "predictions_prod_t065_p80_n5.csv.gz"
    ),
}
# Fallback if short OOS only lives under repo finetune copy
if not PANELS["short_oos_18d"].exists():
    PANELS["short_oos_18d"] = (
        ROOT
        / "finetune/kaggle_c2_top20_consensus_oos/predictions_prod_t065_p80_n5.csv.gz"
    )

SCORE_COL = "predicted_return_10d"
LIMIT_UP = 0.095
COST_PER_SIDE_DEFAULT = 0.003  # 30 bps/side ≈ 双边千分之三
TRADING_DAYS = 252
RNG_SEED = 20261002


def load_panel(path: Path) -> pd.DataFrame:
    cols = [
        "symbol",
        "asof_date",
        "sector",
        "size_decile",
        "return_10d",
        SCORE_COL,
        "predicted_return_d10",
    ] + [f"actual_return_d{i}" for i in range(1, 11)]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    if SCORE_COL not in df.columns and "predicted_return_d10" in df.columns:
        df[SCORE_COL] = df["predicted_return_d10"]
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=[SCORE_COL, "actual_return_d1", "actual_return_d10"])
    # path sanity
    for i in range(1, 11):
        df = df[np.isfinite(df[f"actual_return_d{i}"])]
    return df.reset_index(drop=True)


def path_from_entry(row: pd.Series) -> list[tuple[int, float]]:
    """Return [(horizon_from_signal, return_vs_entry), ...] for h=2..10."""
    d1 = float(row["actual_return_d1"])
    denom = 1.0 + d1
    if denom <= 1e-12:
        return []
    out = []
    for h in range(2, 11):
        rh = (1.0 + float(row[f"actual_return_d{h}"])) / denom - 1.0
        out.append((h, rh))
    return out


def simulate_trade(
    row: pd.Series,
    take_profit: float | None,
    stop_loss: float | None,
    cost_per_side: float,
) -> dict[str, Any] | None:
    """Long-only trade with T+1 entry. None if limit-blocked or bad path."""
    d1 = float(row["actual_return_d1"])
    if d1 >= LIMIT_UP:
        return None  # 涨停买不进
    path = path_from_entry(row)
    if not path:
        return None
    exit_h, exit_r, reason = path[-1][0], path[-1][1], "time"
    if take_profit is not None or stop_loss is not None:
        for h, rh in path:
            if take_profit is not None and rh >= take_profit:
                exit_h, exit_r, reason = h, rh, "take_profit"
                break
            if stop_loss is not None and rh <= -stop_loss:
                exit_h, exit_r, reason = h, rh, "stop_loss"
                break
    net = exit_r - 2.0 * cost_per_side
    return {
        "asof_date": row["asof_date"],
        "symbol": row["symbol"],
        "score": float(row[SCORE_COL]),
        "gross_return": float(exit_r),
        "net_return": float(net),
        "exit_horizon_signal": int(exit_h),
        "hold_sessions_after_entry": int(exit_h - 1),
        "exit_reason": reason,
        "entry_d1": float(d1),
        "return_10d_from_signal": float(row["return_10d"]),
    }


def select_top_k(day: pd.DataFrame, k: int) -> pd.DataFrame:
    k = max(1, min(int(k), len(day)))
    return day.nlargest(k, SCORE_COL)


def select_top_frac(day: pd.DataFrame, frac: float) -> pd.DataFrame:
    k = max(1, int(math.ceil(len(day) * frac)))
    return select_top_k(day, k)


def select_threshold(day: pd.DataFrame, thr: float) -> pd.DataFrame:
    sub = day.loc[day[SCORE_COL] >= thr]
    if sub.empty:
        return sub
    return sub.sort_values(SCORE_COL, ascending=False)


def select_random_k(day: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    k = max(1, min(int(k), len(day)))
    idx = rng.choice(day.index.to_numpy(), size=k, replace=False)
    return day.loc[idx]


def run_rule(
    frame: pd.DataFrame,
    picker,
    take_profit: float | None,
    stop_loss: float | None,
    cost_per_side: float,
    label: str,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    skipped_limit = 0
    selected_n = 0
    for _, day in frame.groupby("asof_date", sort=True):
        picked = picker(day)
        selected_n += len(picked)
        for _, row in picked.iterrows():
            t = simulate_trade(row, take_profit, stop_loss, cost_per_side)
            if t is None:
                skipped_limit += 1
                continue
            t["rule"] = label
            trades.append(t)
    if not trades:
        return {
            "rule": label,
            "n_trades": 0,
            "n_selected": int(selected_n),
            "n_skipped_limit": int(skipped_limit),
            "status": "empty",
        }
    td = pd.DataFrame(trades)
    # Equal-weight daily vintage portfolio (overlapping vintages)
    daily = td.groupby("asof_date")["net_return"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    dd = equity / equity.cummax() - 1.0
    n = len(daily)
    mean_d = float(daily.mean())
    std_d = float(daily.std(ddof=1)) if n > 1 else float("nan")
    sharpe = (
        float(mean_d / std_d * math.sqrt(TRADING_DAYS))
        if n > 1 and std_d > 1e-12
        else float("nan")
    )
    # Turnover proxy: fraction of names that change between consecutive signal dates
    turnover = _mean_turnover(td)
    # Buy-hold baseline on same selected names? reported separately at panel level
    return {
        "rule": label,
        "status": "ok",
        "n_trades": int(len(td)),
        "n_selected": int(selected_n),
        "n_skipped_limit": int(skipped_limit),
        "n_signal_dates": int(n),
        "mean_names_per_day": float(td.groupby("asof_date").size().mean()),
        "hit_rate_net": float((td["net_return"] > 0).mean()),
        "hit_rate_gross": float((td["gross_return"] > 0).mean()),
        "mean_net_return": float(td["net_return"].mean()),
        "median_net_return": float(td["net_return"].median()),
        "mean_gross_return": float(td["gross_return"].mean()),
        "expectancy_net": float(td["net_return"].mean()),  # alias
        "std_net_return": float(td["net_return"].std(ddof=1)),
        "mean_hold_sessions_after_entry": float(td["hold_sessions_after_entry"].mean()),
        "exit_reason_counts": {str(k): int(v) for k, v in td["exit_reason"].value_counts().items()},
        "tp_hit_rate": float((td["exit_reason"] == "take_profit").mean()),
        "compounded_vintage_return_net": float(equity.iloc[-1] - 1.0),
        "max_drawdown_net": float(dd.min()),
        "daily_mean_net": mean_d,
        "daily_std_net": std_d,
        "sharpe_ann_net": sharpe,
        "mean_two_way_name_turnover": turnover,
        "cost_per_side": cost_per_side,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
    }


def _mean_turnover(td: pd.DataFrame) -> float:
    dates = sorted(td["asof_date"].unique())
    if len(dates) < 2:
        return float("nan")
    sets = [set(td.loc[td["asof_date"] == d, "symbol"]) for d in dates]
    vals = []
    for a, b in zip(sets, sets[1:]):
        union = a | b
        if not union:
            continue
        # two-way: (dropped + added) / max(len)
        dropped = len(a - b)
        added = len(b - a)
        denom = max(len(a), len(b), 1)
        vals.append((dropped + added) / denom)
    return float(np.mean(vals)) if vals else float("nan")


def buy_hold_equal_weight(frame: pd.DataFrame, cost_per_side: float) -> dict[str, Any]:
    """Equal-weight all names each day, hold to d10 from signal (no T+1 shift), net costs once."""
    # Simple: use return_10d - 2*cost as trade return (enter T close / exit T+10 close)
    # Also T+1 variant for fairer compare
    rows = []
    for _, day in frame.groupby("asof_date", sort=True):
        # T+1 hold-to-end for every name (not limit-filtered for BH breadth)
        nets = []
        for _, row in day.iterrows():
            path = path_from_entry(row)
            if not path:
                continue
            # if limit-up, still "can't buy" — skip to keep tradable BH
            if float(row["actual_return_d1"]) >= LIMIT_UP:
                continue
            gross = path[-1][1]
            nets.append(gross - 2.0 * cost_per_side)
        if nets:
            rows.append({"asof_date": day["asof_date"].iloc[0], "net": float(np.mean(nets))})
    if not rows:
        return {"status": "empty"}
    daily = pd.Series({r["asof_date"]: r["net"] for r in rows}).sort_index()
    equity = (1.0 + daily).cumprod()
    dd = equity / equity.cummax() - 1.0
    std = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    mean = float(daily.mean())
    return {
        "status": "ok",
        "label": "buy_hold_equal_weight_Tplus1_hold_to_d10",
        "n_signal_dates": int(len(daily)),
        "mean_net_return": mean,
        "hit_rate_net": float((daily > 0).mean()),
        "compounded_vintage_return_net": float(equity.iloc[-1] - 1.0),
        "max_drawdown_net": float(dd.min()),
        "sharpe_ann_net": float(mean / std * math.sqrt(TRADING_DAYS))
        if std > 1e-12
        else float("nan"),
        "cost_per_side": cost_per_side,
        "note": "Equal-weight all tradable names; same T+1 entry & hold-to-d10; no TP/SL",
    }


def panel_summary(name: str, path: Path, cost_per_side: float) -> dict[str, Any]:
    frame = load_panel(path)
    dates = sorted(frame["asof_date"].unique())
    meta = {
        "panel": name,
        "path": str(path),
        "n_rows": int(len(frame)),
        "n_dates": int(len(dates)),
        "n_symbols": int(frame["symbol"].nunique()),
        "date_min": dates[0],
        "date_max": dates[-1],
        "contaminated_vs_c2_train": name == "kairos_val",
        "score_col": SCORE_COL,
        "spearman_pred_vs_return10d": float(
            frame[SCORE_COL].corr(frame["return_10d"], method="spearman")
        ),
        "mean_universe": float(frame.groupby("asof_date").size().mean()),
    }
    rng = np.random.default_rng(RNG_SEED)
    rules: list[tuple[str, Any, float | None, float | None]] = []
    # Top-K absolute
    for k in (10, 20, 50):
        rules.append(
            (
                f"topk{k}_tp10_sl5",
                lambda day, k=k: select_top_k(day, k),
                0.10,
                0.05,
            )
        )
        rules.append(
            (
                f"topk{k}_hold_d10",
                lambda day, k=k: select_top_k(day, k),
                None,
                None,
            )
        )
        rules.append(
            (
                f"topk{k}_tp10_nostop",
                lambda day, k=k: select_top_k(day, k),
                0.10,
                None,
            )
        )
    # Top fraction
    for frac in (0.05, 0.10):
        pct = int(frac * 100)
        rules.append(
            (
                f"top{pct}pct_tp10_sl5",
                lambda day, frac=frac: select_top_frac(day, frac),
                0.10,
                0.05,
            )
        )
        rules.append(
            (
                f"top{pct}pct_hold_d10",
                lambda day, frac=frac: select_top_frac(day, frac),
                None,
                None,
            )
        )
    # Score threshold (absolute predicted 10d return)
    for thr in (0.0, 0.02, 0.05):
        thr_tag = f"thr{int(thr * 100):02d}"
        rules.append(
            (
                f"{thr_tag}_tp10_sl5",
                lambda day, thr=thr: select_threshold(day, thr),
                0.10,
                0.05,
            )
        )
    # Random controls matching top20 / top50
    for k in (20, 50):
        rules.append(
            (
                f"random{k}_tp10_sl5",
                lambda day, k=k, rng=rng: select_random_k(day, k, rng),
                0.10,
                0.05,
            )
        )
        rules.append(
            (
                f"random{k}_hold_d10",
                lambda day, k=k, rng=rng: select_random_k(day, k, rng),
                None,
                None,
            )
        )

    rule_metrics = []
    for label, picker, tp, sl in rules:
        m = run_rule(frame, picker, tp, sl, cost_per_side, label)
        rule_metrics.append(m)

    bh = buy_hold_equal_weight(frame, cost_per_side)

    # Rank rules by expectancy_net among those with enough trades
    ok = [m for m in rule_metrics if m.get("status") == "ok" and m["n_trades"] >= 50]
    ok_sorted = sorted(ok, key=lambda m: m["mean_net_return"], reverse=True)
    best = ok_sorted[0] if ok_sorted else None

    # Hard gate: best Kronos rule vs random same-K hold and vs BH
    kronos_primary = next(
        (m for m in rule_metrics if m["rule"] == "topk20_tp10_sl5" and m.get("status") == "ok"),
        None,
    )
    random_primary = next(
        (m for m in rule_metrics if m["rule"] == "random20_tp10_sl5" and m.get("status") == "ok"),
        None,
    )
    kronos_hold = next(
        (m for m in rule_metrics if m["rule"] == "topk20_hold_d10" and m.get("status") == "ok"),
        None,
    )

    positive_after_costs = bool(
        kronos_primary
        and kronos_primary["mean_net_return"] > 0
        and (
            random_primary is None
            or kronos_primary["mean_net_return"] > random_primary["mean_net_return"]
        )
        and (
            bh.get("status") != "ok"
            or kronos_primary["mean_net_return"] > bh["mean_net_return"]
        )
    )

    return {
        "meta": meta,
        "buy_hold": bh,
        "rules": rule_metrics,
        "best_by_expectancy": best,
        "primary_topk20_tp10_sl5": kronos_primary,
        "primary_random20_tp10_sl5": random_primary,
        "primary_topk20_hold_d10": kronos_hold,
        "positive_expectancy_after_costs_vs_controls": positive_after_costs,
    }


def hard_conclusion(results: dict[str, Any]) -> dict[str, Any]:
    """Decide GO longer confirm vs STOP with cheap tweak suggestion."""
    oos = results["panels"].get("short_oos_18d", {})
    val = results["panels"].get("kairos_val", {})
    oos_p = oos.get("primary_topk20_tp10_sl5") or {}
    oos_r = oos.get("primary_random20_tp10_sl5") or {}
    oos_bh = oos.get("buy_hold") or {}
    val_p = val.get("primary_topk20_tp10_sl5") or {}
    best = oos.get("best_by_expectancy") or {}
    bh_net = oos_bh.get("mean_net_return")
    rand_nets = [
        m.get("mean_net_return")
        for m in oos.get("rules", [])
        if str(m.get("rule", "")).startswith("random") and m.get("status") == "ok"
    ]
    best_rand = max(rand_nets) if rand_nets else oos_r.get("mean_net_return")
    best_net = best.get("mean_net_return")
    best_beats = (
        best_net is not None
        and best_net > 0
        and (best_rand is None or best_net > best_rand)
        and (bh_net is None or best_net > bh_net)
    )
    # A-priori primary (topk20 tp10/sl5) — often hurt by stop-loss on this window
    primary_net = oos_p.get("mean_net_return")
    primary_ok = (
        primary_net is not None
        and primary_net > 0
        and (oos_r.get("mean_net_return") is None or primary_net > oos_r["mean_net_return"])
        and (bh_net is None or primary_net > bh_net)
    )

    if primary_ok or best_beats:
        decision = "PROPOSE_LONGER_CONFIRM"
        focus = "topk20_tp10_sl5" if primary_ok else best.get("rule")
        reason = (
            f"Sealed short OOS rule `{focus}` shows positive net expectancy after "
            f"30bps/side and beats random+buy-hold (best={best.get('rule')} "
            f"net={best_net}); propose ONE longer sealed confirm — no train."
        )
    elif best_net is not None and best_net > 0:
        decision = "CHEAP_TWEAK"
        reason = (
            f"Best OOS rule `{best.get('rule')}` net>0 but fails a control; "
            "cheap K/threshold/hold/cost stress before any longer confirm."
        )
    else:
        decision = "HARD_STOP_NO_EDGE"
        reason = (
            "Sealed short OOS cost-aware top-K rules lack positive net expectancy "
            "vs random/buy-hold; do not deep-train; only cheap K/threshold/hold tweaks."
        )

    if decision == "PROPOSE_LONGER_CONFIRM":
        cn_reason = (
            f"密封短 OOS 上 `{('topk20_tp10_sl5' if primary_ok else best.get('rule'))}` "
            f"成本后净期望为正且优于 random/买持（最佳 `{best.get('rule')}` "
            f"net={best_net:.4f}）；建议开一枪更长密封确认，不训练。"
        )
    elif decision == "CHEAP_TWEAK":
        cn_reason = (
            f"最佳规则 `{best.get('rule')}` net>0 但未稳过对照；先廉价微调 K/阈值/持有/成本。"
        )
    else:
        cn_reason = (
            "密封短 OOS 成本感知 TopK 无可靠正净期望；禁止深训；仅允许廉价 K/阈值/持有微调。"
        )

    return {
        "decision": decision,
        "reason": reason,
        "reason_cn": cn_reason,
        "a_priori_primary_rule": "topk20_tp10_sl5",
        "focus_rule": ("topk20_tp10_sl5" if primary_ok else best.get("rule")),
        "best_oos_rule": best.get("rule"),
        "best_oos_mean_net": best_net,
        "cost_per_side": results["cost_per_side"],
        "short_oos_primary_mean_net": primary_net,
        "short_oos_primary_sharpe_ann": oos_p.get("sharpe_ann_net"),
        "short_oos_primary_max_dd": oos_p.get("max_drawdown_net"),
        "short_oos_primary_hit_rate_net": oos_p.get("hit_rate_net"),
        "short_oos_primary_turnover": oos_p.get("mean_two_way_name_turnover"),
        "short_oos_best_sharpe_ann": best.get("sharpe_ann_net"),
        "short_oos_best_max_dd": best.get("max_drawdown_net"),
        "short_oos_best_hit_rate_net": best.get("hit_rate_net"),
        "short_oos_best_turnover": best.get("mean_two_way_name_turnover"),
        "short_oos_random20_mean_net": oos_r.get("mean_net_return"),
        "short_oos_buyhold_mean_net": bh_net,
        "kairos_val_mean_net_contaminated": val_p.get("mean_net_return"),
        "kairos_val_positive_contaminated": bool(
            val.get("positive_expectancy_after_costs_vs_controls")
        ),
        "caveat_18d_window": (
            "18d sealed OOS is short; overlapping vintages inflate Sharpe; "
            "longer confirm required before capital."
        ),
        "suggested_cheap_tweaks": [
            "Prefer tp10_nostop / hold_d10 over tp10_sl5 (stop hurts OOS)",
            "K=50 beat K=20 on this window — confirm or keep K sweep",
            "threshold predicted_return_10d ∈ {0, 0.02, 0.05} mixed OOS",
            "cost stress 10/20/40/60 bps",
            "stride=5/10 non-overlap sleeves to cut ~1.5 two-way turnover",
        ],
    }


def write_memo_cn(results: dict[str, Any], conclusion: dict[str, Any]) -> str:
    oos = results["panels"]["short_oos_18d"]
    val = results["panels"]["kairos_val"]
    p = oos.get("primary_topk20_tp10_sl5") or {}
    r = oos.get("primary_random20_tp10_sl5") or {}
    bh = oos.get("buy_hold") or {}
    vp = val.get("primary_topk20_tp10_sl5") or {}
    best_oos = oos.get("best_by_expectancy") or {}
    lines = [
        "# Kairos Phase AA：C2 排序分 → 成本感知 TopK 可交易规则回测",
        "",
        f"日期：{results['generated_at_cst']}（北京时间）。",
        "范围：停止 binary mfe10≥10% 决策产品；改用已有 Kronos C2@Seg179 "
        "prod 解码（T=0.65 top_p=0.8 N=5）的 `predicted_return_10d` 做多头 TopK / 阈值规则，"
        "计入 A 股 T+1、涨停过滤、费用+滑点近似，度量净期望。",
        "",
        "## 一句话结论（硬结论）",
        "",
        f"**{conclusion['decision']}** — {conclusion.get('reason_cn') or conclusion['reason']}",
        "",
        f"- 先验主规则 `topk20_tp10_sl5` 短 OOS 净期望 = "
        f"**{p.get('mean_net_return')}**；Sharpe≈**{p.get('sharpe_ann_net')}**；"
        f"MDD≈**{p.get('max_drawdown_net')}**；命中率≈**{p.get('hit_rate_net')}**；"
        f"换手≈**{p.get('mean_two_way_name_turnover')}**",
        f"- 扫描最佳 `{best_oos.get('rule')}` 净期望 = "
        f"**{best_oos.get('mean_net_return')}**；Sharpe≈**{best_oos.get('sharpe_ann_net')}**；"
        f"MDD≈**{best_oos.get('max_drawdown_net')}**；命中率≈**{best_oos.get('hit_rate_net')}**；"
        f"换手≈**{best_oos.get('mean_two_way_name_turnover')}**",
        f"- 对照 random20_tp10_sl5 净期望 = **{r.get('mean_net_return')}**",
        f"- 对照等权买持（T+1→d10）净期望 = **{bh.get('mean_net_return')}**",
        f"- Kairos val（污染窗）先验规则净期望 = **{vp.get('mean_net_return')}**（仅形状）",
        f"- 焦点规则（结论用） = `{conclusion.get('focus_rule')}`",
        f"- 警告：{conclusion.get('caveat_18d_window')}",
        "",
        "## 执行假设",
        "",
        "| 项 | 设定 |",
        "| --- | --- |",
        "| checkpoint | C2 Best Seg@179 sha 4ee469d4…b5a |",
        "| decode | prod T=0.65 top_p=0.8 N=5 |",
        "| 分数 | predicted_return_10d |",
        "| 入场 | 信号日 T 收盘后，T+1 收盘近似成交 |",
        "| 涨停 | actual_return_d1≥9.5% 跳过（买不进） |",
        "| 持有路径 | 相对 T+1 收盘的 d2..d10 收盘路径 |",
        "| 出场 | +10% 触及 / 止损 / 到期 d10 |",
        f"| 成本 | 单边 {results['cost_per_side']}（双边往返 2×） |",
        "| 不做 | 不训练 / 不改解码 / 不碰 TPU WIP / 不复活 mfe10 决策头 |",
        "",
        "## 面板",
        "",
        f"- **short_oos_18d**（主结论）：{oos['meta']['date_min']}..{oos['meta']['date_max']}；"
        f"n={oos['meta']['n_rows']} / {oos['meta']['n_dates']}d / {oos['meta']['n_symbols']} sym；"
        f"Spearman(pred,ret10)≈{oos['meta']['spearman_pred_vs_return10d']:.4f}",
        f"- **kairos_val**（污染）：{val['meta']['date_min']}..{val['meta']['date_max']}；"
        f"n={val['meta']['n_rows']} / {val['meta']['n_dates']}d；"
        f"Spearman≈{val['meta']['spearman_pred_vs_return10d']:.4f}",
        "",
        "## 扫描最佳硬指标（short_oos_18d / best）",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| rule | {best_oos.get('rule')} |",
        f"| n_trades | {best_oos.get('n_trades')} |",
        f"| mean_net_return | {best_oos.get('mean_net_return')} |",
        f"| mean_gross_return | {best_oos.get('mean_gross_return')} |",
        f"| hit_rate_net | {best_oos.get('hit_rate_net')} |",
        f"| sharpe_ann_net | {best_oos.get('sharpe_ann_net')} |",
        f"| max_drawdown_net | {best_oos.get('max_drawdown_net')} |",
        f"| compounded_vintage_return_net | {best_oos.get('compounded_vintage_return_net')} |",
        f"| mean_two_way_name_turnover | {best_oos.get('mean_two_way_name_turnover')} |",
        f"| tp_hit_rate | {best_oos.get('tp_hit_rate')} |",
        "",
        "## 先验主规则硬指标（short_oos_18d / topk20_tp10_sl5）",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| n_trades | {p.get('n_trades')} |",
        f"| mean_net_return（期望） | {p.get('mean_net_return')} |",
        f"| mean_gross_return | {p.get('mean_gross_return')} |",
        f"| hit_rate_net | {p.get('hit_rate_net')} |",
        f"| sharpe_ann_net | {p.get('sharpe_ann_net')} |",
        f"| max_drawdown_net | {p.get('max_drawdown_net')} |",
        f"| compounded_vintage_return_net | {p.get('compounded_vintage_return_net')} |",
        f"| mean_two_way_name_turnover | {p.get('mean_two_way_name_turnover')} |",
        f"| tp_hit_rate | {p.get('tp_hit_rate')} |",
        f"| mean_hold_sessions_after_entry | {p.get('mean_hold_sessions_after_entry')} |",
        f"| n_skipped_limit | {p.get('n_skipped_limit')} |",
        "",
        "## 对照",
        "",
        f"- random20_tp10_sl5 mean_net = {r.get('mean_net_return')}；"
        f"sharpe≈{r.get('sharpe_ann_net')}",
        f"- buy_hold_equal_weight mean_net = {bh.get('mean_net_return')}；"
        f"sharpe≈{bh.get('sharpe_ann_net')}",
        "",
        "## 建议",
        "",
    ]
    if conclusion["decision"] == "PROPOSE_LONGER_CONFIRM":
        lines += [
            "1. **建议**：开一枪更长密封确认窗（仍只用已有分数/同 decode，不训练）。",
            "2. 不碰同事 TPU WIP；不复活 binary mfe10 决策。",
        ]
    elif conclusion["decision"] == "CHEAP_TWEAK":
        lines += [
            "1. **不**开长确认；先做廉价门槛/K/持有微调（见下）。",
            "2. 不深度训练 / 不 ModernBERT long train。",
        ]
    else:
        lines += [
            "1. **HARD STOP 本轴深挖**：成本后净期望未过对照。",
            "2. 仅允许廉价 tweak：K / 分数阈值 / 持有或成本压力测试。",
            "3. 不深度训练；不碰 TPU WIP；不复活 mfe10 决策产品。",
        ]
    lines += ["", "廉价 tweak 清单："] + [f"- {t}" for t in conclusion["suggested_cheap_tweaks"]]
    lines += [
        "",
        "## 产物",
        "",
        f"- 脚本：`modernbert_finance/ablations/phase_aa_kronos_cost_aware_topk_backtest.py`",
        f"- JSON：`modernbert_finance/ablations/kairos_phase_aa_kronos_cost_aware_topk_results.json`",
        f"- 备忘：`modernbert_finance/kairos_phase_aa_kronos_cost_aware_topk_cn.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE_DEFAULT)
    parser.add_argument("--out-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--out-memo", type=Path, default=MEMO_CN)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panels: dict[str, Any] = {}
    for name, path in PANELS.items():
        if not path.exists():
            panels[name] = {"status": "missing", "path": str(path)}
            continue
        panels[name] = panel_summary(name, path, args.cost_per_side)
        # dump best trades sample? skip for size
    results = {
        "phase": "AA",
        "purpose": "cost_aware_topk_tradable_rules_on_existing_kronos_scores",
        "not_mfe10_decision_product": True,
        "training_performed": False,
        "decode_changed": False,
        "tpu_wip_touched": False,
        "cost_per_side": args.cost_per_side,
        "limit_up_threshold": LIMIT_UP,
        "execution": {
            "entry": "T+1 close proxy",
            "path": "relative to actual_return_d1",
            "exits": ["take_profit_10pct", "stop_loss", "time_d10"],
            "cost": "2 * cost_per_side per completed trade",
        },
        "panels": panels,
        "elapsed_sec": time.time() - t0,
        "generated_at_cst": time.strftime("%Y-%m-%d %H:%M:%S CST", time.localtime()),
    }
    conclusion = hard_conclusion(results)
    results["hard_conclusion"] = conclusion
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    memo = write_memo_cn(results, conclusion)
    args.out_memo.write_text(memo)
    # also copy memo/results summary under scratch
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "hard_conclusion": conclusion,
                "short_oos_primary": panels.get("short_oos_18d", {}).get(
                    "primary_topk20_tp10_sl5"
                ),
                "short_oos_best": panels.get("short_oos_18d", {}).get("best_by_expectancy"),
                "kairos_val_primary": panels.get("kairos_val", {}).get(
                    "primary_topk20_tp10_sl5"
                ),
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
