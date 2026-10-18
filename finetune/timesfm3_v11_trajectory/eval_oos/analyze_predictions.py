"""TimesFM 3.0 V11 OOS 离线预测深度分析与免训练集成试验套件。

功能:
1. Block-Bootstrap 95% 置信区间 (按交易日块抽样，杜绝截面自相关假象);
2. 涨跌平衡准确率 (Balanced Accuracy) 与方向偏置分析;
3. 真实截面 Top 50 / Top 100 多空对冲回测 (含交易成本与印花税);
4. 离线无缝多模型免训练集成 (Ensemble: C06 + C07 + C08 等)。

用法示例:
    python finetune/timesfm3_v11_trajectory/eval_oos/analyze_predictions.py \
        --inputs results_c07/eval_predictions_best.npz:c07_best \
                 results_c07/eval_predictions_last.npz:c07_last \
                 results_c08/eval_predictions_best.npz:c08_best \
        --topk 50 100 \
        --bootstrap 1000 \
        --ensemble
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. 基础指标计算与平衡准确率分析
# -------------------------------------------------------------------------
def compute_comprehensive_metrics(pred_ret: np.ndarray, true_ret: np.ndarray, dates: np.ndarray | None = None) -> dict:
    """计算全维指标，包含 10d DirAcc, Pooled IC, 轨迹 Corr, 日度 IC/ICIR, 以及 Balanced Accuracy。"""
    p_10d = pred_ret[:, -1]
    t_10d = true_ret[:, -1]

    # 方向判断
    p_up = p_10d > 0
    t_up = t_10d > 0
    same_dir = (np.sign(p_10d) == np.sign(t_10d))
    direction_acc = float(np.mean(same_dir))

    # 偏置与平衡准确率 (Balanced Accuracy)
    tp = float(np.sum(p_up & t_up))
    fp = float(np.sum(p_up & ~t_up))
    fn = float(np.sum(~p_up & t_up))
    tn = float(np.sum(~p_up & ~t_up))

    tpr = tp / max(tp + fn, 1e-12)  # 实际上涨时的召回率
    tnr = tn / max(tn + fp, 1e-12)  # 实际下跌时的召回率
    balanced_acc = float(0.5 * (tpr + tnr))
    pred_pos_ratio = float(np.mean(p_up))
    market_pos_ratio = float(np.mean(t_up))

    # Pooled IC
    vx = p_10d - np.mean(p_10d)
    vy = t_10d - np.mean(t_10d)
    denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2)) + 1e-12
    pooled_ic = float(np.sum(vx * vy) / denom)

    # 轨迹波形相关度 (点对点 Pearson Corr 均值)
    traj_corrs = []
    # 向量化抽样或直接分块快速计算
    step = max(1, len(pred_ret) // 10000)
    for i in range(0, len(pred_ret), step):
        px, tx = pred_ret[i], true_ret[i]
        std_p, std_t = np.std(px), np.std(tx)
        if std_p > 1e-6 and std_t > 1e-6:
            r = np.corrcoef(px, tx)[0, 1]
            if np.isfinite(r):
                traj_corrs.append(r)
    mean_traj_corr = float(np.mean(traj_corrs)) if traj_corrs else 0.0

    # 振幅与 Skill NRMSE
    err = pred_ret - true_ret
    nrmse = float(np.sqrt(np.mean(err ** 2)))
    naive_nrmse = float(np.sqrt(np.mean(true_ret ** 2)))
    skill_nrmse = float(1.0 - nrmse / max(naive_nrmse, 1e-12))

    pred_std = float(np.mean(np.std(pred_ret, axis=-1)))
    true_std = float(np.mean(np.std(true_ret, axis=-1)))
    amp_ratio = float(pred_std / max(true_std, 1e-12))

    # 日度 IC, ICIR
    daily_ic_mean, daily_icir, pos_ic_ratio = 0.0, 0.0, 0.0
    day_ics = []
    if dates is not None and len(dates) == len(p_10d):
        date_groups = defaultdict(list)
        for idx, dt in enumerate(dates):
            date_groups[dt].append((p_10d[idx], t_10d[idx]))

        for dt, pairs in date_groups.items():
            if len(pairs) >= 50:
                p_sub = np.array([x[0] for x in pairs])
                t_sub = np.array([x[1] for x in pairs])
                if np.std(p_sub) > 1e-6 and np.std(t_sub) > 1e-6:
                    ic_val = np.corrcoef(p_sub, t_sub)[0, 1]
                    if np.isfinite(ic_val):
                        day_ics.append(ic_val)

        if day_ics:
            daily_ic_mean = float(np.mean(day_ics))
            daily_ic_std = float(np.std(day_ics)) + 1e-12
            daily_icir = float(daily_ic_mean / daily_ic_std)
            pos_ic_ratio = float(np.mean([x > 0 for x in day_ics]))

    return {
        "direction_acc": direction_acc,
        "market_natural_up": market_pos_ratio,
        "pred_pos_ratio": pred_pos_ratio,
        "balanced_acc": balanced_acc,
        "tpr_bull_win": tpr,
        "tnr_bear_win": tnr,
        "pooled_ic": pooled_ic,
        "mean_trajectory_corr": mean_traj_corr,
        "daily_ic_mean": daily_ic_mean,
        "daily_icir": daily_icir,
        "annualized_icir": daily_icir * np.sqrt(252),
        "positive_ic_ratio": pos_ic_ratio,
        "amp_ratio": amp_ratio,
        "skill_nrmse": skill_nrmse,
        "trajectory_score": float(direction_acc + 0.5 * max(pooled_ic, 0.0)),
        "day_ics": day_ics,
    }


# -------------------------------------------------------------------------
# 2. Block-Bootstrap 95% 置信区间 (按交易日块抽样)
# -------------------------------------------------------------------------
def run_block_bootstrap(pred_ret: np.ndarray, true_ret: np.ndarray, dates: np.ndarray,
                        n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """按日度块进行 Block-Bootstrap 抽样，评估估计误差与 95% 置信区间。"""
    np.random.seed(seed)
    p_10d = pred_ret[:, -1]
    t_10d = true_ret[:, -1]

    # 按日期分组索引
    date_to_indices = defaultdict(list)
    for idx, dt in enumerate(dates):
        date_to_indices[dt].append(idx)

    unique_dates = np.array(list(date_to_indices.keys()))
    n_days = len(unique_dates)
    if n_days < 5:
        return {}

    date_to_indices_arr = {dt: np.array(idxs, dtype=np.int64) for dt, idxs in date_to_indices.items()}

    boot_dir_accs = []
    boot_pooled_ics = []
    boot_daily_ic_means = []
    boot_daily_icirs = []

    alpha_half = (1.0 - ci) / 2.0

    print(f"[Bootstrap] 正在执行 {n_bootstraps} 次日度块抽样 (总交易日={n_days})...", flush=True)
    t0 = time.time()

    for _ in range(n_bootstraps):
        # 有放回随机抽取 n_days 个日期
        sampled_dates = np.random.choice(unique_dates, size=n_days, replace=True)
        sampled_idxs = np.concatenate([date_to_indices_arr[d] for d in sampled_dates])

        sp = p_10d[sampled_idxs]
        st = t_10d[sampled_idxs]

        # 胜率
        acc = np.mean(np.sign(sp) == np.sign(st))
        boot_dir_accs.append(acc)

        # Pooled IC
        vx = sp - np.mean(sp)
        vy = st - np.mean(st)
        denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2)) + 1e-12
        boot_pooled_ics.append(np.sum(vx * vy) / denom)

        # 日度 IC
        day_ics = []
        for d in sampled_dates:
            idxs = date_to_indices_arr[d]
            dp, dt_val = p_10d[idxs], t_10d[idxs]
            if len(dp) >= 50 and np.std(dp) > 1e-6 and np.std(dt_val) > 1e-6:
                r = np.corrcoef(dp, dt_val)[0, 1]
                if np.isfinite(r):
                    day_ics.append(r)

        if day_ics:
            m_ic = np.mean(day_ics)
            s_ic = np.std(day_ics) + 1e-12
            boot_daily_ic_means.append(m_ic)
            boot_daily_icirs.append(m_ic / s_ic)

    elapsed = time.time() - t0
    print(f"[Bootstrap] 抽样计算完成 (耗时: {elapsed:.2f}s)", flush=True)

    def get_ci_stats(arr):
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, alpha_half * 100)),
            "ci_upper": float(np.percentile(arr, (1.0 - alpha_half) * 100)),
        }

    return {
        "direction_acc": get_ci_stats(boot_dir_accs),
        "pooled_ic": get_ci_stats(boot_pooled_ics),
        "daily_ic_mean": get_ci_stats(boot_daily_ic_means),
        "daily_icir": get_ci_stats(boot_daily_icirs),
    }


# -------------------------------------------------------------------------
# 3. 截面 Top K 多空对冲回测 (含交易成本与印花税)
# -------------------------------------------------------------------------
def run_topk_cross_sectional_backtest(pred_ret: np.ndarray, true_ret: np.ndarray,
                                      dates: np.ndarray, symbols: np.ndarray,
                                      top_k: int = 50, round_trip_cost: float = 0.003,
                                      capital_split: float = 0.5,
                                      trim_days=(0, 1, 3),
                                      split_date: str | None = None) -> dict:
    """按日度截面选股构建 Top K 多头与 Bottom K 空头组合，并计算真实单日与纯多头超额。

    统一口径 (2026-10-18 定稿)：
      - round_trip_cost 默认 0.30%（双边，含冲击成本，比旧版 0.15% 更保守）；
      - capital_split=0.5 表示多头/空头各占 50% 资金，收益按半仓计（旧版按 100/100 满仓）；
      - 同时输出多头腿/空头腿对全市场等权基准的超额，用于区分"挑赢家"与"挑输家"；
      - 对称去极值敏感性（剔除最好 0/1/3 天后重算），防止重尾日主导结论；
      - 分区间（默认按交易日中位切分）单独统计，避免单一行情段掩盖失效。
    """
    p_10d = pred_ret[:, -1]
    t_10d = true_ret[:, -1]
    t_1d = true_ret[:, 0]

    df = pd.DataFrame({
        "date": dates,
        "symbol": symbols,
        "pred": p_10d,
        "true_10d": t_10d,
        "true_1d": t_1d,
    })

    unique_dates = sorted(df["date"].unique())
    daily_ls_10d, daily_ls_1d = [], []
    daily_long_1d, daily_market_1d = [], []
    daily_long_ex, daily_short_ex, used_dates = [], [], []

    for dt in unique_dates:
        sub = df[df["date"] == dt]
        if len(sub) < top_k * 2:
            continue

        sub_sorted = sub.sort_values(by="pred", ascending=False)
        long_basket = sub_sorted.iloc[:top_k]
        short_basket = sub_sorted.iloc[-top_k:]

        # 口径 1: 10d 重叠累计收益
        ls_10d = long_basket["true_10d"].mean() - short_basket["true_10d"].mean() - round_trip_cost
        daily_ls_10d.append(ls_10d)

        # 口径 2: 真实单日调仓多空 (次日实际单日收益结算)
        ls_1d = long_basket["true_1d"].mean() - short_basket["true_1d"].mean() - round_trip_cost
        daily_ls_1d.append(ls_1d)

        # 双腿拆分：相对当日全市场等权基准的超额
        day_mean = sub["true_1d"].mean()
        daily_long_ex.append(long_basket["true_1d"].mean() - day_mean)
        daily_short_ex.append(day_mean - short_basket["true_1d"].mean())
        used_dates.append(dt)

        # 口径 3: 纯多头单日扣费收益与全市场等权基准
        long_1d_cost = long_basket["true_1d"].mean() - (round_trip_cost * 0.67)  # 单边千一
        market_1d = sub["true_1d"].mean()
        daily_long_1d.append(long_1d_cost)
        daily_market_1d.append(market_1d)

    if not daily_ls_1d:
        return {}

    # 口径 1: 10d 重叠统计
    ls_arr_10d = np.array(daily_ls_10d)
    cum_10d = np.cumprod(1.0 + ls_arr_10d)
    sharpe_10d_raw = float(np.mean(ls_arr_10d) / (np.std(ls_arr_10d) + 1e-12) * np.sqrt(252))
    sharpe_10d_indep = float(np.mean(ls_arr_10d) / (np.std(ls_arr_10d) + 1e-12) * np.sqrt(25.2))
    max_dd_10d = float(np.min((cum_10d - np.maximum.accumulate(cum_10d)) / np.maximum.accumulate(cum_10d)))

    # 口径 2: 真实单日持有多空结算 (无重叠放大)
    ls_arr_1d = np.array(daily_ls_1d)
    cum_1d = np.cumprod(1.0 + ls_arr_1d)
    sharpe_1d = float(np.mean(ls_arr_1d) / (np.std(ls_arr_1d) + 1e-12) * np.sqrt(252))
    max_dd_1d = float(np.min((cum_1d - np.maximum.accumulate(cum_1d)) / np.maximum.accumulate(cum_1d)))

    # ---- 统一口径附加统计 (50/50 资金 / 双腿拆分 / 对称去极值 / 分区间) ----
    def _sharpe(x):
        x = np.asarray(x, dtype=float)
        return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(252)) if len(x) > 2 else 0.0

    def _cum(x):
        return float(np.cumprod(1.0 + np.asarray(x, dtype=float))[-1] - 1.0)

    def _mdd(x):
        c = np.cumprod(1.0 + np.asarray(x, dtype=float))
        return float(np.min((c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)))

    ls_scaled = ls_arr_1d * capital_split
    long_ex = np.array(daily_long_ex)
    short_ex = np.array(daily_short_ex)

    trimmed = {}
    for n in trim_days:
        kept = np.sort(ls_arr_1d)[:-n] if n > 0 else np.sort(ls_arr_1d)
        kept_s = kept * capital_split
        trimmed[f"trim_{n}"] = {
            "dropped_best_days": int(n),
            "total_return": _cum(kept_s),
            "sharpe": _sharpe(kept_s),
            "max_drawdown": _mdd(kept_s),
        }

    # 分区间：默认按交易日中位数切分，也可显式指定 split_date
    order = np.argsort(np.array([str(x) for x in used_dates]))
    dates_sorted = [str(used_dates[i]) for i in order]
    ls_sorted = ls_arr_1d[order]
    long_sorted = long_ex[order]
    short_sorted = short_ex[order]
    if split_date is None:
        split_date = dates_sorted[len(dates_sorted) // 2]

    def _regime(mask, label):
        if mask.sum() < 3:
            return {}
        return {
            "label": label,
            "trade_days": int(mask.sum()),
            "ls_mean_daily": float(ls_sorted[mask].mean()),
            "ls_sharpe": _sharpe(ls_sorted[mask]),
            "ls_total_return": _cum(ls_sorted[mask] * capital_split),
            "long_leg_excess_daily": float(long_sorted[mask].mean()),
            "long_leg_sharpe": _sharpe(long_sorted[mask]),
            "short_leg_excess_daily": float(short_sorted[mask].mean()),
            "short_leg_sharpe": _sharpe(short_sorted[mask]),
        }

    dates_arr = np.array(dates_sorted)
    split_idx = int(np.searchsorted(dates_arr, str(split_date)))
    regime = {
        "split_date": str(split_date),
        "first": _regime(np.arange(len(dates_sorted)) < split_idx, "before_split"),
        "second": _regime(np.arange(len(dates_sorted)) >= split_idx, "after_split"),
    }

    # 口径 3: 纯多头超额 (Long-Only Alpha)
    long_arr = np.array(daily_long_1d)
    market_arr = np.array(daily_market_1d)
    excess_arr = long_arr - market_arr
    cum_excess = np.cumprod(1.0 + excess_arr)
    ir_excess = float(np.mean(excess_arr) / (np.std(excess_arr) + 1e-12) * np.sqrt(252))
    max_dd_excess = float(np.min((cum_excess - np.maximum.accumulate(cum_excess)) / np.maximum.accumulate(cum_excess)))

    return {
        "top_k": top_k,
        "trade_days": len(ls_arr_1d),
        "round_trip_cost": float(round_trip_cost),
        "capital_split": float(capital_split),
        # 真实单日持仓多空 (50/50 资金)
        "daily_holding_total_return": _cum(ls_scaled),
        "daily_holding_sharpe": _sharpe(ls_scaled),
        "daily_holding_max_drawdown": _mdd(ls_scaled),
        "daily_holding_win_rate": float(np.mean(ls_arr_1d > 0)),
        # 双腿拆分：谁在赚钱 (相对全市场等权基准)
        "long_leg_excess_daily": float(long_ex.mean()),
        "long_leg_sharpe": _sharpe(long_ex),
        "short_leg_excess_daily": float(short_ex.mean()),
        "short_leg_sharpe": _sharpe(short_ex),
        # 对称去极值敏感性
        "trimmed": trimmed,
        # 分区间稳定性
        "regime": regime,
        # 纯多头超额 (Long-Only Alpha)
        "long_only_cum_return": float(np.cumprod(1.0 + long_arr)[-1] - 1.0),
        "market_benchmark_return": float(np.cumprod(1.0 + market_arr)[-1] - 1.0),
        "long_only_excess_return": float((1.0 + np.cumprod(1.0 + long_arr)[-1]) / (1.0 + np.cumprod(1.0 + market_arr)[-1]) - 1.0),
        "long_only_excess_ir": ir_excess,
        "long_only_excess_max_drawdown": max_dd_excess,
        # 历史 10d 重叠参考口径
        "overlapping_10d_total_return": float(cum_10d[-1] - 1.0),
        "overlapping_10d_sharpe_raw": sharpe_10d_raw,
        "overlapping_10d_sharpe_indep": sharpe_10d_indep,
        "overlapping_10d_max_drawdown": max_dd_10d,
    }


# -------------------------------------------------------------------------
# 主程序入口
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TimesFM 3.0 V11 OOS 深度分析与集成套件")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="输入预测 npz 文件列表，格式: path/to/eval_predictions.npz:tag")
    parser.add_argument("--topk", nargs="+", type=int, default=[50, 100], help="截面回测 Top K 数量")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Block-Bootstrap 重采样次数")
    parser.add_argument("--ensemble", action="store_true", help="是否对输入的各模型预测进行免训练等权融合")
    parser.add_argument("--cost", type=float, default=0.003, help="多空换手摩擦费率 (默认 0.30% 双边，含冲击成本)")
    parser.add_argument("--capital", type=float, default=0.5, help="单腿资金占比 (默认 0.5 = 50/50 多空各半)")
    parser.add_argument("--trim", nargs="+", type=int, default=[0, 1, 3], help="对称去极值敏感性: 剔除最好的 N 天")
    parser.add_argument("--split-date", type=str, default=None, help="分区间切分日 (默认按交易日中位切分)")
    parser.add_argument("--output", type=str, default="analysis_report.json", help="输出指标报告 JSON 路径")
    args = parser.parse_args()

    models_data = {}
    for item in args.inputs:
        if ":" in item:
            path_str, tag = item.split(":", 1)
        else:
            path_str, tag = item, Path(item).stem
        p = Path(path_str)
        if not p.exists():
            print(f"[Warning] 文件不存在: {p}", file=sys.stderr)
            continue
        data = np.load(p)
        models_data[tag] = {
            "pred_ret": data["pred_ret"],
            "true_ret": data["true_ret"],
            "dates": data["dates"],
            "symbols": data["symbols"],
            "path": str(p),
        }

    if not models_data:
        print("[Error] 没有成功加载任何可用的预测 npz 文件！", file=sys.stderr)
        sys.exit(1)

    # 1. 如果指定了 --ensemble 且存在至少两个模型，生成集成模型
    if args.ensemble and len(models_data) >= 2:
        print(f"\n[Ensemble] 正在对 {len(models_data)} 个模型进行预测等权平均集成...")
        first = next(iter(models_data.values()))
        N, H = first["pred_ret"].shape
        ens_pred = np.zeros((N, H), dtype=np.float32)
        for d in models_data.values():
            ens_pred += d["pred_ret"]
        ens_pred /= float(len(models_data))

        ens_tag = "ensemble_" + "_".join(models_data.keys())
        models_data[ens_tag] = {
            "pred_ret": ens_pred,
            "true_ret": first["true_ret"],
            "dates": first["dates"],
            "symbols": first["symbols"],
            "path": "in_memory_ensemble",
        }
        print(f"[Ensemble] 成功构建融合模型: {ens_tag}")

    # 2. 对每个模型执行全维度评估
    full_report = {}
    print("\n" + "=" * 90)
    print("                     TimesFM 3.0 V11 OOS 离线全维深度评估汇总")
    print("=" * 90)

    for tag, d in models_data.items():
        print(f"\n>>> 正在深度分析: [{tag}] (样本数: {len(d['pred_ret']):,}) <<<")
        m = compute_comprehensive_metrics(d["pred_ret"], d["true_ret"], d["dates"])

        # Bootstrap 置信区间
        ci_res = {}
        if args.bootstrap > 0 and len(d["dates"]) > 0:
            ci_res = run_block_bootstrap(d["pred_ret"], d["true_ret"], d["dates"], n_bootstraps=args.bootstrap)

        # Top K 回测
        backtests = {}
        for k in args.topk:
            backtests[f"top_{k}"] = run_topk_cross_sectional_backtest(
                d["pred_ret"], d["true_ret"], d["dates"], d["symbols"], top_k=k,
                round_trip_cost=args.cost, capital_split=args.capital,
                trim_days=tuple(args.trim), split_date=args.split_date,
            )

        full_report[tag] = {
            "source_path": d["path"],
            "metrics": m,
            "bootstrap_ci_95": ci_res,
            "backtests": backtests,
        }

    # 打印格式化对比表格
    print("\n" + "=" * 105)
    print(f"{'指标维度':<26} | " + " | ".join([f"{tag:<22}" for tag in models_data.keys()]))
    print("-" * 105)

    def print_row(label, key, fmt="{:.2%}"):
        vals = []
        for tag in models_data.keys():
            v = full_report[tag]["metrics"].get(key, None)
            vals.append(fmt.format(v) if v is not None else "N/A")
        print(f"{label:<26} | " + " | ".join([f"{v:<22}" for v in vals]))

    print_row("10天走势胜率 (DirAcc)", "direction_acc", "{:.2%}")
    print_row("平衡准确率 (BalancedAcc)", "balanced_acc", "{:.2%}")
    print_row("预测上涨率 (PredPosRatio)", "pred_pos_ratio", "{:.2%}")
    print_row("10天 Pooled IC", "pooled_ic", "{:+.4f}")
    print_row("10天轨迹形态同频度", "mean_trajectory_corr", "{:+.4f}")
    print_row("日度 IC 均值", "daily_ic_mean", "{:+.4f}")
    print_row("日度 ICIR (年化)", "annualized_icir", "{:.2f}")
    print_row("正 IC 交易日胜率", "positive_ic_ratio", "{:.1%}")
    print_row("实测振幅比 (AmpRatio)", "amp_ratio", "{:.4f}")
    print_row("skill_nrmse", "skill_nrmse", "{:.4f}")
    print_row("走势综合评分 (Score)", "trajectory_score", "{:.4f}")

    # 打印 Top K 多空回测 (区分真实单日结算 vs 纯多头超额 vs 10d重叠口径)
    for k in args.topk:
        k_key = f"top_{k}"
        print("-" * 105)
        vals_1d_sharpe, vals_1d_ret, vals_1d_dd = [], [], []
        vals_ex_ret, vals_ex_ir = [], []
        vals_raw_sharpe, vals_raw_ret = [], []

        for tag in models_data.keys():
            bt = full_report[tag]["backtests"].get(k_key, {})
            # 真实单日持仓
            vals_1d_sharpe.append(f"{bt.get('daily_holding_sharpe', 0.0):.2f}")
            vals_1d_ret.append(f"{bt.get('daily_holding_total_return', 0.0)*100:+.2f}%")
            vals_1d_dd.append(f"{bt.get('daily_holding_max_drawdown', 0.0)*100:.2f}%")
            # 纯多头超额
            vals_ex_ret.append(f"{bt.get('long_only_excess_return', 0.0)*100:+.2f}%")
            vals_ex_ir.append(f"{bt.get('long_only_excess_ir', 0.0):.2f}")
            # 10d重叠口径
            vals_raw_ret.append(f"{bt.get('overlapping_10d_total_return', 0.0)*100:+.2f}%")
            vals_raw_sharpe.append(f"{bt.get('overlapping_10d_sharpe_raw', 0.0):.2f}")

        print(f"[真实] Top {k} 单日多空年化夏普| " + " | ".join([f"{v:<22}" for v in vals_1d_sharpe]))
        print(f"[真实] Top {k} 单日多空累计收益| " + " | ".join([f"{v:<22}" for v in vals_1d_ret]))
        print(f"[真实] Top {k} 单日多空最大回撤| " + " | ".join([f"{v:<22}" for v in vals_1d_dd]))
        print(f"[真实] Top {k} 纯多头超额收益(扣费)| " + " | ".join([f"{v:<22}" for v in vals_ex_ret]))
        print(f"[真实] Top {k} 纯多头超额信息比(IR)| " + " | ".join([f"{v:<22}" for v in vals_ex_ir]))

        # 双腿拆分 / 去极值 / 分区间 (统一口径核心表)
        def _row(fmt, getter):
            cells = []
            for tag in models_data.keys():
                bt = full_report[tag]["backtests"].get(k_key, {})
                try:
                    cells.append(fmt.format(getter(bt)))
                except Exception:
                    cells.append("N/A")
            return cells

        print(f"[腿] Top {k} 多头腿超额 SR    | " + " | ".join(f"{v:<22}" for v in _row("{:.2f}", lambda b: b["long_leg_sharpe"])))
        print(f"[腿] Top {k} 空头腿超额 SR    | " + " | ".join(f"{v:<22}" for v in _row("{:.2f}", lambda b: b["short_leg_sharpe"])))
        for n in sorted({int(x) for x in args.trim}):
            if n == 0:
                continue
            print(f"[压测] Top {k} 剔最好{n}天 SR | " + " | ".join(
                f"{v:<22}" for v in _row("{:.2f}", lambda b, n=n: b["trimmed"][f"trim_{n}"]["sharpe"])))
        print(f"[分段] Top {k} 前半段多空 SR  | " + " | ".join(
            f"{v:<22}" for v in _row("{:.2f}", lambda b: (b["regime"]["first"] or {}).get("ls_sharpe", float("nan")))))
        print(f"[分段] Top {k} 后半段多空 SR  | " + " | ".join(
            f"{v:<22}" for v in _row("{:.2f}", lambda b: (b["regime"]["second"] or {}).get("ls_sharpe", float("nan")))))
        print(f"[分段] Top {k} 后半段多头腿SR | " + " | ".join(
            f"{v:<22}" for v in _row("{:.2f}", lambda b: (b["regime"]["second"] or {}).get("long_leg_sharpe", float("nan")))))
        print(f"[重叠] Top {k} 10d重叠累乘虚高夏普| " + " | ".join([f"{v:<22}" for v in vals_raw_sharpe]))

    # 打印 Bootstrap 95% 置信区间
    if args.bootstrap > 0:
        print("-" * 105)
        print(">>> 95% 置信区间 (Block-Bootstrap 按日抽样):")
        for metric_name, m_key in [("DirAcc 95% CI", "direction_acc"), ("Pooled IC 95% CI", "pooled_ic")]:
            ci_strs = []
            for tag in models_data.keys():
                ci_dict = full_report[tag]["bootstrap_ci_95"].get(m_key, {})
                if ci_dict:
                    ci_strs.append(f"[{ci_dict['ci_lower']:.4f}, {ci_dict['ci_upper']:.4f}]")
                else:
                    ci_strs.append("N/A")
            print(f"{metric_name:<26} | " + " | ".join([f"{v:<22}" for v in ci_strs]))

    print("=" * 105 + "\n")

    # 结果持久化
    # 去除 numpy 内部数组对象以便 json 序列化
    serializable = {}
    for tag, d in full_report.items():
        m_copy = dict(d["metrics"])
        m_copy.pop("day_ics", None)
        serializable[tag] = {
            "source_path": d["source_path"],
            "metrics": m_copy,
            "bootstrap_ci_95": d["bootstrap_ci_95"],
            "backtests": d["backtests"],
        }
    Path(args.output).write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"[Done] 全维离线分析大报告已保存至: {args.output}")


if __name__ == "__main__":
    main()
