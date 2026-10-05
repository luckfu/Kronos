# Kairos Phase AA：C2 排序分 → 成本感知 TopK 可交易规则回测

日期：2026-10-02 12:29:28 CST（北京时间）。
范围：停止 binary mfe10≥10% 决策产品；改用已有 Kronos C2@Seg179 prod 解码（T=0.65 top_p=0.8 N=5）的 `predicted_return_10d` 做多头 TopK / 阈值规则，计入 A 股 T+1、涨停过滤、费用+滑点近似，度量净期望。

## 一句话结论（硬结论）

**PROPOSE_LONGER_CONFIRM** — 密封短 OOS 上 `topk50_tp10_nostop` 成本后净期望为正且优于 random/买持（最佳 `topk50_tp10_nostop` net=0.0056）；建议开一枪更长密封确认，不训练。

- 先验主规则 `topk20_tp10_sl5` 短 OOS 净期望 = **-0.0002461948949109889**；Sharpe≈**-0.2031789082484466**；MDD≈**-0.13608706981712126**；命中率≈**0.41690962099125367**；换手≈**1.4610423116615068**
- 扫描最佳 `topk50_tp10_nostop` 净期望 = **0.005593377812257396**；Sharpe≈**3.8687144197897796**；MDD≈**-0.05034128439179719**；命中率≈**0.47368421052631576**；换手≈**1.403881893182805**
- 对照 random20_tp10_sl5 净期望 = **-0.007185414584107231**
- 对照等权买持（T+1→d10）净期望 = **-0.004972844932270409**
- Kairos val（污染窗）先验规则净期望 = **0.01660554048115663**（仅形状）
- 焦点规则（结论用） = `topk50_tp10_nostop`
- 警告：18d sealed OOS is short; overlapping vintages inflate Sharpe; longer confirm required before capital.

## 执行假设

| 项 | 设定 |
| --- | --- |
| checkpoint | C2 Best Seg@179 sha 4ee469d4…b5a |
| decode | prod T=0.65 top_p=0.8 N=5 |
| 分数 | predicted_return_10d |
| 入场 | 信号日 T 收盘后，T+1 收盘近似成交 |
| 涨停 | actual_return_d1≥9.5% 跳过（买不进） |
| 持有路径 | 相对 T+1 收盘的 d2..d10 收盘路径 |
| 出场 | +10% 触及 / 止损 / 到期 d10 |
| 成本 | 单边 0.003（双边往返 2×） |
| 不做 | 不训练 / 不改解码 / 不碰 TPU WIP / 不复活 mfe10 决策头 |

## 面板

- **short_oos_18d**（主结论）：2026-08-11..2026-09-03；n=92751 / 18d / 5167 sym；Spearman(pred,ret10)≈0.2369
- **kairos_val**（污染）：2025-07-03..2026-07-02；n=123836 / 242d；Spearman≈0.4646

## 扫描最佳硬指标（short_oos_18d / best）

| 指标 | 值 |
| --- | ---: |
| rule | topk50_tp10_nostop |
| n_trades | 874 |
| mean_net_return | 0.005593377812257396 |
| mean_gross_return | 0.0115933778122574 |
| hit_rate_net | 0.47368421052631576 |
| sharpe_ann_net | 3.8687144197897796 |
| max_drawdown_net | -0.05034128439179719 |
| compounded_vintage_return_net | 0.10202969304684228 |
| mean_two_way_name_turnover | 1.403881893182805 |
| tp_hit_rate | 0.21967963386727687 |

## 先验主规则硬指标（short_oos_18d / topk20_tp10_sl5）

| 指标 | 值 |
| --- | ---: |
| n_trades | 343 |
| mean_net_return（期望） | -0.0002461948949109889 |
| mean_gross_return | 0.0057538051050890155 |
| hit_rate_net | 0.41690962099125367 |
| sharpe_ann_net | -0.2031789082484466 |
| max_drawdown_net | -0.13608706981712126 |
| compounded_vintage_return_net | -0.01599924499487404 |
| mean_two_way_name_turnover | 1.4610423116615068 |
| tp_hit_rate | 0.22448979591836735 |
| mean_hold_sessions_after_entry | 6.183673469387755 |
| n_skipped_limit | 17 |

## 对照

- random20_tp10_sl5 mean_net = -0.007185414584107231；sharpe≈-4.2023916031345525
- buy_hold_equal_weight mean_net = -0.004972844932270409；sharpe≈-3.451887175398706

## 建议

1. **建议**：开一枪更长密封确认窗（仍只用已有分数/同 decode，不训练）。
2. 不碰同事 TPU WIP；不复活 binary mfe10 决策。

廉价 tweak 清单：
- Prefer tp10_nostop / hold_d10 over tp10_sl5 (stop hurts OOS)
- K=50 beat K=20 on this window — confirm or keep K sweep
- threshold predicted_return_10d ∈ {0, 0.02, 0.05} mixed OOS
- cost stress 10/20/40/60 bps
- stride=5/10 non-overlap sleeves to cut ~1.5 two-way turnover

## 产物

- 脚本：`modernbert_finance/ablations/phase_aa_kronos_cost_aware_topk_backtest.py`
- JSON：`modernbert_finance/ablations/kairos_phase_aa_kronos_cost_aware_topk_results.json`
- 备忘：`modernbert_finance/kairos_phase_aa_kronos_cost_aware_topk_cn.md`
