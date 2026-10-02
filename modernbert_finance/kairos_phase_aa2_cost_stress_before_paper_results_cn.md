# Kairos Phase AA2 成本压力：纸面/影子前 20/30/40/60 bps/单边

日期：2026-10-02 12:44:31 CST（北京时间）。
范围：冻结 AA2 先验规则 `topk50_tp10_nostop`，在 Z2 后半段 + walk-forward 上扫成本；**不**重选 K/止损；**不**训练 / 改解码 / 碰 TPU WIP / 复活 mfe10。

## 成本约定（务必读清）

| 项 | 设定 |
| --- | --- |
| 口径 | **单边（one-way / per side）** bps；往返 = 2× 单边 |
| AA/AA2 基准 | 30 bps/单边 = 60 bps 往返 |
| 本扫 | 20 / 30 / 40 / 60 bps/单边（= 40 / 60 / 80 / 120 bps 往返） |
| 入账 | 每笔完成交易扣 `2 * cost_per_side` |
| 执行 | T+1 收盘近似；涨停≥9.5% 跳过；TP10 nostop / 到期 d10 |

## 一句话结论（硬结论）

**PAPER_OK_ROBUST_TO_60BPS** — 先验 `topk50_tp10_nostop` 在 20/30/40/60bps/单边 均过 Z2 后半+WF 门。建议纸面/影子；实盘杀伤标准按 ≥40bps 压力。

- 建议 = **paper**
- 仍过 late+WF 门的单边 bps = **[20, 30, 40, 60]**
- 后半 mean_net>0 的单边 bps = **[20, 30, 40, 60]**
- 首次挂门单边 bps = **None**
- 污染警告：Z2 Kairos-val overlaps C2 train; IC may be inflated. Cost stress tests frozen rule mechanics — not clean model OOS.

## 硬表：Z2 后半段焦点 `topk50_tp10_nostop` vs random/EW

| bps/单边 | 往返bps | mean_net | Sharpe | MDD | hit | random50 net | EW net | beats | WF正折 | 过门 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 20 | 40 | 0.015755 | 6.686 | -0.427561 | 0.5668 | -0.008555 | -0.010410 | True | 4/4 | True |
| 30 | 60 | 0.013755 | 5.837 | -0.449553 | 0.5560 | -0.010555 | -0.012410 | True | 4/4 | True |
| 40 | 80 | 0.011755 | 4.989 | -0.470744 | 0.5409 | -0.012555 | -0.014410 | True | 4/4 | True |
| 60 | 120 | 0.007755 | 3.292 | -0.510829 | 0.5158 | -0.016555 | -0.018410 | True | 4/4 | True |

## Walk-forward 折净期望（焦点）

- **20bps/单边**：正折 4/4；折净期望 = [0.026807, 0.020468, 0.016413, 0.016394]
- **30bps/单边**：正折 4/4；折净期望 = [0.024807, 0.018468, 0.014413, 0.014394]
- **40bps/单边**：正折 4/4；折净期望 = [0.022807, 0.016468, 0.012413, 0.012394]
- **60bps/单边**：正折 4/4；折净期望 = [0.018807, 0.012468, 0.008413, 0.008394]

## 先验变体（后半段，廉价附报，不重选）

| bps/单边 | topk50_hold_d10 mean_net | topk20_tp10_nostop mean_net |
| ---: | ---: | ---: |
| 20 | 0.020330 | 0.020027 |
| 30 | 0.018330 | 0.018027 |
| 40 | 0.016330 | 0.016027 |
| 60 | 0.012330 | 0.012027 |

## 下一步

1. Paper/shadow: Top50 predicted_return_10d, TP10 nostop, T+1, limit-skip
2. Book assumed cost ≥ max(pass levels) stress; kill if 20d rolling mean_net<0
3. Optional: stride=5/10 sleeves to cut ~1.0 two-way turnover
4. Do NOT deep-train / touch TPU WIP / revive mfe10

## 产物

- 脚本：`modernbert_finance/ablations/phase_aa2_cost_stress_before_paper.py`
- JSON：`modernbert_finance/ablations/kairos_phase_aa2_cost_stress_before_paper_results.json`
- 备忘：`modernbert_finance/kairos_phase_aa2_cost_stress_before_paper_cn.md`
