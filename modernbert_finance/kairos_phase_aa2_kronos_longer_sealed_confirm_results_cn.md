# Kairos Phase AA2：先验 AA 赢家更长密封确认（不重选 K/止损）

日期：2026-10-02 12:32:01 CST（北京时间）。
范围：冻结 Phase AA 短 OOS 选出的先验规则，在 Z2 Kairos-val （C2 Seg@179 prod T=0.65/top_p=0.8/N=5）上做更长确认；**禁止**重扫 K/止损/阈值；**禁止**训练 / 改解码 / 碰 TPU WIP / 复活 mfe10。

## 一句话结论（硬结论）

**PASS_PROPOSE_PAPER_TRADE** — 先验规则 `topk50_tp10_nostop` 在 Z2 后半段成本后净期望仍为正（mean_net=0.013755，优于 random50+等权），且 walk-forward 4/4 折净期望>0。建议廉价纸面交易/影子盘下一步——不深训。

- 焦点先验规则 = `topk50_tp10_nostop`
- Z2 后半段 mean_net = **0.013755224526917725**；Sharpe≈**5.837355681733222**；MDD≈**-0.4495533763493381**；命中≈**0.5559682593280432**；换手≈**0.9927268538621123**；n_trades=5923 / n_dates=121
- 对照 random50_tp10_nostop mean_net = **-0.010555086804049459**
- 对照等权买持 mean_net = **-0.01240954794822861**
- Walk-forward 正折 = **4/4**；折净期望 = [0.02480697981468023, 0.018467659563413296, 0.014412609977193298, 0.01439360086753271]
- 后半 stride10 袖 mean_net = **0.020222584468363277** (Sharpe≈8.701373739538724) — 降重叠 vintage 膨胀
- 污染警告：Z2 Kairos-val dates overlap C2 train window; score IC may be inflated. Confirm tests frozen rule after costs on longer window — not clean model OOS.
- 选择警告：short_oos_18d (18d) selected the AA winner; AA2 verdict uses only Z2 late-half + walk-forward with frozen K/stop — no re-pick.

## 协议（防偷看）

| 项 | 设定 |
| --- | --- |
| 先验规则 | topk50_tp10_nostop, topk50_hold_d10, topk20_tp10_nostop |
| 重选 K/止损 | **否** |
| 主确认窗 | Z2 后半段（时间后半） |
| 辅确认 | Z2 walk-forward 4×~60d；多数折净期望>0 |
| 选择窗 | short_oos_18d 仅作参照，不进 AA2 判决 |
| 成本 | 单边 30bps；双边往返 2× |
| 执行 | T+1 收盘近似；涨停≥9.5% 跳过；TP10 / 到期 d10 |

## Z2 后半段硬指标（焦点）

| 指标 | 值 |
| --- | ---: |
| rule | topk50_tp10_nostop |
| date_range | 2025-12-29..2026-07-02 |
| n_dates | 121 |
| n_trades | 5923 |
| mean_net_return | 0.013755224526917725 |
| mean_gross_return | 0.01975522452691773 |
| hit_rate_net | 0.5559682593280432 |
| sharpe_ann_net | 5.837355681733222 |
| max_drawdown_net | -0.4495533763493381 |
| compounded_vintage_return_net | 3.812900451002066 |
| mean_two_way_name_turnover | 0.9927268538621123 |
| tp_hit_rate | 0.23349653891608982 |
| beats_random_and_ew | True |

## Walk-forward 折

| fold | dates | mean_net | sharpe | MDD | hit | beats |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 2025-07-03..2025-09-24 (60d) | 0.02480697981468023 | 19.238713710395622 | -0.048628629770357756 | 0.6338170772865339 | True |
| 1 | 2025-09-25..2025-12-25 (60d) | 0.018467659563413296 | 10.043551457486243 | -0.21549552219087564 | 0.6110924369747899 | True |
| 2 | 2025-12-26..2026-03-31 (60d) | 0.014412609977193298 | 6.044417098297645 | -0.38096850810908856 | 0.568013468013468 | True |
| 3 | 2026-04-01..2026-06-30 (60d) | 0.01439360086753271 | 5.941892134019452 | -0.4495533763493387 | 0.5485871812543074 | True |

## 先验变体（后半段，不重选）

- `topk50_hold_d10` mean_net=0.018329813979975396；Sharpe≈6.545175836422456；MDD≈-0.44764880418080377；hit≈0.5451629241938207
- `topk20_tp10_nostop` mean_net=0.018026950625909654；Sharpe≈6.779137285999231；MDD≈-0.43934960670838397；hit≈0.5654938533276812

## 下一步

1. Paper-trade / shadow: daily Top50 by predicted_return_10d, TP10 nostop, T+1, limit-skip, 30bps/side book
2. Track live vs sealed metrics weekly; kill if 20d rolling mean_net < 0 and trailing MDD worse than confirm
3. Optional cheap: stride=5/10 sleeves to cut turnover; cost stress 20/40/60 bps
4. Do NOT deep-train / touch TPU WIP / revive mfe10

## 产物

- 脚本：`modernbert_finance/ablations/phase_aa2_kronos_longer_sealed_confirm.py`
- JSON：`modernbert_finance/ablations/kairos_phase_aa2_kronos_longer_sealed_confirm_results.json`
- 备忘：`modernbert_finance/kairos_phase_aa2_kronos_longer_sealed_confirm_cn.md`
