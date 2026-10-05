# Kairos Tokenizer 表达力消融（mfe10≥10%）

日期：2026-10-01 19:42 CST。

## 一句话结论

**Tokenizer bottleneck = `no`。** 最佳 tok Δ=`-0.025916`（tok_full）；最佳 raw Δ=`-0.025825`（raw_full）；对照 Phase G2 comb Δ≈`-0.034130`，失败 deep sidecar Δ≈`+0.008022`。

- 理由：冻结 tokenizer 探针与 raw OHLCV 几乎同水平（tok Δ=-0.025916，raw Δ=-0.025825，G2 comb Δ=-0.034130）；离散码保住了廉价线性信号，因此 deep sidecar Δ=+0.008022 不能归因于 tokenizer 信息损失。

## 设定

- n_samples=`123836`，pos_rate=`0.2529`，seed=`20261001`
- 定义：`y = 1{mfe10 >= 0.1} where mfe10 = max(high[T+1:T+10]) / close[T] - 1`
- tokenizer：`NeoQuasar/Kronos-Tokenizer-base (local scratch copy)`（s1/s2 vocab=1024/1024）
- device=`cpu`；encode_seconds=`227.4`；wall=`300.0`s
- **未**重启 R2 / **未** Kaggle 长训 / **未**动 TPU WIP

## 1) 冻结 Tokenizer 线性探针

| Pack | dim | prior LL | logistic Δ | ridge Δ |
| --- | ---: | ---: | ---: | ---: |
| tok_bit_embed | 60 | 0.565550 | -0.019814 | 0.023844 |
| tok_codes_summary | 83 | 0.565550 | -0.023116 | 0.022100 |
| tok_full | 143 | 0.565550 | -0.025916 | 0.020041 |

## 2) Raw OHLCV（绕过 Tokenizer）

| Pack | dim | logistic Δ | ridge Δ |
| --- | ---: | ---: | ---: |
| raw_summary | 20 | -0.020221 | 0.022365 |
| raw_last10_norm_flat | 60 | -0.016651 | 0.025748 |
| raw_full | 80 | -0.025825 | 0.019394 |

## 3) 对照刷新（同 seed 切分）

- G2 base refresh Δ=`-0.020109348008008898`
- G2 comb refresh Δ=`-0.034130436131804`
- 记录对照：G2 comb=`-0.034130436131804`；sidecar best=`+0.008022`

## 4) 互信息 / 码稳定性

- MI(last s1, y)=`0.020024`；MI(last s2, y)=`0.017069`
- tok pack MI mean/max=`0.003388`/`0.011220`
- raw pack MI mean/max=`0.010181`/`0.022357`
- recon_mse vs y corr=`0.014748`（mean=`0.010648`）
- 邻码停留率 s1/s2/both=`0.1238`/`0.2346`/`0.0619`
- 停留率 vs y corr s1/both=`0.087483`/`0.080788`

## 5) 判决

| 项 | 值 |
| --- | --- |
| tokenizer_bottleneck | **no** |
| tok_best Δ | -0.025916 (tok_full) |
| raw_best Δ | -0.025825 (raw_full) |
| gap(raw−tok) | 0.000091 |
| vs G2 comb | tok gap=0.008214；raw gap=0.008305 |

### 建议

1. Tokenizer **不是**主瓶颈：冻结码已保住廉价线性信号。
2. Deep sidecar 失败应查优化/容量/标签协议，而非先换 tokenizer。
3. **不要**加长训；可回到特征/标签消融。

## 用户要点（中文）

- **判决**：tokenizer bottleneck = `no`
- **tok 最佳 Δ**：`-0.025916`（tok_full）
- **raw 最佳 Δ**：`-0.025825`（raw_full）
- **对照**：G2 comb Δ≈`-0.034130`；deep sidecar Δ≈`+0.008022`
- **邻码稳定性**：both stay≈`0.0619`；vs y corr=`0.080788`
- **下一步**：别扩 sidecar，查优化/标签

