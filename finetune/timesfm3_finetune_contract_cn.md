# TimesFM-3 微调数据契约

更新：2026-10-08 CST

## 目标

微调 TimesFM-3 作为直接价格路径预测器：

```text
过去 120 日：OHLCVA + size_percentile + sector_id
未来 10 日：close
```

Kronos/C2 继续负责横截面排序；TimesFM-3 微调只以价格路径质量为目标。

## 输入

连续特征：

```text
open, high, low, close, volume, amount, size_percentile
```

离散条件：

```text
sector_id
```

`sector_id` 使用整个面板一次性建立稳定编码，训练、验证和测试共享同一映射。
行业编码不能直接当作连续数值喂给模型；训练适配层应将其转换为 embedding，
再广播到 120 日上下文，或作为单独的 static covariate embedding。

## 输出

主输出是未来 10 个交易日的 close 路径。训练时同时保留：

- 原始 close loss；
- 除以起点 close 的归一化 path loss；
- 可选的 log-return 辅助 loss。

验证和报告统一还原为价格路径，不能只看单一终点。

## 时间切分

首轮数据契约默认：

| split | 信号日期 |
|---|---|
| train | 2025-07-01 至 2026-07-02 |
| val | 2026-07-17 至 2026-08-10 |
| test | 2026-08-11 至 2026-08-15 |

训练样本的未来标签必须完整存在；切分按信号日期完成，禁止随机切分。
`test` 只是当前面板中可用的短期保留集，正式微调晋级前应换成新的、
完全未参与选择的数据窗口。

## 训练顺序

1. 先冻结大部分 TimesFM-3 backbone，只训练输出适配层和条件 embedding。
2. 若 val 路径指标改善，再解冻最后一小段 backbone。
3. 最后才比较 full fine-tuning。

第一轮不加入横截面排序 loss，不把 Rank IC 作为 TimesFM-3 的选择指标。

## 主指标

- 归一化全路径 RMSE；
- 归一化全路径 MAE；
- 逐日价格变动方向一致率；
- 预测路径相关性；
- 价格偏差；
- 区间覆盖率与区间宽度的联合校准。

Zero-shot 原始 OHLCVA 基线为：

```text
normalized MAE  = 5.53%
normalized RMSE = 6.29%
path correlation = -0.036
path direction accuracy = 47.07%
```

微调模型必须在时间外验证集上改善全路径指标，不能只降低单一 loss。
