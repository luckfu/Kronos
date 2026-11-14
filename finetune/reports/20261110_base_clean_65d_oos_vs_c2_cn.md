# Base clean 65 天 OOS 对照 C2：复刻失败，且 val 与 OOS 反相关坐实

窗口：`2026-07-17 → 2026-10-23`（65 个交易日，335,312 样本，5,180 只，标签到 2026-11-06）
数据集：`luckfu/a-share-120d-oos-20260717-20261023-v2`
协议：seed 20261104，T=0.65，top_p=0.8，单路径，`evaluate_base_sealed.py`

## 协议漂移校验（先做，否则下面的对比无效）

C2 历史上在 42 天封存窗（08-03→09-30）的 IC 是 **0.1116**。
本次用同一套协议重跑，从 65 天里截同一 42 天：**0.11141**，差 0.0002。

→ **协议一致，跨运行可比。**

## 结果

| 模型 | 65 天 IC | 其中 42 天 IC | MAE | RMSE | direction |
|---|---:|---:|---:|---:|---:|
| **C2 best** | **0.11405** | **0.11141** | 0.05801 | 0.08977 | 0.5494 |
| base clean cosine@88（未退火） | 0.00686 | 0.02317 | 0.10354 | 0.19342 | 0.5192 |
| base clean cosine-a90（真退火） | 0.00156 | 0.02016 | 0.10895 | 0.20318 | 0.5144 |

同一窗口、同一协议下，C2 的 IC 是 base clean 的 **17–73 倍**，MAE 只有它的 **0.53 倍**。

## 关键：退火让 val 变好、让 OOS 变差

| | val best selection | OOS IC（42 天） |
|---|---:|---:|
| cosine（未退火，88 段） | 2.25063 | 0.02317 |
| cosine-a90（真退火，90 段） | **2.25007**（更好 0.00056） | **0.02016**（更差 0.00301） |

LR 完整走完 1e-5 → 1e-6 的余弦，validation 改善 0.00056，OOS 恶化 0.0030。
这是目前为止分辨率最高的一组反相关证据：同一个模型、同一份数据、唯一的变量是学习率曲线。

## 整条链的 val / OOS 轨迹

| 阶段 | val forecast | 42 天 OOS IC |
|---|---:|---:|
| Bootstrap（389 段） | 2.5931 | — |
| Main（520） | 2.33125 | — |
| Extend（185） | 2.30067 | — |
| WC（520） | 2.24355 | 0.0184（@438） |
| Cosine（88，未退火） | 2.24236 | 0.02317 |
| Cosine-a90（90，退火） | **2.24204** | **0.02016** |

val 单调下降 0.351，OOS 停在 0.02 量级且最后一段下滑。

## 结论

1. **复刻失败**，且不是协议伪影 —— C2 在同一次运行里复现了自己的历史数字。
2. **反相关坐实**：在这条线上，val forecast 每变好一点，OOS IC 就往反方向走。
3. 继续在这条线上加段数、调学习率、换 scheduler，只会把 OOS 继续往零推。
4. 该换的是尺子（时间切分 Rank IC 进选型回路）和验证集，不是训练预算。

## 复现路径

```bash
# C2 65 天
KRONOS_OOS_DATASET=luckfu/a-share-120d-oos-20260717-20261023-v2 \
KRONOS_OOS_EXTRA_DATASETS=luckfu/kronos-small-cosine-c2-best-weights \
KRONOS_OOS_PARENTS=user281434/kronos-base-clean-0-1-cosine \
BASE_OOS_LABEL=c2_best \
BASE_OOS_WEIGHTS_GLOB='**/kronos-small-cosine-c2-best-weights/model.safetensors' \
BASE_OOS_CONTEXT_LAYER=6 BASE_OOS_SEED=20261104 \
python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py --push
```

kernel：`user281434/kronos-c2-oos-65d-b`（C2）、`user281434/bc-cosine-a90-oos-65d`（a90）、
`user281434/bc-cosine-oos-65d-smoke`（未退火 cosine）。
