# Kairos MFE≥10% 短 Sidecar 训练交接（单二分类头）

日期：2026-10-01 22:38 CST（北京时间）。Phase K identity 未过闸（best Δ≈−0.003）；用户对齐排序非二分类 → Phase L 本地排序消融已跑通。

## 一句话

**现状**：Phase K COMPLETE 未过闸。**原目标**（二分类）：`y=1{mfe10≥0.10}`，其中 `mfe10=max(high[T+1:T+10])/close[T]-1`（路径触及，非收盘对收盘）。  
**开训**：短 sidecar（≤4 segments / 1 chunk），**不是**旧 8 头 R2 重启。  
**闸门**：验证 Δ logloss ≤ `-0.04` vs 常数先验；正类率约 `25%`。

## 目标定义（必须遵守）

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1` |
| 标签 | `y = 1{mfe10 ≥ 0.10}` |
| 语义 | 买入后 10 个交易日内路径最高价相对入场价触及 +10%（MFE） |
| **不是** | close-to-close `fwd_ret_10 = close[T+10]/close[T]-1`（Phase G 错误目标） |
| **不是** | 旧 8 头 `up_003/005/008/012 + down_*` R2 配方 |
| **不是** | `user281434/kairos-r2-r1-restart-lr-3e-5` 续训 |

标签在训练脚本内从 sidecar parquet 的 `mfe10` 列现场派生，无需重建 targets 数据集。

## 验证集先验（Phase G2 已测）

| 指标 | 值 |
| --- | ---: |
| n (val) | 123 836 |
| 正类率 | **≈0.2529**（~25%） |
| 常数先验 logloss | ≈0.56555 |
| Logistic Base+xsection Δ | −0.03413（未过绝对闸，但相对 E/F 更好 → 仅短 sidecar） |

## 闸门与早停

| 规则 | 值 |
| --- | --- |
| 过关闸门 | `Δ = model_ll − prior_ll ≤ -0.04` |
| 每 eval 必打 | `constant_prior_log_loss` / `delta_vs_prior` / `positive_rate` |
| 早停 | 连续 **2** 次 eval 满足 `\|model_ll − prior_ll\| ≤ 1e-3` → `stuck_at_constant_prior` |
| 预算 | `MAX_SEGMENTS_THIS_RUN=4`，`SEGMENT_SAMPLES=20000`，GPU ≈90min，1 chunk |
| LR | `1e-4`；**BACKBONE_MODE=identity**（绕过 22 层）；fresh embeds+头（不加载 8 头权重） |

## 代码与 Kernel

| 项 | 路径 / 值 |
| --- | --- |
| 助手逻辑 | `modernbert_finance/mfe10_sidecar.py` |
| 训练脚本 | `finetune/kaggle_kairos_mfe10_sidecar/train_mfe10_sidecar.py` |
| Builder | `finetune/build_kairos_mfe10_sidecar.py` |
| Staging | `finetune/stage_kairos_mfe10_sidecar.py`（注入 SwanLab，不入库） |
| Kernel slug | `user281434/kairos-mfe10-sidecar-short-phase-k-identity` |
| 数据 | `luckfu/a-share-120d-temporal-symbol-holdout` + `luckfu/ashare120d-modernbert-targets` |
| kernel_sources | **空**（不挂旧 chunk-8 / R2 权重） |
| 单测 | `tests/test_kairos_mfe10_sidecar.py` |

### 构建 / 推送

```sh
# 1) GitHub 先推代码（Kaggle 会 git clone luckfu/Kronos）
cd /workspace/Kronos
python finetune/build_kairos_mfe10_sidecar.py
git add modernbert_finance/mfe10_sidecar.py \
  modernbert_finance/kairos_mfe10_sidecar_handoff_cn.md \
  finetune/build_kairos_mfe10_sidecar.py \
  finetune/stage_kairos_mfe10_sidecar.py \
  finetune/kaggle_kairos_mfe10_sidecar/ \
  tests/test_kairos_mfe10_sidecar.py
# 勿 add 同事 TPU WIP（beta_v21_c1* / tpu_train_entry 等 dirty）
git commit -m "Add Kairos mfe10≥10% path-touch short sidecar training"
git push origin master

# 2) 私有 staging（写入 SwanLab key）+ Kaggle push
python finetune/stage_kairos_mfe10_sidecar.py
kaggle kernels push -p artifacts/kairos_mfe10_sidecar/private_staging
kaggle kernels status user281434/kairos-mfe10-sidecar-short-phase-k-identity
```

若 CLI push 需要网页确认：打开  
https://www.kaggle.com/code/user281434/kairos-mfe10-sidecar-short-phase-k-identity  
点 **Run** / **Resume**（GPU T4，Internet on）。

## SwanLab 如何看

- 项目：`roc_fu/finance`
- Run id：`kairos-mfe10-sidecar-short-phase-k-identity-20261001`
- URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-sidecar-short-phase-k-identity-20261001
- 关键曲线：`validation/log_loss`、`validation/constant_prior_log_loss`、`validation/delta_vs_prior`、`validation/gate_passed`

## 明确不做

1. **不**重启 `kairos-r2-r1-restart-lr-3e-5` / 旧 8 头标签。
2. **不**全量 R2 长训 / 多 chunk relay。
3. **不**提交同事 TPU WIP（`finetune/kaggle_beta_v21_c1*`、`tpu_train_entry.py` 等 dirty）。
4. **不**把行业中性当本轮硬门槛（本轮无 industry-neutral 要求）。

## 判读

| 结果 | 动作 |
| --- | --- |
| 早停 `stuck_at_constant_prior` | 深度模型未学到信号；停，回消融/特征 |
| 有 Δ 但 `Δ > -0.04` | 弱信号；可再加 1 个短 chunk 或停 |
| `Δ ≤ -0.04` | 过闸；再议是否扩预算（仍非全量 R2） |

## 用户要点

- **定义**：路径 MFE≥10%/10 日，非收盘对收盘。
- **正类率**：val ~25%。
- **闸门**：Δ≤−0.04 vs 常数先验。
- **预算**：短 sidecar only；无 full R2。
