# Kairos Phase Z2：C2 Seg@179 分数物化到 Kairos val

日期：2026-10-02 11:14:25 CST。

## 目标

把 C2 Best@Seg179 生产解码（T=0.65 top_p=0.8 N=5）分数物化到与 Kairos train/val **日期重叠**的面板上，以便对 `y=1{mfe10≥0.10}` 跑 logistic/isotonic（score(+xsec)），测 train→val / val_temporal Δ。

## 启动

| 项 | 值 |
| --- | --- |
| 权重 | ModelScope `luckfu/Kronos-small-0.1-Cosine-C2-Best` → Kaggle dataset `user281434/kronos-small-0-1-cosine-c2-best-seg179`（sha256=4ee469d4…b5a, seg=179） |
| 面板 | holdout `val_data.pkl` 516 sym；signal **2025-07-03..2026-07-02**（242d, n=123836；与 Kairos val mfe10 **100% join**） |
| Kernel | https://www.kaggle.com/code/user281434/kronos-c2-seg179-kairos-val-prod-scores |
| 状态 | **LAUNCHED**（RUNNING）；预计 ~1h / 2×T4，仅 prod arm |
| 同事 TPU | **未触碰** `beta_v21_c1*` |

## 协议边界

- **train→val**：仍需 train 面板全量打分（百万级×多年）→ 本枪只做 **Kairos val 全覆盖**；主协议 train→val 仍记 BLOCKED，直到另开 train 预算。
- **val_temporal**：Kairos val 前半→后半日期，真实 path-touch mfe10（非 OOS close-path 近似）。闸门 Δ≤−0.04。
- 污染注：Kairos val 落在 C2 训练窗内（latest target 2026-07-31）→ 仅服务决策头校准，**不作 Kronos 时间 OOS 声明**。

## 结果

见 `kairos_phase_z2_kronos_val_score_decision_results_cn.md`（kernel 完成后填 Δ）。
