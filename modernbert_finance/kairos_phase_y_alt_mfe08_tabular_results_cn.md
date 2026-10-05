# Kairos Phase Y-alt 结果（软绝对 mfe10≥0.08 一次备选）

日期：2026-10-02 11:04:30 CST（北京时间）。  
前置：Phase Y CS-top 主协议 FAIL（Δ≈−0.031380）。  
协议：仅一次廉价备选 `y=1{mfe10≥0.08}`；同 S/T2 栈；然后硬报，不继续 churn。

## 一句话结论（硬结论）

**主协议 train→val FAIL 且劣于 CS-top / W**（best Δ≈**−0.021921**）。  
次协议 val_temporal 仍过闸（Δ≈**−0.042085**）。  
→ **HARD-REPORT**：换绝对阈值（0.10→0.08）或换 CS top 五分位，均未能清主闸；**停止本标签族空转**。

## 硬指标

### 主协议 train→val

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.021921205051574133** |
| gate | **false** |
| pos_rate_full / val | ≈0.308 / ≈0.329 |
| vs Phase Y CS-top (−0.031380) | **更差** |
| vs Phase W mfe≥0.10 (−0.028500) | **更差** |

### 次协议 val_temporal

| 项 | 值 |
| --- | ---: |
| best_delta_vs_prior | **−0.042085329339235034** |
| gate | **true** |

耗时 ≈401 s。

## 决策

1. **FAIL_HARD_REPORT_LABEL_AXIS**  
2. 不做：加长 confirm；Kaggle kernel；Ranking 产品；ModernBERT；TSLib 序列复刻；TPU WIP。  
3. 若重启需用户点名**新证据轴**（异于已死的绝对/CS-top mfe10 决策标签 + Base+xsection 表格栈）。
