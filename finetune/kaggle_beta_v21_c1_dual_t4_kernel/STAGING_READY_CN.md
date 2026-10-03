# Beta v2.1 C1 Dual-T4 WC 主配方 chunk 1 暂存

用户已停 p1/p1b（拆 LR + warmup ratio 0）。本暂存对齐 small stage2 `main` 单 LR，
并按站令把 `KRONOS_SCHEDULER_WARMUP_RATIO` 设为 **0.05**（不是 0.01）。

## WC 配方（2026-10-03）

- 从 Beta v2.1 pretrained parent 重新开始（`luckfu/Kronos-A-Share-Beta-V2-1`），新 AdamW。
  **不** resume p1 / p1b / v6。
- 新输出目录 / SwanLab：`beta_v2_1_c1_dual_t4_wc`
- `KRONOS_USE_BETA_V21_AUXILIARY=0`（纯 forecast）。Best：`forecast`（weighted_forecast_loss；
  保留 5f044fb aux-off 记录修复）。
- **单 LR**：predictor 与 condition 均为 `1e-5`。`KRONOS_SPLIT_TRUNK_HEAD_LR=0`。
- `warmup_constant`：warmup start `1e-6`（两边），warmup ratio **`0.05`**，之后 hold `1e-5`。不开 cosine。
- 长 chunk：`MAX_SEGMENTS_PER_RUN=250`，`MAX_RUNTIME_SECONDS=39600`。双 T4，eff batch 64（32×2），
  AMP fp16，full val，`KRONOS_COLLECT_VALIDATION_AUXILIARY=0`。
- Feeding：shuffled `coverage_order`，段内不按 `signal_date` 排序。
- 本推送仅 chunk 1 全新启动；后续可从 `last_state` 接力（与 small WC 相同模式）。

| 项 | 值 |
|---|---|
| Kernel slug | `user281434/kronos-beta-v2-1-c1-dual-t4` |
| Staging | `finetune/kaggle_beta_v21_c1_dual_t4_kernel/` |
| Accelerator | dual Tesla T4 |
| Predictor / condition LR | `1e-5` / `1e-5`（split off） |
| Warmup start / ratio | `1e-6` / `0.05` → hold `1e-5` |
| Soft-stop | `39600` s / 250 segments |
| SwanLab run id | `beta_v2_1_c1_dual_t4_wc` |
| SwanLab URL | https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_dual_t4_wc |

重建：`python3 finetune/build_kaggle_beta_v21_c1_dual_t4_kernel.py`
推送：`kaggle kernels push -p finetune/kaggle_beta_v21_c1_dual_t4_kernel`

## 历史

- v6：aux=1，统一 LR 1e-5，best=`beta_v21_score`，跑完 Seg76。
- p1/p1b：拆 LR + warmup ratio 0 短跑；用户停；勿 resume。
- WC：对齐 small stage2 main 单 LR，warmup ratio 0.05，长 chunk。
