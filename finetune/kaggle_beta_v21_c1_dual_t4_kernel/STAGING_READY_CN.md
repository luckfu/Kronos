# Beta v2.1 C1 Dual-T4 Phase 1 暂存（短跑）

v6 kernel `user281434/kronos-beta-v2-1-c1-dual-t4` 已在 Seg76 完成（runtime 39600s）。
Best 停在 Seg1：weighted_forecast_loss 2.31245745 / beta_v21_score 1.00077847。
Forecast 从未打过 Seg1（mean ~2.320）。本暂存是下一阶段 Phase 1，不续 v6 checkpoint。

## Phase 1 配方（2026-10-03）

- 从 Beta v2.1 pretrained parent 重新开始（与 v6 起点相同），**不**加载 v6 last/best。
- 新输出目录 `beta_v2_1_c1_dual_t4_p1b`，避免覆盖 v6 产物。
- `KRONOS_USE_BETA_V21_AUXILIARY=0`（纯 forecast）。loss mode `forecast`，history weight `0.02`，原 horizon weights，shuffled feeding（段内不按日期排序）。
- 拆 LR（`KRONOS_SPLIT_TRUNK_HEAD_LR=1`）：transformer trunk（embedding / time_emb / transformer）`1e-6`；adaptation heads（norm、dep_layer、head、return_head、barrier_head）与 sector/size `1e-5`。aux=0 时 return_head/barrier_head 不建模块，分组代码仍把它们放在 1e-5。warmup start 与目标 LR 相同。`warmup_constant`，warmup ratio `0`。**不开 cosine**（那是 phase 2）。
- `KRONOS_BEST_SELECTION_METRIC=forecast`：选 checkpoint 用 horizon-weighted forecast loss，不是 `beta_v21_score`。
- 短跑：`MAX_SEGMENTS_PER_RUN=12`，`MAX_RUNTIME_SECONDS=9000`。双 T4，eff batch 64（32×2），AMP fp16，`KRONOS_COLLECT_VALIDATION_AUXILIARY=0`，full val。
- SwanLab 新 run：`beta_v2_1_c1_dual_t4_p1b`（不 resume 崩溃的 p1 / v6 board）。
- URL：https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_dual_t4_p1b

| 项 | 值 |
|---|---|
| Kernel slug | `user281434/kronos-beta-v2-1-c1-dual-t4` |
| Staging | `finetune/kaggle_beta_v21_c1_dual_t4_kernel/` |
| Accelerator | dual Tesla T4 |
| Trunk LR | `1e-6` warmup_constant |
| Head / condition LR | `1e-5` warmup_constant |
| Soft-stop | `9000` s / 12 segments |
| SwanLab run id | `beta_v2_1_c1_dual_t4_p1b` |

重建：`python3 finetune/build_kaggle_beta_v21_c1_dual_t4_kernel.py`
推送：`kaggle kernels push -p finetune/kaggle_beta_v21_c1_dual_t4_kernel`

## 历史

- v3：scalar-only full val（`KRONOS_COLLECT_VALIDATION_AUXILIARY=0`），修 CUDA aux all-gather OOM。
- v5：shuffled feeding 后 dens 重校准。
- v6：SwanLab `beta_v2_1_c1_dual_t4_v6`，OUTPUT 仍为 `beta_v2_1_c1_dual_t4`，aux=1，统一 LR 1e-5，best=`beta_v21_score`。已跑完 Seg76。
