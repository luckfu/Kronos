# Beta v2.1 C1 Dual-T4 暂存就绪（OOM fix 已嵌入）

## 根因（ERROR 跑）

校准 full-val 跑完 `[VAL] Processed 1935/1935 batches...` 后，CUDA 路径仍
`collect_validation_auxiliary=True`，对 ~123k 样本做 `torch.cat` +
`dist.all_gather_object`，双进程主机内存被打爆 → `train_predictor.py` SIGKILL -9。
校准已把 consistency=0，所以不是 AR；是 sample-level aux gather。

## v3 修复（2026-10-02）

v2 ERROR: `ModuleNotFoundError: tpu_self_checks` in `train_predictor.py`（校准前即失败，非 SIGKILL/OOM）。

- `train_predictor.py`: `import tpu_self_checks` → try/except，GPU 路径不硬依赖；`master_weight_dtype` 同步守卫
- dual-T4 overlay FILES 补齐 `tpu_self_checks.py` + `drive_cleanup.py`
- 配方不变：LR 1e-5、dual T4、soft-stop 39600、SwanLab `beta_v2_1_c1_dual_t4` resume allow、`KRONOS_COLLECT_VALIDATION_AUXILIARY=0`

## 本暂存修复

- scalar-only full val（`KRONOS_COLLECT_VALIDATION_AUXILIARY=0`，与 TPU 精神一致）
- CUDA AR microbatch 默认 8（`KRONOS_BETA_V21_CONSISTENCY_AR_BATCH=8`）
- `prepare_model_for_validation` 在校准前调用；CUDA `empty_cache`
- 嵌入本地 `train_predictor.py` / `config.py` / `model/` 覆盖 GitHub clone（无需 git commit）

## 配方

| 项 | 值 |
|---|---|
| Kernel slug | `user281434/kronos-beta-v2-1-c1-dual-t4` |
| Staging | `finetune/kaggle_beta_v21_c1_dual_t4_kernel/` |
| Accelerator | dual Tesla T4 |
| LR | `1e-5` warmup_constant |
| Soft-stop | `39600` s |
| SwanLab run id | `beta_v2_1_c1_dual_t4` resume allow |
| URL | https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_dual_t4 |

重建：`python3 finetune/build_kaggle_beta_v21_c1_dual_t4_kernel.py`
推送：`kaggle kernels push -p finetune/kaggle_beta_v21_c1_dual_t4_kernel`
