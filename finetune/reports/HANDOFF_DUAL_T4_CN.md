# Kronos Beta v2.1 C1 双 T4 接手说明（2026-10-02）

## 当前方向
- **已停 TPU**；正式训练走 **双 T4 GPU**。
- Kernel：`user281434/kronos-beta-v2-1-c1-dual-t4`（v4+；去掉段内日期排序）
- 配方：LR **1e-5**、有效 batch **64**（32×2）、soft-stop **39600s**、从 Beta v2.1 预训练 + 新 AdamW
- Best：**`beta_v21_score`（越低越好）**，不是 small stage2 的 `forecast`
- SwanLab 看板（small 同款接力）：https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_dual_t4
- 数据集：`luckfu/a-share-120d-temporal-symbol-holdout`

## 已修故障
1. v1：校准 VAL 后 CUDA 全量 aux gather → 主机 OOM SIGKILL → 默认关闭 `KRONOS_COLLECT_VALIDATION_AUXILIARY`
2. v2：`ModuleNotFoundError: tpu_self_checks` → optional import + dual-T4 overlay 打入该文件
3. v3：RUNNING；Seg1 `beta_v21_score≈1.00678429`
4. v4+：去掉 `set_epoch_seed` 内按 `signal_date_ids` 的整段 chronological argsort，使喂数与 small stage2/main（aux=0、无段内日期排序）一致；保留 beta aux 损失但不再整段按日重排；同日 ranking 需另开 batch 路径

## 关键路径
- Staging / 推送：`finetune/kaggle_beta_v21_c1_dual_t4_kernel/`
- Builder：`finetune/build_kaggle_beta_v21_c1_dual_t4_kernel.py`
- 训练核心：`finetune/train_predictor.py`、`finetune/config.py`
- 勿 drop `stash@{0}`

## 用户站令
出错就改、改完就推，别停；只走 GPU。
