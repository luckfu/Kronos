# Kronos Beta v2.1 C1 双 T4 接手说明（2026-10-02）

## 当前方向
- **已停 TPU**；正式训练走 **双 T4 GPU**。
- Kernel：`user281434/kronos-beta-v2-1-c1-dual-t4`（**v5**：分母按 shuffled 喂数强制重校准）
- 配方：LR **1e-5**、有效 batch **64**（32×2）、soft-stop **39600s**、从 Beta v2.1 预训练 + 新 AdamW
- Best：**`beta_v21_score`（越低越好）**，不是 small stage2 的 `forecast`
- SwanLab 看板（small 同款接力）：https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_dual_t4
- 数据集：`luckfu/a-share-120d-temporal-symbol-holdout`

## 已修故障
1. v1：校准 VAL 后 CUDA 全量 aux gather → 主机 OOM SIGKILL → 默认关闭 `KRONOS_COLLECT_VALIDATION_AUXILIARY`
2. v2：`ModuleNotFoundError: tpu_self_checks` → optional import + dual-T4 overlay 打入该文件
3. v3：RUNNING；Seg1 `beta_v21_score≈1.00678429`（**chronological dens**；跨版本勿用 score 比）
4. v4+：去掉 `set_epoch_seed` 内按 `signal_date_ids` 的整段 chronological argsort，使喂数与 small stage2/main（aux=0、无段内日期排序）一致；保留 beta aux 损失但不再整段按日重排；同日 ranking 需另开 batch 路径
5. **v5**：专家评审后强制分母重校准。旧 chronological dens 下同日 pair 变稀（~0.37 pairs/batch、约 67% 零 batch）会使 `ranking_loss`≈1/5、`beta_v21_score` 虚降 ~0.07–0.08。策略：清空 `KRONOS_BETA_V21_VALIDATION_DENOMINATORS`、`AUTO_CALIBRATE=1`、dens.json 打 `feeding_mode=shuffled_no_segment_date_sort`；不匹配则 wipe 后重校准。**跨 v3/v4/v5 比 `weighted_forecast_loss` 绝对值，不比 score。** 同日 ranking 独立 batch 路径仍 TODO（落地时再 bump feeding_mode 重校准，避免稀 dens 在 pair 变密后爆炸）。不从 score 里丢掉 ranking。

## 关键路径
- Staging / 推送：`finetune/kaggle_beta_v21_c1_dual_t4_kernel/`
- Builder：`finetune/build_kaggle_beta_v21_c1_dual_t4_kernel.py`
- 训练核心：`finetune/train_predictor.py`、`finetune/config.py`、`finetune/dataset.py`
- 日志接手：`logs/HANDOFF.md`
- 巡逻笔记：`scratch/c1_dual_t4_patrol/NOTES.md`
- 勿 drop `stash@{0}`

## 用户站令
出错就改、改完就推，别停；只走 GPU。

## 推送状态（2026-10-02 ~18:10 CST）
- 代码已进 staging；Kaggle **v5 源码已推**（临时 CPU 以卸代码；GPU 并发槽位满）。
- **GPU 双 T4 再推被拦**：`Maximum batch GPU session count of 2 reached`。需网页 Stop 旧 GPU session 后，再 `kaggle kernels push -p finetune/kaggle_beta_v21_c1_dual_t4_kernel`（metadata 已是 GPU/T4）→ 预计 v6 GPU，启动时 wipe 旧 dens 并 AUTO_CALIBRATE。

## Phase 1（2026-10-03）短跑

- v6 已完成 Seg76。下一步不续 v6 checkpoint。
- 新目录 / SwanLab：`beta_v2_1_c1_dual_t4_p1`（https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_dual_t4_p1）。
- 纯 forecast（aux=0）。Trunk 1e-6，heads+sector/size 1e-5，`KRONOS_SPLIT_TRUNK_HEAD_LR=1`，warmup_constant，warmup ratio 0，不开 cosine。
- Best：`forecast`（weighted forecast），不是 `beta_v21_score`。
- 12 segments，soft-stop 9000s，双 T4，eff batch 64，AMP fp16，full val，collect aux 0。
- 仍从 pretrained `luckfu/Kronos-A-Share-Beta-V2-1` + 新 AdamW。shuffled feeding 不变。
