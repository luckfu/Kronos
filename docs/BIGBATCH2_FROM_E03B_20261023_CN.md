# Bigbatch2：从 e03b best 开新一轮 epoch（换 seed）

用户覆盖交接单「不开新训练」。2026-10-23 启动。

## 配置

| 项 | 旧（e01b–e03b） | 新（bigbatch2） |
|---|---|---|
| coverage_seed | `20261020` | **`20261023`** |
| 起点权重 | epoch2 E05 best | **e03b best**（val 2.261617@457） |
| 优化器 | 段内续跑 last_state | **chunk1 全新**（从 best 权重） |
| 实验名 | `small_0.1_clean_bigbatch_20261022` | `small_0.1_clean_bigbatch2_from_e03b_20261023` |
| Kernel slug | `kronos-small-clean-bigbatch-e0Nb` | **`kronos-small-clean-bigbatch2-e0N-from-e03b`**（不用 e04b） |
| SwanLab run | `smce2_big` | **同 id，`resume=allow`** |
| SwanLab 横轴 | 1–520 | **offset 520**（板子接在 e03b 后面） |
| 墙钟 | 40500s 训练 / 43200s hard | 同左（约 12h dual T4） |

## 推送

日历已经冻结，`panel_sha256` 已钉为 `8539246be495db60ac89f23c4a79f19abae730af6d2a23a3847d477abf376784`。`push_chunk.py` 在没设置面板数据集时仍失败关闭。下一轮推送：

```bash
export KRONOS_KAGGLE_OWNER=wynstonliu
export KAGGLE_API_TOKEN=... KAGGLE_USERNAME=wynstonliu
export KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET=wynstonliu/kronos-temporal-rank-ic-panel-v1
python -u finetune/small_clean_bigbatch_relay/push_chunk.py 1
```

Parent kernel：`wynstonliu/kronos-small-clean-bigbatch-e03b`。
训练数据：`smmt315/a-share-120d-temporal-symbol-holdout`。
选型面板：上面的独立 dataset，文件名 `temporal_rank_ic_panel.pkl`。不要把 `val_data.pkl` 当成选型面板。

## 已推送

e01 已取消。下面是取消前那次 val-loss 运行的记录，不要热重推，也不要用 Rank IC 重选它的输出。

- Kernel：https://www.kaggle.com/code/wynstonliu/kronos-small-clean-bigbatch2-e01-from-e03b（已取消）
- Commit：`1d408ab`（relay 配置）+ slug 对齐后续提交
- Resume：`wynstonliu/kronos-small-clean-bigbatch-e03b` best（SHA `f2fb12a5…`）
- Seed：`20261020` → `20261023`
- SwanLab：`smce2_e01e_rankic1p` / project `finance` / workspace `roc_fu` / segment offset 0（新看板，不再续 smce2_big）
- Budget：40500s train / 43200s hard timeout / dual T4

## 选模协议（下一轮）

e01 已取消，当时按 val loss 选 best，**不要**事后用 Rank IC 重选那次输出，也不要在面板就绪前热重推。下一轮启动默认 `KRONOS_CHECKPOINT_SELECTION_PROTOCOL=temporal_rank_ic_v1`：每 10 个 segment（加首段和末段）用时间切分日频 D10 Rank IC（单路径）更新 `best_model`。val loss 仍记录。晋级替换 C2 仍只走 §5.2。

选型日历已冻结：`finetune/manifests/temporal_rank_ic_v1/selection_window.json`，**60** 个交易日（`2026-04-01` → `2026-06-30`），`sha256` `0721b3defca10d2dab4dcaf433f1efab12304340298e6304fca6d3c9a9f3ce0d`，`train_time_upper_bound` `2024-12-31`。日期列表见 [`VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md`](VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md) §7。

`panel_sha256` 已钉住（见上）。下一轮推送需要 `KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET=wynstonliu/kronos-temporal-rank-ic-panel-v1`。构建命令已离线跑过，日期和日历 SHA 不改：

```bash
python finetune/build_temporal_rank_ic_panel.py \
  --dataset-root /path/to/a-share-120d-temporal-symbol-holdout \
  --output /path/to/temporal_rank_ic_panel.pkl
```

脚本会原子更新 manifest 里的 `panel_sha256`，不改日期和日历 SHA。文件缺失、SHA 不符、日期落进禁止窗口、或面板落到 `val_data.pkl` / symbol holdout 时，`push_chunk.py` / `launch.py` 失败关闭，不会退回 val loss。说明见该目录 README 与协议 §9。
