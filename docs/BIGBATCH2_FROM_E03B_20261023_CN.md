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

```bash
export KRONOS_KAGGLE_OWNER=wynstonliu
export KAGGLE_API_TOKEN=... KAGGLE_USERNAME=wynstonliu
python -u finetune/small_clean_bigbatch_relay/push_chunk.py 1
```

Parent kernel：`wynstonliu/kronos-small-clean-bigbatch-e03b`。
面板：`smmt315/a-share-120d-temporal-symbol-holdout`。

## 已推送

- Kernel：**RUNNING** https://www.kaggle.com/code/wynstonliu/kronos-small-clean-bigbatch2-e01-from-e03b
- Commit：`1d408ab`（relay 配置）+ slug 对齐后续提交
- Resume：`wynstonliu/kronos-small-clean-bigbatch-e03b` best（SHA `f2fb12a5…`）
- Seed：`20261020` → `20261023`
- SwanLab：`smce2_big` / project `finance` / workspace `roc_fu` / segment offset 520
- Budget：40500s train / 43200s hard timeout / dual T4

## 选模协议（下一轮）

e01 已取消，当时按 val loss 选 best，**不要**事后用 Rank IC 重选那次输出。下一轮启动默认 `KRONOS_CHECKPOINT_SELECTION_PROTOCOL=temporal_rank_ic_v1`：每 10 个 segment（加首段和末段）用时间切分日频 D10 Rank IC（16 路径中位数）更新 `best_model`。val loss 仍记录。

冻结选型窗 `finetune/manifests/temporal_rank_ic_v1/selection_window.json` 还没落盘。文件缺失时 `push_chunk.py` / `launch.py` 失败关闭，不会退回 val loss 或 symbol holdout。说明见该目录 README 与 [`VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md`](VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md) §9。
