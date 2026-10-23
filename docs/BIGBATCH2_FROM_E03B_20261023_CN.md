# Bigbatch2：从 e03b best 开新一轮 epoch（换 seed）

用户覆盖交接单「不开新训练」。2026-10-23 启动。

## 配置

| 项 | 旧（e01b–e03b） | 新（bigbatch2） |
|---|---|---|
| coverage_seed | `20261020` | **`20261023`** |
| 起点权重 | epoch2 E05 best | **e03b best**（val 2.261617@457） |
| 优化器 | 段内续跑 last_state | **chunk1 全新**（从 best 权重） |
| 实验名 | `small_0.1_clean_bigbatch_20261022` | `small_0.1_clean_bigbatch2_from_e03b_20261023` |
| Kernel slug | `kronos-small-clean-bigbatch-e0Nb` | **`kronos-small-clean-bigbatch2-e0N`**（不用 e04b） |
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
