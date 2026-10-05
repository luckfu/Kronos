# Beta v2.1 C1 Rank True Time-OOS

- Kernel：`luckfu/kronos-beta-v21-c1-rank-oos`
- 对比：Seg155 forecast / Seg19 frozen / Seg8 unfreeze
- 数据：`a-share-120d-temporal-symbol-holdout` 根目录密封包（非 val_data）
- 重建：`python3 finetune/build_kaggle_beta_v21_c1_rank_oos_kernel.py`
- 推送：`kaggle kernels push -p finetune/kaggle_beta_v21_c1_rank_oos_kernel`
