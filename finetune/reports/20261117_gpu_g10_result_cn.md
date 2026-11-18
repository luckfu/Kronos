# GPU 探索 G10 实验结果：冻结 `G1_UNIV_LAST` 线性骨架的残差同行注意力网络（2026-11-17）

- 事前登记：[`20261117_gpu_g10_frozen_skip_peer_prereg_cn.md`](20261117_gpu_g10_frozen_skip_peer_prereg_cn.md)（未触碰 `2026-10-24+` 封存验收窗）
- 脚本与产出：[`g10_peer_resid_entry.py`](../gpu_alpha/g10_peer_resid_entry.py)（Kaggle `luckfu/kronos-alpha-g10-frozen-skip-peer`，2× T4 耗时 `2197.3s`）
- 机器可读结果：[`20261117_gpu_g10_results.json`](20261117_gpu_g10_results.json) / `scratch/gpu_alpha_g10_out/g10_out/g10_summary.json`

---

## 1. 一句话结论

**在未见面验证集有效收敛（`val 0.1448 → 0.1517`），并在同口径对比基线 `k=10` 小幅提升，但非线性残差并未超越 `2021-2026` 全密度闭式岭回归 `FUSE_C2_G1`（诚实记录）：**
- 同口径纯线性对照 `FUSE_G10_LINEAR_G1REF`（`2022-2026 stride=2`）：Rank IC `0.1700`，`k=10 +15.61%`、`k=20 +15.35%`；
- `FUSE_G10_PEER_BLEND`（冻结该骨架后叠加板块/市场残差混合头）：Rank IC `0.1638 (ICIR 1.66)`，`k=10` 净收益提升至 **`+16.88%`（`10/10` 赢，Calmar `1.043`）**；`FUSE_G10_PEER_RAW` 将流动池前 10/前 20 超额提至 `+3.15% / +2.61%`、`ICIR 1.686`；
- 但全系列高密度闭式线性头 `FUSE_C2_G1`（`2021-2026 stride=1`）仍保持明显优势（`k=10 +22.17% / k=20 +20.13%`），说明在固定 512 维特征下，同日板块/全市场残差注意力的泛化增益弱于更长近期历史全密度闭式正则化与跨架构基模型集成。
