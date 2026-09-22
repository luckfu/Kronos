# Stage3 因果 val vol 代理 vs 18d 生产 OOS 失配分析

- 时间（上海）：2026-09-22
- 范围：本地只读；未推 Kaggle / 未开 GPU / 未 git push
- 主文件：`stage3_vol_proxy_vs_18d_mismatch_analysis.json`

## 结论（一句话）

两边公式方向相同（均为 **realized/pred**），realized 几乎一样；**预测路径波动差约 2×**，因为 Stage3「production decode」仍是 **teacher-forced 按 horizon 独立 top-N 拼接的伪路径**，而 18d 是 **自由自回归采样路径**——名义 T/top_p/N 相同，生成对象不同。

## 关键数字

| 表面 | pred_path_vol | realized | ratio (R/P) | 含义 |
|------|---------------|----------|-------------|------|
| Stage3 prod-smoke baseline | 0.02840 | 0.02423 | **0.926** | 过波动 |
| Stage3 uniform（同烟测） | 0.03424 | 0.02423 | ~0.71 | 更过波动 |
| 18d prod T=0.65/p0.8/N=5 | 0.01390 | 0.02486 | **2.102** | 欠波动 |

## 公式（两边）

1. **Stage3 evaluate**：`vol_calibration_ratio = mean(realized_i / pred_i)`，其中 `pred = Σ w_n·std(伪路径_n)`（混合权重）；日志里另有 `uniform_path_vol`、`mixture_mean_path_vol=vol(E[path])`。
2. **18d OOS**：`calibration_ratio_realized_over_predicted = mean(realized / predicted)`，其中 `predicted = mean_n std(AR样本路径_n)`（均匀）。

日收益与 denorm 代数一致；H=10、lookback=120、std ddof=0。

## 假设排序

1. **（主因）teacher-forced 伪路径 vs free AR** — pred 0.028 vs 0.014  
2. 确定性 top-N vs multinomial 采样  
3. mixture vs uniform — **不能翻转方向**（uniform 更过波动）  
4. 因果 val vs 密封 18d 总体 — realized 比 ≈0.975，否决单独致因  
5. E[R/P] vs E[R]/E[P] Jensen — 不改变过/欠方向  

## 建议的下一步（设计 only，现在不要开跑）

同 checkpoint（C2 best）+ 同 decode lock，在**同一批窗口**上并排算：

- Stage3 mixture / uniform / mixture_mean  
- Free-AR `predicted_path_vol`  

窗口：因果 val 分层抽样 512–2048；可选 1–2 个密封信号日。若同窗上仍是 ~0.9 vs ~2.0，则确认代理失配，应先改训练代理再谈 40–60 segment。

## 禁止

- 开 40–60 segment vol 长训 / 第二块 GPU  
- 在密封 18d 上重搜 T/top_p/N  
- 把 causal-val 的 0.93 当成已对齐 18d 欠波动问题  
