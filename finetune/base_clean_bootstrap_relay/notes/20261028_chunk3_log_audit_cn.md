# Bootstrap chunk3 日志审计 + 日志取数整改（2026-10-28）

范围：`base_clean_0.1_bootstrap` 接力第 3 段（luckfu 推送，父块 = user281434 chunk2）。
数据来源：chunk3 kernel 的 console log / `run.log` / `metrics.jsonl` / `summary.json`（本次由
`ListKernelSessionOutput` 直接取回，见 §4）。

---

## 1. 这一棒实际发生了什么（先结论）

| 项 | 值 |
|---|---|
| 父 → 本棒 | `completed=317` → `resume_segment=318`（log 里 `resume from coverage segment 318` 已确认） |
| 本棒区间 | **318 → 353**，共 **36 seg**（`max_segments=36`），`stop_reason=segment_limit` |
| 下一棒 | `resume_segment=354`，`progress.json` 状态 `stopped / complete` |
| 墙钟 | 00:31:24 → 02:54:41 UTC = **2h23m 训练**（= 8904s 含启动 ≈ 240s/seg） |
| 单 seg | 中位 **~3m48s**，含验证 **~4m05s**（`avg20` = 3m53s） |
| 峰值显存 | allocated **2.55 GB** / reserved 2.65 GB（双 T4 DDP，micro4×accum8，global 64） |
| 更新数 | 313 updates/seg（20000 窗口 / 64），**与 chunk1/2 一致** |
| LR | 主干 3.15e-7、条件 2.46e-6（cosine 已走到 520 段的 353/520） |
| OOM / 重启 / 丢步 | **无**。dataloader、父权重加载（192/197 tensors）、`last_state` 续优化器均正常 |

父权重衔接：`best_sha=f618552a…`、`last_sha=f9f25aac…`，`best_equals_last=false`，
`best_metric_segment=21`。**本棒按 `KRONOS_DISABLE_BEST_UPDATES=1` 只续 last，Best 冻结在 seg21**。

## 2. 曲线（chunk3 内，含前后对照）

```
seg    forecast   Δ(seg)      condition Δfull-none / Δfull-shuf
320    2.59658               -1.50e-04 / -4.11e-05
330    2.59550   -1.1e-03    -1.29e-04 / -2.24e-05
340    2.59458   -9.2e-04    -1.30e-04 / -4.53e-05
350    2.59350   -1.1e-03    -1.55e-04 / -1.87e-05
353    2.59314   -3.6e-04    -1.29e-04 / -4.64e-05
```

- 全段：seg1 **2.99579** → seg353 **2.59314**（−13.4%）。近 100 段只降了 **0.0143**。
- 速度：−1.1e-4/seg 且仍在变慢；LR 已到 3e-7，**继续裸跑的边际收益≈0**。
- 条件通道：`Δfull-none ≈ −1.3e-4`、`Δfull-shuf ≈ −4.6e-5`，与 10-27 报告里
  "Seg240–250 时 −1.03e-4 / −3.62e-5" 同量级，**仍在身份噪声地板上**（Small bootstrap 末点
  是 −8.76e-3 / −2.16e-3，差两个数量级）。
  按 10-27 的结论这是**数据暴露差**（Small 51.8M 窗口 vs Base 现在 7.06M），不是配方坏。
- 条件能量：`condition_output_norm` 8.0、`trunk_ratio` 0.54%，`condition_grad_rms` 2.4e-6，
  全程无爆炸；`sector_emb_norm` 2.51、`size_mlp_output_norm` 4.91，单调缓升，健康。

## 3. **Best 是假的，必须注意**

`best_val_loss=2.69015` 是 **seg21** 的旧值，`KRONOS_DISABLE_BEST_UPDATES=1` 让它 353 段都没再更新。
实际 seg353 的 selection（weighted forecast）已是 **2.59366**，比 Best 好 **0.0965**。

→ 任何评估 / 晋级 / 下一阶段初始化，**只能用 `last_model` + `last_state`**；
`checkpoints/best_model/` 现在等于 332 段前的权重，不要用。

## 4. 日志为什么"训练途中拿不到"——根因与整改

### 4.1 根因（三条，互相叠加）

1. **`kaggle kernels output` 只在 session 结束后才落盘 `/kaggle/working`**。跑的过程中该接口
   返回空/404 —— 这是过去"拿不到日志"的直接原因。
2. 就算能拿到，`kaggle kernels logs` 吐出来的是 **JSON 事件数组**（`[{stream_name,time,data},…]`），
   不是纯文本，看着像乱码；2026-10-27 交接里写的 `kaggle kernels logs <ref>` 也没说明这点。
3. 本机 `kaggle` CLI 2.2.4 打 `api.kaggle.com` 会间歇性 `SSLEOFError`（今天 12:xx 起所有
   kernels RPC 全挂，而同一 token 的裸 REST/curl 100% 正常）——所以"取数"不能只依赖 CLI。

### 4.2 现在可用的三条通道（都已在仓库里）

| 通道 | 命令 / 端点 | 用途 |
|---|---|---|
| **实时 SSE** | `GET {api}/v1/kernels/logs/stream/<owner>/<slug>`（running 时 = event-stream，结束后回落为落盘 blob） | 训练途中逐行日志 |
| **落盘 console** | `POST {api}/v1/kernels.KernelsApiService/ListKernelSessionOutput` 的 `log` 字段 | 事后完整日志（无需 output 目录） |
| **SwanLab** | run `base_clean_0_1_bootstrap` | 数值曲线（本棒已补条件消融 key，见 4.4） |

封装脚本（新，本次提交）：

```bash
# 训练途中，一行一个 seg + ETA（Ctrl-C 只是断开，不影响训练）
python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --follow

# 刚推完还没起来：先等 session 再挂上
python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --follow --wait

# 事后复盘最近 12 段
python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --tail 12
```

它直连 REST（不经过 kaggle CLI），凭据读 `KAGGLE_API_TOKEN` 或 `~/.kaggle/credentials.json`。

### 4.3 runner 侧的打印整改（`finetune/kaggle_kronos_base_clean.py`，纯观测、不动数学）

1. **每行加 UTC 时间戳**：`[07:12:33Z] [Rank 0, Segment 354/520, Step 50/313] …`。
   之前 7k 行日志**一个时间戳都没有**，只能靠 `metrics.jsonl` 的 `updated_at` 反推，
   也分不清"卡住"和"在跑"。
2. **每段一行 `[PROGRESS]`（机器可读）**：
   ```
   [PROGRESS] {"segment":354,"total":520,"resume_segment":354,"chunk_end_segment":425,
   "segment_seconds":245,"avg_segment_seconds":236,"run_elapsed_seconds":256,
   "eta_chunk_seconds":16797.6,"eta_total_seconds":38961.1,
   "forecast":2.5931,"selection":2.5940,"best":2.6902,
   "delta_full_none":-1.29e-4,"delta_full_shuffled":-4.64e-5,"cuda_peak_gb":2.55}
   ```
   这是 `monitor_chunk.py` 的首选输入；老日志没有它也能用摘要块解析（向后兼容）。
3. **条件消融上 SwanLab**：`validation/condition_{full,none,shuffled}_loss`、
   `validation/condition_full_minus_{none,shuffled}`、`condition/{output_norm,trunk_norm_ratio,
   grad_rms,sector_emb_norm,size_mlp_norm}`、`progress/{completed_segments,segment_seconds,
   avg_segment_seconds,eta_chunk_minutes,eta_total_minutes,cuda_peak_gb}`。
   交接文档里"曲线里没有 ablation，只有 logs 里有"的坑，就此闭合。
4. 所有解析都在 `try/except` 里，解析失败最多丢一条观测，**不可能中断训练**。

### 4.4 仍然存在的限制（写清楚，别再踩）

- Kaggle 单 session **GPU 上限 12h**；script kernel 的 console 在 GCS 上**有分钟级延迟**，
  所以"实时"= SSE，SSE 断了会自动重连并从头发，脚本按事件序号去重。
- `metrics.jsonl` 会带上父块的整段历史（本棒取回 1197 行 = seg1–353），**不要**把它当成
  "本棒跑了 353 段"；以 `progress.json` 的 `current_segment` 为准。
- 同板回放仍按 `KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT=353` 跳过父段，别改小。

## 5. 下一棒的取舍（给决策用，不代替决策）

到 520 还差 **167 seg ≈ 11h**（按 4m05s/seg），可拆 2–3 棒。但：

- LR 已 3.15e-7、近 100 段只降 0.0143 —— 纯续 Bootstrap 的**边际收益很低**；
- Main 的配方是**重建 optimizer、峰值回到 1e-5**，比在 3e-7 上再跑 11h 更可能推动曲线；
- 唯一支持继续 Bootstrap 的理由是"多 3.3M 窗口的数据暴露"（条件身份靠数据量），
  按 10-27 的结论这条路**不是没道理**，但 11h 只换来 Δ 位移率不变。
