# Base clean 崩塌定位：训练在 10 段内摧毁预训练信号（剂量-效应）

日期：2026-11-10。承接 [`20261110_base_clean_65d_oos_vs_c2_cn.md`](20261110_base_clean_65d_oos_vs_c2_cn.md)
（复刻失败）与 [`20261109_cosine_schedule_horizon_cn.md`](20261109_cosine_schedule_horizon_cn.md)（调度表坑）。

测量基准：65 天干净 OOS（2026-07-17→10-23，335,312 样本）的 **22 天 stride-3 子集**，
统一 decode 协议 seed 20261104 / T=0.65 / top_p=0.8 / 单路径 / `evaluate_base_sealed.py`。
所有探针从同一个 `NeoQuasar/Kronos-base`（SHA `abff193a…`）出发。

## 主结果

| 检查点 | LR（predictor/condition） | 段数 | IC（22 天） | 信号保留 |
|---|---|---:|---:|---:|
| 预训练 base（未微调） | — | 0 | **0.08924** | 100% |
| P2 探针 | **1e-7 / 1e-7** | 10 | **0.05495** | 62% |
| P1 探针 | 1e-6 / 1e-5（原配方） | 10 | **0.02242** | 25% |
| Bootstrap best@21（历史 chunk1） | 1e-6 / 1e-5 | ~21 | 0.0286 | 32% |
| Main 第 1 段 best（历史） | 1e-5/1e-5 | 389+1 | 0.0235 | 26% |
| 全链跑完（cosine@88） | — | 1700+ | 0.0069（65 天全量） | 8% |

（历史点 best@21 与 Main seg1 取自旧 kernel 版本的 `best_model`，其余为本轮实测。）

## 结论

1. **破坏是即时的**：原配方 10 段（约 3,130 优化步）内 IC 从 0.089 → 0.022。
2. **LR 是主凶之一但不充分**：峰值 LR 降 10 倍（1e-6/1e-5 → 1e-7），破坏从 −75% 收窄到 −38%。
   即便 1e-7 也在 10 段内毁掉三分之一 —— **梯度方向本身与 OOS 排序对着干**，
   这与此前"val 降 / OOS 降"的反相关在梯度层面互为印证。
3. 因此嫌疑从"LR 太大"进一步转向**目标函数**：`history_loss_weight=0.02`（几乎放弃历史重建）
   + 前重后轻的 horizon 权重（`1.364,1.364,1.364,…,0.455`），把模型从预训练的
   全序列建模目标上硬拉走。

## 交互拆解（P3/P4 已收尾）

| 探针 | LR | history 权重 | IC（22 天） | 信号保留 |
|---|---|---|---:|---:|
| P1 | 1e-6/1e-5 | 0.02 | 0.02242 | 25% |
| P2 | 1e-7 | 0.02 | 0.05495 | 62% |
| P3s | 1e-6/1e-5 | 1.0 | 0.01668 | 19% |
| P4 | 1e-7 | 1.0 | 0.06140 | 69% |

1. **LR 是唯一能控的主变量**：1e-7 保留 62~69%，1e-6/1e-5 只剩 19~25%。
2. **history 权重不是主凶**：原 LR 下补到 1.0 反而更差（0.022→0.017），只在低 LR 下微有帮忙（0.055→0.061）。
3. **最好的组合（P4）仍在 10 段内丢 31%** —— 没有任何受试配方能"边训边保住"。

## 根因

预训练 base 本身有真信号（IC 0.089，剥离反转后仍 0.066）。我们的微调范式 ——
不论把 LR 降到 1e-7、还是把 history reconstruction 补到 1.0 —— 都会在约 3,000 个优化步内
毁掉三成到七成的 OOS 排序信号。这不是某个旋钮设错，而是**这套微调目标/数据与预训练表征
根本不兼容**：梯度方向本身就与 OOS 排序对着干，与此前观察到的"val 降 / OOS 降"反相关同源。

## 对决策的含义

- 复刻 C2 训练路径在 base 上不成立（`20261110_base_clean_65d_oos_vs_c2_cn.md`）。
- 根因不是选型尺子、不是数据版本、不是条件层、不是单点 LR —— 而是微调本身。
- 可行退路（待 65 天全量确认）：**不微调直接用预训练 base**，其信号为 C2 的 ~78%（22 天口径）。
- 若要走微调，方向不是"调好 LR/损失"，而是**冻结主干、只训顶层**或极短程低 LR —— 需另立探针验证。

65 天全量预训练评测：kernel `wynstonliu/kronos-base-pretrained-oos65-full`。

**结果（65 天全量，335,312 样本）**：预训练 base **IC 0.08973**，direction 0.5192，MAE 0.0605，RMSE 0.0919。
与 22 天子集 0.08924 一致 → 子集有代表性。

最终对照（65 天、同协议）：

| 模型 | IC（65 天） | 相对 C2 |
|---|---:|---:|
| C2 best（Small） | 0.11405 | 100% |
| **预训练 base（未微调）** | **0.08973** | **79%** |
| base_clean 任一 stage | 0.002~0.023 | 2~20% |

→ "不微调直接用预训练 base" 能拿到 C2 约八成的排序能力，且是诚实的时间外样本外结果。
这是 base 线当前唯一站得住的可行路径。

## 排查中顺手修掉/发现的基建问题

- `kaggle_kronos_base_clean.py`：`stop_gpu` 在无 SwanLab key 时未定义，关机时报
  `UnboundLocalError` 使 COMPLETE 变成 ERROR（探针 checkpoint 不可挂载）。已在 HEAD 修复。
- `best_model` 在 bootstrap 阶段只在全量验证时更新；`large_interval=1,000,000` 使 10 段探针
  的 `best_model` 始终是**初始快照**（主干与预训练逐位相同、条件头全零）——第一轮 P1/P2
  评测因此测的是预训练而非训练产物，已改用 `last_model`。
- 评测器 tokenizer 兜底里 `weights` 变量名重用，把 `model_loaded.weights` 日志打成了
  tokenizer 路径（显示问题，不影响实际加载）。
- 预训练权重无 `sector_emb/size_mlp`，走 `BASE_OOS_HF_REPO` + `strict=False` 直读评测。
- Kaggle 数据集 create 有瞬时失败（`No gcs url found` / `Path must be unique` / title 唯一性），
  需重试并回下载校验 SHA。
- kernel 版本输出 zip 在 ~0.5–1.1 GB 处被连接重置且不支持 Range；`zipstream.py` 顺序解压器
  可从截断归档抠出排在前面（`best_model`，约 450 MB 处）的完整条目，`last_model` 排在
  `last_state.pt`（约 1.2 GB）之后而不可达。

## 复现

探针推送：`finetune/base_clean_bootstrap_relay/push_probe.py`（`KRONOS_PROBE_SLUG` 等 env 驱动）。
评测推送：`finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py`（`BASE_OOS_*` env 驱动）。
