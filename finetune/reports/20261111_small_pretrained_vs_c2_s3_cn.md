# 任务 1：未微调 `NeoQuasar/Kronos-small` 的 OOS 基线（2026-11-11）

回答一个问题：**C2 的微调到底带来了多少？**
C2 的 bootstrap 起点就是 `NeoQuasar/Kronos-small`（`finetune/kaggle_kronos_small_v21.py:419`），
所以这个未微调基线的 IC 就是"微调增益"的零点。

窗口：65 天 OOS 的 **22 天 stride-3 子集**（`2026-07-17 → 2026-10-22`，113,480 行），
decode 协议 seed 20261104 / T=0.65 / top_p=0.8 / 单路径。
kernel：`wynstonliu/kronos-small-pretrained-oos65-s3`（`context_layer=6`）。

机器可读结果：[`20261111_small_pretrained_vs_c2_s3.json`](20261111_small_pretrained_vs_c2_s3.json)

## 0. 加载自检（执行单 §1.2 要求）

```text
hf_loaded: missing = [sector_emb.weight, size_mlp.0/2.*], unexpected = []
worker_dates: 两张卡各 11 天（共 22 天）
```

missing 只有我们新增的零初始化条件头，unexpected 为空 → **HF config 可直接喂给本地 `Kronos`，
Small 与 Base 走同一条 `BASE_OOS_HF_REPO` 路径都通**。此项不再存疑。

## 1. 单模型（同 22 天）

| 模型 | IC | ICIR | 正 IC 天数比 | Top−Bottom 10% | 中性化后 IC |
|---|---:|---:|---:|---:|---:|
| 未微调 Kronos-small | **0.05890** | 0.550 | 72.7% | 1.16% | 0.03762 |
| C2 best（微调完成） | **0.10959** | 1.312 | 90.9% | 2.74% | 0.05447 |

**配对差 `c2 − small_pre` = +0.05069，95% CI [+0.02008, +0.12659]（5 日块 bootstrap，2000 次）**，
下界 > 0 → **微调显著改进了 Small**。

## 2. 判定（执行单 §1.4）

| 条件 | 实际 | 命中 |
|---|---|---|
| `small_pre ≥ C2 − 0.01`（微调收益基本不存在） | 0.0589 < 0.0996 | ✗ |
| **`small_pre ≤ 0.08`（微调确实帮到了 Small）** | **0.0589 ≤ 0.08** | **✓ 命中第 2 行** |
| 介于两者之间 | — | ✗ |

**结论：CE 微调对 Small 有效（+86%），Base 上的崩塌是 Base 特有问题，不是"微调本身没用"。**

## 3. 与 Base 的对照（同 22 天）

| 模型 | 预训练 | 微调后 | 变化 |
|---|---:|---:|---:|
| Small（C2 线） | 0.05890 | 0.10959 | **+86%** |
| Base（base_clean 线） | 0.08924 | 0.0069~0.0235 | **−74% ~ −92%** |

同一个训练范式，在 Small 上提升、在 Base 上摧毁。这是本棒最需要用户注意的不对称。

## 4. 诊断性拆解：C2 的增益有多少来自反转因子

本窗口 −10 日反转因子自身 IC = **0.12932**（强因子）。

| 指标 | C2 | small_pre | 差 |
|---|---:|---:|---:|
| 原始 IC | 0.10959 | 0.05890 | +0.05069 |
| 剥离（市值/反转/行业）后 IC | 0.05447 | 0.03762 | **+0.01685** |

**C2 相对预训练 Small 的优势，约 67%（0.0338 / 0.0507）在剥离反转因子后消失。**

即：C2 的"微调增益"有大约三分之二是**学会了更用力地押 10 日反转**，剩余约三分之一是
反转之外的真增量。这是同一窗口上的事后分解（post_hoc），只作诊断，不作为选型依据。

## 5. 融合（22 天）

| 指标 | 融合 | vs small_pre | vs C2 |
|---|---:|---:|---:|
| IC | 0.11629 | +0.05739 [+0.04279, +0.09377] | +0.00670 [−0.03671, +0.02616] |
| 中性化后 | 0.05630 | — | — |

`small_pre ~ c2` 的日度预测秩相关仅 **0.0514** —— C2 几乎完全替换了它起点的信号，
不是"在原信号上精修"。

融合对 C2 的增量在 22 天上不显著（CI 跨 0），与任务 2 的 65 天结论方向一致、幅度更小。

## 6. 本棒顺手修掉的 bug

`evaluate_base_sealed.py` 的 HF 直读路径把 `local_dir` 写死成 `/kaggle/working/hf_model`，
而**两个 CUDA rank 并发下载同一目录**，一个 rank 会在另一个还在写文件时读到
`FileNotFoundError: .../model.safetensors` → 父进程 `RuntimeError: worker failed`。
Base 那几次没有触发（原因未查，可能是下载时序错开）。已改为按 `BASE_OOS_RANK` 隔离目录（`hf_model_r0` / `hf_model_r1`，
tokenizer 同理）。**任何走 `BASE_OOS_HF_REPO` 的评测都可能踩这个竞态。**
