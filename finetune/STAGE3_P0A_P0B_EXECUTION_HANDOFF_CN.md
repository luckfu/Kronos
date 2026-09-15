# Stage 3 P0a / P0b 执行交接文档

日期:2026-09-15。状态:**代码修改已完成且未提交;两个 Kaggle kernel 尚未创建;未执行任何 commit/push**。
本文是唯一执行依据。下一会话从 §5.1 抄文件、按 §5.3 提交、按 §5.4 推 kernel,不必再推导规格。

上一份机制审计:`finetune/reports/stage3_c2_c3_mechanism_audit_20260915.json`。技术报告对应小节 6.7.9–6.7.10、11.1。

## 0. 下一会话开场即做(5.1→5.4)

执行顺序:**P0b → P2(并行,CPU)→ P0a → 依据 P0a 判定树决定 P1**。本会话只完成 5.1–5.4(两个 kernel 落地 + 两个 commit + push + `kaggle kernels push`)。P0a 的 OOS 与 P2 回测不在本轮。

1. 按 §5.2 **先**创建 `finetune/kaggle_stage3_c2_c3_interp/`(P0b),再按 §5.1 创建 `finetune/kaggle_stage3_ce_only_control/`(P0a)。`COMMIT` / `SOURCE_COMMIT` 先写占位符 `__PIN_AFTER_COMMIT_1__`。
2. 按 §5.3 拆两个 commit:先 trainer 三文件 + 交接文档 + control wrapper,取出 SHA 填进两个 kernel,再提交 interp 目录。
3. `git push origin master`。
4. 按 §5.4 **先** `kaggle kernels push` P0b,再 push P0a。配额耗尽则等到 **2026-09-19** 刷新后同一顺序重推。

不要:续训 C3、λ sweep、救 Path、复用 C3 的 SwanLab v2 dashboard、改 evaluator、本地修 conda/numpy。

## 1. 一句话状态

用户已批准 P0a(CE-only 对照)+ P0b(C2×C3 权重插值扫描),并给出硬约束与判定树。trainer/smoke 的 λ 与 milestone 改造已写完(未提交),剩两个 kernel 文件、提交推送和启动。

## 2. 诊断结论(已获用户认可,作为方案依据)

1. **Path Loss 结构性触底**:λ=0.05 EMA 归一化后,加权 Path 梯度仅约 CE 的 2.5%。且实测 raw Path Huber 0.0066 ≈ `δ×(E|e|−δ/2) = 0.02×(0.34−0.01)`,即 Path loss 就是 `0.02×MAE`,其下限由 Top-16 stop-gradient 锚点的量化误差决定,权重再分配无法突破。**不做 λ sweep、不续训 C3、不修 mixture decode,均已定案。**
2. **C3 的 OOS 恶化主嫌疑是 CE 配置混淆**:相对 C2 best 同时改了均匀 horizon 权重(H10 相对权重 +120%,H1 −27%)、去掉 history loss、加了近惰性的 Path 项。by-horizon 证据:符号改善集中 H8–H10,rank 崩塌最深 H4–H6,与远端加权方向一致。
3. **机制结论(论文级)**:验证 CE 与横截面 Alpha 脱钩(alpha-optimal ≠ CE-optimal);分布锐化(熵 1.844→1.761、joint mass 0.570→0.583)用排序信息换符号命中率(D10 +4.0pp)。
4. P1 方向已定为 **C2 原始 CE + 截面 pairwise rank loss**(先朴素 logistic pairwise,不加 |Δr| 加权 / KL / 新 sampler 的复合),但 **必须等 P0a 归因结果**;KL anchor 放 V3-B。

## 3. P0a:严格 CE-only causal control(双 T4)

与 C3 的**唯一训练差别**是 `lambda_path=0`。验证选模量随 λ 变为纯 CE(代码里 `total = CE + λ×normalized`,λ=0 自然退化)。raw path 指标在验证段仍真实计算并记录,保持与 C3 序列可比。训练期跳过候选解码。

### 3.1 对照表(定案,逐项锁定)

| 项 | Stage 2 C2 best(参考,不是本实验) | C3 Path Alignment(已跑完) | P0a CE-only control(要跑) |
|---|---|---|---|
| 起点权重 | C2 best Segment 179 | 同左,SHA `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a` | **同 C3** |
| tokenizer | SHA `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee` | 同左,冻结 | **同 C3** |
| 初始化 | — | 只继承模型权重,fresh AdamW wd=0.01 | **同 C3** |
| LR / scheduler | Stage2 cosine 退火末端 ~1e-6 | 2e-6 **fixed** | **同 C3** |
| seed / coverage | Stage2 自己的 seed | 20260915 | **同 C3** |
| 预算 | — | 15 segments = 4695 steps(每段 313 step) | **同 C3** |
| global batch | 64 | 64(32×2 T4) | **同 C3** |
| 每段窗口 | 20,000 | 20,000 | **同 C3** |
| 验证 | Stage2 forecast NLL,123,836 | 全量因果验证 123,836 | **同 C3** |
| token CE | 带权 horizon + 0.02 history | **均匀 CE,无 history** | **同 C3(均匀 CE,无 history)** |
| `lambda_path` | 无 | **0.05** | **0** |
| 训练期候选解码 | 无 | Top-16 mixture decode | **跳过**(λ=0 分支) |
| 验证 raw Path / MAE / 熵 / joint mass | 无 Path | 真实计算 | **真实计算,与 C3 序列可比** |
| 选模量 | forecast NLL | `CE + 0.05×raw Path Huber` | `CE + 0×…` = 纯 CE |
| AMP | Stage2 fp16 | fp32 无 AMP | **同 C3** |
| DDP | — | `find_unused_parameters=False` | **同 C3**(CE 单独反传对全部可训练参数有梯度) |
| SwanLab run id | — | `small_0.1_stage3_joint_path_alignment_from_c2_best_v2` | **`small_0.1_stage3_ce_only_control_from_c2_best_v1`(严禁复用 v2)** |
| Milestone | — | 无 | seg 0 = 初始 best(`initial_unvalidated`);**seg 5 = step 1565 = 1/3**;**seg 10 = step 3130 = 2/3**;seg 15 = step 4695 = last。由 `--milestone-segments 5,10` 写到 `checkpoints/milestone_seg05\|10/` |
| Kernel 形态 | — | C2 续 C3,resume optimizer | **fresh**,`STAGE3_BASELINE_BEFORE_RESUME=1`(fresh 也会先做一次 C2 best 因果 baseline) |
| 建议时限 | — | C3 单段 train+val 650–678s | `STAGE3_TARGET_SEGMENTS=15`,`STAGE3_MAX_RUNTIME_SECONDS=12600`,`STAGE3_HARD_TIMEOUT_SECONDS=15000`。训练期跳过解码,单段应快于 C3,总时长约 2.5–3h + 预检 |

### 3.2 判定树(用户定案,原文)

P0a 的 15 段因果验证 IC 轨迹(以及随后 §5.5 的 13 日 OOS)出来后,按下面三支写结论并决定 P1。数字是示意锚点,不是硬阈值。

- **A**:CE-only 也崩(如 IC 0.187→0.06–0.08)→ 毒性来自 CE 配置变化;P1 = 恢复 Stage2 原 CE(带权 horizon + 0.02 history)+ 小权重 rank 辅助。
- **B**:CE-only 保住 alpha(≈0.17–0.19)而 C3 崩 → Path Alignment 负迁移,正式归档;P1 = C2 CE + rank。
- **C**:CE-only 中度下降(如 0.12)而 C3 更差 → CE 配置为主害、Path 放大;同样汇聚到 C2 CE + rank,但论文要写放大效应。

三支都不授权继续救 Path,也不授权 λ sweep。

## 4. P0b:C2×C3 权重插值扫描(先跑)

`W_α = (1−α)·W_C2 + α·W_C3`,α ∈ {0, 0.25, 0.50, 0.75, 1},**执行顺序 [0, 1, 0.5, 0.25, 0.75]**(sanity 在前,中点最优先)。同一 sealed 生产 evaluator(13 signal dates / 66,986 identity,2026-08-11 至 08-27)。

**结果定位:exploratory candidate;13 日集已有设计污染,production candidate 必须另走 sealed 新时间窗。**

### 4.1 五条硬约束与机器校验(全部写入 kernel 日志/summary)

| # | 硬约束 | 机器校验(写进脚本,不要靠人眼) |
|---|---|---|
| 1 | 插值前 C2/C3 state dict **key 集与 shape 完全一致**;统一转 float32 再线性组合 | `set(c2)==set(c3)`;逐 key `shape` 相等;`(1−α)*a.float()+α*b.float()`;逐 tensor `torch.isfinite` |
| 2 | tokenizer、optimizer、scheduler、EMA **一律不插值**;checkpoint 目录模板(config.json 等)取自 **C2 best**,仅替换 `model.safetensors` | 只 `load_file` 两个 `model.safetensors`;拷贝 C2 best 目录中除该文件外的全部文件;不读取 C3 的 optimizer/EMA/state |
| 3 | 逐 tensor NaN/Inf 检查;`α=0` 重载后逐 tensor 与 W_C2 `torch.equal`,`α=1` 同理 | 插值后 `save_file` → `load_file` 再比。`α=0`: `torch.equal(reloaded[k], c2[k].float())`;`α=1` 对 C3。这替代「重跑必须逐位相等」 |
| 4 | evaluator 代码**零改动**;`α=0`/`α=1` 作为端到端 sanity,与参考预测比对 | 直接 `from finetune.evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples`。调用签名与 C3 OOS 相同:`evaluate_predictions(label, model, tokenizer, date_records, store, device, 64, 1, 20260906, True)` |
| 5 | shard 粒度 = `(α, date)`,可断点续跑;超硬限优雅退出 | `shards/{label}_{date}.csv.gz` 存在即跳过;每 shard 后若 `elapsed>39600` 写 `summary.json(status=partial)` 后 `sys.exit(0)`;全部完成写 `status=complete` |

**不要**把 α=0/1 的生产预测与参考文件做硬失败比对。原因:生产 evaluator 带 **fp16 autocast** + `torch.manual_seed(seed+date_index)` 按日期播种采样;C2 参考跑在 **P100**,本 kernel 是 **T4**。跨卡允许微小数值差。按 `identity` merge(唯一性断言)后报告逐 horizon `max|Δ| / mean|Δ| / spearman`(rank-agreement),写入 summary 的 `sanity` 块,**不硬失败**。

α=0/1 的权威正确性门禁是 **tensor `torch.equal`**,不是预测逐位相等。

### 4.2 输入布局(精确到 glob 与来源)

Kaggle 把 `dataset_sources` 和 `kernel_sources` 都挂到 `/kaggle/input/**`。全部用 `find_one`,匹配数不是 1 就失败。

| 输入 | 来源 | glob / 选取规则 |
|---|---|---|
| W_C2 | kernel `smmt315/kronos-small-0-1-stage2-cosine-refinement-c2` 输出 | `**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors`。模板目录 = 该文件的 parent(含 `config.json` 等) |
| W_C3 | kernel `smmt315/kronos-small-0-1-stage3-joint-path-c3` 输出 | `**/stage3_joint_path_smoke/checkpoints/last_model/model.safetensors` |
| tokenizer | 同上 C2 kernel 输出(C3 输出不含 tokenizer) | `**/Kronos-Tokenizer-base/model.safetensors`,加载用 `.parent` |
| C2 参考预测 | kernel `smmt315/kronos-small-0-1-c2-alpha-oos-evaluation` 输出 | `**/kronos_small_0_1_stage2_oos/predictions.csv.gz`,路径必须含 `c2-alpha-oos-evaluation`;取 `model=='c2_best_segment_179'`;行数必须 == 66,986 |
| C3 参考预测 | kernel `smmt315/kronos-small-0-1-stage3-c2-c3-oos` 输出 | `**/kronos_small_0_1_stage3_c2_c3_oos/predictions.csv.gz`;取 `model=='c3_last_segment_15'`(该文件还拼了 C2 行,必须过滤) |
| sealed 评估包 | **dataset** `luckfu/a-share-120d-temporal-symbol-holdout`(不是某个 kernel 输出) | 唯一同时满足:存在 `evaluation_manifest.json` + 同目录 `evaluation_panel.pkl` + `evaluation_samples.jsonl`,且 manifest.`temporal_isolation.incremental_signal_start` 有值。再断言 `strictly_after_parent_latest_training_target` 或 `targets_strictly_after_training_target_end` |

label = `interp_alpha_000/025/050/075/100`。推理:`Kronos.from_pretrained(dir, num_sectors=86, num_size_buckets=0, context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64)`。每个 α 前后 `del model; gc.collect(); torch.cuda.empty_cache()`。

每 α 完成后:拼接行数断言 == `len(records)` → 复用 C3 OOS 的 `summarize()`(逐 horizon + `by_signal_date_d10`)→ 立即写 `metrics_{label}.json`(partial 运行也能交付已完成 α)。

summary 顶层必须写:

```json
"purpose": "exploratory_only; 13-date OOS design-contaminated; production candidate requires sealed fresh-window OOS",
"oos_used": true
```

环境策略与 `stage3_c2_c3_oos.py` 相同:**不装 torch**,clone 仓库到 pin 的 commit,用 Kaggle 预装环境。

## 4.3 已完成的代码修改(全部未提交,`git diff` 可见)

1. **`finetune/stage3_training_model.py`** — `forward()` 新增分支:`config.weight == 0 and self.training` 时跳过候选解码,直接返回 `total=token_loss`,metrics 用同形状零张量填充(s1 熵仍真实计算);验证分支不变(raw path 照常计算)。依据:CE 单独反传对全部可训练参数有梯度(梯度诊断已分别算过 CE 梯度),DDP `find_unused_parameters=False` 安全。λ>0 路径逐位不变。
2. **`finetune/train_stage3_path_alignment.py`** — 新增 `--lambda-path`(默认 0.05)与 `--milestone-segments`(默认空);`Stage3TrainingModel(…, config=PathAlignmentConfig(weight=a.lambda_path))`;启动日志、SwanLab config(`lambda_path`、动态 `validation_objective` 字符串)、`best_metric.json` definition 同步参数化;seg 结束后 rank0 保存 milestone(权重 + milestone.json);`validate_resume` 校验键加入 `lambda_path`。
3. **`finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py`** — `RUN_ID`→env `STAGE3_SWANLAB_RUN_ID`;新增 `STAGE3_LAMBDA_PATH`(默认 0.05)、`STAGE3_MILESTONE_SEGMENTS`;manifest 的 `loss` 字符串动态化并新增 `lambda_path`/`milestone_segments` 字段;训练命令追加 `--lambda-path`(及 milestone);`--baseline-before-resume` 移出 `if RESUME_KERNEL:` 使 fresh run 也可用。默认值全部保持旧行为,C2/C3 复现链路不受影响。

**本地测试回归被环境阻断**:`pytest tests/test_stage3_*` 在 numpy C 模块导入阶段 SIGABRT(exit 134,conda base env 损坏),未触及任何项目代码,**与本次改动无关**。处置:不必修本地环境;权威门禁是 kernel preflight(smoke entry 训练前本就跑同样两个测试文件,失败即退出)。

另:工作区还有未提交的 C3 OOS / 梯度诊断 kernel 与 `reports/stage3_c2_c3_mechanism_audit_20260915.json`。它们已经跑过,与 P0a/P0b 启动无关,**不要塞进下面两个 commit**。技术报告 `SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md` 的本地修改可随 commit 1 一起归档。

## 5. 剩余步骤(按序执行)

### 5.1 创建 `finetune/kaggle_stage3_ce_only_control/`(P0a,P0b 之后启动)

下面两份文件原样落盘。`COMMIT` 先写 `__PIN_AFTER_COMMIT_1__`,§5.3 填真 SHA。

`kernel-metadata.json`:

```json
{
  "id": "smmt315/kronos-small-0-1-stage3-ce-only-control",
  "title": "Kronos Small 0 1 Stage3 Ce Only Control",
  "code_file": "stage3_ce_only_control.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["luckfu/a-share-120d-temporal-symbol-holdout"],
  "kernel_sources": ["smmt315/kronos-small-0-1-stage2-cosine-refinement-c2"]
}
```

`stage3_ce_only_control.py`(照抄 C3 wrapper 模式:fetch pinned smoke entry 后 runpy;fresh,不 resume):

```python
"""Fresh Stage3 CE-only causal control from C2 best, dual T4."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = '__PIN_AFTER_COMMIT_1__'


def main():
    os.environ.update(
        PYTHONUNBUFFERED='1',
        STAGE3_SOURCE_COMMIT=COMMIT,
        STAGE3_TARGET_SEGMENTS='15',
        STAGE3_CHUNK='ce_only_control',
        STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_ce_only_control_from_c2_best_v1',
        STAGE3_LAMBDA_PATH='0',
        STAGE3_MILESTONE_SEGMENTS='5,10',
        STAGE3_BASELINE_BEFORE_RESUME='1',
        STAGE3_MAX_RUNTIME_SECONDS='12600',
        STAGE3_HARD_TIMEOUT_SECONDS='15000',
    )
    os.environ.pop('STAGE3_RESUME_KERNEL', None)
    os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)
    os.environ.pop('STAGE3_EXPECTED_PROGRESS', None)
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)
    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as f: f.write(line + '\n')
    log('started', source_commit=COMMIT, segments=15, lambda_path=0,
        milestone_segments='5,10', run_id='small_0.1_stage3_ce_only_control_from_c2_best_v1',
        hard_timeout_seconds=15000, max_runtime_seconds=12600, baseline_before_resume=True)
    url = f'https://raw.githubusercontent.com/luckfu/Kronos/{COMMIT}/finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py'
    for attempt in range(3):
        log('fetch_pinned_entry', attempt=attempt + 1)
        try:
            with urllib.request.urlopen(url, timeout=20) as response: code = response.read()
            break
        except Exception:
            if attempt == 2: raise
            time.sleep(2)
    path = output / 'pinned_entry.py'
    path.write_bytes(code)
    log('pinned_entry_ready', sha256=hashlib.sha256(code).hexdigest())
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
```

### 5.2 创建 `finetune/kaggle_stage3_c2_c3_interp/`(P0b,先启动)

`kernel-metadata.json`(`kernel_sources` 顺序即输入,缺一不可;评估包来自 dataset 不是 kernel):

```json
{
  "id": "smmt315/kronos-small-0-1-stage3-c2-c3-interp",
  "title": "Kronos Small 0 1 Stage3 C2 C3 Interp",
  "code_file": "stage3_c2_c3_interp.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["luckfu/a-share-120d-temporal-symbol-holdout"],
  "kernel_sources": [
    "smmt315/kronos-small-0-1-stage2-cosine-refinement-c2",
    "smmt315/kronos-small-0-1-stage3-joint-path-c3",
    "smmt315/kronos-small-0-1-c2-alpha-oos-evaluation",
    "smmt315/kronos-small-0-1-stage3-c2-c3-oos"
  ]
}
```

`stage3_c2_c3_interp.py`(自包含,与 C3 OOS 同环境策略,不装 torch):

```python
"""C2×C3 float32 weight interpolation on the sealed 13-date production OOS evaluator."""
import gc
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE_COMMIT = "__PIN_AFTER_COMMIT_1__"
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage3_c2_c3_interp")
INPUT = Path("/kaggle/input")
HARD_LIMIT_SECONDS = 39600
ALPHAS = [(0.0, "interp_alpha_000"), (1.0, "interp_alpha_100"),
          (0.5, "interp_alpha_050"), (0.25, "interp_alpha_025"),
          (0.75, "interp_alpha_075")]
PURPOSE = ("exploratory_only; 13-date OOS design-contaminated; "
           "production candidate requires sealed fresh-window OOS")


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-interp-source-")) / "repo"
        try:
            run(["git", "clone", "--depth", "8", "--branch", "master",
                 "https://github.com/luckfu/Kronos.git", str(repo)])
            run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=repo)
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError(f"Source commit mismatch: {actual}")
            return repo
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def metric_or_none(value):
    value = float(value)
    return None if not math.isfinite(value) else value


def summarize(frame, top_fraction=0.10):
    import numpy as np
    import pandas as pd

    by_horizon = []
    by_date_h10 = []
    for horizon in range(1, 11):
        pred_col = f"predicted_return_d{horizon}"
        actual_col = f"actual_return_d{horizon}"
        pred, actual = frame[pred_col], frame[actual_col]
        daily = []
        for date, rows in frame.groupby("asof_date", sort=True):
            rank_ic = rows[pred_col].corr(rows[actual_col], method="spearman")
            count = max(1, int(math.ceil(len(rows) * top_fraction)))
            ranked = rows.sort_values(pred_col)
            spread = ranked.tail(count)[actual_col].mean() - ranked.head(count)[actual_col].mean()
            daily.append({
                "asof_date": str(date),
                "samples": int(len(rows)),
                "rank_ic": metric_or_none(rank_ic),
                "direction_accuracy": float((np.sign(rows[pred_col]) == np.sign(rows[actual_col])).mean()),
                "top_bottom_10pct": float(spread),
            })
        daily_ic = pd.Series([row["rank_ic"] for row in daily], dtype=float).dropna()
        ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        metrics = {
            "horizon_day": horizon,
            "samples": int(len(frame)),
            "direction_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "mae": float((pred - actual).abs().mean()),
            "pooled_rank_ic": metric_or_none(pred.corr(actual, method="spearman")),
            "daily_rank_ic_mean": metric_or_none(daily_ic.mean()),
            "daily_rank_ic_std": metric_or_none(ic_std),
            "daily_icir": metric_or_none(daily_ic.mean() / ic_std),
            "daily_icir_annualized_sqrt252": metric_or_none(daily_ic.mean() / ic_std * math.sqrt(252)),
            "positive_ic_ratio": float((daily_ic > 0).mean()),
            "mean_top_bottom_10pct": float(np.mean([row["top_bottom_10pct"] for row in daily])),
        }
        by_horizon.append(metrics)
        if horizon == 10:
            by_date_h10 = daily
    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "by_horizon": by_horizon,
        "by_signal_date_d10": by_date_h10,
    }


def interpolate(c2, c3, alpha):
    import torch
    if set(c2) != set(c3):
        missing = sorted(set(c2) ^ set(c3))
        raise RuntimeError(f"State dict keys differ: {missing[:20]}")
    merged = {}
    for key, left in c2.items():
        right = c3[key]
        if tuple(left.shape) != tuple(right.shape):
            raise RuntimeError(f"Shape mismatch {key}: {tuple(left.shape)} vs {tuple(right.shape)}")
        tensor = (1.0 - alpha) * left.float() + alpha * right.float()
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite interpolated tensor: {key}")
        merged[key] = tensor
    return merged


def write_checkpoint(c2_dir, tensors, dest):
    from safetensors.torch import save_file
    dest.mkdir(parents=True, exist_ok=True)
    for path in c2_dir.iterdir():
        if path.name == "model.safetensors" or not path.is_file():
            continue
        shutil.copy2(path, dest / path.name)
    save_file(tensors, str(dest / "model.safetensors"), metadata={"format": "pt"})


def assert_reload(path, alpha, c2, c3):
    import torch
    from safetensors.torch import load_file
    reloaded = load_file(str(path))
    if set(reloaded) != set(c2):
        raise RuntimeError("Reloaded keys differ from C2")
    if alpha == 0.0:
        reference, tag = c2, "C2"
    elif alpha == 1.0:
        reference, tag = c3, "C3"
    else:
        return {"checked": False}
    for key, tensor in reference.items():
        if reloaded[key].dtype != tensor.float().dtype:
            raise RuntimeError(f"alpha={alpha} dtype mismatch {key}")
        if not torch.equal(reloaded[key], tensor.float()):
            raise RuntimeError(f"alpha={alpha} torch.equal failed vs {tag}: {key}")
    return {"checked": True, "equal_to": tag, "tensors": len(reloaded)}


def agreement(left, right, name):
    pred_cols = [f"predicted_return_d{h}" for h in range(1, 11)]
    merged = left[["identity"] + pred_cols].merge(
        right[["identity"] + pred_cols], on="identity",
        suffixes=("_new", "_ref"), validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"{name} identity coverage mismatch: {len(merged)} vs {len(left)}/{len(right)}")
    rows = []
    for horizon in range(1, 11):
        delta = (merged[f"predicted_return_d{horizon}_new"] - merged[f"predicted_return_d{horizon}_ref"]).abs()
        spearman = merged[f"predicted_return_d{horizon}_new"].corr(
            merged[f"predicted_return_d{horizon}_ref"], method="spearman"
        )
        rows.append({
            "horizon_day": horizon,
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "spearman": metric_or_none(spearman),
        })
    return {"name": name, "identities": int(len(merged)), "by_horizon": rows}


def write_summary(payload):
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = OUTPUT / "shards"
    weights = OUTPUT / "weights"
    shards.mkdir(exist_ok=True)
    weights.mkdir(exist_ok=True)

    repo = clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import pandas as pd
    import torch
    from safetensors.torch import load_file
    from finetune.evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda:0")

    c2_file = find_one("**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors")
    c3_file = find_one("**/stage3_joint_path_smoke/checkpoints/last_model/model.safetensors")
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors")
    c2_predictions = find_one(
        "**/kronos_small_0_1_stage2_oos/predictions.csv.gz",
        lambda path: "c2-alpha-oos-evaluation" in str(path),
    )
    c3_predictions = find_one("**/kronos_small_0_1_stage3_c2_c3_oos/predictions.csv.gz")

    manifests = []
    for path in INPUT.glob("**/evaluation_manifest.json"):
        if not ((path.parent / "evaluation_panel.pkl").is_file()
                and (path.parent / "evaluation_samples.jsonl").is_file()):
            continue
        candidate = json.loads(path.read_text())
        if candidate.get("temporal_isolation", {}).get("incremental_signal_start"):
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one sealed evaluation manifest, found {manifests}")
    evaluation_root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    isolation = manifest["temporal_isolation"]
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")

    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    sample_groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    sample_key = "incremental_future_all" if "incremental_future_all" in sample_groups else "future_all"
    start = isolation.get("incremental_signal_start", isolation.get("future_signal_start"))
    end = isolation.get("incremental_signal_end", isolation.get("future_signal_end"))
    records = [row for row in sample_groups[sample_key] if start <= row["asof_date"] <= end]
    if not records:
        raise RuntimeError("No sealed OOS records")
    dates = sorted({row["asof_date"] for row in records})

    c2_ref = pd.read_csv(c2_predictions)
    c2_ref = c2_ref[c2_ref["model"] == "c2_best_segment_179"].copy()
    if c2_ref["identity"].duplicated().any() or len(c2_ref) != 66986:
        raise RuntimeError(f"C2 reference invalid: rows={len(c2_ref)}")
    if len(c2_ref) != len(records):
        raise RuntimeError(f"C2 prediction count mismatch: {len(c2_ref)} != {len(records)}")
    c3_ref = pd.read_csv(c3_predictions)
    c3_ref = c3_ref[c3_ref["model"] == "c3_last_segment_15"].copy()
    if c3_ref["identity"].duplicated().any() or len(c3_ref) != len(records):
        raise RuntimeError(f"C3 reference invalid: rows={len(c3_ref)}")

    c2_tensors = load_file(str(c2_file))
    c3_tensors = load_file(str(c3_file))
    provenance = {
        "c2_weights": {"path": str(c2_file), "sha256": sha256_file(c2_file),
                       "dtype": sorted({str(t.dtype) for t in c2_tensors.values()})},
        "c3_weights": {"path": str(c3_file), "sha256": sha256_file(c3_file),
                       "dtype": sorted({str(t.dtype) for t in c3_tensors.values()})},
        "tokenizer": {"path": str(tokenizer_file), "sha256": sha256_file(tokenizer_file)},
        "c2_reference_predictions": {"path": str(c2_predictions), "sha256": sha256_file(c2_predictions)},
        "c3_reference_predictions": {"path": str(c3_predictions), "sha256": sha256_file(c3_predictions)},
        "evaluation_root": str(evaluation_root),
    }
    print(json.dumps({"phase": "inputs_resolved", **provenance}, ensure_ascii=False), flush=True)

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    metrics = {}
    sanity = {}
    remaining = []
    timed_out = False

    for alpha, label in ALPHAS:
        dest = weights / label
        merged = interpolate(c2_tensors, c3_tensors, alpha)
        write_checkpoint(c2_file.parent, merged, dest)
        check = assert_reload(dest / "model.safetensors", alpha, c2_tensors, c3_tensors)
        print(json.dumps({"phase": "interpolated", "label": label, "alpha": alpha,
                          "reload_check": check, "sha256": sha256_file(dest / "model.safetensors")}), flush=True)

        pending = [date for date in dates if not (shards / f"{label}_{date}.csv.gz").is_file()]
        if pending:
            model = Kronos.from_pretrained(
                dest, num_sectors=86, num_size_buckets=0, context_layer=6,
                use_size_percentile=True, size_mlp_hidden_dim=64,
            ).to(device).eval()
            for date in pending:
                if time.time() - started > HARD_LIMIT_SECONDS:
                    remaining = [{"label": label, "alpha": alpha, "dates": [date] + pending[pending.index(date)+1:]}]
                    remaining.extend({"label": rest_label, "alpha": rest_alpha}
                                     for rest_alpha, rest_label in ALPHAS[ALPHAS.index((alpha, label))+1:])
                    timed_out = True
                    break
                date_records = [row for row in records if row["asof_date"] == date]
                result = evaluate_predictions(
                    label, model, tokenizer, date_records, store, device,
                    64, 1, 20260906, True,
                )
                shard = shards / f"{label}_{date}.csv.gz"
                result.to_csv(shard, index=False, compression="gzip")
                print({"saved": str(shard), "rows": len(result), "elapsed_sec": time.time() - started}, flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if timed_out:
                break

        parts = [pd.read_csv(path) for path in sorted(shards.glob(f"{label}_*.csv.gz"))]
        if len(parts) != len(dates):
            remaining = [{"label": label, "alpha": alpha, "have_shards": len(parts), "need": len(dates)}]
            break
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != len(records):
            raise RuntimeError(f"{label} prediction count mismatch: {len(frame)} != {len(records)}")
        metrics[label] = summarize(frame)
        (OUTPUT / f"metrics_{label}.json").write_text(
            json.dumps(metrics[label], ensure_ascii=False, indent=2) + "\n"
        )
        if alpha == 0.0:
            sanity[label] = agreement(frame, c2_ref, "alpha0_vs_c2_best_segment_179")
        elif alpha == 1.0:
            sanity[label] = agreement(frame, c3_ref, "alpha1_vs_c3_last_segment_15")
        print(json.dumps({"phase": "alpha_complete", "label": label, "d10": metrics[label]["by_horizon"][9]},
                         ensure_ascii=False), flush=True)

    payload = {
        "status": "partial" if timed_out or remaining else "complete",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "elapsed_sec": time.time() - started,
        "alpha_order": [{"alpha": alpha, "label": label} for alpha, label in ALPHAS],
        "provenance": provenance,
        "reload_checks": "alpha=0/1 use torch.equal on float32 tensors; not bit-identical re-inference",
        "sanity_policy": "identity-merge max|delta|/mean|delta|/spearman; soft report only; fp16 autocast + date seed + cross-GPU",
        "sanity": sanity,
        "metrics": metrics,
        "remaining": remaining,
    }
    write_summary(payload)
    if timed_out:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### 5.3 提交与推送

风格对齐近期提交(`Add …` / `Pin …` / `Launch …`)。**不要**把 C3 OOS、梯度诊断、audit JSON 塞进来。

```bash
# 1) 占位符文件已按 5.1/5.2 落盘后,先提交 trainer + control + 交接文档 + 技术报告
git add \
  finetune/stage3_training_model.py \
  finetune/train_stage3_path_alignment.py \
  finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py \
  finetune/kaggle_stage3_ce_only_control \
  finetune/STAGE3_P0A_P0B_EXECUTION_HANDOFF_CN.md \
  finetune/SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md
git commit -m "$(cat <<'EOF'
Add Stage3 lambda_path override, milestone checkpoints and fresh-run baseline flag

EOF
)"
PIN=$(git rev-parse HEAD)
echo "PIN=$PIN"

# 2) 把 PIN 填进两个 kernel(必须是 commit 1 的 SHA:smoke entry 与 trainer 都在该 commit)
python - <<PY
from pathlib import Path
pin = "$PIN".strip()
assert len(pin) == 40, pin
for path in [
    Path("finetune/kaggle_stage3_ce_only_control/stage3_ce_only_control.py"),
    Path("finetune/kaggle_stage3_c2_c3_interp/stage3_c2_c3_interp.py"),
]:
    text = path.read_text()
    path.write_text(text.replace("__PIN_AFTER_COMMIT_1__", pin))
    assert pin in path.read_text()
PY

# 3) 提交 interp 目录 + 已 pin 的 control wrapper
git add \
  finetune/kaggle_stage3_c2_c3_interp \
  finetune/kaggle_stage3_ce_only_control/stage3_ce_only_control.py
git commit -m "$(cat <<'EOF'
Add C2-C3 weight interpolation OOS scan kernel

EOF
)"
git push origin master
```

Kaggle 上传的是**本地文件**,control wrapper 里的 `COMMIT` 指向 commit 1(含改造后的 smoke entry)。interp 的 `SOURCE_COMMIT` 同样 pin commit 1;evaluator 本次未改动,clone 该 commit 即可。

### 5.4 启动与监控

**先 push P0b**(用户指定顺序):

```bash
cd finetune/kaggle_stage3_c2_c3_interp && kaggle kernels push && cd -
cd finetune/kaggle_stage3_ce_only_control && kaggle kernels push && cd -
kaggle kernels status smmt315/kronos-small-0-1-stage3-c2-c3-interp
kaggle kernels status smmt315/kronos-small-0-1-stage3-ce-only-control
# 取日志/产物
kaggle kernels output smmt315/kronos-small-0-1-stage3-c2-c3-interp -p /tmp/kronos-small-0-1-stage3-c2-c3-interp
kaggle kernels output smmt315/kronos-small-0-1-stage3-ce-only-control -p /tmp/kronos-small-0-1-stage3-ce-only-control
```

若因本周 GPU 配额耗尽启动失败(status 报错),把两个 kernel 挪到 **2026-09-19 配额刷新后**第一批重推,P0b 仍在前。不要为了抢配额改规格。

预期:P0b 约 2–6h(5 次推理 × ~1053 batch),P0a 约 3–3.5h。

### 5.5 P0a 完成后的 OOS(届时再建第三个 kernel,本轮不要建)

复用 interp kernel 的全部机制(同一 sealed 包、同一 evaluator 调用、同一 shard 粒度、同一 `summarize`),输入换为:

- kernel_sources 增加 `smmt315/kronos-small-0-1-stage3-ce-only-control`
- 权重:`checkpoints/best_model`(seg 0 初始)、`checkpoints/milestone_seg05`、`checkpoints/milestone_seg10`、`checkpoints/last_model`(共 4 个)
- 不再做 α 插值,单模型推理
- 参考预测仍用 C2 `c2_best_segment_179`
- 产出 CE-only 在 13 日口径的 D10 / by-horizon 表,按 §3.2 判定树写结论
- 13 日结果仍是 exploratory,不能当 production claim

### 5.6 P2:C2 best 生产回测(CPU,可与 kernel 并行)

本轮 5.1–5.4 不必做;可另开会话并行。

- **预测**:19 日 OOS,不重跑模型。
  - first 6:`artifacts/kronos_small_0_1_c2_alpha_oos_output/kronos_small_0_1_stage2_oos/predictions.csv.gz`
  - next 13:`artifacts/kronos_small_0_1_c2_alpha_oos_output_smmt315_20260914/kronos_small_0_1_stage2_oos/predictions.csv.gz`
  - 口径与重建:`finetune/audit_small_0_1_oos.py` + `finetune/reports/small_0_1_oos_raw_audit_20260915.json`
  - 标签:`model` 映射后取 `c2_best_segment_179`
- **价格 / panel**:
  - first 6:`data/a_share_v1_beta_eval_20260826/package`
  - next 13:`data/a_share_v1_beta_eval_20260914_incremental/package`
- **规则骨架**:`finetune/backtest_august_costs.py` + `finetune/backtest_august_path.py` 的 `simulate()`。规则 `hold_d10 / tp3_sl2 / tp5_sl3`;输出 turnover、gross、cost、net、Sharpe、max drawdown;按日期等权。
- 把 C2 best 定成 money baseline,作为后续一切 candidate 的「比 C2 多赚多少净收益」基准。
- **明确标注**:19 日集已用于设计讨论,数字用于建立 baseline 口径,不作为 sealed 确认。

## 6. 关键基础设施事实(执行时直接引用)

- **C2 best**:`small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors`,SHA `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a`;tokenizer SHA `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee`;数据 SHA 见 smoke 脚本 `HASHES`。
- **evaluator**:`finetune/evaluate_v1_beta_checkpoints.py`。生产自回归解码 `auto_regressive_inference`(T=0.6, top_p=0.9, sample_count=1, fp16 autocast,`torch.manual_seed(seed+date_index)` 按日期播种)。C2 OOS 与 C3 OOS 都是**按日调用** `evaluate_predictions`,因此单次调用内 `date_index` 恒为 0,与 C2 参考同一模式。输出列含 `identity, symbol, asof_date, target_date, …, predicted_return_d1..d10, actual_return_d1..d10`,按 asof_date 排序。**P1 的训练 rank score 将来必须与该定义数学同源**(`predicted_closes[-1]/last_close − 1`)。
- **C3 现状**:15 段完成,step 4695,best==last,验证 CE 2.279028;13 日 OOS:D10 方向 51.959% vs C2 47.932%,pooled Rank IC 0.0630 vs 0.1868,Top-Bottom 2.122% vs 3.489%。
- **P0b 参考数字**(判定插值结果时对照):C2 best pooled IC 0.1868 / daily ICIR 2.512 / 正 IC 13/13;C3 last 0.0630 / 1.391 / 12/13。
- 近期提交风格:`kaggle kernels push` 的 kernel 目录均在 `finetune/kaggle_*`,wrapper 通过 raw.githubusercontent pin commit。
- smoke 默认:`STAGE3_SWANLAB_RUN_ID` → `small_0.1_stage3_joint_path_alignment_from_c2_best_v2`,`STAGE3_LAMBDA_PATH` → `0.05`,`STAGE3_MILESTONE_SEGMENTS` → 空。P0a 必须覆盖这三项。

## 7. 纪律红线(用户定案,不得违反)

原文固化,下一会话不得改写或「顺手优化」:

1. **不救 Path**:不续训 C3、不做 λ sweep、不修 mixture decode、不把 Path-only adaptation 作为 V3 主线。当前 V2 Path Alignment 退出主线;C3 归档为 negative result。
2. **13 日 OOS 只出 exploratory candidate**:该 13 日集已有设计污染。任何 production claim 必须走未参与设计的新时间窗 sealed OOS。summary 必须带 `purpose=exploratory_only` 与 `oos_used=true`。
3. **不再以验证 loss 作为第一选模标准**:主指标改为 Rank IC → Top-Bottom → ICIR/正 IC 比例 → 净收益/Sharpe/回撤;CE/MAE/熵降级为 diagnostic。C2 best Segment 179 重新锁定为 Alpha reference 与 production candidate。
4. **P1 必须等 P0a 归因**:第一版 rank loss 只做朴素 pairwise logistic,不加 |Δr| 加权 / KL / 新 sampler 的复合。KL anchor 放 V3-B。batch 内必须真实同 date 截面,`unique signal dates / stocks per date / valid pairs / pair count / return dispersion` 全部进日志。
5. **每个 kernel 保持完整血缘**(manifest、SHA、run id 独立)。严禁复用已废弃 C1 的 dashboard(`small_0.1_stage3_path_alignment_from_c2_best`),也严禁复用 C3 的 v2 dashboard。P0a 必须用 `small_0.1_stage3_ce_only_control_from_c2_best_v1`。
6. **α=0/1 的硬门禁是 tensor `torch.equal`**,不是预测逐位相等。跨卡 / fp16 采样差异只做 identity-merge 软比对。
7. **本地 conda/numpy SIGABRT 不修**。权威门禁是 Kaggle kernel 训练前的 preflight。

## 8. 本地环境备注

conda base(`# /opt/miniconda3`,Python 3.12)在 import numpy 时 SIGABRT,pytest 无法本地运行;与本次改动无关(崩溃发生在任何项目代码导入之前)。修复或换 env 属可选事项;执行门禁以 Kaggle preflight 为准。
