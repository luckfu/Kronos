# 条件身份 2×2 探针：运行时代码快照（2026-10-27）

本目录只存**代码**，不含任何权重、数据或产物。权重继续放在 Kaggle kernel output
或 ModelScope，不进 Git。

## 为什么有这一页

10-27 的四个 2×2 点（Base Seg281 / Small Bootstrap-final / Small Main-final /
Small C2）都是在 Kaggle 上跑的纯前向探针。它们的源码此前**只存在于 kernel 的
base64 payload 里**，`scratch/` 又被 `.gitignore` 整体排除，所以换台机器后
`git clone` 拿不到任何能复现这些数字的代码。这一页把那段代码原样落库。

来源（四个 kernel 各解包一次）：

| 点 | kernel | payload sha256 | 入库脚本 |
|---|---|---|---|
| Base Seg281 | `user281434/kronos-base-cond-sector-gain-sweep` v2 | `aadf6d3a…648d3` | `launch_cond_identity_2x2.py` |
| Small Bootstrap-final | `wynstonliu/kronos-small-bootfinal-cond-2x2` v1 | `a768f10e…01ba3` | `launch_cond_identity_2x2_bootfinal.py` |
| Small Main-final | `luckfu/kronos-small-mainfinal-cond-2x2` v1 | `8b418327…f6a7ad` | `launch_cond_identity_2x2_mainfinal.py` |
| Small C2 | `user281434/kronos-small-c2-cond-identity-2x2` v1 | `34124ae2…6127b` | `launch_cond_identity_2x2_small_c2.py` |

四个 kernel 的 `model/kronos.py`、`finetune/config.py`、`finetune/train_predictor.py`
**逐字节相同**（同一份 sha：`7d88cff0…` / `8b293373…` / `47003988…`），所以只存一份：

```
payload …/launch_*.py      →  四个 launch_cond_identity_2x2*.py
model/kronos.py            →  payload_snapshot/model/kronos.py
finetune/config.py         →  payload_snapshot/finetune/config.py
finetune/train_predictor.py→  payload_snapshot/finetune/train_predictor.py
```
payload 里其余文件（`requirements.txt`、`dataset.py`、`asset_metadata.py`、
`beta_v21.py`、`drive_cleanup.py`、`validation_precision.py`、
`export_last_model.py`、`power_law_sampler.py`、`temporal_rank_ic_selection.py`、
`utils/`）与仓库 HEAD 逐字节一致，故未重复入库。

## 与仓库 HEAD 的差异（重要）

`payload_snapshot/` 里的三个文件**不是** HEAD 的版本，它们包含当时还没提交的改动：

| 文件 | 差异 | 影响 |
|---|---|---|
| `model/kronos.py` | 新增 `use_condition_scale` / `condition_scale_init` / `condition_last_layer_init` / `condition_last_layer_init_std` 四个构造参数与对应初始化分支（默认值 = 旧的零初始化 + bare add） | HEAD 的 `Kronos.__init__` 不接受 `use_condition_scale`，而脚本显式传入 → 直接跑会 `TypeError` |
| `finetune/config.py` | 新增 `KRONOS_CONDITION_SCALE_INIT` / `KRONOS_USE_CONDITION_SCALE` / `KRONOS_CONDITION_LAST_LAYER_INIT*` 环境变量 | 默认关闭；不设环境变量时行为与 HEAD 相同 |
| `finetune/train_predictor.py` | ① `model_config` 透传上面四个字段；② 续训 guard 的 opt-in 字段；③ **`evaluate_validation` 的 shuffle 从 `torch.roll(...,1)` 改成 `torch.randperm` 真批置换** | ③ 会改变 `Full−Shuffled`、`condition_shuffled_forecast_loss` 的数值口径；用 HEAD 跑会得到不同的数 |

第 ③ 条是关键：四个点的 `Full−Shuffled`（以及由此得到的判别地板）是在
**真批置换**口径下测的，HEAD 仍是 roll-by-1。只拿 HEAD 复跑，数字对不上不是
噪声，是口径不同。

另外，脚本里没有任何 `manual_seed`；`randperm` 每次前向都不同，所以同一权重
两次运行本来就会有 ~1e-5 量级的差异（报告里 Base `Full−Shuffled` 两次
−4.0e-5 / −1.4e-5 即来源于此）。**四点的 `fs_*` 应视为单次抽样，不是可逐位
复现的确定值。**

## 怎么复现

最干净的方式是在一个干净 checkout 上把快照覆盖回去：

```bash
git clone https://github.com/luckfu/Kronos.git
cd Kronos

# 1) 还原运行时代码（只覆盖 3 个文件；不动权重、不动数据）
cp -R finetune/cond_identity/payload_snapshot/* .

# 2) 数据：公开 dataset（~700MB），不要提交
kaggle datasets download smmt315/a-share-120d-temporal-symbol-holdout -p /tmp/holdout --unzip

# 3) 权重：对应 kernel output（私有，需账号权限），见下表
```

跑法就是脚本的 `main()`：在 Kaggle GPU 上执行对应点的脚本（见下表），或用原
kernel 的 wrapper（base64 自解压 + `runpy`）执行同一路径。

### 四个点各自的取数位置

| 点 | 脚本 | checkpoint | Δ |
|---|---|---|---|
| Base Seg281 | `launch_cond_identity_2x2.py` | `user281434/kronos-base-clean-0-1-bootstrap` → `last_model` | 0.00843 |
| Small Bootstrap-final | `launch_cond_identity_2x2_bootfinal.py` | `wynstonliu/kronos-small-0-1-bootstrap-final` → `best_model` | 0.01669 |
| Small Main-final | `launch_cond_identity_2x2_mainfinal.py` | `luckfu/kronos-small-0-1-main-final` → `best_model` | 0.02695 |
| Small C2 | `launch_cond_identity_2x2_small_c2.py` | Kaggle dataset `user281434/kronos-small-0-1-cosine-c2-best-seg179`（sha `4ee469d4…`） | 0.07489 |

结果数字在 `finetune/reports/20261027_cond_identity_2x2_*.json`，判读见
`finetune/reports/20261027_cond_identity_handover_cn.md`。

> **口径更正**：`20261027_cond_identity_2x2_small_c2.json` 的
> `protocol.shuffle` 写的是 `torch.roll(shifts=1)`，但该 kernel 实际 payload 里
> 已是 `torch.randperm` 真批置换（见 `launch_cond_identity_2x2_small_c2.py` 对应的
> snapshot）。四个点因此确实同口径；那段 JSON 文本是旧模板没更新，**以代码为准**。

### 四个变体之间的差异（留作核对 / 新增点参考）

四个脚本已全部入库，不再需要手工还原。它们的差异只有下表这些：

| 位置 | Base | Small（bootstrap / main / C2） |
|---|---|---|
| `kernel_sources`（push 时的 metadata） | `user281434/kronos-base-clean-0-1-bootstrap` | 对应 Small kernel output |
| `resolve_c2()` 的 glob 后缀 | `checkpoints/last_model/model.safetensors` | `checkpoints/best_model/model.safetensors`（C2 变体按 sha256 解析） |
| `resolve_c2()` 的关键词 | `"base_clean_0.1_bootstrap"` | `"small_0.1_bootstrap"` / `"small_0.1_main"`（C2 用 dataset 名） |
| `KRONOS_CONTEXT_LAYER` / `config["context_layer"]` / `model_kwargs["context_layer"]` | `10` | `6`（Small 8 层，注入第 6 层） |

> 已知小瑕疵：main-final 变体的 `EXPERIMENT` 常量仍是 `small_boot_cond_identity_ablation`
> （从 bootstrap kernel 复制后没改），只影响输出目录命名，不影响数值；为保持
> 逐字节来源一致，未改。

| 位置 | Base | Small |
|---|---|---|
| `kernel_sources`（push 时的 metadata） | `user281434/kronos-base-clean-0-1-bootstrap` | 对应 Small kernel output |
| `resolve_c2()` 的 glob 后缀 | `checkpoints/last_model/model.safetensors` | `checkpoints/best_model/model.safetensors` |
| `resolve_c2()` 的关键词 | `"base_clean_0.1_bootstrap"` | `"small_0.1_bootstrap"` / `"small_0.1_main"` |
| `KRONOS_CONTEXT_LAYER` / `config["context_layer"]` / `model_kwargs["context_layer"]` | `10` | `6`（Small 8 层，注入第 6 层） |

其余不动：`num_sectors=86`、`use_size_percentile=True`、`use_condition_scale=False`、
val batch 64、n=2000、α 网格 `{1, 1.5, 2, 3, 4, 8}`。

## 边界

- 只读权重 + 只做前向，不改 `model/`、不训练、不碰 live Bootstrap 与本机 relay。
- 本目录不包含 kernel 的 push 脚本；重建 payload 的流程见
  [`../reports/20261027_cond_identity_next_step_handover_cn.md`](../reports/20261027_cond_identity_next_step_handover_cn.md) §5。
- 权重、数据、`artifacts/` 一律不入库。
