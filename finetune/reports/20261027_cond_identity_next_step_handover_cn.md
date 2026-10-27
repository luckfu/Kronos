# 下一步交接：补 Small 的低 Δ 参照点（2026-10-27 收尾）

上一页：`20261027_cond_identity_handover_cn.md`（全局现状）。本文只讲**一件事**：怎么补上那个缺口。

---

## 1. 为什么非补不可

今天测出来（Base Seg281，Δ=0.0084）：

- `fs_sector = +1.2e-5`（符号反，且**小于**自身零假设 1.5e-5）
- `fs_size = −9.6e-6`（约 2× 零假设）
- 8× 行业增益**拉不动**：`fs_sector` 仍在噪声，CE 只降 2.5e-5
- 按单位注入归一，Base 的行业效率 **0.015**，Small 是 **0.168** → 弱 **11 倍**

但 **Base 的 Δ=0.0084 低于 Small 我能量到的最早的点**（cum 516 时 Δ=0.0167）。所以：

> **“Small 在这个阶段也是这样” 和 “Base 的身份通道被压住了” 现在无法区分。**

补上 Small@Δ≈0.0167 的 `fs_sector`，二选一立刻成立：

| 那个点量出来 | 结论 | 下一步 |
|---|---|---|
| `fs_sector` 也在噪声（≤3e-5） | Base 现在**正常**，只是没走到 | 继续跑，别动配方 |
| `fs_sector` 已有 −2e-4 量级 | **Base 的身份通道确实被压住** | 才值得动 γ / 注入位点 |

---

## 2. 那个点在哪（已确认存在）

```
kernel:  luckfu/kronos-small-0-1-bootstrap        (COMPLETE)
输出:    kronos_small_v21/outputs/models/small_0.1_bootstrap/checkpoints/
         ├─ best_model/    ← 用这个（≈seg 516，对应的正是 Δ=0.0167）
         └─ last_model/
```

`Δ=0.0167` 是从 `small_0.1_main` 的第一个 Condition Monitor 点反推的，而 Main 加载的是 **Bootstrap best_model**。所以**用 best_model**，不要用 last_model。

---

## 3. 怎么做（改 4 处，其余照抄 Base 那一版）

拿 `user281434/kronos-base-cond-sector-gain-sweep` 的 v2 当模板，改：

| 改哪 | 从 | 改成 |
|---|---|---|
| `kernel-metadata.json` → `kernel_sources` | `["user281434/kronos-base-clean-0-1-bootstrap"]` | `["luckfu/kronos-small-0-1-bootstrap"]` |
| `resolve_c2()` 的 glob 后缀 | `checkpoints/last_model/model.safetensors` | `checkpoints/best_model/model.safetensors` |
| `resolve_c2()` 的路径关键词 | `"base_clean_0.1_bootstrap"` | `"small_0.1_bootstrap"` |
| `context_layer`（两处：env + config） | `10` | **`6`**（Small 是 8 层、第 6 层注入） |

其余不动：`num_sectors=86`、`use_size_percentile=True`、`use_condition_scale=False`、val batch 64、n=2000、α 网格 `{1, 1.5, 2, 3, 4, 8}`。

---

## 4. 今天踩过的坑（照抄就不会再踩）

1. **`global C2_SHA` 用在赋值前** → `SyntaxError`。用一个局部 `c2_sha`，别用 global。
2. **`context_layer` 写死 6** 是最容易漏的：Base 是 12 层注入第 10 层，套 Small 的 6 会静默算错（不报错，只是数值不对）。
3. **权重走 `kernel_sources`，不走 dataset**：`kernel_sources` 把上一个 kernel 的 output 挂到 `/kaggle/input/<slug>/`，glob 就能找到；改成 dataset 要多一步发布。
4. **不要按 sha256 匹配权重**：Base/Small 的 sha 每次 chunk 都变；用路径关键词匹配。
5. **`sweep` 写盘时别引用函数内的 `total`**（会 NameError，但所有格子已打印，数据不丢——真要重跑再修）。
6. **置换零假设 ≠ 地板**：Base 上同一个量测两次差 2.7e-5，是 null 的 15 倍。判读时地板取 `max(3·null, 复测差)`，Base 约 **3e-5**。

---

## 5. 重建 payload 的最小脚本

kernel 源码不在仓库里，靠 wrapper 里那段 base64 的自解压 zip。完整流程：

```bash
export KAGGLE_API_TOKEN=<user281434 或 luckfu>
.venv-swanlab-api/bin/kaggle kernels pull user281434/kronos-base-cond-sector-gain-sweep -p /tmp/kb
```

然后：

```python
import re, base64, zipfile, io, pathlib, hashlib, json
src = pathlib.Path('/tmp/kb/kernel.py').read_text()
m = re.search(r"payload = base64\.b64decode\('([^']+)'\)", src)
zin = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1))))
src_dir = pathlib.Path('/tmp/kb/src'); zin.extractall(src_dir)      # 改这里的 launch_*.py
# ... 按 §3 改 4 处 ...
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in zin.namelist():
        z.writestr(n, (src_dir / n).read_bytes())
payload = buf.getvalue(); sha = hashlib.sha256(payload).hexdigest()
out = re.sub(r"payload = base64\.b64decode\('[^']+'\)",
             lambda _: f"payload = base64.b64decode('{base64.b64encode(payload).decode()}')", src, count=1)
out = re.sub(r"assert hashlib\.sha256\(payload\)\.hexdigest\(\) == '[0-9a-f]{64}'",
             f"assert hashlib.sha256(payload).hexdigest() == '{sha}'", out, count=1)
push = pathlib.Path('/tmp/kb/push'); push.mkdir(exist_ok=True)
(push / 'kernel.py').write_text(out)
(push / 'kernel-metadata.json').write_text(json.dumps({
    "id": "user281434/kronos-small-boot-cond-2x2",
    "title": "Kronos Small Bootstrap-end Cond 2x2",
    "code_file": "kernel.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["smmt315/a-share-120d-temporal-symbol-holdout"],
    "competition_sources": [],
    "kernel_sources": ["luckfu/kronos-small-0-1-bootstrap"],
}, indent=2))
```

```bash
.venv-swanlab-api/bin/kaggle kernels push -p /tmp/kb/push
.venv-swanlab-api/bin/kaggle kernels logs <ref> > /tmp/boot.log   # 找 phase=sweep_cell / decompose_done
```

**耗时**：装 torch ~8 分钟 + 计算 ~2 分钟。

---

## 6. 判读（先写下，再看数）

把 Small@Bootstrap-best 的结果和这两行并排：

| | Δ | 相对注入 | `fs_sector` | `fs_size` |
|---|---|---|---|---|
| Small Bootstrap best（待测） | 0.0167 | ~0.075% | **?** | **?** |
| Small C2（已知） | 0.0749 | 0.332% | −5.58e-4 | −1.56e-3 |
| Base Seg281（已知） | 0.0084 | 0.017% | +1.2e-5（噪声） | −9.6e-6（噪声） |

- 若 Small 在 0.0167 就已经有明显 `fs_sector`（比如 −2e-4）→ **Base 落后是有结构的**，可以开始考虑 γ/注入位点；
- 若它也只有噪声 → **Base 完全在正常轨道上**，把这段争论收口，回去算预算（§4.2 外推 ~2,325 段）。

---

## 7. 环境与凭据

- `KAGGLE_API_TOKEN`：`luckfu`（读 Small 那个 kernel 的 output）与 `user281434`（推新 kernel）。**都不在仓库里**，需重新提供。
- `SWANLAB_API_KEY`：只有拉日志才需要；**曲线里没有** `sector_embedding_weight_norm`，数据只在 logs 的 `Condition Monitor JSON` 行里。
- 本机工作目录：`.kaggle_home/`（gitignore），token、payload 中间件、旧日志都在里面，换机不用带。
- 仓库当前 HEAD 之后无未提交改动。

---

## 8. 边界

这一跑**只读权重 + 只做前向**：不改 `model/`、`finetune/`、relay，不碰 live Bootstrap，不动 Small 生产冻结。
