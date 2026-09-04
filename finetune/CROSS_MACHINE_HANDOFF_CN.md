# Kronos 跨机器开发与 Beta v2.1 交接

更新时间：2026-09-02 19:25 UTC+8。

## 当前结论

- GitHub `master` 是代码、脚本和文档的共享主线。
- Beta v2.0 Best@687 是另一台机器当前应使用的部署基线；公开权重在 ModelScope，
  不随 Git 仓库下载。
- Beta v2.1 训练已安全停止，Best@475 已完成隔离未来评估并发布到 ModelScope；
  仍属于研究权重，不应未经成本与成交约束验证直接用于生产交易。
- 当前机器继续负责 A800 训练与 SwanLab 转发；另一台机器负责部署和前端开发。
  两边通过 GitHub 合并代码，通过 ModelScope 或移动介质传输大权重，不把权重提交
  到 Git。

## 另一台机器如何开始

```bash
git clone https://github.com/luckfu/Kronos.git
cd Kronos
git switch master
git pull --ff-only origin master

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r webui/requirements.txt
python -m pip install modelscope
```

下载 Beta v2.0 Best@687：

```bash
modelscope download luckfu/Kronos-A-Share-Beta-V2-0 \
  --repo-type model \
  --local-dir models/a_share_v1_beta/releases/beta_v2.0
```

该仓库是公开仓库，下载不依赖本次会话使用过的上传令牌。下载后建议校验：

```bash
cd models/a_share_v1_beta/releases/beta_v2.0
shasum -a 256 -c SHA256SUMS
cd ../../../..
```

本地启动 Web UI 时显式指定新权重，避免继续使用默认的 Beta v1.2 路径：

```bash
export KRONOS_MODEL_ID="$PWD/models/a_share_v1_beta/releases/beta_v2.0"
export KRONOS_TOKENIZER_ID="$PWD/models/a_share_v1_beta/releases/beta_v2.0/tokenizer"
PYTHONPATH=. python webui/app.py
```

模型加载合同保持：`num_sectors=86`、`num_size_buckets=0`、
`use_size_percentile=True`、`size_mlp_hidden_dim=64`、`context_layer=10`。输入为 120 日
OHLCVA、行业 ID 和 `[0, 1]` 连续市值百分位，输出未来 10 日路径。

## Beta v2.0 发布基线

- ModelScope：<https://modelscope.cn/models/luckfu/Kronos-A-Share-Beta-V2-0>
- checkpoint：Best@687，仅含部署权重，不含 Last 和优化器状态。
- 模型 SHA-256：
  `e603fe3178d61ee7feb8a5b0ad520d13166d533785f12d0a4f51d85db0a91ed3`。
- Predictor 参数量：102,437,248。
- 固定 symbol-holdout 验证：123,982 个窗口，4,678 只训练股票与 520 只验证股票
  无代码交集。
- Best@687：combined objective `2.3532605171`，forecast loss `2.3008320332`。
- ModelScope 发布后已重新下载，13 个文件哈希全部通过，模型与 tokenizer 实际加载
  通过。

这些指标证明的是固定验证合同下的预测损失表现，不证明可交易盈利。生产晋级仍需要
未参与调参的未来数据、方向与 Rank IC 指标以及含成本 Top5 回测。

## Beta v2.1 训练快照

已发布 Best@475：<https://modelscope.cn/models/luckfu/Kronos-A-Share-Beta-V2-1>。
模型 SHA-256：`e1bd55842996b7690a21c34c4d74e1128702bca9c16164788b741e3b5d052f97`。
发布包只含 Best@475、tokenizer、配置和评估说明，不含 Last 或优化器状态。

以下是 2026-09-02 19:25 UTC+8 的只读快照，恢复工作时必须重新查询：

- 状态：运行中，Segment `56 / 946`，正在 full-only 验证。
- 当前 Best：Segment 50，`beta_v21_score=0.7826530298`，越低越好。
- Best@50：forecast loss `2.3245618343`、return loss `0.3174758554`、
  return bias loss `0.0403778926`、barrier loss `0.8397740126`、ranking loss
  `0.6901906133`。
- 预计总耗时约 41 小时；按快照速度估计在 2026-09-04 上午完成，实际时间以
  `progress.json` 为准。
- 训练、GPU 0、远端 tmux 与本地 relay 在快照时均健康，未发现 NaN/Inf。

训练基座是 Beta v2.0 Best@687，不是 Last。Beta v2.1 保留原 10 日自回归路径头，
新增收益和 barrier 辅助头；return bias、barrier、ranking 等目标会参与反向传播。
完整公式与定版边界见
[`BETA_V2_1_TRAINING_PLAN_CN.md`](BETA_V2_1_TRAINING_PLAN_CN.md)。

远端运行：

```text
tmux: kronos_beta_v21
run: /nfsdata/models/2026/kronos-v1-beta/runs/beta_v2_1_best687_decision_heads_twopass_seed100
output: outputs/models/a_share_beta_v2_1_best687_decision_heads_twopass_120d_to_10d
```

本机 relay：

```text
tmux: kronos_beta_v21_swanlab
state: finetune/artifacts/swanlab_beta_v2_1_best687_decision_heads_twopass_a800/state.json
log: finetune/artifacts/swanlab_beta_v2_1_best687_decision_heads_twopass_a800/relay.log
```

看板：<https://swanlab.cn/@roc_fu/finance/runs/kronos-beta-v2-1-best687-decision-heads-twopass-a800>

看板的最简读法：`beta_v21_score` 总体向下代表综合训练目标改善；同时必须看
`generated_bias` 是否接近 0。辅助收益头变好但生成路径仍系统性偏空时，不能认为偏空
问题已经解决。任何 loss 或 score 下降也不能单独证明盈利。

## 两台机器的协作边界

当前训练机：

- 保持 A800 SSH、`kronos_beta_v21` 和本地 SwanLab relay 存活。
- 训练结束后核验 summary、progress、metrics、Best、Last、哈希和血缘。
- 完成固定验证与隔离未来评估后，再决定是否发布 Beta v2.1。
- 股票数据只允许从本机准备后传入 A800，禁止让 A800 主动抓取外部股票数据。

部署开发机：

- 从 GitHub `master` 获取代码，从 ModelScope 获取 Beta v2.0 权重。
- 前端、部署和服务端改动通过普通 Git commit 推回 `master`，提交前先拉取并处理冲突。
- 不提交模型权重、数据集、缓存、运行日志、访问令牌或本机虚拟环境。
- 不把 Beta v2.1 当前中间 checkpoint 当作正式部署版本。

## Git 合并约定

两台机器都直接使用 `master`，因此每次提交前执行：

```bash
git status --short --branch
git pull --ff-only origin master
```

只暂存本次明确修改的文件，禁止使用会顺带收集本地数据和实验产物的宽泛暂存命令。
发生非 fast-forward 时停止推送，先检查双方提交，不要 reset 或清理另一台机器的修改。

## 凭据与不可移植内容

- GitHub、A800 SSH、SwanLab、ModelScope 的登录态属于机器本地状态，不在 GitHub 中。
- 本次 ModelScope 上传令牌没有写入仓库或发布包；另一台机器需要上传权限时应独立登录，
  不复用聊天中的明文令牌。
- `finetune/artifacts/`、数据集、训练输出和本机虚拟环境默认不会被 Git 同步。
- GitHub 负责代码和报告的版本协作；大权重使用 ModelScope 或移动介质传输，并以
  SHA-256 校验完整性。
