# Kronos A 股预测工作台

这是一个基于 [Kronos](https://github.com/shiyu-coder/Kronos) 的 A 股日线预测项目。当前交付版本固定使用经过市值分层增训的 `Kronos-base`，提供单股区间预测、自选股票池横截面排序、增训脚本和样本外回测脚本。

> 本项目用于研究和回测，不构成投资建议。模型预测、回测收益和排名都可能失效，不能直接替代交易系统或风险管理。

## 当前模型

- 模型：A-share Size-Conditioned Kronos-base
- 输入：90 个交易日的 `open/high/low/close/volume/amount`
- 输出：未来 10 个交易日的采样路径及 P10/P50/P90 区间
- 条件：按交易日横截面流通市值划分为 10 桶（`0` 最小，`9` 最大）
- 基础训练期：2015-01-05 至 2025-12-31
- 增量数据行：2026-01-05 至 2026-07-31；V4 连续面板保留了信号日前 90 日上下文，并加入 20% 分层历史回放
- 生产 checkpoint：`outputs/models/a_share_v4_corrected_2026_replay20_latest/checkpoints/last_model`
- 增量覆盖：V4 B 训练混合 249,280 个近期窗口和 62,320 个历史回放窗口，生产使用两遍覆盖后的 `last_model`
- 设备：自动选择 `MPS → CUDA → CPU`

公开的 90 日增量权重发布在 [ModelScope: `luckfu/a-share-size-kronos-base-earlystop50`](https://modelscope.cn/models/luckfu/a-share-size-kronos-base-earlystop50)。本机预测前端当前加载上面的 V4 B `last_model`；两者 SHA-256 不同，V5 训练默认使用本机生产基座包，不会静默回退到 ModelScope。

旧 V3+2026 增量权重存在已确认的数据构造限制：2026 面板在生成 101 行窗口前被裁到 2026 年，90 日观察长度使 1 月至 5 月上旬无法成为监督信号，最终只训练了 39 个偏空信号日。V4 已修正连续上下文并加入历史回放；当前生产切到了 V4 B 两遍覆盖后的 `last_model`，但仍只应作为待持续验证的横截面排序模型。

## 市值桶如何接入原版 Kronos

市值桶不是 OHLCVA 之外的第七个连续行情字段，也不是十个分别训练的模型。项目保留原版 tokenizer 的六维输入和归一化方式，在同一个 `Kronos-base` 中增加一个市值类别 Embedding：

```text
原版：归一化 OHLCVA → Tokenizer → K线 Token + 时间编码 → 12层 Transformer → 预测
当前：归一化 OHLCVA → Tokenizer → 前10层 Transformer
                                       + 市值桶 Embedding
                                     → 后2层 Transformer → 预测
```

### 数据侧

BaoStock 不提供稳定的历史流通市值字段，因此数据准备阶段使用 `收盘价 × 成交量 ÷ 换手率` 估算流通市值。这个代理值只用于同一交易日的横截面排名，不作为精确财务市值使用。每天将可用股票从小到大划分为十个近似等量分组，`0` 表示最小的约 10%，`9` 表示最大的约 10%。

每个训练样本使用其 90 日观察窗口最后一个交易日对应的桶，不使用预测窗口中的未来市值信息。具体实现见 [`finetune/prepare_a_share.py`](finetune/prepare_a_share.py) 和 [`finetune/dataset.py`](finetune/dataset.py)。

### 模型侧

增训模型增加一张 `11 × 832` 的 Embedding 表：十个有效市值桶加一个未知桶，每个桶对应一个与模型隐藏维度相同的向量。该向量会广播到序列的所有时间步，并在 12 层 Transformer 的第 10 层之后注入，使顶部两层能够针对不同市值风格调整预测。

新增 Embedding 从全零初始化，因此刚加载原始 `Kronos-base` 权重时不会改变原始输出。增训时冻结 tokenizer 和底部十层，只训练市值 Embedding、顶部两层、归一化层、依赖层和输出头。这样可以保留基础模型已经学到的通用 K 线规律，同时让顶部网络学习小盘股与大盘股在波动、流动性和走势持续性方面的差异。实现位于 [`model/kronos.py`](model/kronos.py) 和 [`finetune/train_predictor.py`](finetune/train_predictor.py)。

市值条件解决的是归一化造成的体量信息缺失：两只绝对市值差异很大的股票可能具有相似的归一化 K 线，而桶标签可以让模型区分它们所属的规模风格。当前 `6.082%` 的验证 loss 改善来自“市值条件 + 顶部两层增训”的整体效果；若要单独量化市值桶的贡献，还需要训练一个数据和参数完全相同、但关闭市值条件的消融对照模型。

### 连续百分位消融

数据管线同时保留 `size_percentile`（当日横截面市值百分位）。实验模型在离散桶 Embedding 之外增加一个两层 MLP，将 `[size_percentile, is_known]` 映射到 832 维并在同一位置注入。该分支同样以零输出层初始化，旧离散 checkpoint 默认关闭，因此向后兼容。

混合模型从原始 `Kronos-base` 独立训练，连续运行 26 轮后早停。固定随机序列评估中，旧离散模型 loss 为 `2.951176`，混合模型为 `2.951893`；混合模型高 `0.000717`，没有胜出。129 日滚动回测中，混合模型 Rank IC 从 `0.04152` 小幅升至 `0.04250`，但方向准确率从 `55.31%` 降至 `55.05%`，计成本净收益从 `-33.30%` 恶化至 `-43.41%`。因此当前生产页面和 ModelScope 使用完整覆盖的离散桶模型，混合 checkpoint 仅作为消融实验保留。

### 早停耐心消融

离散桶模型还测试了“连续 5 轮无改善才早停”。候选模型运行 43 轮，最佳点出现在第 38 轮；固定种子 loss 为 `2.954815`，比现有生产模型的 `2.951176` 高 `0.003639`（约 `0.123%`），十个市值桶全部略差。因此默认训练耐心调整为 5，避免未来实验过早退出，但本次长训候选不替换生产 checkpoint。

## 快速部署

环境要求：Python 3.10+。Apple Silicon 建议安装支持 MPS 的 PyTorch；NVIDIA 机器安装对应 CUDA 版本的 PyTorch；没有加速设备时使用 CPU。

```bash
git clone https://github.com/luckfu/Kronos.git
cd Kronos
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r webui/requirements.txt
python -m pip install modelscope
```

下载增训模型到默认目录（也可以直接使用已有的本地 checkpoint）：

```bash
modelscope download luckfu/a-share-size-kronos-base-earlystop50 \
  --repo-type model \
  --local-dir outputs/models/a_share_v3_2026_incremental_latest/checkpoints/best_model
```

启动页面：

```bash
PYTHONPATH=. python webui/app.py
```

浏览器打开 <http://127.0.0.1:7070/>。生产部署时建议使用反向代理和进程管理器（例如 Nginx + systemd），不要把 Flask 开发服务器直接暴露到公网。服务监听地址和端口可在 `webui/app.py` 的启动入口中调整。

## 页面功能

页面只有两个工作 Tab，不需要选择模型或设备：

### 单股预测

输入六位 A 股代码（如 `300395`）后点击预测。系统会优先通过 BaoStock 拉取该股票最新复权日线和未来交易日；接口失败时回退到本地面板并明确标记数据来源。股票不在训练面板时，系统使用最新收盘价、成交量和换手率估算流通市值，再与仓库内 `webui/size_reference.json` 的参考横截面比较，得到可审计的百分位和市值桶；页面会标记“训练面板外”。该路径不要求部署完整训练面板，但至少需要 90 个有效交易日，且超出 CSI800 历史分布的股票预测风险更高。

### 股票池排序

输入 2–64 只股票，支持逗号、空格或换行分隔。系统优先通过 BaoStock 刷新每只股票的行情，并以股票池中最早的最新可用交易日作为共同截面；训练面板外股票使用成交量和换手率估算流通市值，再映射到便携式参考横截面的市值桶。系统生成三条路径并按未来 10 日预测收益中位数排序，榜单提供正/负预测、上涨路径比例、Top 8 标记，并可跳转到单股预测。本地面板内外股票可以混合排序；外部行情不可用且没有本地缓存的股票会明确报错。

## API

- `POST /api/load-model`：加载唯一生产模型并返回实际运行设备。
- `POST /api/predict`：提交 `{"symbol": "300395"}`，刷新行情并返回单股区间预测。
- `POST /api/a-share/rankings`：提交 `{"symbols": ["300395", "600519"]}`，返回自选股票池排名。
- `GET /api/a-share/symbols`：返回本地面板中的可用股票和最新市值桶。

## 增训与数据准备

A 股数据准备和实验口径详见 [`finetune/A_SHARE_PLAN_CN.md`](finetune/A_SHARE_PLAN_CN.md)。典型流程如下：

### 全市场市值分层 V3（训练中）

CSI800 历史成分仍偏向大中盘，不能代表完整的小微盘风格。V3 在 2015–2025 年 CSI800 历史成分并集之外，按 2025 年末市值代理值补入微盘、小盘和中小盘股票各 300 只，并为这三个层级各保留 80 只整股票样本外集合。训练集共 2,389 只股票，整股票样本外集合 240 只，两者没有股票重叠。

V3 行情从 2015-01-01 开始，使训练同时覆盖 2015 年快速上涨和随后大幅下跌的市场状态；训练截止 2025-12-31，2026 数据不参与参数更新，只用于严格的时间外验证。处理后训练集包含 5,233,538 个有效窗口，按十个市值桶均衡编排，每个窗口在单遍覆盖中只出现一次。Colab 正式训练默认完整遍历两遍，共 524 个覆盖分段，之后才应用 5 段早停耐心。完整配置入口是 `finetune/train_a_share_v3.sh`，候选模型完成跨时间、跨股票评估前不会替换生产模型。

正式长训放在 Colab CUDA 上，本机只负责准备、校验和打包数据。Kaggle P100 三段吞吐基准和后续迁移说明见 [`finetune/KAGGLE_V3.md`](finetune/KAGGLE_V3.md)；完整 Colab 命令见 [`finetune/COLAB_V3.md`](finetune/COLAB_V3.md)。本机 `train_a_share_v3.sh` 只用于短跑通或有意的 MPS 实验，不作为默认完整训练入口。

滚动生产增训默认使用 `80%` 近期窗口和 `20%` 分层历史回放窗口，并允许在 `10%-30%` 历史占比内调整。历史回放覆盖不同牛熊震荡、波动状态和市值桶，用于降低灾难性遗忘；固定股票 holdout 与最近至少 20 个完整标签交易日分别负责跨股票和跨时间验证。当前市场 Rank IC、方向准确率和含成本交易收益是主要晋级指标，旧年份表现用于调整回放比例和回滚判断，不作为绝对否决条件。完整规则见 [`finetune/A_SHARE_PLAN_CN.md`](finetune/A_SHARE_PLAN_CN.md)。

V4 修正实验使用连续的 2015-2026 面板，但只允许信号日期在配置区间内生成样本。训练信号为 `2026-01-05` 至 `2026-06-17`，共 249,280 个窗口；`2026-06-18` 至 `2026-07-16` 的最后 20 个完整标签交易日作为时间验证。A 组只使用修正后的近期窗口，B 组在完全相同的近期窗口上加入 62,320 个按年份和市值桶分层抽取的历史窗口，占训练集 20%。P100 A/B 实验已完成：时间外 Rank IC 分别为 `-0.04796` 和 `-0.03966`，均低于增训前 V3 Last 的 `0.01124`，两个候选均不晋级生产。Kaggle 一键 A/B 入口和复现命令见 [`finetune/KAGGLE_V3.md`](finetune/KAGGLE_V3.md)。

增训是独立 App，不与预测进程共享模型或生命周期。运行 `python webui/finetune_app.py` 后访问 `http://127.0.0.1:7071/`；预测 App 继续独立运行在 7070。增训页面可选择本地 `NeoQuasar/Kronos-base` 或已有的完整 checkpoint 作为训练起点，再选择离散市值桶或“桶 + 连续百分位”；模型列表只展示名称和来源，不向前端返回本机路径。设备自动使用 MPS、CUDA 或 CPU，并提供规模预估、实时训练/验证/最佳 Loss 曲线、原始日志、完整覆盖进度、停止保存和 checkpoint 恢复。默认每段训练 20,000 个无重复窗口；训练器沿固定随机排列依次推进，76 段覆盖当前 1,509,252 个训练窗口一遍，全部窗口完成覆盖后才开始计算 5 段早停耐心。

Colab GPU 重建数据、下载基础模型和长训流程见 [`finetune/COLAB.md`](finetune/COLAB.md)。该流程不把大数据文件提交进 Git，而是在 Colab 中从 BaoStock 重建真实面板，并支持 Google Drive checkpoint 和 `google-colab-cli` 远程会话。下一版 `120+10` 上下文候选使用独立的 2014 补充数据和 [`finetune/COLAB_V5.md`](finetune/COLAB_V5.md)，不运行 `V5-90-control`。

```bash
python finetune/download_a_share_baostock.py \
  --universe csi800 \
  --start 2020-01-01 \
  --end 2026-07-31 \
  --output ./data/a_share/a_share_daily.csv

python finetune/prepare_a_share.py \
  --input ./data/a_share/a_share_daily.csv \
  --output-dir ./data/a_share/processed_datasets \
  --metadata-out ./data/a_share/asset_metadata.csv

KRONOS_DATASET_PATH=./data/a_share/processed_datasets \
KRONOS_METADATA_PATH=./data/a_share/asset_metadata.csv \
KRONOS_PREDICTOR_PATH=./Kronos-base \
python finetune/train_predictor.py
```

启用“离散桶 + 连续百分位”消融实验：

```bash
KRONOS_USE_SIZE_PERCENTILE=1 \
KRONOS_PREDICTOR_SAVE_FOLDER=a_share_size_hybrid_kronos_base_earlystop50 \
KRONOS_PREDICTOR_PATH=./Kronos-base \
python finetune/train_predictor.py
```

训练脚本默认使用完整覆盖分段：每段内部无放回、段间沿同一排列继续，完成要求的覆盖遍数后再应用验证集早停。`KRONOS_TRAIN_SAMPLES_PER_SEGMENT`、`KRONOS_COVERAGE_PASSES`、`KRONOS_VALIDATION_SAMPLES` 和 `KRONOS_EARLY_STOPPING_PATIENCE` 可覆盖页面默认值。Apple Silicon 直接运行即可自动使用 MPS；CUDA 可使用 `torchrun`。停止请求会在当前 batch 完成后立即暂停计算并保存 `last_state.pt`；状态包含模型、优化器、学习率调度器、分段和 batch 位置及随机数状态，设置 `KRONOS_RESUME_TRAINING=1` 可从下一 batch 恢复。训练指标同时写入 `metrics.jsonl`，旧任务可从 `training.log` 恢复曲线。`Kronos-base` 基础权重可放在任意目录，通过 `KRONOS_PREDICTOR_PATH` 指定。

V3 全覆盖基线 checkpoint 已完成一轮全覆盖训练：1,509,252 个有效窗口按固定排列无放回遍历，共推进 81 个保存分段。早期“每轮随机抽取 800 个窗口”的 checkpoint 仅作为历史实验保留；当前预测页面加载的是后续 V4 B `last_model`。

## 评估与回测

V3 全覆盖基线共使用 1,509,252 个窗口，最终 `latest` checkpoint 位于第 81 个覆盖分段；训练过程记录的最佳验证 loss 为 `2.877660`。针对从未进入 1,176 只训练面板、且不属于对应 CSI800 截面的 24 只股票，2025 年 16 个截面上 `latest` 的平均 Rank IC 为 `0.16549`，高于全覆盖 `best` 的 `0.15849` 和旧在线模型的 `0.09122`；预测前后 20% 的平均十日收益差分别为 `4.46%`、`4.04%` 和 `2.44%`。2024 年三个模型的 Rank IC 都接近零，2026 熊市压力测试中排序能力同样明显减弱，说明模型表现具有行情依赖性。

2024、2025 的股票本身没有参与增训，但模型见过同期 CSI800 行情，因此这些结果检验的是跨股票泛化，不是严格的时间外验证。2026 是时间外压力测试，也不能单独代表所有市场环境。当前证据支持将全覆盖 `latest` 用于股票池排序，但不支持把单股绝对涨跌或预测价格直接作为交易指令。

复现实验：

```bash
PYTHONPATH=. python finetune/backtest_a_share_2026.py \
  --universe-selection smallest_market_cap \
  --output-dir ./outputs/backtest_results/a_share_2026_smallcap

PYTHONPATH=. python finetune/backtest_a_share_daily.py \
  --output-dir ./outputs/backtest_results/a_share_2026_daily_smallcap
```

## 目录说明

```text
model/                 Kronos 模型与预测器
webui/                 Flask 页面、接口和前端资源
finetune/              A 股数据准备、增训、评估和回测
data/a_share/          本地数据（不纳入 Git）
outputs/               checkpoint 与实验结果（不纳入 Git）
```

旧版通用 Kronos 示例仍保留在 `examples/`，用于兼容原项目 API；本项目实际部署入口是 `webui/app.py`。

## 致谢与许可

底层 Kronos 代码和预训练模型来自原作者项目，相关研究请参阅 [论文](https://arxiv.org/abs/2508.02739)。本仓库沿用 [MIT License](LICENSE)；ModelScope 上的增训权重使用 Apache-2.0 模型卡声明。
