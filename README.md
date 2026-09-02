# Kronos A 股预测工作台

> 在 Kaggle 上进行任何训练、续训、调参或验证前，必须先执行 [`finetune/KAGGLE_RUNBOOK_CN.md`](finetune/KAGGLE_RUNBOOK_CN.md) 的统一检查流程。

当前可跨机器获取的最新正式研究权重为 **Beta v2.0 Best@687**。它使用整股票
90/10 隔离验证选点，已发布到
[ModelScope](https://modelscope.cn/models/luckfu/Kronos-A-Share-Beta-V2-0)；模型 SHA-256 为
`e603fe3178d61ee7feb8a5b0ad520d13166d533785f12d0a4f51d85db0a91ed3`。Beta v2.1
决策目标训练仍在 A800 上运行，尚不是发布版本。跨机器开发、部署与训练监控的当前
交接入口见 [`finetune/CROSS_MACHINE_HANDOFF_CN.md`](finetune/CROSS_MACHINE_HANDOFF_CN.md)。

当前 Web UI 与 Serverless 推理契约仍按 **Beta V1.2** 运行，默认使用 `Best@871`，并保留
`Last@1056` 作为完整两遍训练终点。稳定本地入口为
`models/a_share_v1_beta/releases/beta_v1.2/best_model` 和
`models/a_share_v1_beta/releases/beta_v1.2/last_model`。发布依据见
[`finetune/BETA_V1_2_RELEASE_CN.md`](finetune/BETA_V1_2_RELEASE_CN.md)。Beta V1.2 的 Modal
配置使用独立 App 名，不会覆盖旧 V6 服务；线上切换仍需显式执行部署。

Beta 系列的推理输入和全市场市值百分位准备规则见
[`finetune/MODEL_USAGE_V1_BETA_CN.md`](finetune/MODEL_USAGE_V1_BETA_CN.md)。该契约与 V6
不同，不接受离散 `size_bucket`。

2026-08-28 从 A800 迁回的 v1-beta 模型、严格未来评估数据和全部评估结果已直接合并
到项目目录。当前模型血缘、精确 SHA-256、晋级状态与路径事件结论见
[`finetune/V1_BETA_MODEL_LINEAGE_CN.md`](finetune/V1_BETA_MODEL_LINEAGE_CN.md)；机器可读
清单为 [`models/a_share_v1_beta/LINEAGE.json`](models/a_share_v1_beta/LINEAGE.json)。

这是一个基于 [Kronos](https://github.com/shiyu-coder/Kronos) 的 A 股日线预测项目，提供单股区间预测、自选股票池横截面排序、增训脚本和样本外回测脚本。

> 本项目用于研究和回测，不构成投资建议。模型预测、回测收益和排名都可能失效，不能直接替代交易系统或风险管理。

## 当前推理部署

- 模型：A-share Full-Market Beta V1.2（约 102.4M 参数）
- 输入：120 个交易日的 `open/high/low/close/volume/amount`
- 输出：未来 10 个交易日的采样路径及 P10/P50/P90 区间
- 条件：信号日行业 ID + 同日全市场连续市值百分位
- 行业配置：86 个行业标签，未知行业 ID 为 `86`
- 模型配置：`num_sectors=86`、`num_size_buckets=0`、`use_size_percentile=True`、`context_layer=10`
- 默认 checkpoint：`models/a_share_v1_beta/releases/beta_v1.2/best_model`（`Best@871`）
- 完整训练终点：`models/a_share_v1_beta/releases/beta_v1.2/last_model`（`Last@1056`）
- 设备：自动选择 `MPS → CUDA → CPU`

公开权重发布在 [ModelScope: `luckfu/Kronos-A-Share-Beta-V1-2`](https://modelscope.cn/models/luckfu/Kronos-A-Share-Beta-V1-2)。`Best@871` 在固定 24k 验证集上优于 `Last@1056`，因此作为页面和 Serverless 默认模型。它尚未在全新严格未来时间段完成评估，仍是研究候选，不构成生产交易系统。

## 行业与连续市值条件

Beta V1.2 保留 tokenizer 的六维 OHLCVA 输入。行业和规模不是额外行情列，而是在 Transformer 第 10 层之后注入的静态条件：

```text
原版：归一化 OHLCVA → Tokenizer → K线 Token + 时间编码 → 12层 Transformer → 预测
当前：归一化 OHLCVA → Tokenizer → 前10层 Transformer
                                       + 行业 Embedding
                                       + 连续市值百分位 MLP
                                     → 后2层 Transformer → 预测
```

### 数据侧

BaoStock 不提供稳定的历史流通市值字段，因此使用同一交易日的
`market_cap_proxy = amount / (turn / 100)` 作为代理值。代理值仅用于全市场横截面排序，
不能当作精确财务市值。每个样本使用 120 日观察窗口最后一个交易日的行业和市值百分位，
不读取未来窗口信息。

Web 网关将行业配置拆成两层：`webui/sector_vocabulary.json` 固定保存 Beta V1.2 训练时
使用的 86 项行业顺序，`webui/symbol_sector_map.json` 保存可更新的股票行业映射；
`webui/size_reference.json` 保存 2026-07-31 全市场市值横截面。股票不在映射内，或数据源
返回训练词表之外的行业时，使用未知行业 ID `86`；所有结果都会记录条件来源与参考日期。
行业映射可通过 `python webui/update_sector_mapping.py` 从 BaoStock 手动更新，模型词表不会
被该脚本修改。

### 模型侧

模型使用 `87 × 832` 行业 Embedding（86 个有效行业加 1 个未知行业），并用两层 MLP
把 `[size_percentile, is_known]` 映射到 832 维。两个条件广播到序列各时间步，在第 10 层
之后与隐藏状态相加。Beta V1.2 对完整 Predictor 继续训练；实现见
[`model/kronos.py`](model/kronos.py) 和 [`finetune/train_predictor.py`](finetune/train_predictor.py)。

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

下载 Beta V1.2 模型仓库到默认发布目录：

```bash
modelscope download luckfu/Kronos-A-Share-Beta-V1-2 \
  --repo-type model \
  --local-dir models/a_share_v1_beta/releases/beta_v1.2
```

启动页面：

```bash
PYTHONPATH=. python webui/app.py
```

浏览器打开 <http://127.0.0.1:7070/>。生产部署时建议使用反向代理和进程管理器（例如 Nginx + systemd），不要把 Flask 开发服务器直接暴露到公网。服务监听地址和端口可在 `webui/app.py` 的启动入口中调整。

## 页面功能

页面只有两个工作 Tab，不需要选择模型或设备：

### 单股预测

输入一个代码时执行单股买点判断。系统由 Web 网关优先从 Eastmoney 获取最新复权日线，失败时回退到 BaoStock；行情按股票写入 `webui/market_data_cache/`，后续请求只补缓存末日之后的增量并保留约 7 天重叠区间来覆盖数据修订。网关按 `amount / (turn / 100)` 估算市值代理，与 `webui/size_reference.json` 比较得到连续百分位，并从 `webui/symbol_sector_map.json` 取得行业条件。该路径至少需要 120 个有效交易日，页面会展示行业、百分位和参考日期。

### 股票池排序

输入 2–12 只股票，支持逗号、空格或换行分隔，前端自动切换为横截面排序。系统为每只股票增量刷新行情，并以股票池中最早的最新可用交易日作为共同截面，再按该截止日重新计算行业和连续市值百分位。每只股票生成 50 条路径，按未来 10 日预测收益的 P50 中位数排序，并展示行业、百分位和正向路径比例。前端的单股和横截面排序共用一份采样配置，当前为 `sample_count=50, temperature=0.65, top_p=0.8`。

预测历史不区分单股和批量入口，统一按行情截止日（信号日）归组。同一天分开提交的股票
会与批量提交结果进入同一个截面并按预测收益排序；同日同股重复预测只展示最近一次结果。
历史日期筛选同样使用信号日，而不是请求提交时间。

## API

- `POST /api/load-model`：加载 Beta V1.2 `Best@871` 并返回实际运行设备。
- `POST /api/predict`：提交 `{"symbol": "300395"}`，刷新行情并返回单股区间预测。
- `POST /api/a-share/rankings`：提交 `{"symbols": ["300395", "600519"]}`，返回自选股票池排名。
- `GET /api/a-share/symbols`：返回当前行业映射中的股票、行业 ID 和行业标签。

行情采集、缓存和预测结果保存都属于 Web 网关职责；`webui/market_data_cache/` 与
`webui/prediction_results/` 是运行时目录，不纳入 Git。Modal Serverless 只接收已整理的
OHLCVA 数据、未来时间戳、行业 ID 和连续市值百分位，执行模型调用并返回预测路径。

### 轻量 Serverless 推理服务

Serverless 入口与上面的完整 Web UI 分离，位于 `api/index.py`。它只负责模型调用：
调用方必须传入已经采集并清洗好的 OHLCVA 历史数据、未来时间戳、行业 ID 和连续市值
百分位。该服务不会按股票代码采集行情、读取本地数据面板、计算条件、生成图表或保存
预测结果。

安装最小依赖并启动：

```bash
python -m pip install -r requirements-serverless.txt
flask --app api.index run --port 8080
```

`POST /predict` 请求结构如下；`data` 必须恰好包含 120 行，`future_timestamps` 必须恰好
包含 10 个递增交易日。可执行的完整样例见 [`deploy/modal/curl_test.sh`](deploy/modal/curl_test.sh)。

```json
{
  "data": "120 OHLCVA row objects",
  "future_timestamps": "10 ISO-8601 timestamp strings",
  "pred_len": 10,
  "sample_count": 50,
  "temperature": 0.65,
  "top_p": 0.8,
  "sector_id": 42,
  "size_percentile": 0.5
}
```

默认模型是 `models/a_share_v1_beta/releases/beta_v1.2/best_model`，默认 tokenizer 是
同一发布目录内的 `tokenizer`。两者可分别通过 `KRONOS_MODEL_ID`、
`KRONOS_TOKENIZER_ID` 显式覆盖，设备可通过 `KRONOS_DEVICE` 指定。运行时会懒加载一次
并在热实例中复用。服务会拒绝非 `120→10`、缺少条件或包含旧 `size_bucket` 的请求。

### 部署到 Modal

Modal 部署已经独立整理到 [`deploy/modal/`](deploy/modal/README.md)。该目录只包含
线上模型调用所需的入口、最小依赖和测试脚本；本地训练、评估、数据与 Web UI 不会
进入 Modal 镜像。镜像构建时直接从公开 ModelScope 仓库
`luckfu/Kronos-A-Share-Beta-V1-2` 下载 `Best@871` 和配套 tokenizer，不上传本地
checkpoint，也不会覆盖旧 V6 App。

```bash
python -m pip install modal
modal token info
modal deploy deploy/modal/modal_app.py
./deploy/modal/curl_test.sh
```

部署后的公网地址是 `https://luckfu--kronos-beta-v1-2-inference-web.modal.run`，提供
`/health`、`/predict` 和 `/predict-batch`。鉴权、日志、强制重建和发布检查见独立说明。

## 增训与数据准备

A 股数据准备和实验口径详见 [`finetune/A_SHARE_PLAN_CN.md`](finetune/A_SHARE_PLAN_CN.md)。典型流程如下：

当前 Beta v2.1 决策头训练的标签、反向传播目标、固定分母 Best 规则、纯观察指标
与 Sealed Future 终审边界见
[`finetune/BETA_V2_1_TRAINING_PLAN_CN.md`](finetune/BETA_V2_1_TRAINING_PLAN_CN.md)。

### 全市场市值分层 V3（训练中）

CSI800 历史成分仍偏向大中盘，不能代表完整的小微盘风格。V3 在 2015–2025 年 CSI800 历史成分并集之外，按 2025 年末市值代理值补入微盘、小盘和中小盘股票各 300 只，并为这三个层级各保留 80 只整股票样本外集合。训练集共 2,389 只股票，整股票样本外集合 240 只，两者没有股票重叠。

V3 行情从 2015-01-01 开始，使训练同时覆盖 2015 年快速上涨和随后大幅下跌的市场状态；训练截止 2025-12-31，2026 数据不参与参数更新，只用于严格的时间外验证。处理后训练集包含 5,233,538 个有效窗口，按十个市值桶均衡编排，每个窗口在单遍覆盖中只出现一次。Colab 正式训练默认完整遍历两遍，共 524 个覆盖分段，之后才应用 5 段早停耐心。完整配置入口是 `finetune/train_a_share_v3.sh`，候选模型完成跨时间、跨股票评估前不会替换生产模型。

正式长训放在 Colab CUDA 上，本机只负责准备、校验和打包数据。Kaggle P100 三段吞吐基准和后续迁移说明见 [`finetune/KAGGLE_V3.md`](finetune/KAGGLE_V3.md)；完整 Colab 命令见 [`finetune/COLAB_V3.md`](finetune/COLAB_V3.md)。本机 `train_a_share_v3.sh` 只用于短跑通或有意的 MPS 实验，不作为默认完整训练入口。

滚动生产增训默认使用 `80%` 近期窗口和 `20%` 分层历史回放窗口，并允许在 `10%-30%` 历史占比内调整。历史回放覆盖不同牛熊震荡、波动状态和市值桶，用于降低灾难性遗忘；固定股票 holdout 与最近至少 20 个完整标签交易日分别负责跨股票和跨时间验证。当前市场 Rank IC、方向准确率和含成本交易收益是主要晋级指标，旧年份表现用于调整回放比例和回滚判断，不作为绝对否决条件。完整规则见 [`finetune/A_SHARE_PLAN_CN.md`](finetune/A_SHARE_PLAN_CN.md)。

V4 修正实验使用连续的 2015-2026 面板，但只允许信号日期在配置区间内生成样本。训练信号为 `2026-01-05` 至 `2026-06-17`，共 249,280 个窗口；`2026-06-18` 至 `2026-07-16` 的最后 20 个完整标签交易日作为时间验证。A 组只使用修正后的近期窗口，B 组在完全相同的近期窗口上加入 62,320 个按年份和市值桶分层抽取的历史窗口，占训练集 20%。P100 A/B 实验已完成：时间外 Rank IC 分别为 `-0.04796` 和 `-0.03966`，均低于增训前 V3 Last 的 `0.01124`，两个候选均不晋级生产。Kaggle 一键 A/B 入口和复现命令见 [`finetune/KAGGLE_V3.md`](finetune/KAGGLE_V3.md)。

增训是独立 App，不与预测进程共享模型或生命周期。运行 `python webui/finetune_app.py` 后访问 `http://127.0.0.1:7071/`；预测 App 继续独立运行在 7070。增训页面可选择本地 `NeoQuasar/Kronos-base` 或已有的完整 checkpoint 作为训练起点，再选择离散市值桶或“桶 + 连续百分位”；模型列表只展示名称和来源，不向前端返回本机路径。设备自动使用 MPS、CUDA 或 CPU，并提供规模预估、实时训练/验证/最佳 Loss 曲线、原始日志、完整覆盖进度、停止保存和 checkpoint 恢复。默认每段训练 20,000 个无重复窗口；训练器沿固定随机排列依次推进，76 段覆盖当前 1,509,252 个训练窗口一遍，全部窗口完成覆盖后才开始计算 5 段早停耐心。

Colab GPU 重建数据、下载基础模型和长训流程见 [`finetune/COLAB.md`](finetune/COLAB.md)。该流程不把大数据文件提交进 Git，而是在 Colab 中从 BaoStock 重建真实面板。下一版 `120+10` 上下文候选使用独立的 2014 补充数据，正式训练使用 Kaggle P100，入口见 [`finetune/KAGGLE_V5.md`](finetune/KAGGLE_V5.md)，不运行 `V5-90-control`。

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

V3 全覆盖基线 checkpoint 已完成一轮全覆盖训练：1,509,252 个有效窗口按固定排列无放回遍历，共推进 81 个保存分段。早期“每轮随机抽取 800 个窗口”的 checkpoint 仅作为历史实验保留；V6 Segment 542 `last_model` 曾作为页面默认模型，现已由 Beta V1.2 `Best@871` 替代。

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
