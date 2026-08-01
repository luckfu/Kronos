# Kronos A 股预测工作台

这是一个基于 [Kronos](https://github.com/shiyu-coder/Kronos) 的 A 股日线预测项目。当前交付版本固定使用经过市值分层增训的 `Kronos-base`，提供单股区间预测、自选股票池横截面排序、增训脚本和样本外回测脚本。

> 本项目用于研究和回测，不构成投资建议。模型预测、回测收益和排名都可能失效，不能直接替代交易系统或风险管理。

## 当前模型

- 模型：A-share Size-Conditioned Kronos-base
- 输入：90 个交易日的 `open/high/low/close/volume/amount`
- 输出：未来 10 个交易日的采样路径及 P10/P50/P90 区间
- 条件：按交易日横截面流通市值划分为 10 桶（`0` 最小，`9` 最大）
- 训练期：2020-01-02 至 2025-12-31
- 验证期：2026-01-01 至 2026-07-31
- 设备：自动选择 `MPS → CUDA → CPU`

公开模型权重已发布到 [ModelScope: `luckfu/a-share-size-kronos-base-earlystop50`](https://modelscope.cn/models/luckfu/a-share-size-kronos-base-earlystop50)。模型卡记录了加载方式、训练口径和限制。

## 快速部署

环境要求：Python 3.10+。Apple Silicon 建议安装支持 MPS 的 PyTorch；NVIDIA 机器安装对应 CUDA 版本的 PyTorch；没有加速设备时使用 CPU。

```bash
git clone <your-repository-url>
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
  --local-dir outputs/models/a_share_size_kronos_base_earlystop50/checkpoints/best_model
```

启动页面：

```bash
PYTHONPATH=. python webui/app.py
```

浏览器打开 <http://127.0.0.1:7070/>。生产部署时建议使用反向代理和进程管理器（例如 Nginx + systemd），不要把 Flask 开发服务器直接暴露到公网。服务监听地址和端口可在 `webui/app.py` 的启动入口中调整。

## 页面功能

页面只有两个工作 Tab，不需要选择模型或设备：

### 单股预测

输入六位 A 股代码（如 `300395`）后点击预测。系统会优先通过 BaoStock 拉取该股票最新复权日线和未来交易日；接口失败时回退到本地面板并明确标记数据来源。页面显示历史 K 线与成交量、未来 10 日预测区间、路径中位数和逐日明细。

### 股票池排序

输入 2–64 只股票，支持逗号、空格或换行分隔。系统使用股票池共同的最新交易日和每只股票对应的市值桶，生成三条路径并按未来 10 日预测收益中位数排序。榜单提供正/负预测、上涨路径比例、Top 8 标记，并可跳转到单股预测。

## API

- `POST /api/load-model`：加载唯一生产模型并返回实际运行设备。
- `POST /api/predict`：提交 `{"symbol": "300395"}`，刷新行情并返回单股区间预测。
- `POST /api/a-share/rankings`：提交 `{"symbols": ["300395", "600519"]}`，返回自选股票池排名。
- `GET /api/a-share/symbols`：返回本地面板中的可用股票和最新市值桶。

## 增训与数据准备

A 股数据准备和实验口径详见 [`finetune/A_SHARE_PLAN_CN.md`](finetune/A_SHARE_PLAN_CN.md)。典型流程如下：

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

训练脚本默认最多运行 50 轮并按验证集早停（可用 `KRONOS_EPOCHS` 覆盖）。Apple Silicon 直接运行即可自动使用 MPS；CUDA 可使用 `torchrun`。`Kronos-base` 基础权重可放在任意目录，通过 `KRONOS_PREDICTOR_PATH` 指定。

## 评估与回测

本次验证集 token loss 从原始模型的 `3.142226` 降至 `2.951122`，改善 `6.082%`，十个市值桶均有改善。129 个信号日的滚动小市值试验中，平均 Rank IC 为 `0.04152`，10 日方向准确率为 `55.31%`；按正预测持仓并计入换手成本后组合收益为 `-33.30%`，同期等权基准为 `-5.98%`。高换手和信号反转是主要问题，当前结果不支持直接实盘。

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
