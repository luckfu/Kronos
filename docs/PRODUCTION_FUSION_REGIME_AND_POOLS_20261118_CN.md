# 生产系统升级与运维手册：隐状态融合、市场宽度门控（`G_BREADTH`）、自定义股池云端同步与 ETL 故障恢复（2026-11-18）

本文档汇总并记录生产链路（`model/`、`serverless/`、`deploy/oracle-kronos/`、`webui/`）在提交 `4bafb83` 中的全部架构变更、参数约定、部署核对与 `2026-11-17` 线上 ETL 故障恢复实录。

---

## 1. 架构总览

生产推理采用“轻量网关（Oracle ARM：`oracleArm` / `oracle4C24G`）+ 无状态 GPU Serverless（Modal T4）”分离架构：

```text
Cron (20:30 CST) -> start_daily_sync_systemd.sh -> Imperial ETL (Supabase ashare_daily_k)
Cron (21:30 Mon-Fri) -> daily_prediction_scheduler.sh (轮询 etl_health_status.py ready_for_downstream=true)
  -> deploy/oracle-kronos/prediction_drill.py (全市场 120d 窗口, 12只/批, 16路径)
     -> Modal GPU (https://luckfu--kronos-beta-v1-2-inference-web.modal.run, luckfu/Kronos-small-0.1-Cosine-C2-Best)
        -> 返回 predictions (16路径) + hidden_state (512d context_layer=6 末位隐向量)
  -> 本地截面前向融合: FUSE_RIDGE_C2 (50% Rank(C2 D10) + 50% Rank(zscore(hidden_state) @ head_val_alpha1e4.npy))
  -> 本地因果市场状态门控: G_BREADTH (全市场有效交易股中 close*adj_factor > MA20 占比)
  -> 原子写入 /opt/kronos-web/data/prediction_shadow/<date>/full_market/{ranking.csv,predictions.jsonl,summary.json}
  -> kronos-web (Gunicorn 127.0.0.1:7072 -> Nginx /kronos/) 读取展示
```

> [!NOTE]
> **当前线上融合头状态与封存窗纪律**：
> `deploy/oracle-kronos/head_val_alpha1e4.npy`（与 `finetune/c2_ceiling/frozen/head_val_alpha1e4.npy` SHA-256 同为 `5b372471215668ff45238cc086521865c8f8244e4a6761794a62bd7df1a89722`）目前仍为 `val_only` 基线 512 维权重头，保持生产链路稳定。`G1–G18+` GPU 探索中涌现的新一代头部权重的替换与升级，需待全部探索结束并在严格封存验收窗（`2026-10-24+`）一次性评审后再做生产换头。

---

## 2. 核心模块改动说明

### 2.1 模型与 Modal Serverless 隐状态回传（零额外前向算力）

- [`model/kronos.py`](../model/kronos.py)：
  - `auto_regressive_inference(...)`、`KronosPredictor.generate(...)`、`predict(...)`、`predict_batch(...)` 新增布尔参数 `return_hidden=False`。
  - 在自回归第 0 步（`i == 0`）即提取 `context[::sample_count, -1, :]`（已注入行业/市值条件之后的 `512` 维末位隐藏状态），**不增加任何额外的神经网络前向计算**，仅增加极小 CPU/网络传输开销。
- [`serverless/service.py`](../serverless/service.py)：
  - `InferenceRequest` 与 `parse_request` 新增可选字段 `include_hidden: bool = False`（默认关闭，完全向后兼容老客户端）。
  - `predict_batch` 签名校验要求同一批次所有条目的 `include_hidden` 一致。
  - 开启时，`predict()` 在响应体顶层返回 `hidden_state: list[float]`，`predict_batch()` 在 `results[i]["hidden_state"]` 返回每只股票的 `512` 维向量。
- [`deploy/modal/modal_app.py`](../deploy/modal/modal_app.py)：
  - Modal App 名为 `kronos-beta-v1-2-inference`，GPU 为 `T4`，从 ModelScope 加载 `luckfu/Kronos-small-0.1-Cosine-C2-Best`，Tokenizer 使用 `NeoQuasar/Kronos-Tokenizer-base`。

### 2.2 日度批量预测、混合排名与市场宽度门控（`G_BREADTH`）

- [`deploy/oracle-kronos/prediction_drill.py`](../deploy/oracle-kronos/prediction_drill.py)：
  1. **隐状态请求与融合（`FUSE_RIDGE_C2`）**：
     - 默认启用 `include_hidden=True` 与 `--hidden-head head_val_alpha1e4.npy`（可用 `--no-hidden` 回退纯 C2；支持 `--hidden-head <path>` 指定权重文件，要求形状严格为 `(512,)`）。
     - 在 `build_outputs()` 中对全市场 `hidden_state` 矩阵执行当期截面标准化 `zscore_hidden_matrix(..., clip_val=5.0)`，与权重做内积得到 `score_hidden`。
     - 计算 `0.5 * RankPct(predicted_return_d10) + 0.5 * RankPct(score_hidden)` 得到 `score_fused`，按降序生成 `rank_fused` 与 `rank_percentile_fused`，同时保留原 `rank_d10` 与 `rank_percentile_d10` 以便前端随时切换比较。
     - 将结果模式写入 `summary.json["fusion"]`（如 `enabled: true, method: "FUSE_RIDGE_C2", head_file: "head_val_alpha1e4.npy"`；缺失隐向量或权重文件时安全降级为 `enabled: false`）。
  2. **因果市场宽度状态识别（`compute_market_regime` / `G_BREADTH`）**：
     - 在 SQL `load_audit_rows` 中统计截面全市场近 20 日复权均价 `ma20_adj_close` 与当日复权收盘价 `asof_adj_close`。
     - 严格过滤停牌（`trade_status == "1"`）、零成交额、存在 OHLCVA/复权因子异常或来源跳变（`max_source_switch_log_jump <= 1e-6`）及不足 20 日样本的个股后，统计 `asof_adj_close > ma20_adj_close` 的比例 `breadth_ma20`：
       - `breadth_ma20 < 0.30` → `level=0, position_pct=0, action="空仓防御"`；
       - `0.30 <= breadth_ma20 < 0.45` → `level=1, position_pct=50, action="半仓观察"`；
       - `breadth_ma20 >= 0.45` → `level=2, position_pct=100, action="满仓运行"`。
     - 该指标写入 `audit.json["regime"]` 与 `summary.json["regime"]`，零前视泄漏。

### 2.3 WebUI 升级：混合/原版切换、仓位横幅、30% 换仓缓冲提示与跨端自定义股池同步

- [`webui/app.py`](../webui/app.py) & [`webui/templates/daily_rankings.html`](../webui/templates/daily_rankings.html)：
  1. **排序切换与信号展示**：
     - `/api/daily-rankings` 与 `/api/daily-consensus` 新增 `sort=fused|c2` 查询参数（当数据包含 `rank_fused / score_fused` 时默认使用 `fused`，否则自动降级 `c2`）。
     - 日期列表、排行、共识、详情接口一并返回 `summary.json` 中的 `fusion` 与 `regime`。
     - 前端顶部展示市场宽度门控横幅（`#regime-banner`，含持仓建议胶囊与 `<30% 空仓 / 30-45% 半仓 / >=45% 满仓` 及 `Top 30%` 换仓缓冲规则提示），个股详情面板新增“30% 保留区”判定（领先百分位 `>= 70%`）与“隐藏头打分”。
  2. **自定义股池多端云端持久化（`/api/custom-pools`）**：
     - 数据存储路径由 `KRONOS_CUSTOM_POOLS_PATH`（默认 `/opt/kronos-web/data/custom_pools.json`）控制，使用 `fcntl.flock` 文件锁与原子 `os.replace` 按登录会话用户（`user:<username>` 或 `default`）隔离存储，彻底解决原仅存浏览器 `localStorage` 无法在电脑/手机 PWA 间同步的问题。
     - 支持：`GET /api/custom-pools` 读取、`POST /api/custom-pools` 新建/重命名编辑（`original_name`）/批量首次迁移合并（`merge: true`）、`DELETE /api/custom-pools/<name>` 删除；前端股池弹窗新增逐行“编辑”与“清空转为新建”交互，自动规范化 6 位 A 股代码为 `(sh|sz|bj).xxxxxx` 并去重。
  3. **PWA 图标补全**：[`deploy.sh`](../deploy/oracle-kronos/deploy.sh) 同步发布 `apple-touch-icon.png`、`icon-192.png`、`icon-512.png`、`logo-128.png` 与 `head_val_alpha1e4.npy`。

---

## 3. 部署与运维命令

### 3.1 Modal GPU 推理端部署

```bash
# 从仓库根目录执行（需本机已配置 ~/.modal.toml 鉴权）
modal deploy deploy/modal/modal_app.py

# 健康检查
curl -fsS https://luckfu--kronos-beta-v1-2-inference-web.modal.run/health
```

### 3.2 Oracle Web 网关与调度器部署

```bash
# 上传 webui/ + deploy/oracle-kronos/ 到 /opt/kronos-web，校验 nginx 并重启 kronos-web
bash deploy/oracle-kronos/deploy.sh

# 查看服务与定时器状态
ssh oracleArm 'sudo systemctl status kronos-web --no-pager'
ssh oracleArm 'curl -fsS http://127.0.0.1:7072/health'
ssh oracleArm 'crontab -l'
```

Oracle 服务器核心定时任务（已在 `oracleArm` 验证）：
- `30 20 * * * bash /data/openclaw_workspace/imperial_data_engine/scripts/start_daily_sync_systemd.sh`（每晚 20:30 触发 A 股 ETL 入库 Supabase）；
- `30 21 * * 1-5 /opt/kronos-web/deploy/daily_prediction_scheduler.sh`（工作日 21:30 轮询 `etl_health_status.py status --date <today>` 的 `ready_for_downstream=true`，就绪后自动执行全市场 `prediction_drill.py`，超时上限 `MAX_WAIT_SECONDS=57600` 秒即 16 小时，支持断点续跑）。

---

## 4. `2026-11-17` 线上 ETL 故障与手动恢复实录

1. **故障现象与根因**：
   - `2026-11-17 20:30 CST` 晚间定时任务启动 `imperial-daily-sync`，`kline_sync` 于 `20:34 CST` 中断退出（原因：Tushare `daily_basic` 触发 `5/day` 与 `1/hr` 限频，回退 Baostock 时遇到 `10002007 网络接收错误`），导致次日 `etl_health_status.py status --date 2026-11-17` 未放行，`daily_prediction_scheduler.sh` 处于等待状态。
2. **恢复处置流程**：
   - 在 `oracleArm` 上人工检查 ETL 状态确认仅停在日线/市值补采，通过 `systemd-run --user` 拉起恢复任务执行补同步（`imperial-recovery-20261117-20261118-112507`）；
   - 验证 ETL 健康检查通过：`etl_health_status.py status --date 2026-11-17` 返回 `status="healthy"`, `healthy=true`, `ready_for_downstream=true`（`kline_rows=4605`, `adj_factor_rows=4605`, `turnover_valid=4605`, `factor_rows=4410`, `universe_missing=0`）；
   - `daily_prediction_scheduler.sh` 于 `2026-11-18 12:08:18` 自动检测到 `ready_for_downstream=true` 启动预测，至 `13:02` 完成全部 `360/360` 批次：
     - `summary.json`：`status="complete"`, `prediction_count=4318`, `fusion.enabled=true`（`FUSE_RIDGE_C2`, `head_val_alpha1e4.npy`）；
     - `regime`：`breadth_ma20=0.1879`（`851 / 4529` 只站上 MA20），`action="空仓防御"`（`position_pct=0`）；
     - Web 端 `kronos-web` 与 `/health` 检查均正常上线提供查询。
