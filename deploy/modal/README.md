# Kronos V6 Modal Serverless 部署

这个目录只负责把 V6 Segment 542 生产模型发布为 Modal 公网 API。训练、评估、
回测、行情采集、数据清洗、Web UI 和 checkpoint 选择仍在本地完成，不进入线上服务。

## 目录职责

- `modal_app.py`：Modal 镜像和 FastAPI 入口。
- `requirements.txt`：线上推理的最小 Python 依赖。
- `curl_test.sh`：生成 120 行测试 OHLCVA，并依次验证 `/health` 和 `/predict`。
- `../../serverless/service.py`：本地与 Modal 共用的模型调用和请求校验边界。
- `../../model/`：本地与 Modal 共用的 Kronos 模型实现。

Modal 镜像不会包含 `finetune/`、`webui/`、本地数据集或训练输出。模型权重在镜像
构建阶段直接从固定的公开 ModelScope 生产仓库
`luckfu/Kronos-A-Share-Forecast` 下载，不上传本地 checkpoint。这个仓库名保持不变，
其中的内容始终代表当前生产模型。

## 首次准备

在仓库根目录执行：

```bash
python -m pip install modal
modal setup
modal token info
```

如果 Modal 账号已经登录，只需确认 `modal token info` 正常。公开 ModelScope 仓库下载
不需要在 Modal 中保存 ModelScope token。

## 部署

始终从仓库根目录执行：

```bash
modal deploy deploy/modal/modal_app.py
```

模型下载层启用了 `force_build`。因此，即使生产仓库名不变，每次部署也会重新读取
ModelScope 当前快照，避免 Modal 复用包含旧权重的镜像层。

当前 App 名称为 `kronos-v6-inference`，公网地址为：

```text
https://luckfu--kronos-v6-inference-web.modal.run
```

如需强制替换所有旧容器：

```bash
modal deploy deploy/modal/modal_app.py --strategy recreate
```

查看部署历史和日志：

```bash
modal app history kronos-v6-inference
modal app logs kronos-v6-inference --timestamps
```

## API 鉴权

默认是公开 API。若需要 Bearer Token，先创建一个包含 `KRONOS_API_KEY` 的 Modal
Secret，再在部署时指定 Secret 名称：

```bash
export MODAL_KRONOS_SECRET_NAME=kronos-inference-secret
modal deploy deploy/modal/modal_app.py --strategy recreate
```

不要把 API key、ModelScope token 或其他凭证提交到本目录。

## curl 端到端测试

测试当前公网部署：

```bash
./deploy/modal/curl_test.sh
```

测试另一个 URL：

```bash
./deploy/modal/curl_test.sh https://example.modal.run
```

启用 API 鉴权后：

```bash
KRONOS_API_KEY='your-api-key' ./deploy/modal/curl_test.sh
```

脚本使用 `curl` 发送请求，使用 Python 标准库生成和格式化 JSON，不依赖 `jq`。
一次冷启动会加载 V6 权重和 tokenizer，首次 `/predict` 通常比热实例请求慢。
容器在请求结束 10 秒后缩容到零，避免低频请求继续占用 T4。`/predict-batch` 支持
一次提交 2–12 组等长行情，在同一次 GPU batch 中完成推理；行情采集仍由调用方负责。

## 发布新生产模型

本地完成训练和测试后，只有明确晋级为生产模型的 checkpoint 才能发布：

1. 把晋级 checkpoint 更新到固定 ModelScope 仓库
   `luckfu/Kronos-A-Share-Forecast`。
2. 校验远端权重 SHA-256 与晋级 checkpoint 一致。
3. 运行本地 serverless 测试。
4. 使用 `--strategy recreate` 部署并运行 `curl_test.sh`。
5. 检查 `/health` 返回的 `model` 是固定仓库名，并确认 `/predict` 返回 200、
   `model_device` 为 `cuda:0`。

这个流程不会修改本地训练默认值，也不会让 Modal 参与训练或数据采集。
