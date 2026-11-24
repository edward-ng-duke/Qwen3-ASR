# Qwen3-ASR vLLM Hotword Gateway

两容器离线部署：vLLM 0.26.0 引擎负责推理，FastAPI 网关提供原生兼容转写、热词安全编排、
SSE、WebSocket、VAD 和 Prometheus 指标。

```bash
docker compose -f vllm_hotword/docker-compose.yml config
docker compose -f vllm_hotword/docker-compose.yml up -d
curl -fsS http://127.0.0.1:6001/health/ready
```

v1.2.2 镜像为 `mygpt-qwen3-asr:v1.2.2`。HTTP、文件 SSE、realtime WS、热词字符串/数组、
VAD 提交时间戳和 readiness capabilities 与 Transformers 部署使用同一协议。

realtime 默认首个非空 preview 在每个新 VAD 语音段累计 0.5 秒时开始尝试，后续周期为 1 秒；
分别使用 `ASR_REALTIME_FIRST_UPDATE_SECONDS` 和 `ASR_REALTIME_UPDATE_SECONDS` 调整。

完整统一接口文档见仓库根目录 `QWEN3_ASR_UNIFIED_API.md`。
