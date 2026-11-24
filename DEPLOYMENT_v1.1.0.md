# Qwen3-ASR v1.1.0 部署记录

部署时间：2026-08-08 11:45 CST

## 镜像

| 服务 | 旧标签 Image ID | v1.1.0 Image ID | 端口 |
| --- | --- | --- | --- |
| vLLM | `sha256:7c32d95b16d45ca3019faab90a34d8beb2e435226e422be695dd993d16594e0d` | `sha256:7b2a549638defe4d25b505e5f9846b49514dabb4bd248807367f8c594c4567ce` | 6001 |
| Transformers | `sha256:507211db2cc56b2efcf3faeda0261326886768ff88093df88fbbf9d05ad8bd35` | `sha256:0e9ca8181e31d5361f6ce0c864350323916069edd059f42ee44d42324696b99c` | 6002 |

旧 `v1.0` 标签保留，用于镜像级回滚。

## 验证结果

- 6001、6002 容器均通过 Docker 健康检查。
- 同一段 8 秒中文音频分别通过 multipart HTTP、整文件 SSE 和 realtime WebSocket 验证。
- HTTP 验证了独立 `prompt` 与 JSON 数组格式 `hotwords`。
- SSE 验证了 `started -> partial* -> segment_final+ -> final`，且 `segment_final` 的毫秒时间范围合法、单调、不重叠。
- WebSocket 验证了 start 帧中的 `prompt`、`hotwords`、`api_key`，以及 ready、partial、segment_final、final 事件。
