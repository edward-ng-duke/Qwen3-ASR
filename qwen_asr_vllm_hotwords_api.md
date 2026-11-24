# Qwen3-ASR vLLM 热词兼容 API（Agent 使用）

本文档描述 `mygpt-qwen3-asr:v1.2.1` 的 vLLM 热词兼容服务。服务对普通 OpenAI/vLLM
客户端保持兼容，并为需要热词、文件 SSE 或实时 PCM 的 Agent 提供增强接口。

交付 Image ID：`sha256:200b78ff0d649cd3de35dd26857a0b2bb1991c5b2eb3d6700f2a5e0f9e0dee5c`。
v1.2.1 的 preview 两单位修订、两轮确认、累计重放防重复和 finish/VAD 规则以
[`QWEN3_ASR_UNIFIED_API.md`](QWEN3_ASR_UNIFIED_API.md) 为准。

## 1. 地址与能力选择

当前 Linux 主机的现场局域网地址为 `10.0.0.93`，容器端口映射为 `6001:8000`：

```text
HTTP Base URL: http://10.0.0.93:6001
WebSocket:     ws://10.0.0.93:6001/v1/audio/realtime
```

不要把 `10.0.0.32:6001` 当成本次部署地址：2026-08-08 现场复核时，该地址的
`/health/ready` 返回 404，且不是当前 Linux 主机的地址。本服务应以 readiness 返回中
`backend=vllm`、`vllm_version=0.26.0` 为准；机器地址变化后先按本节命令重新确认。

不要使用 Docker 的 `172.x`/`192.168.x` 网桥地址。若 DHCP 地址发生变化，在服务器执行：

```bash
ip -br -4 addr show dev wlo1
ss -ltnp | grep ':6001'
```

接口选择：

| 场景 | 接口 |
|---|---|
| 现有 OpenAI/vLLM 客户端、完整文件 | `POST /v1/audio/transcriptions` |
| 完整文件、需要安全的增量事件 | `POST /v1/audio/transcriptions/stream` |
| 麦克风或电话 PCM | `WS /v1/audio/realtime` |
| readiness | `GET /health/ready` |
| 模型发现 | `GET /v1/models` |
| Prometheus | `GET /metrics` |

连通性检查：

```bash
curl -fsS http://10.0.0.93:6001/health/live
curl -fsS http://10.0.0.93:6001/health/ready
curl -fsS http://10.0.0.93:6001/v1/models
```

局域网无法访问时依次检查服务器本机请求、6001 是否监听 `0.0.0.0`、客户端是否在同一网段、
路由器 AP/访客隔离及 Linux 防火墙。若使用 UFW，只开放局域网来源：

```bash
sudo ufw allow from 10.0.0.0/24 to any port 6001 proto tcp
```

## 2. 兼容原则

- 不传 `prompt` 和 `hotwords`：请求直接交给 vLLM，响应结构、`response_format` 和原生流式行为
  由 vLLM 决定，不增加 VAD 或第二次推理。
- 传 `prompt` 或 `hotwords`：由安全网关执行 VAD、上下文泄漏校验和必要的无上下文回退。
- 支持两个模型名：`models/Qwen3-ASR-1.7B`、`Qwen/Qwen3-ASR-1.7B`。
- `/v1/chat/completions` 等非 ASR 路径透明转发，不应用热词或 VAD。
- 热词是 contextual bias，不是强制字符串替换。

## 3. 普通 HTTP 转写

### 3.1 无热词，兼容原客户端

```bash
curl -sS http://10.0.0.93:6001/v1/audio/transcriptions \
  -F 'file=@meeting.wav' \
  -F 'model=models/Qwen3-ASR-1.7B' \
  -F 'language=auto'
```

默认 OpenAI JSON 响应：

```json
{"text":"识别结果"}
```

### 3.2 字符串热词

```bash
curl -sS http://10.0.0.93:6001/v1/audio/transcriptions \
  -F 'file=@meeting.wav' \
  -F 'model=Qwen/Qwen3-ASR-1.7B' \
  -F 'language=zh' \
  -F 'hotwords=DeepSeek, 智谱, OpenAI'
```

也可以直接写标准前缀：

```text
hotwords=Vocabulary: DeepSeek, 智谱, OpenAI
```

### 3.3 JSON 数组热词

```bash
curl -sS http://10.0.0.93:6001/v1/audio/transcriptions \
  -F 'file=@meeting.wav' \
  -F 'language=zh' \
  -F 'hotwords=["DeepSeek","智谱","OpenAI"]'
```

### 3.4 prompt 与热词同传

```bash
curl -sS http://10.0.0.93:6001/v1/audio/transcriptions \
  -F 'file=@meeting.wav' \
  -F 'language=zh' \
  -F 'prompt=这是一场人工智能基础设施会议。' \
  -F 'hotwords=["DeepSeek","智谱","英伟达"]'
```

服务端实际传给 Qwen 的 system prompt 为：

```text
这是一场人工智能基础设施会议。
Vocabulary: DeepSeek, 智谱, 英伟达
```

数组热词会去除首尾空白、保持顺序并精确去重。限制为 100 项、每项 64 个 Unicode 字符、合并后
4096 字符。以 `[` 开头但不是合法 JSON 字符串数组时返回 422。

带上下文请求支持 `json`、`verbose_json`、`text`。带上下文且设置原生 `stream=true` 时返回 422，
必须改用下一节的安全 SSE 接口。无上下文 `stream=true` 仍按 vLLM 原生协议工作。

## 4. 文件 SSE

SSE 文件必须先完整上传，随后才返回事件：

```bash
curl -N http://10.0.0.93:6001/v1/audio/transcriptions/stream \
  -F 'file=@meeting.wav' \
  -F 'model=Qwen/Qwen3-ASR-1.7B' \
  -F 'language=auto' \
  -F 'hotwords=DeepSeek, 智谱'
```

事件顺序为：

```text
started → partial* → segment_final+ → final
```

错误时最后一个事件为 `error`。公共字段包括：

```json
{
  "type": "partial",
  "language": "Chinese",
  "confirmed": "不会撤回的文本",
  "delta": "本次新增",
  "preview": "可能变化的预览",
  "text": "当前完整显示文本",
  "raw_text": "模型当前输出",
  "final": false,
  "sequence": 2,
  "duration": 8.2,
  "start_ms": null,
  "end_ms": null,
  "processing_ms": 135.4,
  "model": "Qwen/Qwen3-ASR-1.7B",
  "request_id": "..."
}
```

`start_ms/end_ms` 仅在 `segment_final` 中为整数毫秒，其他事件为 `null`。它们描述原始解码音频中
本次提交的近似 VAD/推理区间，不是词级对齐；服务会裁掉长块的 2 秒推理重叠，保证范围单调且不重叠。
按事件创建业务 segment 时使用 `segment_final.delta`，不要重复保存累计 `confirmed`，也不要持久化
`preview`。必须读到 `final` 后才能把临时事件原子写入业务存储。

## 5. WebSocket 实时 PCM

连接后第一帧必须是初始化 JSON：

```json
{
  "type": "start",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "language": "auto",
  "prompt": "技术会议",
  "hotwords": ["DeepSeek", "智谱"]
}
```

`hotwords` 也可以是字符串 `"DeepSeek, 智谱"`。后续帧必须是二进制 PCM；结束时发送：

```json
{"type":"finish"}
```

保活控制帧：

```json
{"type":"ping"}
```

服务端事件为 `ready`、`partial`、`segment_final`、`pong`、`final`、`error`。`confirmed` 单调且
不会撤回；`segment_final` 已去重。默认最多 4 个会话。8 路实测无错误和串话，但更新间隔 p95
超过约 1.5 秒，因此没有作为正式默认值；可通过 `ASR_MAX_REALTIME_SESSIONS=8` 显式开启。

限制：单帧 1 MiB、会话最长 1 小时、空闲超时 5 分钟。支持 `pcm_s16le`、`pcm_f32le`，
8–96 kHz、1/2 声道；服务端降混并重采样到 16 kHz mono。

## 6. 语言与鉴权

`language=auto` 为默认值。普通话用 `zh`，粤语用 `yue`，英文用 `en`；也接受模型英文语言名，
例如 `Chinese`、`Cantonese`、`English`。

若部署设置了 `ASR_API_KEY`，HTTP 使用：

```bash
-H 'Authorization: Bearer YOUR_KEY'
```

WS 可使用同一 Authorization header，或在初始化 JSON 中传 `api_key`。健康检查不需要 Key；
`/v1/*` 与 `/metrics` 需要。生产 TLS 由外部反向代理负责。

## 7. 安全降级语义

带上下文时服务可能自动改用无上下文结果：

1. 主 VAD 没检测到语音时，对短音频无上下文识别一次，避免误杀短句。
2. 输出疑似回显 prompt 或热词表时丢弃，并无上下文重试一次。
3. 英文识别配非 ASCII prompt 时，用无上下文结果复核，拦截 prompt 英文译文泄漏。
4. WS 候选 VAD 只提交主 VAD 未覆盖、持续至少 250ms 且 RMS 不低于 0.0015 的区间。

每段最多一次回退，不递归。回退不会泄漏本次 prompt/hotwords，但无上下文模型仍可能返回普通语气词，
例如 `嗯。`、`Hmm.` 或 `Ahem.`。

## 8. 错误处理

网关错误结构：

```json
{"error":{"code":"invalid_parameters","message":"..."}}
```

常见状态码：400 音频/请求格式错误；401 Key 错误；413 文件超限；422 参数、模型或安全流式模式错误；
429 网关队列已满；503 引擎尚未 ready。Agent 对 429/503 可指数退避重试，对 400/401/413/422
必须修正请求后再试。

最大文件 8 GiB、最长音频 4 小时；长文件按 VAD 和最多约 30 秒分段处理。运行时不接受远程音频 URL，
只能上传文件或向 WS 发送 PCM。

## 9. 运维核对

```bash
docker inspect --format '{{.Image}} {{.State.Health.Status}}' mygpt-qwen3-asr
docker inspect --format '{{.Image}} {{.State.Health.Status}}' mygpt-qwen3-asr-engine
docker compose -f /home/edward/research/Qwen3-ASR/vllm_hotword/docker-compose.yml ps
curl -fsS http://127.0.0.1:6001/health/ready
nvidia-smi
```

判断版本必须记录实际 Image ID，不要只看可变 tag。
