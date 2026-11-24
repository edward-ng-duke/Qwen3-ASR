# Qwen3-ASR vLLM / Transformers 统一 API 接入文档

本文档面向业务后端、Agent、桌面端和前端配套服务的开发者，描述 Qwen3-ASR `v1.2.2` 的统一接入协议。
同一套客户端可连接 vLLM 或 Transformers 部署，不应根据后端类型编写分支。

文档对应的现场服务：

| 部署 | Base URL | 模型配置值 | 镜像 |
| --- | --- | --- | --- |
| vLLM（office 默认） | `http://10.0.0.93:6001` | `Qwen/Qwen3-ASR-1.7B` | `mygpt-qwen3-asr:v1.2.2` |
| Transformers（回滚服务） | `http://10.0.0.93:6002` | `Qwen/Qwen3-ASR-1.7B-hf` | `mygpt-qwen3-asr-hotword:v1.2.2` |

> `10.0.0.93` 是当前局域网地址。生产接入必须从配置读取 URL，不要在业务代码中硬编码地址。

## 1. 接入结论

调用方只需要支持以下四个公共接口：

| 场景 | 方法与路径 | 请求类型 | 返回类型 |
| --- | --- | --- | --- |
| 完整音频转写 | `POST /v1/audio/transcriptions` | `multipart/form-data` | JSON |
| 整文件增量转写 | `POST /v1/audio/transcriptions/stream` | `multipart/form-data` | SSE |
| 实时音频转写 | `WS /v1/audio/realtime` | JSON 控制帧 + 二进制 PCM | JSON 事件 |
| 就绪检查 | `GET /health/ready` | 无 | JSON |

统一原则：

- `prompt` 和 `hotwords` 都是独立字段。
- HTTP/SSE 的 `hotwords` 推荐使用“JSON 字符串数组”，也兼容普通字符串；WS 推荐使用真正的 JSON 数组，也兼容字符串。
- 不使用 `/v1/chat/completions` 做语音识别。
- 不发送 Base64、`audio_url`、data URL 或客户端拼接的 system message。
- 热词是上下文偏置，不是强制替换，不能保证每个热词一定出现。
- HTTP 成功响应只把 `text` 视为必需字段；其余字段按可选字段解析。
- SSE 必须收到 `final` 才算成功；HTTP 200 中的 `error` 事件仍然表示失败。
- 调用方不应探测或依赖 `backend=vllm/transformers` 来改变业务逻辑。

## 2. 应用配置

推荐配置：

```yaml
ASR_TYPE: "qwen3"
QWEN3_ASR_SERVER_URL: "http://10.0.0.93:6001"
QWEN3_ASR_MODEL: "Qwen/Qwen3-ASR-1.7B"
QWEN3_ASR_API_KEY: ""
QWEN3_ASR_HOTWORDS_ENABLED: true
ASR_REALTIME_ENABLED: true
ASR_SSE_IDLE_TIMEOUT_SECONDS: 120
ASR_REALTIME_HANDSHAKE_TIMEOUT_SECONDS: 10
```

切换到 Transformers 时只修改：

```yaml
QWEN3_ASR_SERVER_URL: "http://10.0.0.93:6002"
QWEN3_ASR_MODEL: "Qwen/Qwen3-ASR-1.7B-hf"
```

字段说明：

| 配置 | 必填 | 默认值 | 说明 |
| --- | ---: | --- | --- |
| `ASR_TYPE` | 是 | — | 固定为 `qwen3`，不表达具体推理后端 |
| `QWEN3_ASR_SERVER_URL` | 是 | — | HTTP Base URL；不得包含接口路径 |
| `QWEN3_ASR_MODEL` | 是 | — | 随部署切换的模型标识 |
| `QWEN3_ASR_API_KEY` | 否 | 空 | 服务未启用鉴权时为空 |
| `QWEN3_ASR_HOTWORDS_ENABLED` | 否 | `true` | 应用级紧急回退开关；关闭后应用不发送热词 |
| `ASR_REALTIME_ENABLED` | 否 | `true` | 是否允许应用使用 realtime WS |
| `ASR_SSE_IDLE_TIMEOUT_SECONDS` | 否 | `120` | 文件 SSE 连续无新事件的最大等待秒数 |
| `ASR_REALTIME_HANDSHAKE_TIMEOUT_SECONDS` | 否 | `10` | WS 建连并等待 `ready` 的最大秒数 |

旧 Qwen 类型名仅作为 `qwen3` 的弃用别名：

```text
qwen3_vllm / qwen3_transformers / qwen3_hotword -> qwen3
```

旧连接字段不再兼容读取。启动时发现 `QWEN3_VLLM_*`、`QWEN3_TRANSFORMERS_*` 或
`QWEN3_HOTWORD_ASR_*` 会拒绝启动，只列出字段名并提示迁移到 `QWEN3_ASR_*`，不会记录字段值。

若调用方接入的是 `mygpt-audio-streaming-frontend`，可通过应用接口读取面向 UI 的能力开关：

```http
GET /api/config/features
```

```json
{
  "success": true,
  "features": {
    "hotwords_enabled": true,
    "asr_prompt_enabled": true
  }
}
```

实际响应可能还包含其他应用功能开关。调用方只读取自己认识的字段；该接口不会暴露 vLLM 或
Transformers 类型。

## 3. 通用请求约定

### 3.1 鉴权

未设置 API Key 时不需要鉴权。启用后，HTTP 和 SSE 使用 Bearer Token：

```http
Authorization: Bearer YOUR_API_KEY
```

WebSocket 可以使用 `Authorization` 请求头，也可以在第一帧 start JSON 中传 `api_key`。浏览器原生
WebSocket 难以设置自定义请求头，推荐在 start 帧传递。

健康检查 `/health/live` 和 `/health/ready` 不需要 API Key；`/v1/*` 与 `/metrics` 在启用 Key 后需要鉴权。

### 3.2 language

省略或传 `auto` 表示自动识别。常用值：

| 值 | 语言 | 值 | 语言 |
| --- | --- | --- | --- |
| `zh` | 普通话 / Chinese | `yue` | 粤语 / Cantonese |
| `en` | English | `ja` | Japanese |
| `ko` | Korean | `fr` | French |
| `de` | German | `es` | Spanish |

完整支持代码：

```text
ar, yue, zh, cs, da, nl, en, fil, fi, fr, de, el, hi, hu,
id, it, ja, ko, mk, ms, fa, pl, pt, ro, ru, es, sv, th, tr, vi
```

也可以传模型英文语言名，如 `Chinese`、`Cantonese`、`English`，不区分大小写。未知语言返回 422。

### 3.3 prompt

`prompt` 用于描述语音的语义背景，例如：

```text
这是一场关于大模型推理基础设施和 GPU 调度的技术会议。
```

推荐写法：

- 描述会议领域、上下文或可能出现的人名/项目背景。
- 保持简短、客观，不写输出格式要求。
- 不要求摘要、翻译、润色、分点或角色扮演。
- 英文识别配非 ASCII prompt 会触发额外一致性复核，可能增加推理并抑制 partial。

空白会被去除，清理后为空等价于不传。

### 3.4 hotwords

HTTP 和 SSE 中推荐发送 JSON 数组序列化后的字符串：

```text
["邓郁松","阿莱","Qwen3-ASR"]
```

WebSocket start 帧中推荐发送真正的数组：

```json
{"hotwords":["邓郁松","阿莱","Qwen3-ASR"]}
```

为兼容旧调用，HTTP/SSE/WS 也接受普通字符串。字符串可用逗号或分号分隔，并可带
`Vocabulary:` 前缀，例如 `Vocabulary: 邓郁松, 阿莱`。两种部署共用同一清理、去重与限制规则。

服务端规范化规则：

- 最多 100 项。
- 每项最多 64 个 Unicode 字符。
- 去除每项首尾空白。
- 清理后为空的项目会被忽略。
- 保持顺序，按精确字符串去重，区分大小写。
- `prompt + "\n" + "Vocabulary: ..."` 合并后最多 4096 个 Unicode 字符。

实际送入模型的上下文等价于：

```text
这是一场技术会议。
Vocabulary: 邓郁松, 阿莱, Qwen3-ASR
```

调用方仍应自行做相同校验，以便在上传音频前向用户返回明确错误；服务端校验是最终权威。

### 3.5 音频约束

文件接口：

- 最大文件：8 GiB。
- 最大解码时长：4 小时。
- 可接受 FFmpeg 能解码的常见 WAV、MP3、M4A、FLAC、OGG 等格式。
- 为获得两种后端最一致的行为，推荐调用前转换为 16 kHz、单声道 WAV。
- 服务端会按 VAD 和最长约 30 秒的窗口处理长音频。

推荐转换命令：

```bash
ffmpeg -y -i input.m4a -ac 1 -ar 16000 -c:a pcm_s16le normalized.wav
```

实时接口只接受原始 PCM，不接受 WAV/MP3/M4A 文件头：

- 编码：`pcm_s16le` 或 `pcm_f32le`。
- 采样率：8000–96000 Hz。
- 声道：1 或 2；双声道由服务端降混。
- 单个二进制 WS 帧最大 1 MiB。
- 单会话最长 1 小时，空闲超时 5 分钟。
- 默认最多 4 个并发 realtime 会话。

## 4. 普通 HTTP 转写

### 4.1 请求

```http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
Authorization: Bearer <optional>
```

公共字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | ---: | --- | --- |
| `file` | 文件 | 是 | — | 待识别音频 |
| `model` | 字符串 | 建议传 | 当前模型 | 使用配置中的 `QWEN3_ASR_MODEL` |
| `language` | 字符串 | 否 | `auto` | 自动识别、代码或英文语言名 |
| `prompt` | 字符串 | 否 | 空 | 语义背景 |
| `hotwords` | 字符串 | 否 | 空 | 推荐 JSON 字符串数组；兼容普通分隔字符串 |

`response_format`、`stream` 等 OpenAI/vLLM 扩展不属于两种部署的公共契约。需要跨 6001/6002 切换时不要依赖它们。

### 4.2 curl：无上下文

```bash
curl -sS http://10.0.0.93:6001/v1/audio/transcriptions \
  -F 'file=@/absolute/path/meeting.wav' \
  -F 'model=Qwen/Qwen3-ASR-1.7B' \
  -F 'language=auto'
```

### 4.3 curl：prompt + 热词

```bash
curl -sS http://10.0.0.93:6001/v1/audio/transcriptions \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -F 'file=@/absolute/path/meeting.wav' \
  -F 'model=Qwen/Qwen3-ASR-1.7B' \
  -F 'language=zh' \
  -F 'prompt=这是一场关于语音识别部署的技术会议。' \
  -F 'hotwords=["邓郁松","阿莱","Qwen3-ASR"]'
```

当前未启用 API Key 时删除 `Authorization` 这一行。

### 4.4 成功响应兼容

vLLM 最小响应：

```json
{
  "text": "这是识别结果。"
}
```

Transformers 完整响应：

```json
{
  "language": "Chinese",
  "text": "这是识别结果。",
  "raw_text": "language Chinese<asr_text>这是识别结果。",
  "duration": 8.0,
  "processing_ms": 382.4,
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "request_id": "70d927a135784587be1bf46b81ca05d8"
}
```

跨后端客户端必须按下面的规则解析：

- 响应必须是 JSON 对象，`text` 必须存在且类型为字符串；显式空字符串是合法的“无文本”结果。
- `language`：可选；缺失时返回 `null`，不要猜测。
- `duration`：可选，单位秒；缺失时可从本地音频计算。
- `raw_text`、`processing_ms`、`model`、`request_id`：诊断字段，不参与业务正确性判断。
- HTTP 响应头可能包含 `x-request-id`；vLLM 上下文请求还可能包含 `x-asr-processing-ms`。

### 4.5 Python/httpx 示例

```python
from pathlib import Path
import json
import wave

import httpx


base_url = "http://10.0.0.93:6001"
model = "Qwen/Qwen3-ASR-1.7B"
api_key = ""
audio_path = Path("/absolute/path/meeting.wav")

headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
data = {
    "model": model,
    "language": "zh",
    "prompt": "这是一场关于语音识别部署的技术会议。",
    "hotwords": json.dumps(["邓郁松", "阿莱"], ensure_ascii=False),
}

with audio_path.open("rb") as audio_file:
    response = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": (audio_path.name, audio_file, "audio/wav")},
        data=data,
        headers=headers,
        timeout=90,
    )
response.raise_for_status()
payload = response.json()

if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
    raise RuntimeError("invalid ASR response: string text is required")
text = payload["text"]
language = payload.get("language")
duration = payload.get("duration")
if duration is None:
    with wave.open(str(audio_path), "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()

print({"text": text, "language": language, "duration": duration})
```

## 5. 整文件 SSE 转写

### 5.1 行为说明

SSE 适合文件导入进度和服务端 VAD 分段。它不是“边上传边识别”：客户端先上传完整文件，服务端完成
解码后再持续返回事件。

```http
POST /v1/audio/transcriptions/stream
Content-Type: multipart/form-data
Accept: text/event-stream
```

multipart 字段与普通 HTTP 的五个公共字段一致。

正常事件顺序：

```text
started -> partial* -> segment_final* -> final
```

失败可能有两种形式：

1. 建立 SSE 前返回非 200 HTTP 和 JSON 错误。
2. 已返回 HTTP 200，流内最后出现 `error` 事件。

两者都必须视为失败。

### 5.2 curl 示例

```bash
curl -N --no-buffer http://10.0.0.93:6001/v1/audio/transcriptions/stream \
  -F 'file=@/absolute/path/meeting.wav' \
  -F 'model=Qwen/Qwen3-ASR-1.7B' \
  -F 'language=zh' \
  -F 'prompt=这是一场技术会议。' \
  -F 'hotwords=["邓郁松","阿莱"]'
```

服务端返回标准 SSE：

```text
event: started
data: {"type":"started","sequence":1,...}

event: segment_final
data: {"type":"segment_final","sequence":2,"delta":"...","start_ms":0,"end_ms":8000,...}

event: final
data: {"type":"final","sequence":3,"final":true,...}
```

### 5.3 事件公共结构

```json
{
  "type": "segment_final",
  "language": "Chinese",
  "confirmed": "累计且不会撤回的文本",
  "delta": "本段新增的去重文本",
  "preview": "",
  "text": "当前完整显示文本",
  "raw_text": "模型原始输出",
  "final": false,
  "sequence": 3,
  "duration": 62.4,
  "start_ms": 10200,
  "end_ms": 17860,
  "processing_ms": 431.7,
  "model": "Qwen/Qwen3-ASR-1.7B",
  "request_id": "70d927a135784587be1bf46b81ca05d8",
  "error": null
}
```

字段语义：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | 字符串 | `started`、`partial`、`segment_final`、`final`、`error` |
| `sequence` | 整数 | 从 1 开始，整个流严格递增 |
| `duration` | 浮点数 | 原始解码音频总时长，单位秒 |
| `language` | 字符串或 null | 当前检测语言 |
| `confirmed` | 字符串 | 当前累计、已确认文本 |
| `delta` | 字符串 | 本事件新增文本；`segment_final.delta` 用于业务分段 |
| `preview` | 字符串 | 可能变化的临时预览，禁止持久化 |
| `text` | 字符串 | 当前完整显示文本 |
| `raw_text` | 字符串 | 模型原始输出，仅诊断使用 |
| `final` | 布尔值 | 仅最终成功事件为 `true`；错误事件也可能为 `true` |
| `start_ms` / `end_ms` | 整数或 null | 仅 `segment_final` 有值，单位毫秒 |
| `processing_ms` | 浮点数 | 当前事件或总请求推理耗时 |
| `model` | 字符串 | 实际服务模型 |
| `request_id` | 字符串 | 请求追踪 ID |
| `error` | 对象或 null | `{code, message}` |

### 5.4 各事件含义

`started`

- 表示上传、解码和初始校验完成。
- `duration` 可用于初始化进度条。
- `start_ms/end_ms` 为 `null`。

`partial`

- 仅用于临时显示或进度反馈，可能完全不出现。
- `preview` 可能变化，禁止写入数据库。
- 跨语言 prompt 安全复核等场景可能主动抑制 partial。

`segment_final`

- 表示一个不可撤回的业务分段。
- 使用 `delta` 创建该段文本，不要再次保存累计 `confirmed`。
- `start_ms/end_ms` 是原始解码音频中的近似 VAD/推理提交区间，不是词级强制对齐。
- 服务端会裁掉长段推理重叠，保证范围单调且互不重叠。
- 合法条件：`0 <= start_ms < end_ms <= round(duration * 1000)`。

`final`

- 表示整个文件成功完成。
- `final=true`，`text/confirmed` 是最终累计文本。
- `start_ms/end_ms` 为 `null`。
- 只有收到该事件后才能原子写入业务存储。
- 无语音时允许没有 `segment_final`，但最终事件必须满足 `final=true` 且最终文本为空；调用方应将其映射为“未检测到可识别语音”，不要再做 HTTP 复核。

`error`

- 即使 HTTP 状态为 200，也表示本次流失败。
- 丢弃所有临时事件，不得保存部分识别结果。

### 5.5 持久化规则

文件导入推荐流程：

1. 把完整标准 WAV 上传到 SSE 接口。
2. 所有事件先保存在当前任务内存中。
3. 用 `segment_final.end_ms / started.duration` 更新进度。
4. `partial` 只做状态显示，不落库。
5. 收到 `final` 后，逐个使用 `segment_final.delta` 作为文本。
6. 按 `start_ms/end_ms` 从同一标准 WAV 切出对应 PCM，封装成片段 WAV。
7. 跳过空 `delta`、空文本或非法时间范围。
8. 为本次导入生成唯一 `import_batch_id`，segment 与 GridFS metadata 先作为 staging 批次写入。
9. 所有片段与音频成功后，用一条 meeting 更新写入 `published_import_batch_id`，作为原子可见性切换。
10. 所有读取只返回 legacy 数据和当前 published 批次；定时清理超出安全窗口且未被任何 meeting 引用的 staging 数据。

若缺少 `final`、`final=false`、JSON 损坏、未知事件、sequence 非递增、时间戳非法、网络中断，或连续
`ASR_SSE_IDLE_TIMEOUT_SECONDS` 没有新事件，本次流不完整，必须丢弃临时结果。sequence 必须是非布尔整数，
第一条为 `started/1`；后续只要求严格递增，不要求连续。

分块 HTTP 回退中，HTTP 200 且 `text=""` 表示该区间无可识别语音；连接异常、超时或非 200 重试耗尽则
记录失败范围并让整场导入失败，不能保存其余成功片段。所有分块均为空时也应报告“未检测到可识别语音”。

### 5.6 严格 Python SSE 示例

```python
import json
from pathlib import Path

import httpx


async def transcribe_sse(
    base_url: str,
    model: str,
    audio_path: Path,
    *,
    language: str = "auto",
    prompt: str | None = None,
    hotwords: list[str] | None = None,
    api_key: str = "",
) -> list[dict]:
    data = {"model": model, "language": language}
    if prompt and prompt.strip():
        data["prompt"] = prompt.strip()
    if hotwords:
        data["hotwords"] = json.dumps(hotwords, ensure_ascii=False)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = httpx.Timeout(connect=10, read=None, write=60, pool=10)
    events: list[dict] = []
    previous_sequence = 0
    previous_end_ms = 0
    duration_ms: int | None = None
    seen_started = False
    seen_final = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        with audio_path.open("rb") as audio_file:
            async with client.stream(
                "POST",
                f"{base_url.rstrip('/')}/v1/audio/transcriptions/stream",
                files={"file": (audio_path.name, audio_file, "audio/wav")},
                data=data,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors="replace")[:500]
                    raise RuntimeError(f"SSE HTTP {response.status_code}: {detail}")

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("SSE event JSON is broken") from exc

                    event_type = event.get("type")
                    sequence = event.get("sequence")
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= previous_sequence:
                        raise RuntimeError("SSE sequence is not strictly increasing")
                    previous_sequence = sequence

                    if not seen_started:
                        if event_type != "started":
                            raise RuntimeError("SSE stream does not start with started")
                        seen_started = True
                        duration_ms = round(float(event.get("duration", 0)) * 1000)
                    elif event_type == "started":
                        raise RuntimeError("duplicate started event")

                    if event_type == "error":
                        raise RuntimeError("SSE error event")
                    if event_type not in {"started", "partial", "segment_final", "final"}:
                        raise RuntimeError("unknown SSE event")

                    if event_type == "segment_final":
                        start_ms = event.get("start_ms")
                        end_ms = event.get("end_ms")
                        if not isinstance(start_ms, int) or not isinstance(end_ms, int):
                            raise RuntimeError("segment timestamp is missing")
                        if start_ms < previous_end_ms or end_ms <= start_ms:
                            raise RuntimeError("segment timestamp overlaps or is empty")
                        if duration_ms is not None and end_ms > duration_ms:
                            raise RuntimeError("segment timestamp exceeds duration")
                        previous_end_ms = end_ms

                    events.append(event)
                    if event_type == "final":
                        if not event.get("final"):
                            raise RuntimeError("final event flag is false")
                        seen_final = True
                        break

    if not seen_final:
        raise RuntimeError("incomplete SSE stream: final event is missing")
    return events
```

## 6. Realtime WebSocket

### 6.1 建立连接

HTTP Base URL 到 WS URL 的推导：

```text
http://host:port  -> ws://host:port/v1/audio/realtime
https://host      -> wss://host/v1/audio/realtime
```

连接成功后，第一帧必须是 start JSON，且必须在 10 秒内发送：

```json
{
  "type": "start",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "language": "auto",
  "prompt": "这是一场技术会议。",
  "hotwords": ["邓郁松", "阿莱"],
  "api_key": ""
}
```

收到 `ready` 后发送二进制 PCM 帧。推荐每帧 20–100 ms，避免把整小时音频作为一个帧发送。

结束发送：

```json
{"type":"finish"}
```

保活：

```json
{"type":"ping"}
```

服务端会返回 `pong`。

### 6.2 服务端事件

```text
ready -> partial* -> segment_final* -> final
```

中间可穿插 `pong`，失败时返回 `error` 后关闭连接。

WS 事件字段与 SSE 基本相同。`segment_final` 含整数毫秒 `start_ms/end_ms`，表示当前 realtime
PCM 时间轴上的非重叠确认范围；`duration` 表示当前已接收音频时长。

处理规则：

- `ready`：初始化成功，可以开始或继续发送 PCM。
- `partial`：临时显示，不持久化 `preview`。
- `segment_final`：使用 `delta` 追加已确认文本。
- `final`：`final=true`，会话成功结束。
- `error`：只依据安全的 `error.code` 分类，本次会话失败；不要记录上游 message。
- `confirmed` 应保持单调；客户端不应把累计值重复追加。
- `ready.sequence=0`；`partial/segment_final/pong/final` 从 1 开始严格递增，布尔值不是合法 sequence。
- 客户端应验证事件为对象、类型在允许集合内、`final=true`、时间戳合法且 `confirmed` 前缀单调。

#### Realtime preview 稳定与首帧语义（v1.2.2）

- `confirmed` 永不回滚，可以作为累计确认状态；不要把累计值整段重复追加。
- `preview` 已经过显示稳定处理，但仍然只能用于临时 UI，禁止持久化。
- 中文一字、英文/数字一词算一个显示单位；标点和空白不单独计数。
- preview 末尾不超过两个显示单位的修订可立即出现；更大的修订必须连续两轮兼容后才下发。
- 如果模型假设改写了 confirmed 内部，服务保留上一版可见 preview，等待 `segment_final` 收口，不发送瞬时空尾部。
- 极短 VAD 残段若重放已确认的累计开头，服务只按可信末尾锚点提取新增尾部；找不到可信锚点时保留 confirmed，绝不再次拼接整份累计前缀。
- 10 秒推理窗口完成 `segment_final` 后，新窗口若返回旧 confirmed 的精确重叠或修订版累计开头，服务会剔除重叠或暂时抑制该 preview；客户端不会看到已转正文本作为新话轮 preview 再次出现。
- `segment_final` 和 `final` 是权威结果，可能一次性收敛尚未展示的修订。
- 相同 partial 可能重复出现以维持事件节奏；UI 应跳过相同内容的 DOM 重绘。
- `delta` 只在新 `text` 是上一版已下发文本的扩展时表示新增后缀；修订场景允许为空。
- 每个主 VAD 语音段默认在累计 0.5 秒时尝试首个非空 partial；首帧为空或被上下文安全校验抑制时仍按 0.5 秒重试。
- 首个非空 partial 下发后，后续默认每 1 秒更新一次；配置项为 `ASR_REALTIME_FIRST_UPDATE_SECONDS` 和 `ASR_REALTIME_UPDATE_SECONDS`，必须满足 `0 < FIRST <= UPDATE`。
- 不足首帧周期就结束的短语音允许直接发送 `segment_final`，客户端不得假设每段一定存在 partial。

### 6.3 WebSocket 关闭码

| 关闭码 | 常见原因 | 处理建议 |
| ---: | --- | --- |
| `1000` | 正常完成 | 成功 |
| `1008` | start/control/音频参数无效、超时 | 修正请求，不盲目重试 |
| `1009` | 单帧超过 1 MiB | 减小 PCM 帧 |
| `1013` | 服务未 ready 或会话容量已满 | 短暂退避，必要时降级 HTTP |
| `4401` | API Key 错误 | 修正鉴权 |
| `1011` | 服务端内部错误 | 记录 request ID，退避或降级 |

### 6.4 Python websockets 示例

先生成 16 kHz mono PCM：

```bash
ffmpeg -y -i input.m4a -f s16le -acodec pcm_s16le -ac 1 -ar 16000 input.pcm
```

客户端：

```python
import asyncio
import json
from pathlib import Path

import websockets


async def realtime_transcribe() -> str:
    ws_url = "ws://10.0.0.93:6001/v1/audio/realtime"
    pcm_path = Path("input.pcm")
    start = {
        "type": "start",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
        "language": "zh",
        "prompt": "这是一场技术会议。",
        "hotwords": ["邓郁松", "阿莱"],
        "api_key": "",
    }

    async with websockets.connect(ws_url, max_size=2 * 1024 * 1024) as ws:
        await ws.send(json.dumps(start, ensure_ascii=False))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready" or ready.get("sequence") != 0:
            raise RuntimeError("realtime init failed")

        # 100 ms 的 pcm_s16le/16k/mono = 1600 samples * 2 bytes。
        with pcm_path.open("rb") as pcm:
            while chunk := pcm.read(3200):
                await ws.send(chunk)
                await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "finish"}))

        confirmed = ""
        async for raw in ws:
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError("realtime error")
            if event_type == "segment_final":
                # delta 是新增文本；confirmed 是累计文本，二者不要同时追加。
                confirmed += event.get("delta") or ""
            if event_type == "final":
                return str(event.get("text") or event.get("confirmed") or confirmed)

        raise RuntimeError("connection closed before final")


print(asyncio.run(realtime_transcribe()))
```

生产客户端应让“发送 PCM”和“接收事件”并发执行，避免服务端事件积压。示例为了突出协议而采用顺序写法。

## 7. 健康检查与模型发现

### 7.1 readiness

```bash
curl -fsS http://10.0.0.93:6001/health/ready
```

公共成功条件：HTTP 200、响应为合法 JSON 对象，并且 `status` 严格等于 `ready`。

```json
{
  "status": "ready",
  "capabilities": {
    "http": true,
    "file_sse": true,
    "realtime": true,
    "realtime_segment_timestamps": true,
    "hotwords": true,
    "prompt": true
  }
}
```

响应还可能包含模型、revision、GPU 或后端信息，但这些是诊断扩展。应用启动时缓存 `capabilities`；
旧服务缺少该字段时保守按 HTTP-only 处理，不先试 SSE/WS。

未完成模型加载时返回 HTTP 503：

```json
{
  "status": "loading",
  "detail": null
}
```

### 7.2 liveness

```bash
curl -fsS http://10.0.0.93:6001/health/live
```

```json
{"status":"live"}
```

`live` 只表示进程存在，不代表模型可接收请求。业务流量必须以 `ready` 为准。

### 7.3 模型发现

```bash
curl -sS http://10.0.0.93:6001/v1/models
```

该接口适合诊断，不用于自动决定前端业务分支。本次统一接入不实现自动模型发现；模型仍由配置明确指定。

## 8. 错误结构与重试策略

HTTP/SSE 建流前，由统一网关主动生成的错误结构为：

```json
{
  "error": {
    "code": "invalid_parameters",
    "message": "hotwords must be a string or string array"
  }
}
```

multipart 缺少必填字段时，FastAPI 参数校验可能返回 `{"detail":[...]}`；6001 的无上下文请求直接
代理给原生 vLLM 时，也可能保留 OpenAI 兼容的上游错误结构。客户端只保留安全的 HTTP status 和
`error.code` 用于分类，不能把上游 message、detail、URL、prompt 或 API Key 写入用户错误或日志；
重试判断以 HTTP 状态码为主。

常见状态：

| HTTP 状态 | 含义 | 是否自动重试 |
| ---: | --- | --- |
| 400 | multipart、文件、音频解码或请求格式错误 | 否 |
| 401 | API Key 错误 | 否 |
| 403 | 无访问权限 | 否 |
| 404 / 405 | 接口能力缺失或方法不支持 | SSE 可降级普通 HTTP |
| 413 | 整文件 SSE 超限 | 不重试整文件，直接降级 10 秒分块 HTTP |
| 422 | model/language/prompt/hotwords 参数错误 | 否 |
| 429 | 调度队列或 realtime 容量已满 | 建流前最多退避重试两次 |
| 503 | 模型未 ready | 建流前最多退避重试两次 |
| 5xx | 服务端或上游错误 | 幂等文件请求可有限退避 |

推荐退避：首次等待 1 秒，第二次等待 2 秒，并加入少量随机抖动。不要对 400/401/403/422 重复执行
GPU 推理；必须先修正请求。

文件 SSE 推荐降级规则：

- 404/405、连接断开、流中 `error`、缺少 `final`、JSON/sequence/时间戳错误：丢弃所有临时事件，
  从头走普通 HTTP 或兼容分块 HTTP 链路。
- 429/503：只在尚未消费任何事件时最多退避两次，再降级。
- 413：不重试整文件，清空临时事件后直接走 10 秒分块 HTTP。
- 400/401/403/422：直接失败并向调用方返回安全、可诊断的状态与错误 code。
- 无语音 `final=true` 空结果：直接报告“未检测到可识别语音”，不走 HTTP 复核。
- 降级请求必须继续传递相同的 `prompt` 和 `hotwords`。
- 降级前不得写入 segment、MongoDB、GridFS 或其他持久化存储，否则会产生重复数据和孤儿音频。

## 9. 上下文安全行为

带 `prompt` 或 `hotwords` 时，服务端会启用上下文安全处理：

1. 使用 VAD 判断有效语音区间。
2. 检测输出是否疑似回显 prompt 或整组热词。
3. 可疑结果会被丢弃，对同一分段最多执行一次无上下文回退。
4. 英文识别配非 ASCII prompt 时执行额外无上下文一致性复核。

因此带上下文与不带上下文可能得到不同结果，这是安全策略的一部分。调用方不得自行恢复被服务端拦截的
上下文结果，也不要把空音频结果替换成 prompt 或热词。

日志要求：

- 不记录 prompt、热词正文、识别文本、raw_text、confirmed、preview 或 delta。
- 可以记录 prompt 是否存在、热词数量、HTTP 状态、错误类型、request ID 和耗时。
- API Key 绝不能进入日志。

## 10. 旧调用迁移

旧写法：

```json
{
  "model": "qwen3-asr-1.7b",
  "messages": [
    {"role": "system", "content": "Vocabulary: 邓郁松"},
    {
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {"url": "data:audio/m4a;base64,..."}
        }
      ]
    }
  ]
}
```

新写法：

```bash
curl -sS "$QWEN3_ASR_SERVER_URL/v1/audio/transcriptions" \
  -F 'file=@test.m4a' \
  -F "model=$QWEN3_ASR_MODEL" \
  -F 'hotwords=["邓郁松"]'
```

迁移检查：

- 删除 `/v1/chat/completions` ASR 调用。
- 删除音频 Base64/data URL 编码。
- 删除客户端 `Vocabulary:` system message 拼接。
- 使用 multipart `file`、独立 `prompt` 和 JSON 字符串 `hotwords`。
- 业务配置只保留一个 `ASR_TYPE=qwen3`。
- 仅通过 URL 和 model 在 6001/6002 间切换。

## 11. 接入验收清单

接入方发布前至少验证：

- `/health/ready` 返回 200 和 `status=ready`。
- HTTP 无上下文可以识别。
- HTTP 带热词可以识别人名或专有词。
- HTTP 同时传 prompt 和热词可以识别，且响应解析不依赖可选字段。
- SSE sequence 严格递增，`segment_final` 时间范围合法且无重叠。
- SSE 收到 `final` 前不落库。
- SSE 断流、缺 final、404 和 503 的处理符合降级规则。
- WS 收到 `ready` 后发送原始 PCM，最终收到 `final`。
- WS start 帧包含同一份 prompt、hotwords 和可选 API Key。
- 日志中没有 prompt、热词、识别正文或 API Key。
- 将 URL/model 切到 6002 后无需修改任何业务代码即可重复通过以上测试。

## 12. 当前发布记录

v1.2.2 的实际镜像 ID、构建时间、首帧延迟和多话轮/四并发验证结果见
[DEPLOYMENT_v1.2.2.md](DEPLOYMENT_v1.2.2.md)。v1.2.1 的 preview 稳定性基线见
[DEPLOYMENT_v1.2.1.md](DEPLOYMENT_v1.2.1.md)，v1.2.0 的已部署基线仍保留在
[DEPLOYMENT_v1.2.0.md](DEPLOYMENT_v1.2.0.md)。

判断运行版本时以 Image ID 为准，不要只看可变镜像标签。
