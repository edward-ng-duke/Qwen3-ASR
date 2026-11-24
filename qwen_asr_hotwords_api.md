# Qwen3-ASR 热词版 API 文档（Agent 使用）

本文档描述 Qwen3-ASR Transformers 热词服务 v1.2.1 交付镜像，面向需要自动调用、
生成客户端代码、排查连接或集成业务流程的 Agent。

文档对应镜像：`mygpt-qwen3-asr-hotword:v1.2.1`
交付 Image ID：`sha256:eebfe20632c830b7f2718370afba603ca8ee14176a0364ef0657b9f27046a57c`
模型：`Qwen/Qwen3-ASR-1.7B-hf`
协议版本：服务实现 `1.2.1`

> **判断版本请以 Image ID 为准，不要只看 tag。** 本次没有恢复或替换真实 6002；共享代码与
> 契约测试已经通过，真实 Transformers 现场验收仍待服务恢复。核对部署后的实际版本：
>
> ```bash
> docker inspect --format '{{.Image}}' qwen3-asr-hotword
> ```

## 1. 快速结论

当前局域网访问地址：

```text
http://10.0.0.93:6002
```

WebSocket 地址：

```text
ws://10.0.0.93:6002/v1/audio/realtime
```

当前服务器网络信息：

| 项目 | 当前值 |
|---|---|
| Linux 局域网接口 | `wlo1` |
| Linux 局域网 IPv4 | `10.0.0.93/24` |
| 默认网关 | `10.0.0.1` |
| 局域网网段 | `10.0.0.0/24` |
| Docker 发布端口 | `0.0.0.0:6002`、`[::]:6002` |
| 容器内部端口 | `8000` |
| 当前 API Key | 未设置，开放访问 |

`10.0.0.93` 是本文档生成时的现场值。如果服务器使用 DHCP，该地址以后可能变化。Agent 在连接失败时
应先在服务器上重新执行：

```bash
ip -4 addr show dev wlo1
ip route show default
```

不要把 Docker 网桥地址（例如 `172.x.x.x`、`192.168.32.1`）当作局域网服务地址。

## 2. Agent 应该选择哪个接口

| 需求 | 推荐接口 | 说明 |
|---|---|---|
| 上传一个音频并等待完整结果 | `POST /v1/audio/transcriptions` | 最简单、最稳定 |
| 上传完整文件后逐步消费生成结果 | `POST /v1/audio/transcriptions/stream` | SSE；上传完成后才开始返回事件 |
| 麦克风、电话或实时 PCM | `WS /v1/audio/realtime` | 真正双向传输，但模型内部会重识别窗口 |
| 判断进程是否存活 | `GET /health/live` | 不代表模型已加载完成 |
| 判断能否接收识别请求 | `GET /health/ready` | 调用识别接口前应检查此接口 |
| 查询当前模型和运行后端 | `GET /v1/models` | 可用于能力发现 |
| 抓取监控指标 | `GET /metrics` | Prometheus 文本格式 |
| 人工浏览器测试 | `GET /` | 完全离线测试页，不是 Agent API |

如果调用方不需要边生成边展示，优先使用普通 HTTP 接口。不要为了“流式”而使用 SSE：SSE 文件接口
仍然需要先上传完整文件。只有 WebSocket 能在音频仍在产生时持续发送音频。

## 3. 局域网访问方法

### 3.1 最小连通性检查

在与服务器处于同一局域网的电脑上执行：

```bash
curl --connect-timeout 3 http://10.0.0.93:6002/health/live
curl --connect-timeout 3 http://10.0.0.93:6002/health/ready
```

正常返回：

```json
{"status":"live"}
```

```json
{
  "status": "ready",
  "id": "Qwen/Qwen3-ASR-1.7B-hf",
  "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
  "transformers_revision": "c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d",
  "device": "NVIDIA GeForce RTX 4090",
  "dtype": "bfloat16",
  "attention_backend": "flash_attention_2",
  "torch_compile": false,
  "torch_compile_mode": "default"
}
```

现场已经从服务器通过 `10.0.0.93:6002` 验证健康和模型接口可以访问。局域网其他设备能否访问还取决于
无线路由器是否开启客户端隔离，以及 Linux 防火墙是否允许 6002/TCP。

### 3.2 局域网客户端无法访问时

按以下顺序排查。

1. 在服务器上确认服务健康：

   ```bash
   curl -fsS http://127.0.0.1:6002/health/ready
   docker ps --filter name=qwen3-asr-hotword
   ```

2. 确认服务器当前局域网 IP：

   ```bash
   ip -br -4 addr show dev wlo1
   ```

3. 确认端口不是只监听回环地址：

   ```bash
   ss -ltnp '( sport = :6002 )'
   ```

   应看到 `0.0.0.0:6002`，而不是仅有 `127.0.0.1:6002`。

4. 确认客户端的地址通常也在 `10.0.0.0/24`，并检查路由器的“AP 隔离”“客户端隔离”或
   “访客网络隔离”。Ping 被禁用不等于 TCP 服务不可达，应以 `curl` 结果为准。

5. 检查 Linux 防火墙。本文档生成时，读取 UFW 状态需要 root 权限，因此没有擅自修改防火墙。

   ```bash
   sudo ufw status verbose
   sudo firewall-cmd --state
   ```

   如果确实需要为 UFW 添加规则，建议只允许当前局域网，不要直接向所有来源开放：

   ```bash
   sudo ufw allow from 10.0.0.0/24 to any port 6002 proto tcp
   ```

   使用 firewalld 时可按现场 zone 配置 rich rule；执行前应先确认 `wlo1` 所属 zone。

### 3.3 地址稳定性建议

给长期运行的 Agent 配置以下任一种方式：

- 在路由器中为服务器网卡设置 DHCP 静态租约，使 `10.0.0.93` 不变。
- 使用内部 DNS 名称，并让 Agent 从配置项读取 Base URL。
- 不要在 Agent 源码的多个位置硬编码 IP；统一使用 `QWEN_ASR_BASE_URL`。

推荐环境变量：

```bash
export QWEN_ASR_BASE_URL='http://10.0.0.93:6002'
export QWEN_ASR_WS_URL='ws://10.0.0.93:6002/v1/audio/realtime'
```

## 4. 通用约定

### 4.1 模型名

唯一允许的模型名：

```text
Qwen/Qwen3-ASR-1.7B-hf
```

HTTP 请求可以省略 `model`。如果提供，必须与上述值完全一致，否则返回 HTTP 422。

### 4.2 语言

`language` 可以是：

- `auto`：让模型自动识别语言，推荐默认值。
- 下表中的 ISO 代码。
- 对应的英文模型语言名，不区分大小写，例如 `Chinese`、`English`、`Cantonese`。

| 代码 | 模型语言名 | 代码 | 模型语言名 |
|---|---|---|---|
| `ar` | Arabic | `yue` | Cantonese |
| `zh` | Chinese | `cs` | Czech |
| `da` | Danish | `nl` | Dutch |
| `en` | English | `fil` | Filipino |
| `fi` | Finnish | `fr` | French |
| `de` | German | `el` | Greek |
| `hi` | Hindi | `hu` | Hungarian |
| `id` | Indonesian | `it` | Italian |
| `ja` | Japanese | `ko` | Korean |
| `mk` | Macedonian | `ms` | Malay |
| `fa` | Persian | `pl` | Polish |
| `pt` | Portuguese | `ro` | Romanian |
| `ru` | Russian | `es` | Spanish |
| `sv` | Swedish | `th` | Thai |
| `tr` | Turkish | `vi` | Vietnamese |

普通话使用 `zh`，粤语使用 `yue`。不支持任意 ISO 代码，未知值返回 422。

### 4.3 prompt 和 hotwords

`prompt` 是自由文本上下文；`hotwords` 是热词数组。服务端使用以下固定规则：

1. 去除每个热词首尾空白。
2. 保持原始顺序。
3. 按完全相同的字符串精确去重，区分大小写。
4. 忽略清理后为空的热词。
5. 最多 100 个热词。
6. 每个热词最多 64 个 Unicode 字符。
7. prompt 与热词合并后最多 4096 个 Unicode 字符。

最终送入 Processor 的内容为：

```text
{prompt}
Vocabulary: 热词1, 热词2, 热词3
```

如果没有 prompt，则只有 `Vocabulary: ...`；如果没有热词，则只有 prompt；两者都没有时不传 system
prompt。热词属于 contextual bias，不是百分之百强制替换。

重要差异：

- HTTP multipart 中的 `hotwords` 是“JSON 数组序列化后的字符串”，例如
  `["Qwen3-ASR","张三"]`。
- WebSocket 初始化 JSON 中的 `hotwords` 是真正的 JSON 字符串数组。
- HTTP 不接受重复的 `hotwords=词1&hotwords=词2` 表单字段。

**安全门控与复核（HTTP / SSE / WS 三条路径通用）**

模型本身存在上下文幻觉：在“有声音但无有效语音”的音频上（环境噪声、纯音、咳嗽、清嗓等），
它可能把 system prompt——也就是 `prompt` 正文和 `Vocabulary:` 热词表——当成待转写内容输出。
服务层为此建立了两道防线，**只要请求带了 `prompt` 或 `hotwords` 就会启用**：

1. **前置 VAD 门控**：判定无有效语音的区间不做带上下文推理。
2. **后置结果校验**：疑似回显上下文的结果被丢弃，并对同一音频改用**无上下文**重跑一次，
   返回重跑结果。

由此产生三条调用方必须知道的后果：

- **同一段音频，带 `hotwords`/`prompt` 与不带，可能得到不同结果。** 带上下文的请求在被判定
  可疑时会自动降级为无上下文结果，这是预期行为，不是故障。
- **`language=en` 且 `prompt` 含非 ASCII 字符（例如英语识别配中文 prompt）时会额外触发一次
  一致性复核**，该分段的推理次数翻倍，并且 **SSE/WS 在该会话不再产生 `partial` 事件**
  （详见 6.3 与 7.4）。注意音频实际语种不影响判定，**误把中文音频标成 `language=en` 同样会
  触发**。
- **仅传 `hotwords`、不传 `prompt` 不会触发跨语言复核**，也不增加推理次数。若无明确的
  prompt 收益需求，推荐只用 `hotwords`：可同时避开复核开销与 `partial` 缺失。

服务不会把 `prompt`/`hotwords` 回填进空结果；被门控拦下的分段只会得到无上下文结果或空文本。

### 4.4 音频约束

文件接口：

- 最大上传大小：8 GiB。
- 最大解码后音频时长：4 小时。
- 由 FFmpeg 解码，因此可接受 FFmpeg 支持的常见 WAV、MP3、M4A、FLAC、OGG 等格式。
- 服务端统一转换为 16 kHz、单声道、float32 PCM。
- 长文件按最长约 30 秒切分，优先使用 VAD 边界，并带短重叠去重。
- 上传和解码使用临时磁盘，不把 4 小时文件整体读入内存。
- 带有 prompt 或 hotwords、且不超过 30 秒的音频会先经过主 Silero VAD。主 VAD 未确认到语音时，
  服务不会直接返回空文本，而是自动以 `prompt=None` 对整段做一次无上下文识别。
- 带上下文的模型输出如果原样包含 prompt，或呈现为多热词列表回显，会被丢弃；同一音频分段随后
  以 `prompt=None` 自动重试一次。每个分段最多回退一次，不递归校验或重试。
- 当最终识别语言为英语，且请求 prompt 含有非 ASCII 字母（例如中文、日文或阿拉伯文）时，服务会
  对同一分段额外做一次 `prompt=None` 复核。服务比较两次规范化后的文本；一致度低于默认阈值
  0.45 时，判定带 prompt 结果可能是 prompt 的翻译或语义改写回显，丢弃它并采用无上下文结果。
  这项检查不依赖固定短语或预置译文，未知 prompt 也适用。
- 被判定为上下文回显的结果不会返回；回退推理本身不接收 prompt/hotwords。这里不表示最终文本绝不
  出现某个热词：真实语音可能本来就包含它，模型也可能受 contextual bias 影响只输出单个热词，而单个
  热词不会被当作整表泄漏拦截。无上下文模型还可能产生自身的普通幻觉或 `嗯。`、`Hmm.`、`Ahem.` 等
  语气词。Agent 应按普通 ASR 输出处理，不能自行恢复被拦截的上下文结果。
- 英语跨文字 prompt 复核会增加一次模型推理。仅传 hotwords、不传 prompt 的请求不触发该复核，
  其他语言或普通安全路径也不增加推理。若复核排队或推理失败，服务返回现有错误，绝不返回未经
  复核的带 prompt 结果。
- 长文件仍先按 VAD 和最长约 30 秒切段。整段 VAD 失败时不会把最长 4 小时文件直接送入模型；仅已生成
  的单个分段在命中上下文回显时按段回退。

实时接口：

- 只接受原始 `pcm_s16le` 或 `pcm_f32le`，不接受 WAV/MP3 文件头。
- 输入采样率范围 8000–96000 Hz。
- 声道数只能是 1 或 2；双声道会降混为单声道。
- 服务端使用有状态重采样转换为 16 kHz。
- 单个二进制 WebSocket 帧最大 1 MiB。
- 单会话最长 1 小时，空闲超时 5 分钟。
- 最多同时 4 个实时会话。

### 4.5 鉴权

当前服务的 `ASR_API_KEY` 为空，因此目前无需鉴权。Agent 仍应支持可选 API Key，避免服务以后启用
鉴权时修改协议。

启用 API Key 后：

- `/v1/*` 和 `/metrics` 受保护。
- `/health/live`、`/health/ready` 和测试页保持开放。
- HTTP/SSE 使用 `Authorization: Bearer <key>`。
- WebSocket 可以使用 Authorization Header，或在初始化 JSON 的 `api_key` 字段传入。
- 浏览器原生 WebSocket API 不能自由设置 Authorization Header，应使用初始化 JSON。

HTTP 示例：

```bash
curl -H 'Authorization: Bearer replace-me' \
  http://10.0.0.93:6002/v1/models
```

局域网接口当前没有 TLS。不要把 6002 直接映射到公网；跨不可信网络时应使用反向代理提供 HTTPS、
WSS、访问控制和限流。

## 5. 普通文件转写

### 5.1 请求

```http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
Authorization: Bearer <optional-api-key>
```

字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `file` | 文件 | 是 | — | 待识别音频 |
| `model` | 文本 | 否 | 当前模型 | 提供时必须是唯一模型名 |
| `language` | 文本 | 否 | `auto` | ISO 代码、模型语言名或 `auto` |
| `prompt` | 文本 | 否 | 空 | 自由文本上下文 |
| `hotwords` | 文本 | 否 | `[]` | JSON 字符串数组，不是多个表单字段 |

### 5.2 curl 示例

```bash
curl -sS http://10.0.0.93:6002/v1/audio/transcriptions \
  -F 'file=@/absolute/path/meeting.wav' \
  -F 'model=Qwen/Qwen3-ASR-1.7B-hf' \
  -F 'language=zh' \
  -F 'prompt=这是一场关于医疗废物处理的会议。' \
  -F 'hotwords=["Qwen3-ASR","日产日清","张三"]'
```

文件名包含空格时仍应使用完整路径，并正确处理 shell 引号。

### 5.3 Python/httpx 示例

```python
from pathlib import Path
import json
import httpx

base_url = "http://10.0.0.93:6002"
audio_path = Path("/absolute/path/meeting.wav")
api_key = ""  # 当前为空；启用鉴权后填写

headers = {}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

with audio_path.open("rb") as audio:
    response = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        headers=headers,
        files={"file": (audio_path.name, audio, "application/octet-stream")},
        data={
            "model": "Qwen/Qwen3-ASR-1.7B-hf",
            "language": "zh",
            "prompt": "这是一场关于医疗废物处理的会议。",
            "hotwords": json.dumps(
                ["Qwen3-ASR", "日产日清", "张三"], ensure_ascii=False
            ),
        },
        timeout=300,
    )

response.raise_for_status()
result = response.json()
print(result["text"])
```

对于超长文件，客户端超时应根据音频长度调整，不要固定为几秒。请求是可安全重试的只读推理，但重试
会产生重复 GPU 工作；只有在连接错误、429 或 503 时进行带抖动的指数退避重试。

### 5.4 成功响应

```json
{
  "language": "Chinese",
  "text": "山东持续提升医疗废物处置能力，保障涉医废物日产日清，守牢安全底线。",
  "raw_text": "山东持续提升医疗废物处置能力，保障涉医废物日产日清，守牢安全底线。<|im_end|>",
  "duration": 7.009,
  "processing_ms": 376.1,
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "request_id": "33d11152906d4c229648cc73d69dc969"
}
```

字段语义：

| 字段 | 说明 |
|---|---|
| `language` | 检测到或强制指定的模型语言名；自动检测失败时可能为 `null` |
| `text` | 清理后的最终转写，业务应优先使用此字段 |
| `raw_text` | 模型原始解码结果，可能包含特殊 token；不要直接展示给终端用户 |
| `duration` | FFmpeg 解码后的音频时长，单位秒 |
| `processing_ms` | 服务端处理耗时，单位毫秒；不等于客户端完整网络 RTT |
| `model` | 实际使用的模型 ID |
| `request_id` | 本次推理请求 ID，可用于问题定位 |

响应不包含词级或句级时间戳，也不支持 Forced Aligner。

**`text` 可能为空字符串，此时 HTTP 仍是 200。** 纯数字静音、以及被安全门控判定为无有效语音
的音频都会返回 `""`。调用方不能假设 `text` 非空，也不应把空结果当作错误重试。

**无有效语音的音频不一定返回空，更常见的是短幻觉文本。** 实测环境噪声、纯音、和弦会得到
`"嗯。"`，咳嗽/清嗓会得到 `"Ahem."`、`"Cough."`、`"咳咳咳。"`。这是模型固有行为，与是否传
`hotwords`/`prompt` 无关（不传时同样出现），安全门控只负责阻止上下文回显，不负责消除这类
幻觉。**若这些短语气词会污染下游（会议纪要、待办抽取等），调用方需自行过滤**，例如按
最小字数、结合 `duration` 与文本长度的比值，或维护一份语气词停用表。

**响应中没有字段标识本次请求是否触发了门控、降级或复核。** 三种情况下的响应结构与正常请求
完全相同，`processing_ms` 也不足以区分。因此调用方无法仅凭响应判断“这段确实没人说话”还是
“被门控拦下了”；有此类监控需求时，需要在服务端指标侧观察，或对同一音频再发一次不带
`hotwords`/`prompt` 的请求做对照。

## 6. SSE 文件转写

### 6.1 请求语义

```http
POST /v1/audio/transcriptions/stream
Content-Type: multipart/form-data
Accept: text/event-stream
```

multipart 字段与普通文件转写完全相同。服务会先把文件完整上传并解码到临时磁盘，然后才返回 SSE
事件。因此该接口适合逐步展示模型生成文本，但不能降低文件上传阶段的等待时间。

如果主 VAD 拒绝短音频，无上下文回退会正常产生 `partial → segment_final`。如果带上下文推理直到
final 才被判断为回显，服务不会再发送第二套 partial，只发送采用无上下文回退结果的一个
`segment_final`，避免 UI 预览反复跳变。`processing_ms` 包含首次推理和回退的全部耗时。

对于“英语识别 + 非 ASCII prompt”，带 prompt 的 partial 在安全复核完成前全部抑制，可信文本从
`segment_final` 开始发送。因此 Agent 不能假设每个 SSE 请求一定存在 `partial`。

### 6.2 curl 示例

```bash
curl -N --no-buffer \
  http://10.0.0.93:6002/v1/audio/transcriptions/stream \
  -F 'file=@/absolute/path/meeting.wav' \
  -F 'language=zh' \
  -F 'prompt=公司技术会议' \
  -F 'hotwords=["Qwen3-ASR","Transformers"]'
```

### 6.3 事件顺序

成功时固定为：

```text
started
partial（零个或多个）
segment_final（一个或多个）
final
```

失败可能得到：

```text
started
...
error
```

SSE 已经开始响应后发生的推理错误通常仍然具有 HTTP 200 状态，因此 Agent 不能只检查 HTTP 状态；
必须持续解析到 `final` 或 `error`。

**`partial` 真的可能一个都没有。** 当 `language=en` 且 `prompt` 含非 ASCII 字符时（见 4.3
的安全复核），本次请求的 `partial` 会被全部抑制，事件序列退化为
`started → segment_final → final`。实测同一段 4.8 秒英文音频：仅 `hotwords` 时 8 个
`partial`，配 ASCII 英文 prompt 时 8 个，配中文 prompt 时 **0 个**；7.7 秒中文音频从 28 个
降到 0 个。`final` 内容不受影响，仍然正确。

因此**依赖 SSE 做渐进展示的调用方，不要在该组合下使用**：要么改用 ASCII prompt，要么只传
`hotwords`。同时解析逻辑必须容忍零个 `partial`，不能把“没收到 partial”当成超时或故障。

每个事件格式：

```text
event: partial
data: {"type":"partial",...}

```

事件之间以空行分隔，`data` 是单行 UTF-8 JSON。

### 6.4 SSE 事件字段

所有事件共享以下结构，未使用的文本字段为空字符串：

```json
{
  "type": "partial",
  "language": "Chinese",
  "confirmed": "",
  "delta": "持续",
  "preview": "山东持续",
  "text": "山东持续",
  "raw_text": "山东持续",
  "final": false,
  "sequence": 3,
  "duration": 7.009,
  "start_ms": null,
  "end_ms": null,
  "processing_ms": 44.3,
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "request_id": "e011fd45ac9c4b099e34ab2261ea008f",
  "error": null
}
```

| 字段 | Agent 处理规则 |
|---|---|
| `type` | `started`、`partial`、`segment_final`、`final` 或 `error` |
| `confirmed` | 已完成片段的累计文本 |
| `delta` | 相对于上次预览或确认结果新增的内容；必要时可为空 |
| `preview` | 当前片段尚未最终确认的预览 |
| `text` | 当前最完整的可展示文本；UI 最简单的做法是直接替换为此字段 |
| `raw_text` | 当前模型原始文本，可能含特殊 token |
| `final` | 只有整个请求的 `final`/`error` 终态应视为结束 |
| `sequence` | 从 1 开始递增；Agent 可用它丢弃重复或乱序事件 |
| `start_ms` / `end_ms` | 仅 `segment_final` 为整数毫秒；表示原始解码音频中本次提交的非重叠区间，其他事件为 `null` |
| `processing_ms` | partial/segment 多为该次模型生成耗时；final 为流式推理阶段总耗时 |
| `error` | 正常为 `null`；错误时是 `{"code":"...","message":"..."}` |

`segment_final` 的 `final` 字段仍为 `false`，因为它只表示一个分段完成。只有 `type=final` 才表示整个
文件完成。

`segment_final.start_ms/end_ms` 来自 VAD/推理分段的样本范围，不是词级强制对齐时间戳。长音频推理块
可能有 2 秒重叠，服务会把后一个 `start_ms` 裁到前一个 `end_ms`，保证所有提交区间单调且不重叠。
业务端如按事件建立 segment，应保存该事件的 `delta`，不要把累计 `confirmed` 重复保存。

### 6.5 Python SSE 解析示例

```python
from pathlib import Path
import json
import httpx

base_url = "http://10.0.0.93:6002"
audio_path = Path("/absolute/path/meeting.wav")

timeout = httpx.Timeout(connect=10, read=None, write=None, pool=10)
with httpx.Client(timeout=timeout) as client, audio_path.open("rb") as audio:
    with client.stream(
        "POST",
        f"{base_url}/v1/audio/transcriptions/stream",
        files={"file": (audio_path.name, audio, "application/octet-stream")},
        data={
            "language": "zh",
            "hotwords": json.dumps(["Qwen3-ASR", "张三"], ensure_ascii=False),
        },
    ) as response:
        response.raise_for_status()
        event_name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if event_name == "error":
                    raise RuntimeError(payload["error"])
                if event_name in {"partial", "segment_final", "final"}:
                    print(payload["text"])
                if event_name == "final":
                    result = payload
```

客户端取消或断开 SSE 后，服务会清理对应临时文件。Agent 不应在读到第一个 partial 后主动断开，除非
确实要取消任务。

## 7. WebSocket 实时转写

### 7.1 连接和初始化

连接地址：

```text
ws://10.0.0.93:6002/v1/audio/realtime
```

连接成功后，客户端必须在 10 秒内发送第一帧文本 JSON：

```json
{
  "type": "start",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "language": "zh",
  "prompt": "这是一场技术会议",
  "hotwords": ["Qwen3-ASR", "Transformers", "张三"],
  "api_key": null
}
```

字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `type` | string | 否 | `start` | 只能是 `start` |
| `encoding` | string | 是 | — | `pcm_s16le` 或 `pcm_f32le` |
| `sample_rate` | integer | 是 | — | 8000–96000 |
| `channels` | integer | 否 | 1 | 只能是 1 或 2 |
| `language` | string | 否 | `auto` | 语言代码、模型语言名或 `auto` |
| `prompt` | string/null | 否 | null | 会话上下文 |
| `hotwords` | string[] | 否 | `[]` | 真正的 JSON 数组 |
| `api_key` | string/null | 否 | null | 浏览器或不便设置 Header 的客户端使用 |

服务返回 `ready` 后才能开始发送音频。初始化后的 language、prompt、hotwords 不允许修改；需要修改时
关闭连接并建立新会话。

### 7.2 音频二进制帧

后续音频必须使用 WebSocket 二进制帧：

- `pcm_s16le`：小端有符号 16 位整数，取值 -32768 到 32767。
- `pcm_f32le`：小端 IEEE float32，推荐范围 -1.0 到 1.0；服务会裁剪异常值。
- 双声道必须按帧交错排列：`L0,R0,L1,R1,...`。
- 不要发送 WAV RIFF Header、MP3 数据或 Base64 文本。
- 每个二进制帧必须结束在完整采样帧边界。
- 推荐每帧 20–100 ms，按真实时间节奏发送并处理网络背压。

16 kHz、单声道、`pcm_s16le` 的 100 ms 帧大小：

```text
16000 samples/s × 0.1s × 2 bytes = 3200 bytes
```

### 7.3 控制帧

控制消息使用文本 JSON。

探活：

```json
{"type":"ping"}
```

服务响应：

```json
{"type":"pong", "sequence": 1, ...}
```

结束输入并要求最终结果：

```json
{"type":"finish"}
```

发送 `finish` 后不要再发送音频。继续读取，直到收到 `type=final`，然后服务以正常状态关闭连接。

### 7.4 实时事件

服务端事件类型：

| 类型 | 含义 |
|---|---|
| `ready` | 初始化成功，可以发送音频 |
| `partial` | 当前工作窗口的非最终识别 |
| `segment_final` | VAD 句尾或 10 秒窗口已提交 |
| `final` | 整个会话最终结果 |
| `pong` | ping 响应 |
| `error` | 结构化错误，通常随后关闭连接 |

实时事件示例：

```json
{
  "type": "partial",
  "language": "Chinese",
  "confirmed": "山东持续提升",
  "delta": "医疗废物",
  "preview": "医疗废物",
  "text": "山东持续提升医疗废物",
  "raw_text": "山东持续提升医疗废物",
  "final": false,
  "sequence": 4,
  "duration": 4.0,
  "processing_ms": 218.5,
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "request_id": "...",
  "error": null
}
```

Agent 的推荐状态机：

1. `confirmed` 是不可回滚部分，可以持久化或发送给下游。
2. `preview` 是经过稳定处理的临时尾部，只用于显示，绝不能持久化。
3. 末尾两个显示单位内的修订可立即出现；更大的修订连续两轮兼容后才显示。中文按字、英文/数字按词计数，标点和空白不单独计数。
4. 如果新假设改写 confirmed 内部，服务保留上一版可见 preview，不会用瞬时空字符串清空 UI；旧服务客户端也应对“confirmed 不变、preview 突然为空”做保留兜底。
5. 相同 partial 可能重复发送，UI 应跳过相同内容的 DOM 重绘，只对真正变化的后缀做动画。
6. 如果做增量下游消费，只消费新增的 confirmed，不要把 preview 当最终事实。
7. `segment_final` 后可提交该段，但整个会话仍未结束。
8. 只有 `type=final && final=true` 才是完整终态。

模型不是原生流式音频编码器。服务约每秒重新识别当前工作窗口，使用 10 秒窗口和 2 秒重叠；连续两次
结果的公共前缀用于生成 confirmed，并保留两个显示单位的回滚余量。`segment_final`/`final` 是权威收口，
可能一次性提交未曾作为 preview 展示的大幅修订。

带 prompt/hotwords 的会话同时运行两个接收相同 PCM 的 VAD：

- 主 VAD 默认阈值 0.5；命中的区间带上下文推理。
- 候选 VAD 默认阈值 0.1；仅候选命中而未被主 VAD 覆盖的差集区间做无上下文推理。
- 主 VAD 句段结束时会重置与该主句重叠、但仍未闭合的候选状态，候选区间不会跨过可信主句后继续
  吞入尾部静音。主/候选重叠区间始终以主 VAD 为准。
- 候选独有段短于 250 ms 时不推理；归一化 float PCM 的 RMS 低于默认值 0.0015 时也不推理。
  `ASR_CANDIDATE_MIN_RMS` 可调整该门槛，允许范围为 `[0, 1)`，设为 0 表示禁用能量门槛。
- 候选段最长 10 秒；调度窗口保留 2 秒观察重叠，但已提交范围会在推理前裁掉，避免同一音频区间重复推理或重复提交。
- `finish` 会刷新尚未结束的主/候选区间，短会话末尾不会因等不到静音事件而丢失。
- 主区间的最终上下文结果若被安全过滤，会对原区间无上下文重试一次；可疑 partial 只抑制，回退集中在句段结束时。
- “英语识别 + 非 ASCII prompt”的主区间也会抑制未复核 partial，并在句段结束时用无 prompt 结果
  做一致度复核；`segment_final` 只发送最终选中的可信结果。

无上下文会话只运行主 VAD。两个 VAD 的最短语音均按 250 ms、静音 600 ms 处理。所有回退结果仍进入
现有稳定器，`confirmed` 保持单调，`segment_final` 不会把整段再次重复拼接。

极短残段偶尔会让模型重放上一段的累计开头。服务把“与现有 confirmed 共享至少三个起始显示单位”的
结果视为累计修订，只从可信的末尾锚点后提取新增内容；找不到至少两个显示单位的精确锚点，也无法用
最近兼容假设完成可信对齐时，保留原 confirmed，不会退化成 `confirmed + 完整重放文本`。真正独立、
不共享累计开头的下一段仍按普通非重叠 segment 追加。

**“抑制未复核 partial”在实测中表现为该会话完全没有 `partial` 事件**，而不是只少发几个。
`language=en` 且 `prompt` 含非 ASCII 字符时，事件序列退化为只有 `segment_final` 和 `final`：

| 会话配置（4.8 秒英文音频） | `partial` 数 | 首个 `partial` |
|---|---:|---|
| `en` + 仅 hotwords | 4 | 约 1.1 秒 |
| `en` + ASCII 英文 prompt | 4 | 约 1.1 秒 |
| **`en` + 中文 prompt（触发复核）** | **0** | **无** |
| `zh` + 中文 prompt（9.9 秒中文音频，不触发） | 9 | 约 1.4 秒 |
| **中文音频误标 `en` + 中文 prompt** | **0** | **无** |

`final` 与 `confirmed` 的正确性不受影响，这是**实时性退化而非正确性问题**：整段话要等到
`segment_final` 才第一次出现。**需要实时字幕的场景，请在 `language=en` 下改用 ASCII prompt
或只传 `hotwords`。** 客户端的接收循环也必须能在零个 `partial` 的情况下正常走到 `final`，
不要用“多久没收到 partial”作为超时判据。

### 7.5 Python WebSocket 完整示例

下面把任意音频先用 FFmpeg 转成 16 kHz 单声道 PCM，再模拟真实时间发送。生产麦克风客户端可以直接
用采集到的 PCM 替换 `pcm`。

```python
import asyncio
import json
import subprocess
import time

import websockets

WS_URL = "ws://10.0.0.93:6002/v1/audio/realtime"
AUDIO_FILE = "/absolute/path/meeting.wav"


def decode_pcm_s16le(path: str) -> bytes:
    return subprocess.check_output(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "pipe:1",
        ]
    )


async def main() -> None:
    pcm = decode_pcm_s16le(AUDIO_FILE)
    frame_bytes = 3200  # 16kHz mono s16le 的 100ms

    async with websockets.connect(WS_URL, max_size=4 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "prompt": "这是一场技术会议",
                    "hotwords": ["Qwen3-ASR", "Transformers", "张三"],
                    "api_key": None,
                },
                ensure_ascii=False,
            )
        )

        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"WebSocket 初始化失败: {ready}")

        async def receive_events() -> dict:
            last_confirmed = ""
            async for message in ws:
                event = json.loads(message)
                if event["type"] == "error":
                    raise RuntimeError(event["error"])
                if event.get("confirmed", "").startswith(last_confirmed):
                    last_confirmed = event.get("confirmed") or last_confirmed
                if event["type"] in {"partial", "segment_final"}:
                    print("实时:", event["text"])
                if event["type"] == "final":
                    return event
            raise RuntimeError("连接在 final 之前关闭")

        receiver = asyncio.create_task(receive_events())
        started = time.monotonic()
        for offset in range(0, len(pcm), frame_bytes):
            await ws.send(pcm[offset : offset + frame_bytes])
            sent_audio_seconds = min(offset + frame_bytes, len(pcm)) / 32000
            delay = started + sent_audio_seconds - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        await ws.send(json.dumps({"type": "finish"}))
        final = await asyncio.wait_for(receiver, timeout=300)
        print("最终:", final["text"])


asyncio.run(main())
```

## 8. 模型、健康和指标接口

服务没有开启 `/docs` 或 `/redoc`；本文档是当前 API 契约。`GET /` 返回供人工测试文件、SSE 和麦克风
输入的离线网页，不建议 Agent 解析网页来发现能力。

### 8.1 `GET /health/live`

不鉴权。进程启动后立即返回 200：

```json
{"status":"live"}
```

此接口成功不代表模型已加载。

### 8.2 `GET /health/ready`

不鉴权。模型加载和预热完成后返回 200。加载中或加载失败返回 503：

```json
{
  "status": "loading",
  "detail": null
}
```

Agent 启动任务前应以 ready 为准。推荐 2–5 秒间隔轮询，最长允许约 10 分钟首次启动时间，不要高频
毫秒级轮询。

### 8.3 `GET /v1/models`

启用 API Key 后需要鉴权：

```bash
curl -sS http://10.0.0.93:6002/v1/models
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-ASR-1.7B-hf",
      "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
      "transformers_revision": "c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d",
      "device": "NVIDIA GeForce RTX 4090",
      "dtype": "bfloat16",
      "attention_backend": "flash_attention_2",
      "torch_compile": false,
      "torch_compile_mode": "default"
    }
  ]
}
```

### 8.4 `GET /metrics`

Prometheus 文本格式。启用 API Key 后需要鉴权。主要指标包括：

- `asr_requests_total{mode,status}`
- `asr_request_duration_seconds{mode}`
- `asr_realtime_factor{mode}`
- `asr_audio_seconds_total{mode}`
- `asr_queue_depth{priority}`
- `asr_active_realtime_sessions`
- `asr_gpu_memory_bytes{kind}`
- `asr_ready`
- `asr_compile_enabled`
- `asr_context_leaks_suppressed_total{mode}`
- `asr_context_fallback_total{mode,reason,outcome}`
- `asr_context_fallback_duration_seconds{mode,reason}`
- `asr_prompt_verification_total{mode,outcome}`
- `asr_prompt_verification_duration_seconds{mode}`
- `asr_realtime_candidate_segments_total{outcome}`

`mode` 可能是 `offline`、`sse` 或 `realtime`。
`reason` 只有 `vad_reject` 或 `context_echo`；`outcome` 只有 `nonempty`、`empty` 或 `error`。
prompt 复核的 `outcome` 是 `kept_context`、`used_no_context` 或 `error`。
候选 VAD 的 `outcome` 是 `submitted`、`too_short`、`low_energy` 或 `overlap_reset`。
回退指标不包含音频、prompt、热词或转写正文。

## 9. 错误处理

### 9.1 HTTP 错误结构

服务自身校验错误通常为：

```json
{
  "error": {
    "code": "invalid_parameters",
    "message": "hotwords must be a JSON string array"
  }
}
```

常见状态码：

| HTTP 状态 | 常见 code/原因 | Agent 策略 |
|---:|---|---|
| 400 | `invalid_audio`，FFmpeg 无法解码、空音频 | 不自动重试，检查输入 |
| 401 | `unauthorized` | 刷新或修正 API Key |
| 413 | 文件或音频时长超过限制 | 缩短或拆分输入 |
| 422 | `invalid_parameters`、`unsupported_model`，语言/热词错误 | 修正请求，不重试 |
| 429 | `queue_full` | 指数退避并加入随机抖动 |
| 503 | `not_ready` | 轮询 ready 后重试 |
| 500 | 未预期内部错误 | 记录 request ID，有限次数重试 |

FastAPI 在 multipart 缺失等框架级校验失败时可能返回标准 `detail` 数组，而不是上述 `error` 对象；
Agent 应同时兼容两种错误结构。

建议重试参数：

- 仅对连接失败、429、503 和有限的 5xx 重试。
- 初始等待 0.5–1 秒，指数增长，加入 20% 左右抖动。
- 普通请求最多重试 2–3 次。
- 422、413、401 不应盲目重试。

可以发送 `X-Request-ID` Header 用于 HTTP 日志链路追踪。服务会把该值写入响应 Header；JSON 中的
推理 `request_id` 是服务生成的独立 ID，也应一并记录。

### 9.2 WebSocket 错误和关闭码

错误事件：

```json
{
  "type": "error",
  "final": true,
  "error": {
    "code": "invalid_init",
    "message": "..."
  },
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "request_id": "..."
}
```

常见错误码：

| code | 含义 | 常见关闭码 |
|---|---|---:|
| `too_many_sessions` | 已达到 4 路实时上限 | 1013 |
| `not_ready` | 模型未就绪 | 1013 |
| `invalid_init` | 初始化 JSON、语言、热词或格式错误 | 1008 |
| `unauthorized` | API Key 错误 | 4401 |
| `frame_too_large` | 单个二进制帧超过 1 MiB | 1009 |
| `session_too_long` | 超过 1 小时 | 1008 |
| `invalid_control` | 文本控制帧不是 ping/finish | 通常保持连接 |
| `idle_timeout` | 5 分钟未收到数据 | 1008 |
| `invalid_request` | PCM、队列或请求错误 | 1008 |
| `internal_error` | 未预期服务错误 | 1011 |

正常完成关闭码为 1000。网络断线后可以建立新连接，但服务不支持从旧连接的音频位置续传；需要由客户端
决定重发尚未确认的音频。不要把已经消费的 confirmed 再次无条件拼接，以免重复。

## 10. 并发、性能和调用建议

- 单 GPU worker，实时任务优先于文件任务。
- 单路请求立即执行，不主动等待凑 batch。
- 已经同时进入队列、长度桶相近的请求最多动态合批 4 路。
- 最多 4 个活动 WebSocket 会话。
- GPU 队列默认最多 32 个待处理请求，超过后返回 429 或 WebSocket 结构化错误。
- 长文件每个片段完成后重新排队，因此可以让实时任务插队。
- 回退及 prompt 复核沿用原请求的优先级和长度桶。普通安全路径和仅 hotwords 路径不会增加 GPU
  推理次数；VAD 拒绝、字符串回显或英语跨文字 prompt 复核路径对每个音频分段最多增加一次
  无上下文推理。任务队列满或失败时返回现有 HTTP/SSE/WS 错误，绝不恢复已判定或未经复核的结果。

RTX 4090 实测参考：

| 场景 | p50 | p95/更新 p95 |
|---|---:|---:|
| 7 秒文件单路 | 0.366 秒 | 0.381 秒 |
| 30 秒文件单路 | 1.598 秒 | 1.622 秒 |
| 单路实时首结果 | 1.300 秒 | 更新间隔 1.056 秒 |
| 四路实时首结果 | 1.322 秒 | 更新间隔 1.160 秒 |

这些是**服务器本机**参考值，不是 SLA。Agent 设计超时时应保留上传、网络、队列和长音频切分余量。

**跨局域网调用会明显高于上表。** 从同网段另一台机器、上传 48 kHz/s32le 单声道 WAV（未预先
降采样，解码由服务端 FFmpeg 承担）实测：

| 场景（局域网客户端视角，端到端墙钟） | p50 | p95 |
|---|---:|---:|
| 7.7 秒音频 单路 无上下文 | 516 ms | — |
| 7.7 秒音频 单路 仅 hotwords | 528 ms | — |
| 12 路并发 中文 hotwords+prompt（不触发复核） | 1054 ms | 1400 ms |
| 20 路并发 中文 hotwords | 1198 ms | 1776 ms |

客户端若先把音频降采样为 16 kHz 单声道再上传，可省掉一部分传输与解码时间。

**跨语言 prompt 复核会让耗时接近翻倍**，与“每个分段最多增加一次无上下文推理”的设计一致。
同一批 4.8 秒英文音频、12 路并发：

| 配置 | p50 | p95 | 相对仅 hotwords |
|---|---:|---:|---|
| 无上下文 | 750 ms | 1052 ms | — |
| 仅 hotwords | 992 ms | 1337 ms | 基准 |
| hotwords + 中文 prompt（触发复核） | 1890 ms | 2228 ms | **+91%** |

并发下未观察到失败或串话：12 路与 20 路均 100% 成功、无 429，且并发结果与单路基准逐字一致。
若吞吐是硬约束，优先只传 `hotwords`——既不触发复核，也不损失热词命中率。

## 11. 与 OpenAI/vLLM API 的差异

该服务有意不保证与 OpenAI 或原 vLLM ASR 接口完全兼容：

- 路径和 multipart 基本相似，但只接受一个固定模型。
- `hotwords` 是本服务定义的 JSON 字符串数组字段。
- 不支持 `response_format`、`temperature`、采样参数或时间戳粒度。
- 不返回 segments、words 或时间戳。
- SSE 和 WebSocket 事件结构是本服务专用协议。
- WebSocket 输入是原始 PCM，不是封装音频或 Base64。

Agent 不应把 OpenAI SDK 的响应模型直接套在本服务上；建议按本文档定义独立适配器。

## 12. Agent 集成检查清单

在宣告集成完成前，至少检查：

- [ ] Base URL 从配置读取，当前值为 `http://10.0.0.93:6002`。
- [ ] 任务开始前检查 `/health/ready`，而不是只检查 live。
- [ ] HTTP hotwords 使用 `json.dumps(list)` 后放入 multipart 文本字段。
- [ ] WebSocket hotwords 使用真正的 JSON 数组。
- [ ] 普通话使用 `zh`，粤语使用 `yue`，未知语言不自动猜代码。
- [ ] 业务展示使用 `text`，不直接展示 `raw_text`。
- [ ] SSE 一直读取到 `final` 或 `error`，不只依赖 HTTP 200。
- [ ] WebSocket 等待 `ready` 后再发音频，并在 `finish` 后读取最终事件。
- [ ] 实时下游只持久化 confirmed，preview 允许变化。
- [ ] 能接受服务把带上下文请求自动降级为无上下文结果，且不会把 prompt/hotwords 回填到空结果。
- [ ] **默认只传 `hotwords`、不传 `prompt`**；确有 prompt 收益需求时，在 `language=en` 下
      改用 ASCII prompt，以免触发复核（推理翻倍 + `partial` 全部消失）。
- [ ] 处理 `text` 为空字符串的情况（HTTP 200），不把空结果当错误重试。
- [ ] 对无有效语音产生的短幻觉（`"嗯。"`、`"Ahem."`、`"Cough."` 等）有过滤策略，避免污染
      会议纪要、待办抽取等下游。
- [ ] SSE/WS 接收循环能容忍**零个 `partial`**，不用“多久没收到 partial”当超时判据。
- [ ] `language` 传值与音频实际语种一致；误标 `en` 会触发跨语言复核并丢失所有 `partial`。
- [ ] PCM 格式、采样率、声道和字节序与初始化 JSON 一致。
- [ ] 对 429/503 做有限指数退避，对 401/413/422 不盲目重试。
- [ ] 日志记录服务 request ID，但不记录敏感音频、完整 prompt、热词或转写正文。
- [ ] 如果启用 API Key，HTTP Header 和 WebSocket 初始化都已适配。
- [ ] 不把无 TLS 的 6002 端口直接暴露到公网。

## 13. 服务端运维位置

实现目录：

```text
/home/edward/research/Qwen3-ASR/transformers_hotword
```

常用只读检查：

```bash
cd /home/edward/research/Qwen3-ASR/transformers_hotword
docker compose ps
docker logs --tail 100 qwen3-asr-hotword
docker logs --tail 100 qwen3-asr-hotword-gateway
curl -fsS http://127.0.0.1:6002/health/ready
curl -fsS http://127.0.0.1:6002/metrics
```

服务端默认日志只记录请求 ID、模式、时长、耗时、状态和资源指标，不应记录音频、完整 prompt、热词
或转写正文。
