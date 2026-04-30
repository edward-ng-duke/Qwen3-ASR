# Qwen3-ASR API 参考文档

## 目录

- [1. 服务概述](#1-服务概述)
- [2. API 端点列表](#2-api-端点列表)
- [3. 核心接口：音频转录](#3-核心接口音频转录)
- [4. 语言支持](#4-语言支持)
- [5. 热词功能](#5-热词功能)
- [6. 调用示例](#6-调用示例)
- [7. 响应解析](#7-响应解析)
- [8. 伪流式识别](#8-伪流式识别)
- [9. 服务启动参数](#9-服务启动参数)
- [10. Docker 部署](#10-docker-部署)
- [11. 错误处理与常见问题](#11-错误处理与常见问题)

---

## 1. 服务概述

Qwen3-ASR 通过 `qwen-asr-serve` 命令启动，底层基于 **vLLM OpenAI 兼容 API 服务器**。启动后提供标准的 OpenAI Audio API 接口，可直接使用 OpenAI SDK 或 curl 调用。

### 启动命令

```bash
qwen-asr-serve models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.6 \
  --host 0.0.0.0 \
  --port 6001
```

### 工作原理

`qwen-asr-serve` 内部做了两件事：
1. 将自定义的 `Qwen3ASRForConditionalGeneration` 模型注册到 vLLM 的 ModelRegistry
2. 委托给 vLLM 的 `vllm serve` 命令启动 OpenAI 兼容 API 服务

因此，所有 vLLM 的服务端参数均可使用（如 `--max-model-len`、`--max-num-seqs` 等）。

> **注意：** 不能直接使用 `vllm serve` 启动 Qwen3-ASR 模型，必须通过 `qwen-asr-serve` 以确保模型正确注册。

---

## 2. API 端点列表

服务启动后，默认基础 URL 为 `http://<host>:<port>/v1`。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/transcriptions` | POST | **核心接口** — 音频转录 |
| `/v1/models` | GET | 查看当前加载的模型列表 |
| `/health` | GET | 服务健康检查 |

### GET /v1/models

返回当前加载的模型信息。

**请求：**
```bash
curl http://localhost:6001/v1/models
```

**响应示例：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "models/Qwen3-ASR-1.7B",
      "object": "model",
      "created": 1700000000,
      "owned_by": "vllm"
    }
  ]
}
```

### GET /health

服务健康检查，返回 HTTP 200 表示服务正常。

```bash
curl http://localhost:6001/health
```

---

## 3. 核心接口：音频转录

### POST /v1/audio/transcriptions

将音频文件转录为文本，兼容 OpenAI Whisper API 格式。

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | **是** | 模型路径，如 `models/Qwen3-ASR-1.7B` |
| `file` | binary | **是** | 音频文件数据（支持 WAV、MP3、FLAC、OGG、M4A） |
| `language` | string | 否 | 语言代码（ISO 639-1），不填则自动检测 |
| `prompt` | string | 否 | 热词/上下文提示，空格分隔多个词 |

### 音频格式要求

- **支持格式：** WAV、MP3、FLAC、OGG、M4A 等常见格式
- **内部处理：** 统一转为 **16kHz 单声道 float32 PCM**
- **最大时长：** 单次请求最长 300 秒（5 分钟）
- **最短时长：** 不少于 0.5 秒

推荐预先使用 ffmpeg 转换为标准格式以获得最佳性能：

```bash
ffmpeg -i input.mp3 -f wav -ar 16000 -ac 1 output.wav
```

### 响应格式

**Content-Type:** `application/json`

```json
{
  "text": "language Chinese<asr_text>你好世界"
}
```

响应的 `text` 字段包含模型的原始输出，格式为：

```
language {语言名}<asr_text>{转录文本}
```

- `language` 后跟检测到的语言英文名（如 `Chinese`、`English`、`Japanese`）
- `<asr_text>` 标签后跟实际转录文本

如果指定了 `language` 参数，模型会以指定语言输出；否则自动检测。

### 内部提示词机制

当指定 `language` 参数时，服务端构建如下提示引导模型输出：

```
<|im_start|>assistant
language Chinese<asr_text>
```

不指定语言时，模型自行检测并输出完整的 `language {name}<asr_text>{text}` 格式。

---

## 4. 语言支持

### 主流语言（30 种，ISO 639-1 代码）

| 代码 | 语言 | Language | 代码 | 语言 | Language |
|------|------|----------|------|------|----------|
| `zh` | 中文 | Chinese | `en` | 英语 | English |
| `yue` | 粤语 | Cantonese | `ja` | 日语 | Japanese |
| `ko` | 韩语 | Korean | `de` | 德语 | German |
| `fr` | 法语 | French | `es` | 西班牙语 | Spanish |
| `pt` | 葡萄牙语 | Portuguese | `ru` | 俄语 | Russian |
| `ar` | 阿拉伯语 | Arabic | `it` | 意大利语 | Italian |
| `id` | 印尼语 | Indonesian | `th` | 泰语 | Thai |
| `vi` | 越南语 | Vietnamese | `tr` | 土耳其语 | Turkish |
| `hi` | 印地语 | Hindi | `ms` | 马来语 | Malay |
| `nl` | 荷兰语 | Dutch | `sv` | 瑞典语 | Swedish |
| `da` | 丹麦语 | Danish | `fi` | 芬兰语 | Finnish |
| `pl` | 波兰语 | Polish | `cs` | 捷克语 | Czech |
| `tl` | 菲律宾语 | Filipino | `fa` | 波斯语 | Persian |
| `el` | 希腊语 | Greek | `ro` | 罗马尼亚语 | Romanian |
| `hu` | 匈牙利语 | Hungarian | `mk` | 马其顿语 | Macedonian |

### 中文方言（22 种）

支持安徽话、东北话、福建话、甘肃话、广西话、贵州话、海南话、河北话、河南话、湖北话、湖南话、江西话、辽宁话、内蒙古话、宁夏话、青海话、山东话、山西话、陕西话、四川话、云南话、浙江话。

> 方言通过 Qwen3ASRModel 的 Python API 传入全名（如 `"Sichuan"`），vLLM HTTP API 的 `language` 参数使用 ISO 639-1 代码。

---

## 5. 热词功能

通过 `prompt` 参数传入热词（Hotword），引导模型更准确地识别特定词汇。适用于：

- 专业术语（如 "量化交易"、"GPT-4"）
- 人名、地名、品牌名
- 容易误识别的同音词

### 使用方式

热词以**空格分隔**传入 `prompt` 参数：

```python
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
    prompt="量化交易 特斯拉 GPT-4",
    language="zh",  # 可选
)
```

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav" \
  -F "prompt=量化交易 特斯拉 GPT-4"
```

热词会被放入模型的 system message 中引导识别，与 Transformers 后端的 `context` 参数效果一致。

---

## 6. 调用示例

### 6.1 Python（OpenAI SDK）

安装依赖：

```bash
pip install openai soundfile
```

#### 基本转录（自动检测语言）

```python
import subprocess
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")

# 用 ffmpeg 转换音频为 16kHz 单声道 WAV
audio_bytes = subprocess.run(
    ["ffmpeg", "-i", "audio.mp3", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
    capture_output=True, check=True,
).stdout

# 转录
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
)
print(result.text)
# 输出示例: "language Chinese<asr_text>你好世界"
```

#### 指定语言

```python
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
    language="zh",
)
```

#### 使用热词

```python
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
    language="zh",
    prompt="量化交易 特斯拉 GPT-4",
)
```

#### 检查服务连接

```python
client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")

try:
    models = client.models.list()
    print(f"连接成功，可用模型: {[m.id for m in models.data]}")
except Exception as e:
    print(f"连接失败: {e}")
```

### 6.2 curl

#### 基本转录

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav"
```

#### 指定语言

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav" \
  -F "language=zh"
```

#### 指定语言 + 热词

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "prompt=Tesla OpenAI GPT-4"
```

#### 通过 ffmpeg 管道传入

```bash
ffmpeg -i input.mp3 -f wav -ar 16000 -ac 1 pipe:1 2>/dev/null | \
  curl -X POST http://localhost:6001/v1/audio/transcriptions \
    -F "model=models/Qwen3-ASR-1.7B" \
    -F "file=@-;type=audio/wav"
```

### 6.3 JavaScript / Node.js

```javascript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI({
  baseURL: "http://localhost:6001/v1",
  apiKey: "EMPTY",
});

const transcription = await client.audio.transcriptions.create({
  model: "models/Qwen3-ASR-1.7B",
  file: fs.createReadStream("audio.wav"),
  language: "zh",
});

console.log(transcription.text);
```

---

## 7. 响应解析

模型原始输出格式为 `language {语言名}<asr_text>{转录文本}`，需要解析提取纯文本和语言信息。

### 解析函数

```python
import re

_ASR_TEXT_TAG = "<asr_text>"

def parse_asr_output(raw: str) -> tuple[str, str]:
    """
    解析模型原始输出为 (语言, 转录文本)。

    参数:
        raw: 模型原始输出文本

    返回:
        (language, text) 元组
        - language: 检测到的语言名（如 "Chinese"），可能为空
        - text: 纯转录文本

    示例:
        >>> parse_asr_output("language Chinese<asr_text>你好世界")
        ("Chinese", "你好世界")
        >>> parse_asr_output("language English<asr_text>hello world")
        ("English", "hello world")
    """
    if not raw:
        return "", ""
    s = raw.strip()
    if _ASR_TEXT_TAG in s:
        meta, text = s.split(_ASR_TEXT_TAG, 1)
        lang = ""
        m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
        if m and m.group(1).lower() != "none":
            lang = m.group(1)
        return lang, text.strip()
    return "", s
```

### 使用示例

```python
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
)

language, text = parse_asr_output(result.text)
print(f"语言: {language}")   # "Chinese"
print(f"文本: {text}")       # "你好世界"
```

---

## 8. 伪流式识别

vLLM 的 `/v1/audio/transcriptions` 接口本身不支持真流式输出。但可以通过**客户端累积音频**的方式实现伪流式效果——每隔固定时间将累积的音频发送给 API，获取当前的完整转录。

### 实现原理

1. 从音频开头开始，每次增加一段音频（如 2 秒）
2. 将从头到当前位置的累积音频发送给 API
3. 对比前后两次结果，提取稳定的公共前缀
4. 显示增量文本

### 完整示例

```python
import io
import re
import numpy as np
import soundfile as sf
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")

def numpy_to_wav_bytes(wav: np.ndarray, sr: int = 16000) -> bytes:
    """将 numpy 数组转为 WAV 字节"""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def parse_asr_output(raw: str) -> tuple[str, str]:
    if not raw:
        return "", ""
    s = raw.strip()
    if "<asr_text>" in s:
        meta, text = s.split("<asr_text>", 1)
        lang = ""
        m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
        if m and m.group(1).lower() != "none":
            lang = m.group(1)
        return lang, text.strip()
    return "", s

# 加载音频（16kHz mono float32）
wav16k, _ = sf.read("audio.wav", dtype="float32")

step = 16000 * 2  # 每次推进 2 秒
pos = 0
prev_text = ""
stable_text = ""
call_id = 0

while pos < wav16k.shape[0]:
    pos = min(pos + step, wav16k.shape[0])
    call_id += 1

    # 发送从头到当前位置的累积音频
    wav_bytes = numpy_to_wav_bytes(wav16k[:pos])
    result = client.audio.transcriptions.create(
        model="models/Qwen3-ASR-1.7B",
        file=wav_bytes,
    )
    _, text = parse_asr_output(result.text or "")

    # 前缀稳定化：前 2 次调用后，公共前缀被冻结
    if call_id > 2 and prev_text:
        cp = min(len(prev_text), len(text))
        for i in range(cp):
            if prev_text[i] != text[i]:
                cp = i
                break
        if cp > len(stable_text):
            stable_text = text[:cp]

    prev_text = text
    print(f"[{pos/16000:.1f}s] {text}")

print(f"\n最终结果: {text}")
```

### 性能注意

伪流式模式下，每次请求都会重新处理全部历史音频：
- 第 1 秒：处理 1 秒音频
- 第 10 秒：处理 10 秒音频
- 第 60 秒：处理 60 秒音频

**推理耗时随录音时长线性增长**，长时间录音（>30 秒）可能出现明显延迟。

---

## 9. 服务启动参数

`qwen-asr-serve` 支持所有 vLLM 服务端参数。以下为常用参数：

### 基本参数

| 参数 | 说明 | 示例 |
|------|------|------|
| 第一个位置参数 | 模型路径 | `models/Qwen3-ASR-1.7B` |
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `6001` |
| `--gpu-memory-utilization` | GPU 显存使用比例 | `0.6` |

### GPU 显存推荐值

| 模型 | 推荐值 | 说明 |
|------|--------|------|
| Qwen3-ASR-0.6B | `0.3` | 小模型，30% 显存即可 |
| Qwen3-ASR-1.7B | `0.5 ~ 0.8` | 按实际显存调整，16GB+ 显卡建议 0.6 |

### 高级参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--max-model-len` | 最大序列长度 | `16384` |
| `--max-num-batched-tokens` | 批量 token 数上限 | `8192` |
| `--max-num-seqs` | 最大并发序列数 | `1` ~ `128` |
| `--tensor-parallel-size` | 张量并行 GPU 数量 | `2`（多 GPU） |
| `--dtype` | 模型精度 | `bfloat16`（默认） |

### 启动示例

```bash
# 基本启动（1.7B 模型）
qwen-asr-serve models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.6 \
  --host 0.0.0.0 \
  --port 6001

# 小模型（0.6B，更快）
qwen-asr-serve models/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.3 \
  --host 0.0.0.0 \
  --port 6001

# 高并发生产配置
qwen-asr-serve models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 6001 \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32
```

---

## 10. Docker 部署

### 生产部署命令

```bash
docker run -d --gpus device=0 --name mygpt-qwen3-asr \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 6001:6001 \
  --network host \
  --add-host=host.docker.internal:host-gateway \
  --restart unless-stopped \
  --volume /etc/localtime:/etc/localtime:ro \
  --volume /etc/timezone:/etc/timezone:ro \
  --shm-size=4gb \
  mygpt-qwen3-asr:v1.0 \
  qwen-asr-serve models/Qwen3-ASR-1.7B \
    --gpu-memory-utilization 0.6 \
    --host 0.0.0.0 \
    --port 6001
```

### Docker 参数说明

| 参数 | 说明 |
|------|------|
| `--gpus device=0` | 指定使用 GPU 0 |
| `--network host` | 使用主机网络（避免端口映射问题） |
| `--shm-size=4gb` | 共享内存大小，PyTorch 需要足够的共享内存 |
| `--restart unless-stopped` | 自动重启（除非手动停止） |
| `--volume /etc/localtime:/etc/localtime:ro` | 同步时区 |

> **重要：** 使用 `--shm-size=4gb` 或 `--ipc=host`，否则 PyTorch 共享内存分配会失败。

### 构建自定义镜像

基于 vLLM 官方镜像：

```bash
docker build -t vllm-qwen3-asr:v1.0 -f docker/Dockerfile-vllm-qwen3-asr docker/
```

### 依赖

容器内需要以下系统包和 Python 包：

```bash
# 系统依赖
apt-get install -y libsndfile1 ffmpeg

# Python 依赖
pip install qwen-asr[vllm]
# 或
pip install "vllm[audio]"
```

---

## 11. 错误处理与常见问题

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Connection refused` | 服务未启动或端口不对 | 检查服务是否正常运行，确认端口号 |
| `Model not found` | 模型路径错误 | 调用 `GET /v1/models` 确认模型 ID |
| `CUDA out of memory` | GPU 显存不足 | 降低 `--gpu-memory-utilization` 值 |
| `ffmpeg 错误` | 音频格式不支持或文件损坏 | 检查音频文件是否完整，尝试用 ffmpeg 手动转换 |
| `SharedMemory error` | Docker 共享内存不足 | 添加 `--shm-size=4gb` 或 `--ipc=host` |

### 服务检查

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")

# 检查连接
try:
    models = client.models.list()
    print(f"服务正常，可用模型: {[m.id for m in models.data]}")
except Exception as e:
    print(f"服务异常: {e}")
```

```bash
# 健康检查
curl -s http://localhost:6001/health

# 查看可用模型
curl -s http://localhost:6001/v1/models | python -m json.tool
```

### api_key 说明

vLLM 服务默认不验证 API Key，但 OpenAI SDK 要求传入 `api_key` 参数。使用任意非空字符串即可，推荐使用 `"EMPTY"`：

```python
client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")
```

### 性能调优建议

1. **指定语言**：已知音频语言时始终传入 `language` 参数，可提高识别准确率并加快推理速度
2. **预转换音频**：使用 ffmpeg 预先转为 16kHz mono WAV，避免服务端重复转换
3. **调整并发**：通过 `--max-num-seqs` 控制并发请求数，根据 GPU 资源调整
4. **短音频优先**：单次请求音频越短，推理越快；长音频建议分段处理
