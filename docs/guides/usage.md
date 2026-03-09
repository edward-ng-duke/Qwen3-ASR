# Qwen3-ASR vLLM 使用文档

## 安装

```bash
# 安装（含 vLLM 后端）
pip install -U qwen-asr[vllm]

# 或从源码安装
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e ".[vllm]"

# 推荐安装 FlashAttention 2 加速推理
pip install -U flash-attn --no-build-isolation
```

## 启动服务

```bash
# 基本启动
qwen-asr-serve models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.6 \
  --host 0.0.0.0 \
  --port 6001

# 0.6B 小模型（更快，精度稍低）
qwen-asr-serve models/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.3 \
  --host 0.0.0.0 \
  --port 6001
```

服务启动后提供 OpenAI 兼容的 `/v1/audio/transcriptions` 接口。

## 基本调用

### Python（OpenAI SDK）

```python
import subprocess
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")

# 用 ffmpeg 转换音频为 16kHz 单声道 WAV
audio_bytes = subprocess.run(
    ["ffmpeg", "-i", "audio.mp3", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
    capture_output=True, check=True,
).stdout

# 转录（自动检测语言）
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
)
print(result.text)
```

### curl

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav"
```

## 指定语言

通过 `language` 参数指定音频语言，可提高识别准确率和速度：

```python
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
    language="zh",
)
```

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav" \
  -F "language=zh"
```

### 支持的语言代码 (ISO 639-1)

| 代码 | 语言 | 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|------|------|
| `zh` | 中文 | `en` | 英语 | `yue` | 粤语 |
| `ja` | 日语 | `ko` | 韩语 | `de` | 德语 |
| `fr` | 法语 | `es` | 西班牙语 | `pt` | 葡萄牙语 |
| `ru` | 俄语 | `ar` | 阿拉伯语 | `it` | 意大利语 |
| `id` | 印尼语 | `th` | 泰语 | `vi` | 越南语 |
| `tr` | 土耳其语 | `hi` | 印地语 | `ms` | 马来语 |
| `nl` | 荷兰语 | `sv` | 瑞典语 | `da` | 丹麦语 |
| `fi` | 芬兰语 | `pl` | 波兰语 | `cs` | 捷克语 |
| `fil` | 菲律宾语 | `fa` | 波斯语 | `el` | 希腊语 |
| `ro` | 罗马尼亚语 | `hu` | 匈牙利语 | `mk` | 马其顿语 |

## 热词（Hotword）

通过 `prompt` 参数传入热词，引导模型更准确地识别特定词汇。适用于专业术语、人名、产品名等容易误识别的词语。

```python
result = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_bytes,
    prompt="量化交易 特斯拉 GPT-4",  # 空格分隔的热词
    language="zh",                    # 可选
)
```

```bash
curl -X POST http://localhost:6001/v1/audio/transcriptions \
  -F "model=models/Qwen3-ASR-1.7B" \
  -F "file=@audio.wav" \
  -F "prompt=量化交易 特斯拉 GPT-4"
```

热词通过 OpenAI API 的 `prompt` 字段传入，会被放入模型的 system message 中引导识别，与 transformers 后端的 `context` 参数效果一致。

## 伪流式识别（客户端）

通过不断发送累积音频实现伪流式效果，适用于 vLLM 服务端点：

```python
import io
import re
import numpy as np
import soundfile as sf
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6001/v1", api_key="EMPTY")

def numpy_to_wav_bytes(wav: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def parse_asr_output(raw: str) -> tuple[str, str]:
    """解析模型输出 -> (language, text)"""
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

## 命令行示例

```bash
# 基本转录
python examples/example_vllm_with_language.py --audio data/test.mp3

# 指定语言
python examples/example_vllm_with_language.py --audio data/test.mp3 --language zh

# 使用热词
python examples/example_vllm_with_language.py --audio data/test.mp3 --prompt "交易 停滞"

# 伪流式
python vllm_req_streaming.py --audio data/test.wav --step-ms 2000

# 多种步长对比
python vllm_req_streaming.py --step-ms 500 1000 2000
```

## 音频格式要求

- 支持 MP3、WAV、FLAC、OGG、M4A 等常见格式
- 内部统一转换为 **16kHz 单声道 WAV**
- 推荐使用 ffmpeg 预转换：

```bash
ffmpeg -i input.mp3 -f wav -ar 16000 -ac 1 output.wav
```

## 服务启动参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gpu-memory-utilization` | GPU 显存使用比例 | `0.6`（1.7B）/ `0.3`（0.6B）|
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `6001` |

## API 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型路径，如 `models/Qwen3-ASR-1.7B` |
| `file` | bytes | 是 | 音频文件数据 |
| `language` | string | 否 | 语言代码（ISO 639-1），不填则自动检测 |
| `prompt` | string | 否 | 热词/上下文提示，空格分隔 |
