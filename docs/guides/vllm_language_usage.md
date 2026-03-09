# vLLM 服务语言指定指南

本文档介绍如何在使用 vLLM 服务时指定转录语言。

## 1. 启动 vLLM 服务

```bash
qwen-asr-serve models/Qwen3-ASR-1.7B --gpu-memory-utilization 0.6 --host 0.0.0.0 --port 6001
```

参数说明：
- `models/Qwen3-ASR-1.7B`: 模型路径
- `--gpu-memory-utilization`: GPU 显存使用率
- `--host`: 监听地址
- `--port`: 服务端口

## 2. Python 调用方法

### 基本调用（自动检测语言）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6001/v1",
    api_key="EMPTY"
)

transcription = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_file,
)

print(transcription.text)
```

### 指定语言调用

```python
transcription = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_file,
    language="zh",  # 指定中文
)
```

## 3. 支持的语言代码 (ISO 639-1)

| 代码 | 语言 | Language |
|------|------|----------|
| `zh` | 中文 | Chinese |
| `en` | 英语 | English |
| `yue` | 粤语 | Cantonese |
| `ja` | 日语 | Japanese |
| `ko` | 韩语 | Korean |
| `de` | 德语 | German |
| `fr` | 法语 | French |
| `es` | 西班牙语 | Spanish |
| `pt` | 葡萄牙语 | Portuguese |
| `ru` | 俄语 | Russian |
| `ar` | 阿拉伯语 | Arabic |
| `it` | 意大利语 | Italian |
| `id` | 印尼语 | Indonesian |
| `th` | 泰语 | Thai |
| `vi` | 越南语 | Vietnamese |
| `tr` | 土耳其语 | Turkish |
| `hi` | 印地语 | Hindi |
| `ms` | 马来语 | Malay |
| `nl` | 荷兰语 | Dutch |
| `sv` | 瑞典语 | Swedish |
| `da` | 丹麦语 | Danish |
| `fi` | 芬兰语 | Finnish |
| `pl` | 波兰语 | Polish |
| `cs` | 捷克语 | Czech |
| `tl` | 菲律宾语 | Filipino |
| `fa` | 波斯语 | Persian |
| `el` | 希腊语 | Greek |
| `ro` | 罗马尼亚语 | Romanian |
| `hu` | 匈牙利语 | Hungarian |
| `mk` | 马其顿语 | Macedonian |

此外还支持 22 种中文方言（安徽、东北、福建、甘肃等）。

## 4. 内部实现原理

当指定 `language` 参数时，服务端会构建包含语言信息的提示词：

```
<|im_start|>assistant
language Chinese<asr_text>
```

这会强制模型以指定语言输出转录文本。

如果不指定 `language` 参数（或设为 `None`），模型会自动检测语言并输出：

```
<|im_start|>assistant
language {detected_language}<asr_text>{transcription_text}
```

## 5. 完整示例

参见 [examples/example_vllm_with_language.py](../examples/example_vllm_with_language.py)
