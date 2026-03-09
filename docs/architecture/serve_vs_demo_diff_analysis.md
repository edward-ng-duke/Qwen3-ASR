# 分析：`qwen-asr-serve` 与 `qwen-asr-demo` 结果差异的原因

## 背景

使用以下两种方式启动 Qwen3-ASR 服务，对同一段音频（< 30 秒）进行转录，结果存在细微差别。两种方式各自都是确定性的（方法二连续测试 10 次结果完全一致），但彼此之间有些许不同。

**方法一（`qwen-asr-serve`）：**
```bash
qwen-asr-serve models/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 6001
```

**方法二（`qwen-asr-demo`）：**
```bash
qwen-asr-demo \
  --asr-checkpoint models/Qwen3-ASR-1.7B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization":0.8,"max_inference_batch_size":8,"max_new_tokens":2048}' \
  --ip 0.0.0.0 --port 6001
```

## 结论

两种方法都使用 `temperature=0.0`（贪心解码），因此各自是确定性的。但它们产生略有不同的结果，**根本原因是 prompt 模板不一致**。

---

## 差异 1（最关键）：Prompt 模板不同

### 方法二（`qwen-asr-demo`）的 prompt

通过 `_build_text_prompt()` 构建 prompt，会包含一个**空的 system message**：

```
<|im_start|>system
<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
```

代码路径：`qwen_asr/inference/qwen3_asr.py:448-462`

```python
def _build_messages(self, context, audio_payload):
    return [
        {"role": "system", "content": context or ""},   # <-- 空 system message
        {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
    ]

def _build_text_prompt(self, context, force_language):
    msgs = self._build_messages(context=context, audio_payload="")
    base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}<asr_text>"
    return base
```

当 `context=""` 时，`context or ""` 为空字符串，但 system message 的特殊 token（`<|im_start|>system\n<|im_end|>\n`）仍然会被添加到 prompt 中。

### 方法一（`qwen-asr-serve`）的 prompt

通过模型的 `get_generation_prompt()` 构建 prompt，**没有 system message**：

```
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
```

代码路径：`qwen_asr/core/vllm_backend/qwen3_asr.py:960-997`

```python
@classmethod
def get_generation_prompt(cls, audio, model_config, stt_config, language, task_type, request_prompt, to_language):
    tokenizer = cached_tokenizer_from_config(model_config)
    audio_placeholder = cls.get_placeholder_str("audio", 0)

    if to_language is None:
        prompt = (
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\nlanguage {full_lang_name_to}<asr_text>"
        )

    prompt_token_ids = tokenizer.encode(prompt)
    return {"prompt_token_ids": prompt_token_ids, "multi_modal_data": {"audio": audio}}
```

### 影响分析

空 system message 多了约 4-5 个 token（`<|im_start|>`, `system`, `\n`, `<|im_end|>`, `\n`），这改变了所有后续 token 的位置编码（RoPE）。模型在略微不同的位置视角下对相同的音频特征做贪心解码，最终输出产生细微差别。

**这是导致两种方法结果不一致的主要原因。**

---

## 差异 2：音频分块策略不同

| | 方法一 (serve) | 方法二 (demo) |
|---|---|---|
| 最大分块长度 | **30 秒**（vLLM `SpeechToTextConfig` 默认值） | **1200 秒**（20 分钟） |
| 分块重叠 | 1 秒 overlap | 无 overlap，在低能量点切分 |
| 分块逻辑来源 | `vllm/entrypoints/openai/speech_to_text.py` | `qwen_asr/inference/utils.py` |

**本次测试中不影响**（测试音频 < 30 秒，不触发任何分块逻辑）。

但如果使用超过 30 秒的音频，方法一会将其拆分成多个 chunk 分别推理再拼接结果，而方法二不会。这会导致更大的结果差异。

---

## 差异 3：SamplingParams 细节不同

| 参数 | 方法一 (serve) | 方法二 (demo) |
|---|---|---|
| temperature | 0.0 | 0.0 |
| top_p | 1.0 | vLLM SamplingParams 默认 (1.0) |
| top_k | **0** | vLLM SamplingParams 默认 (**-1**) |
| repetition_penalty | 1.0 | vLLM SamplingParams 默认 (1.0) |
| frequency_penalty | 0.0 | 未显式设置 |
| presence_penalty | 0.0 | 未显式设置 |
| max_tokens | vLLM 引擎默认值 | **2048** |

方法一的 SamplingParams 来自 vLLM 的 `TranscriptionRequest.to_sampling_params()`（`vllm/entrypoints/openai/protocol.py:2117-2164`），方法二来自 `Qwen3ASRModel.LLM()` 中显式创建的 `SamplingParams(temperature=0.0, max_tokens=2048)`。

虽然 `temperature=0.0` 意味着贪心解码，但 `top_k=0` vs `top_k=-1` 在 vLLM 内部可能走不同的代码路径。在贪心模式下这个差异的实际影响应该很小。

---

## 差异 4：音频预处理路径不同

- **方法一**：客户端 `vllm_req.py` 先通过 **ffmpeg** 将音频转为 16kHz mono WAV → 以 bytes 发送到服务端 → vLLM 服务端用 **librosa.load()** 解码 WAV bytes
- **方法二**：demo 端直接用 **librosa.load()** 加载原始音频文件并重采样到 16kHz

如果原始音频不是 16kHz WAV 格式，两条路径使用了不同的重采样算法（ffmpeg 默认使用 soxr，librosa 默认使用 resampy 或 soxr），可能产生微小的 PCM 样本值差异，进而导致 mel-spectrogram 输入略有不同。

在 `temperature=0.0` 的贪心解码下，即使输入的微小差异也可能在某些边界 token 上导致不同的解码选择。

---

## 总结

| 差异来源 | 影响程度 | 是否影响本次测试 |
|---|---|---|
| **Prompt 模板**（空 system message） | **高** | **是** |
| 音频分块策略 | 中-高 | 否（音频 < 30 秒） |
| SamplingParams 细节 | 低 | 可能 |
| 音频预处理路径 | 低 | 可能 |

**根本原因**：方法二（demo）在 prompt 开头多了空的 `<|im_start|>system\n<|im_end|>\n`，这额外的 token 改变了后续所有 token 的 RoPE 位置编码，导致模型在略微不同的条件下做贪心解码，最终输出产生细微差别。

---

## 关键文件索引

| 文件 | 行号 | 说明 |
|---|---|---|
| `qwen_asr/core/vllm_backend/qwen3_asr.py` | 960-997 | serve 端的 prompt 构建（无 system message） |
| `qwen_asr/inference/qwen3_asr.py` | 448-465 | demo 端的 prompt 构建（有空 system message） |
| `qwen_asr/cli/serve.py` | 40-42 | serve 入口，直接委托给 vLLM CLI |
| `qwen_asr/cli/demo.py` | 227-240 | demo 默认 backend kwargs |
| `qwen_asr/inference/qwen3_asr.py` | 226-288 | `Qwen3ASRModel.LLM()` 初始化（SamplingParams 创建） |
| `vllm/.../protocol.py` | 2069-2164 | vLLM TranscriptionRequest 默认采样参数 |
| `vllm/.../speech_to_text.py` | 249-313 | vLLM 服务端音频预处理与分块逻辑 |
| `models/Qwen3-ASR-1.7B/generation_config.json` | - | 模型默认 generation config（temperature=0.000001） |
