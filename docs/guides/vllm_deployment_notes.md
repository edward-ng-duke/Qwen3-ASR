# vLLM 启动 Qwen3-ASR 注意事项

## 1. vLLM 版本锁定

```toml
# pyproject.toml
vllm = ["vllm==0.14.0"]
```

必须使用 `vllm==0.14.0`，其他版本可能因 API 变更导致模型注册失败。安装方式：
```bash
pip install qwen-asr[vllm]
# 或
pip install vllm==0.14.0
```

## 2. 音频依赖（容易遗漏）

系统级依赖必须安装，否则音频加载会报错：
```bash
apt-get install libsndfile1 ffmpeg
pip install "vllm[audio]"
```

Dockerfile 中已包含这些：
```dockerfile
# docker/Dockerfile-vllm-qwen3-asr
FROM vllm/vllm-openai:nightly
RUN apt-get install -y libsndfile1 ffmpeg
RUN pip install "vllm[audio]"
```

## 3. GPU 显存配置

`--gpu-memory-utilization` 是最关键的参数，设太高会 OOM，设太低浪费资源：

| 模型 | 推荐值 | 说明 |
|------|--------|------|
| Qwen3-ASR-0.6B | `0.3` | 小模型，30% 显存即可 |
| Qwen3-ASR-1.7B | `0.5 ~ 0.8` | 按实际显存调整 |

用户命令用了 `0.8`，如果 GPU 显存不够大（如 < 16GB），可能需要降低。

## 4. 模型注册机制

代码中通过以下方式注册自定义模型到 vLLM：

```python
# qwen_asr/inference/qwen3_asr.py
from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
from vllm import ModelRegistry
ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
```

这意味着：
- **不能直接用 `vllm serve` 命令启动**，因为 vLLM 不认识这个自定义模型架构
- 必须通过 `qwen-asr-demo-streaming` 或 `qwen-asr-serve` 命令启动（它们内部会注册模型）
- 或者在 Python 代码中手动导入注册后再使用

## 5. 三种启动方式的区别

```bash
# 方式1：流式 Demo（你选的）— Flask Web UI + 流式 API
qwen-asr-demo-streaming --asr-model-path models/Qwen3-ASR-1.7B --port 6001

# 方式2：OpenAI 兼容 API 服务 — 适合生产部署
qwen-asr-serve models/Qwen3-ASR-1.7B --gpu-memory-utilization 0.6 --port 6001

# 方式3：Gradio Demo — 交互式 Web 界面（非流式）
qwen-asr-demo --asr-checkpoint models/Qwen3-ASR-1.7B --backend vllm
```

| 方式 | 协议 | 流式 | 适用场景 |
|------|------|------|----------|
| `demo-streaming` | Flask REST | 支持 | 实时语音转文字演示 |
| `serve` | OpenAI `/v1/audio/transcriptions` | 不支持 | 生产 API 服务 |
| `demo` | Gradio | 不支持 | 开发调试 |

## 6. max_new_tokens 差异

- **`demo-streaming`** 用的 `max_new_tokens=32`（每次只生成少量 token）
- **`serve`** 和 **`LLM()` 默认** 是 `max_new_tokens=4096`
- 流式场景 32 够用，因为每 1 秒 chunk 只需几个字的增量输出

## 7. 流式推理的限制

- **仅 vLLM 后端支持**，Transformers 后端不支持流式
- **不支持时间戳**（forced aligner 不能与流式结合）
- **不支持批量推理**，每个 session 独立推理
- **音频会越来越长**：每次推理都重新送入全部历史音频（`audio_accum`），长时间录音会导致推理变慢

## 8. 长音频性能问题

流式模式下 `audio_accum` 会无限增长：
- 第 1 秒：送 1 秒音频
- 第 10 秒：送 10 秒音频
- 第 60 秒：送 60 秒音频

每次推理都重新处理全部音频，**推理耗时会随录音时长线性增长**。长时间录音（>30秒）可能出现明显延迟。

## 9. 并发限制

`demo-streaming` 用的是 Flask + `threaded=True`：
- vLLM 的 `model.generate()` 是同步阻塞的
- 多个 session 会排队等待 GPU
- 不适合高并发场景，仅适合 Demo 演示

如果需要高并发，应该用 `qwen-asr-serve`（基于 vLLM 的 AsyncLLMEngine）。

## 10. Docker 部署注意

```bash
docker run -d \
  --runtime nvidia --gpus all --ipc=host \  # GPU 透传 + 共享内存
  -p 6001:8000 \
  -v /path/to/models:/models:ro \           # 模型挂载为只读
  vllm-qwen3-asr:v1.0 \
  --model /models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.5 \
  --max-model-len 16384 \                   # 限制最大序列长度
  --max-num-batched-tokens 8192 \           # 限制批量 token 数
  --max-num-seqs 1                          # 限制并发序列数
```

关键：`--ipc=host` 必须加，否则 PyTorch 共享内存分配会失败。

## 11. 支持的音频格式

输入支持：MP3, WAV, FLAC, OGG, M4A（自动通过 ffmpeg 转换）
内部处理：统一转为 **16kHz 单声道 float32 PCM**

## 12. 支持的语言（52 种）

包括中文、英文、粤语、日语、韩语、德语、法语等主流语言，以及 22 种中国方言。
完整列表见 `qwen_asr/inference/utils.py` 中的 `SUPPORTED_LANGUAGES`。
