# Qwen3-ASR Streaming Demo 实现分析

## 1. 系统架构总览

```
┌─────────────────────┐     HTTP POST      ┌─────────────────────┐      vLLM       ┌─────────────┐
│   浏览器 (JS)        │ ─────────────────→ │   Flask 服务端       │ ──────────────→ │  Qwen3-ASR  │
│                     │                    │                     │                │  (GPU)      │
│ - 麦克风采集         │  /api/start        │ - 会话管理           │  audio_accum   │             │
│ - 重采样到 16kHz     │  /api/chunk        │ - 音频缓冲           │  + prompt      │  音频编码器   │
│ - 500ms 分块发送     │  /api/finish       │ - 推理调度           │  ────────────→ │  + LLM 解码  │
│                     │ ←───────────────── │                     │ ←──────────── │             │
│ - 更新 UI 显示       │  JSON {text, lang} │ - 结果解析返回        │  生成文本       │             │
└─────────────────────┘                    └─────────────────────┘                └─────────────┘
```

**关键特征**：
- 协议：**HTTP 轮询**（非 WebSocket）
- 入口：`qwen_asr/cli/demo_streaming.py` → Flask 应用
- 推理引擎：vLLM（流式 ASR 仅支持 vLLM 后端）
- 模型：`Qwen3-ASR-1.7B`

## 2. 数据流详细分析

### 2.1 客户端（浏览器 JavaScript）

代码位置：`qwen_asr/cli/demo_streaming.py` 第 211-405 行（内嵌 JS）

```
麦克风 → WebAudio API (ScriptProcessorNode, 4096 samples/回调)
    → 线性插值重采样到 16kHz
    → 累积到 buf (Float32Array)
    → 每积满 500ms (8000 samples) 发送一次 POST /api/chunk
```

**音频采集参数**：
- 单声道、回声消除、噪声抑制、自动增益（第 317-323 行）
- `CHUNK_MS = 500`：客户端每 500ms 发送一块音频（第 222 行）
- `TARGET_SR = 16000`：目标采样率 16kHz（第 223 行）

**重采样方式**：线性插值（第 253-266 行），非 librosa 高质量重采样

### 2.2 HTTP API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/start` | POST | 创建会话，初始化流式状态，返回 `session_id` |
| `/api/chunk` | POST | 接收 float32 PCM 二进制数据，触发推理，返回当前转录结果 |
| `/api/finish` | POST | 刷新剩余音频，返回最终结果，销毁会话 |

### 2.3 服务端处理流程

```
POST /api/chunk 接收 binary float32 PCM
    → np.frombuffer(raw, dtype=np.float32)
    → asr.streaming_transcribe(wav, state)
        → 追加到 state.buffer
        → while buffer >= chunk_size_samples (默认 16000，即 1 秒):
            → 取出一个 chunk
            → 追加到 state.audio_accum（全部历史音频）
            → 构建 prefix（前缀回滚策略）
            → prompt = prompt_raw + prefix
            → vLLM.generate(prompt + 全部 audio_accum)  ← 关键：每次都喂全部音频
            → 解析输出 → 更新 state.language, state.text
    → 返回 JSON {language, text}
```

## 3. 流式推理核心机制

### 3.1 分块策略

代码位置：`qwen_asr/inference/qwen3_asr.py` 第 720-763 行

两层缓冲机制：

1. **客户端缓冲**：每 500ms 发送一次（`CHUNK_MS = 500`）
2. **服务端缓冲**：积累到 `chunk_size_sec`（默认 1.0 秒）才触发推理

客户端发送频率（500ms）和服务端推理阈值（1.0s）不同步，因此大约每 2 次 `/api/chunk` 请求触发 1 次实际推理。

### 3.2 音频累积（无截断）

```python
# qwen3_asr.py 第 724-728 行
if state.audio_accum.shape[0] == 0:
    state.audio_accum = chunk
else:
    state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)
```

每次新 chunk 到来，都追加到 `audio_accum`。推理时输入的是**从流开始到当前的全部音频**：

```python
# qwen3_asr.py 第 751 行
inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}
```

### 3.3 前缀回滚策略（Prefix Rollback）

代码位置：`qwen_asr/inference/qwen3_asr.py` 第 730-746 行

目的：减少分块边界处的转录抖动（jitter）。

工作方式：
- **前 N 个 chunk**（`unfixed_chunk_num`，默认 4）：不使用前缀，即 `prefix = ""`
- **第 N+1 个 chunk 起**：取上一轮的解码文本，回滚最后 K 个 token（`unfixed_token_num`，默认 5），将截断后的文本作为 prompt 前缀

```python
# 伪代码
if chunk_id < unfixed_chunk_num:
    prefix = ""
else:
    token_ids = tokenizer.encode(上一轮解码文本)
    prefix = tokenizer.decode(token_ids[:-unfixed_token_num])

prompt = prompt_raw + prefix
# 然后用 prompt + 全部音频 做推理
```

**效果**：模型在生成时会看到之前已确认的文本前缀，倾向于延续而非重新生成，减少边界处文字跳动。

### 3.4 输出解析

代码位置：`qwen_asr/inference/utils.py` → `parse_asr_output()`

模型输出格式：`language {语言名}<asr_text>{转录文本}`

解析后更新 `state.language` 和 `state.text`。

### 3.5 生成参数

```python
# demo_streaming.py 第 497-501 行
asr = Qwen3ASRModel.LLM(
    model=args.asr_model_path,
    gpu_memory_utilization=0.8,
    max_new_tokens=32,    # 每次推理最多生成 32 个 token
)
```

`max_new_tokens=32` 是一个关键限制——每次推理只输出最多 32 个新 token。

## 4. 截断问题深度分析

### 4.1 问题本质

**streaming_transcribe() 没有任何音频截断/分段逻辑。** `audio_accum` 随时间单调增长，每次推理都将全部历史音频重新喂给模型。

### 4.2 数学估算

| 录制时长 | audio_accum 采样数 | 内存占用 (float32) | 每次推理的音频输入 |
|---------|-------------------|-------------------|------------------|
| 1 分钟   | 960,000           | 3.7 MB            | 960K samples     |
| 5 分钟   | 4,800,000         | 18.3 MB           | 4.8M samples     |
| 10 分钟  | 9,600,000         | 36.6 MB           | 9.6M samples     |

注意：以上仅是 PCM 原始数据。音频编码器会将其转换为特征序列，再与文本 token 一起占用 GPU 显存。实际 GPU 显存消耗远大于 PCM 数据本身。

### 4.3 具体后果

1. **推理延迟线性增长**
   - 音频编码器处理时间与音频长度成正比
   - Attention 计算量与序列长度的平方成正比
   - 录制 5 分钟后，每秒的推理可能需要数秒甚至更长

2. **GPU 显存持续增长**
   - 音频特征序列 + KV cache 随音频长度线性增长
   - 可能导致 OOM（显存不足）

3. **模型上下文长度溢出**
   - Qwen3-ASR 有最大上下文长度限制
   - 超长音频的特征序列可能超出限制，导致推理失败或结果质量下降

4. **`max_new_tokens=32` 的瓶颈**
   - 每次推理最多生成 32 个 token
   - 对于 1 分钟以上的音频，32 token 远不够覆盖全部转录文本
   - 但由于前缀回滚策略，模型会在之前文本的基础上继续生成，所以短期内可以工作
   - 长期来看，当新增内容超过 32 token 时，转录将不完整

5. **前缀回滚的局限性**
   - 回滚策略依赖 `_raw_decoded` 中的累积文本
   - 当 `max_new_tokens` 不足以覆盖新增内容时，`_raw_decoded` 会落后于实际音频
   - 形成恶性循环：前缀越来越长 → 可用生成空间越来越少

### 4.4 根因总结

这个 streaming demo 的设计思路是**"伪流式"**：

- 它不是真正的流式解码（像 CTC/RNN-T 那样逐帧解码）
- 而是一种**滑动窗口式的批量推理**，但窗口只扩大不缩小
- 每次推理都是对全部历史音频的"重新翻译"，用前缀来引导模型延续之前的结果
- 适用于短时间（1-2 分钟以内）的实时转录场景
- 不适用于长时间连续录制场景

## 5. 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 命令入口定义 | `pyproject.toml` | 52 |
| Flask 服务启动 | `qwen_asr/cli/demo_streaming.py` | 485-503 |
| 前端 JS（音频采集+发送） | `qwen_asr/cli/demo_streaming.py` | 211-405 |
| HTTP API 路由 | `qwen_asr/cli/demo_streaming.py` | 417-469 |
| 会话管理 + TTL 清理 | `qwen_asr/cli/demo_streaming.py` | 38-72 |
| `ASRStreamingState` 状态定义 | `qwen_asr/inference/qwen3_asr.py` | 78-128 |
| `init_streaming_state()` | `qwen_asr/inference/qwen3_asr.py` | 584-655 |
| `streaming_transcribe()` 核心循环 | `qwen_asr/inference/qwen3_asr.py` | 657-765 |
| 音频累积（无截断） | `qwen_asr/inference/qwen3_asr.py` | 724-728 |
| 前缀回滚策略 | `qwen_asr/inference/qwen3_asr.py` | 730-746 |
| vLLM 推理调用 | `qwen_asr/inference/qwen3_asr.py` | 751-754 |
| `finish_streaming_transcribe()` | `qwen_asr/inference/qwen3_asr.py` | 767-829 |
| 输出解析 `parse_asr_output()` | `qwen_asr/inference/utils.py` | 403-470 |
| 重复检测 `detect_and_fix_repetitions()` | `qwen_asr/inference/utils.py` | 335-400 |
