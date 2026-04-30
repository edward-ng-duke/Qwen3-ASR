# realtime_demo_1.py 实现分析

## 1. 系统架构总览

```
┌─────────────────────┐     WebSocket        ┌─────────────────────┐   OpenAI API    ┌─────────────┐
│   浏览器 (JS)        │ ◄══════════════════► │  FastAPI 服务端       │ ──────────────→ │  vLLM 推理   │
│                     │                      │  (async, uvicorn)   │                │  (Qwen3-ASR) │
│ - AudioWorklet 采集  │  binary: float32 PCM │ - 会话 = WS 连接     │  AsyncOpenAI   │             │
│ - 重采样到 16kHz     │  JSON: 控制消息       │ - 滑动窗口截断       │  audio.trans-  │  音频编码器   │
│ - chunk_ms 分块发送  │                      │ - VAD 门控推理       │  criptions     │  + LLM 解码  │
│                     │ ◄────────────────── │ - 增量输出           │ ◄──────────── │             │
│ - delta 追加显示     │  JSON {delta,preview}│   (delta + preview)  │  转录文本       │             │
│ - preview 灰色预览   │                      │                     │                │             │
└─────────────────────┘                      └──────────┬──────────┘                └─────────────┘
                                                        │ WebSocket (可选)
                                                        ▼
                                              ┌─────────────────┐
                                              │  外部 VAD 服务    │
                                              │  (Silero 等)     │
                                              └─────────────────┘
```

**关键特征**：
- 协议：**WebSocket**（双向流式，非 HTTP 轮询）
- 框架：FastAPI + uvicorn（全异步）
- 推理客户端：AsyncOpenAI（通过 OpenAI 兼容 API 调用 vLLM）
- 会话管理：WebSocket 连接即会话，无需全局字典
- 音频管理：**滑动窗口截断**（默认 10 秒上下文窗口）
- VAD：外部 VAD 服务 + energy-based 降级方案
- 输出模式：**增量输出**（delta 已确认文本 + preview 预览文本）

## 2. 数据流详细分析

### 2.1 客户端（浏览器 JavaScript + AudioWorklet）

代码位置：`realtime_demo_1.py` 第 902-1217 行（内嵌 HTML/JS）+ 第 827-896 行（AudioWorklet）

```
麦克风 → AudioContext.getUserMedia (单声道, 16kHz, 回声消除/噪声抑制/自动增益)
    → AudioWorkletProcessor (PCMProcessor)
        → 累积到内部 buffer
        → 每积满 targetSamples (chunk_ms 对应的采样数) 触发一次
        → 如果 sourceSr ≠ targetSr，在 Worklet 内做线性插值重采样
        → postMessage 发送 Float32Array
    → 主线程 onmessage
        → 检查 WebSocket backpressure (bufferedAmount > 256KB 则丢弃)
        → ws.send(samples.buffer) 发送二进制帧
```

**音频采集参数**：
- 单声道、回声消除、噪声抑制、自动增益（第 1107-1114 行）
- `CHUNK_MS`：可配置，默认 1000ms（第 991 行，由服务端注入）
- `TARGET_SR = 16000`：目标采样率 16kHz（第 992 行）

**AudioWorklet vs ScriptProcessorNode**：
- 官方 demo 使用 ScriptProcessorNode（已弃用，在主线程运行）
- 本实现使用 AudioWorkletProcessor（独立音频线程，不阻塞 UI）

**Backpressure 控制**（第 1139-1141 行）：
- 当 WebSocket 发送缓冲 > 256KB 时，丢弃音频块
- 防止网络慢时客户端内存溢出

### 2.2 WebSocket 协议

单一端点 `/ws/asr`，双向通信：

| 方向 | 帧类型 | 内容 |
|------|--------|------|
| 客户端 → 服务端 | binary | float32 PCM 音频数据 |
| 客户端 → 服务端 | text/JSON | `{ "type": "config", "context_sec": 10 }` 动态调整窗口 |
| 客户端 → 服务端 | text/JSON | `{ "type": "finish" }` 结束录制 |
| 服务端 → 客户端 | text/JSON | `{ "language": "...", "delta": "...", "preview": "...", "text": "..." }` |
| 服务端 → 客户端 | text/JSON | `{ "error": "..." }` 错误信息 |
| 服务端 → 客户端 | text/JSON | `{ "type": "ping" }` 心跳 |

**对比官方**：官方使用 3 个独立 HTTP 端点（`/api/start`、`/api/chunk`、`/api/finish`），每次都是请求-响应模式。WebSocket 支持服务端主动推送、更低延迟。

### 2.3 服务端处理流程

```
WebSocket 连接建立
    → 创建 StreamingSession（连接即会话）
    → 可选：连接外部 VAD 服务
    → 启动 watchdog (5 分钟空闲超时)
    → 进入消息循环:
        收到 binary 帧 (音频):
            → np.frombuffer(raw, dtype=np.float32)
            → 数据清理 (NaN/Inf → 0, clip [-1,1])
            → session.append_audio(chunk)
                → 追加到 _audio_chunks 列表
                → 触发 _apply_sliding_window() 检查是否需要截断
            → 转发音频给 VAD 服务 (float32 → int16)
            → VAD 判定是否有语音活动
            → 如果 should_infer 且无正在执行的推理 且间隔足够:
                → asyncio.create_task(_do_infer_and_send())
                    → engine.transcribe(session)
                        → 音频 → WAV 字节 → AsyncOpenAI audio.transcriptions.create()
                        → _parse_asr_output() 解析
                        → _stabilize() 稳定化处理
                        → 计算 delta (已确认增量) 和 preview (待确认预览)
                    → ws.send_json(result)
        收到 JSON "config":
            → 动态调整 context_window_sec
        收到 JSON "finish":
            → 取消进行中的推理 → 执行最终推理 → 发送结果 → 断开
```

## 3. 流式推理核心机制

### 3.1 推理调用方式

代码位置：`realtime_demo_1.py` 第 387-435 行（`ASREngine.transcribe`）

**与官方的根本差异**：本实现不直接调用 vLLM Python API，而是通过 **OpenAI 兼容 HTTP API** 调用：

```python
# realtime_demo_1.py 第 402-405 行
transcription = await self.client.audio.transcriptions.create(
    model=self.config.model,
    file=wav_bytes,  # WAV 格式字节
)
```

- 音频先转为 WAV 字节（`_to_wav()`，第 471-474 行）
- 通过 `AsyncOpenAI` 发送到 vLLM 的 `/v1/audio/transcriptions` 端点
- 支持并发控制：`asyncio.Semaphore(max_concurrent_infer)`（默认 10）

### 3.2 滑动窗口截断（核心改进）

代码位置：`realtime_demo_1.py` 第 119-149 行（`_apply_sliding_window`）

**官方问题**：`audio_accum` 无限增长，每次推理都喂全部历史音频，导致延迟线性增长、GPU 显存溢出。

**本实现方案**：滑动窗口，默认保留最近 10 秒音频。

```python
# 第 119-149 行
def _apply_sliding_window(self) -> None:
    if self.context_window_sec <= 0:        # 0 = 无限模式（兼容官方行为）
        return
    if self._infer_since_trim < 2:          # 至少 2 次推理后才允许截断
        return                               # 保证 common-prefix 稳定化有 prev_text
    max_samples = self.context_window_sec * self.config.sample_rate
    if self._audio_len <= max_samples:
        return

    # 将 stable_text 提交到 finalized_text（已确认，不会再变）
    if self.stable_text:
        self.finalized_text = _append_with_overlap(self.finalized_text, self.stable_text)

    # 截断音频：只保留最后 max_samples
    self._set_audio(self.audio_accum[-max_samples:])
    # 重置稳定化状态
    self.prev_text = ""
    self.stable_text = ""
    self._infer_since_trim = 0
```

**截断安全保障**：
1. `_infer_since_trim < 2` — 截断后至少进行 2 次推理才允许再次截断，确保 common-prefix 稳定化机制有 `prev_text` 可比较
2. 截断前先将 `stable_text` 提交到 `finalized_text`，避免丢失已确认文本
3. 用户可通过前端按钮动态切换窗口大小（3s / 5s / 10s / ∞）

### 3.3 文本稳定化（Common-Prefix Stabilization）

代码位置：`realtime_demo_1.py` 第 437-465 行（`_stabilize`）

与官方的"前缀回滚"类似但机制不同：

```
官方方案（Prefix Rollback）：
    把上一轮解码文本回滚 K 个 token → 作为 prompt 前缀 → 喂给模型引导延续

本实现方案（Common-Prefix + Token Rollback）：
    比较本轮文本与上轮文本 → 找最长公共前缀 → 回滚 K 个 token → 作为 stable_text
```

详细流程：

```python
# 第 444-458 行
if session.call_id > unfixed_chunk_num and session.prev_text:
    cp = _common_prefix_len(session.prev_text, text)  # 两轮文本的公共前缀长度
    if cp > len(session.stable_text):
        candidate = text[:cp]
        candidate = self._rollback_tokens(candidate, unfixed_token_num)  # 回滚 5 个 token
        if len(candidate) > len(session.stable_text):
            session.stable_text = candidate     # stable 只增不减

    # 强制已稳定的前缀不变
    if not text.startswith(session.stable_text):
        text = session.stable_text + text[len(session.stable_text):]
```

**关键区别**：
- 官方将前缀注入到 prompt 中引导模型；本实现在输出端做后处理
- 本实现的 `stable_text` 单调递增（只增不减），更稳定
- 支持 tokenizer-based 回滚（精确）和 char-based 回滚（降级）

### 3.4 增量输出（Delta + Preview）

代码位置：`realtime_demo_1.py` 第 415-435 行

官方每次返回完整文本 `text`，客户端直接替换显示。本实现返回增量：

```
delta   = 新确认的文本增量（追加到已显示文本后，不会再变）
preview = 尚未确认的预览文本（灰色显示，可能被下次推理修改或删除）
```

计算逻辑：

```python
# 第 418-427 行
confirmed = _merge_finalized_with_window(session.finalized_text, session.stable_text)
delta = _compute_incremental_delta(confirmed, session._emitted_text)
if delta:
    session._emitted_text += delta

preview = window_text[len(session.stable_text):]  # stable 之后的部分是未确认的
```

**文本层次**：
```
finalized_text  ← 被滑动窗口截断时提交的历史文本（不可逆）
stable_text     ← 当前窗口内已通过 common-prefix 确认的文本
window_text     ← 当前窗口的完整转录（包含未确认部分）
preview         ← window_text 中 stable_text 之后的部分

客户端显示 = 已追加的 delta 累积 + 当前 preview（灰色）
```

### 3.5 去重与合并辅助函数

| 函数 | 位置 | 作用 |
|------|------|------|
| `_common_prefix_len(a, b)` | 第 247 行 | 两个字符串的公共前缀长度 |
| `_suffix_prefix_overlap(a, b)` | 第 255 行 | a 的后缀与 b 的前缀的最长重叠 |
| `_remove_output_duplicates(delta, recent)` | 第 264 行 | 去除 delta 与已输出文本的重复前缀 |
| `_compute_incremental_delta(confirmed, emitted)` | 第 275 行 | 计算需要新追加的增量文本 |
| `_append_with_overlap(base, addition)` | 第 292 行 | 带重叠检测的字符串拼接 |
| `_merge_finalized_with_window(finalized, window)` | 第 301 行 | 合并 finalized 与窗口文本（处理子串匹配） |
| `_split_at_last_sentence_boundary(text)` | 第 329 行 | 在最后一个句号处切分（完整句 + 尾部片段） |

### 3.6 输出解析

代码位置：`realtime_demo_1.py` 第 228-244 行（`_parse_asr_output`）

与官方 `parse_asr_output()` 功能相同，解析模型输出格式 `language {语言名}<asr_text>{转录文本}`。额外集成了 `detect_and_fix_repetitions()`（第 155-218 行）用于清除 ASR 模型常见的重复输出。

## 4. VAD（语音活动检测）机制

代码位置：`realtime_demo_1.py` 第 572-605 行（初始化）、第 663-731 行（运行时）

双层 VAD 方案，外部服务优先：

### 4.1 外部 VAD 服务

```
连接建立时：
    → WebSocket 连接到 vad_url/vad_realtime_ws
    → 发送 init 消息 {type: "init", sample_rate, chunk_size_ms}

每个音频 chunk：
    → float32 → int16 转换
    → 发送到 VAD 服务
    → 接收响应: {status: "detected", segments: [[start, end], ...]}
        → end == -1: 语音开始 → _vad_speech_active = True
        → end != -1: 语音结束 → _vad_speech_active = False
    → 无语音时递增 _vad_chunks_since_speech_end
```

### 4.2 Energy-based 降级

当外部 VAD 不可用时，使用简单的 RMS 能量检测（第 221-225 行）：

```python
def _check_energy_vad(chunk, threshold=0.01):
    return float(np.sqrt(np.mean(chunk**2))) > threshold
```

### 4.3 推理门控逻辑

```python
# 第 708-731 行
if vad_ws:
    should_infer = speech_active or (ever_had_speech and chunks_since_end <= 3)
else:
    is_speech = _check_energy_vad(chunk, threshold)
    should_infer = energy_speech_active or chunks_since_end <= 3
```

**效果**：静默期间不触发推理，节省 GPU 资源。语音结束后额外推理 3 个 chunk 以捕获尾部内容。

## 5. 推理节流与并发控制

### 5.1 推理间隔控制

```python
# 第 706-744 行
min_infer_interval = config.chunk_ms / 1000.0  # 与 chunk 间隔相同

if (should_infer
    and (infer_task is None or infer_task.done())  # 上一次推理已完成
    and (now - last_infer_time >= min_infer_interval)):  # 间隔足够
    infer_task = asyncio.create_task(_do_infer_and_send(...))
```

- 上一次推理未完成时不启动新推理（避免并发修改 session）
- 推理间隔不小于 `chunk_ms`

### 5.2 全局并发信号量

```python
# 第 349 行
self._semaphore = asyncio.Semaphore(config.max_concurrent_infer)  # 默认 10
```

多个 WebSocket 会话共享此信号量，防止同时发起过多 vLLM 请求导致服务过载。

## 6. 会话管理与生命周期

### 6.1 会话 = WebSocket 连接

```python
# 第 556 行
session = StreamingSession(config=config)
```

官方使用全局 `sessions` 字典 + `session_id` + TTL 清理线程。本实现中 WebSocket 连接本身就是会话的生命周期边界，连接断开即销毁，无需清理。

### 6.2 超时与心跳

- **Watchdog**（第 609-615 行）：5 分钟无活动自动断开
- **心跳**（第 631-635 行）：60 秒无消息发送 ping 保活
- **Finish 流程**（第 766-783 行）：取消进行中的推理 → 执行最终推理 → 发送结果 → 断开

### 6.3 指标收集

代码位置：第 480-491 行（`Metrics`）、第 529-541 行（`/metrics` 端点）

```python
@dataclass
class Metrics:
    active_sessions: int = 0
    total_sessions: int = 0
    total_infer_calls: int = 0
    total_infer_errors: int = 0
    infer_latency_sum: float = 0.0
    audio_duration_processed: float = 0.0
```

通过 `GET /metrics` 暴露 JSON 格式的运行统计。

## 7. 与官方实现的关键差异对比

| 维度 | 官方 `demo_streaming.py` | `realtime_demo_1.py` |
|------|--------------------------|----------------------|
| **通信协议** | HTTP 轮询（3 个端点） | WebSocket 双向流 |
| **Web 框架** | Flask（同步） | FastAPI + uvicorn（全异步） |
| **推理客户端** | vLLM Python API 直接调用 | AsyncOpenAI HTTP API |
| **音频采集** | ScriptProcessorNode（已弃用） | AudioWorkletProcessor（独立线程） |
| **音频截断** | 无（audio_accum 无限增长） | 滑动窗口（默认 10s，可动态调整） |
| **文本稳定化** | Prefix Rollback（注入 prompt） | Common-Prefix + Token Rollback（输出端后处理） |
| **输出方式** | 每次返回完整 text | 增量 delta + 预览 preview |
| **VAD** | 仅能量 VAD | 外部 VAD 服务 + energy 降级 |
| **推理节流** | 无（每个 chunk 都可能触发） | VAD 门控 + 间隔限制 + 并发信号量 |
| **会话管理** | 全局字典 + TTL 清理 | WebSocket = Session，自动清理 |
| **长时录制** | 不适用（>2min 延迟爆炸） | 滑动窗口 + finalized_text，理论无上限 |
| **SSL 支持** | 无 | 自动检测 cert.pem/key.pem |
| **Backpressure** | 无 | WebSocket bufferedAmount > 256KB 丢弃 |
| **运行指标** | 无 | `/metrics` 端点 |

## 8. 长时录制问题的解决

### 8.1 官方问题回顾

官方实现中 `audio_accum` 随时间单调增长，导致：
- 推理延迟 O(n) 增长（音频编码）、O(n²) 增长（Attention）
- GPU 显存持续增长直至 OOM
- `max_new_tokens=32` 瓶颈导致转录不完整

### 8.2 滑动窗口方案

```
时间线：
[-------------- 已截断 (finalized_text) --------------][-- 当前窗口 (10s) --]
                                                        ↑ stable_text ↑ preview
```

- **finalized_text**：被截断时提交的累积文本，只增不减，存储在 session 内存中（纯文本，极小）
- **audio_accum**：只保留最近 `context_window_sec` 秒的音频
- **每次推理**：只处理窗口内的音频，延迟恒定
- **最终输出**：`finalized_text` + 当前窗口转录，通过 `_merge_finalized_with_window()` 智能合并

### 8.3 估算对比

| 录制时长 | 官方方案每次推理音频 | 本方案每次推理音频 |
|---------|--------------------|--------------------|
| 1 分钟   | 960K samples (60s) | 160K samples (10s) |
| 5 分钟   | 4.8M samples (300s) | 160K samples (10s) |
| 30 分钟  | OOM / 超时          | 160K samples (10s) |

## 9. 关键代码位置索引

| 功能 | 行号 |
|------|------|
| `ASRConfig` 配置数据类 | 45-64 |
| `StreamingSession` 会话状态 | 73-149 |
| 滑动窗口截断 `_apply_sliding_window()` | 119-149 |
| 重复检测 `detect_and_fix_repetitions()` | 155-218 |
| Energy VAD `_check_energy_vad()` | 221-225 |
| 输出解析 `_parse_asr_output()` | 231-244 |
| 文本去重/合并辅助函数 | 247-338 |
| `ASREngine` 推理引擎 | 345-468 |
| Token 回滚 `_rollback_tokens()` | 366-385 |
| 推理主逻辑 `transcribe()` | 387-435 |
| 文本稳定化 `_stabilize()` | 437-465 |
| WAV 编码 `_to_wav()` | 471-474 |
| 运行指标 `Metrics` | 480-491 |
| FastAPI 应用与路由 | 496-541 |
| WebSocket 主循环 `/ws/asr` | 544-804 |
| 外部 VAD 连接 | 572-605 |
| 音频接收与 VAD 门控 | 642-749 |
| Finish 处理 | 766-783 |
| 推理执行与发送 `_do_infer_and_send()` | 807-821 |
| AudioWorklet JS (PCMProcessor) | 827-896 |
| 前端 HTML/CSS/JS | 902-1217 |
| CLI 参数解析 | 1223-1280 |
| 启动入口 `main()` | 1283-1350 |
