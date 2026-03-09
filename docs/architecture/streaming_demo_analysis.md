# `qwen-asr-demo-streaming` 源代码深度解析

## 概述

用户执行的命令：

```bash
qwen-asr-demo-streaming \
  --asr-model-path models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 6001
```

该命令启动一个基于 Flask 的流式语音识别 (ASR) Web 服务，使用 vLLM 作为推理后端，加载 Qwen3-ASR-1.7B 模型，在 `0.0.0.0:6001` 上提供实时语音转文字服务。

---

## 第一部分：CLI 入口点注册

```toml
# pyproject.toml (第 50-52 行)
[project.scripts]
qwen-asr-demo = "qwen_asr.cli.demo:main"
qwen-asr-demo-streaming = "qwen_asr.cli.demo_streaming:main"
qwen-asr-serve = "qwen_asr.cli.serve:main"
```

**解释：** `pyproject.toml` 的 `[project.scripts]` 段定义了 pip 安装后生成的可执行命令。`qwen-asr-demo-streaming` 映射到 `qwen_asr/cli/demo_streaming.py` 模块的 `main()` 函数。当你在终端输入 `qwen-asr-demo-streaming` 时，Python 会直接调用该函数。

---

## 第二部分：命令行参数解析

```python
# qwen_asr/cli/demo_streaming.py (第 472-482 行)
def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR Streaming Web Demo (vLLM backend)")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")

    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=1.0)
    return p.parse_args()
```

**解释：** 定义 7 个命令行参数：

| 参数 | 用户传的值 | 默认值 | 作用 |
|------|-----------|--------|------|
| `--asr-model-path` | `models/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-1.7B` | 本地模型路径（也可以是 HuggingFace 仓库 ID） |
| `--gpu-memory-utilization` | `0.8` | `0.8` | vLLM 使用 GPU 显存的比例，0.8 = 80% |
| `--host` | `0.0.0.0` | `0.0.0.0` | 绑定所有网络接口，允许外部访问 |
| `--port` | `6001` | `8000` | Web 服务监听端口 |
| `--unfixed-chunk-num` | 未指定 (默认 4) | `4` | 前 N 个音频块不使用前缀引导 |
| `--unfixed-token-num` | 未指定 (默认 5) | `5` | 回退 token 数量，减少边界抖动 |
| `--chunk-size-sec` | 未指定 (默认 1.0) | `1.0` | 每个音频块的长度（秒） |

---

## 第三部分：main() 函数 — 模型加载与服务启动

```python
# qwen_asr/cli/demo_streaming.py (第 485-503 行)
def main():
    args = parse_args()

    global asr
    global UNFIXED_CHUNK_NUM
    global UNFIXED_TOKEN_NUM
    global CHUNK_SIZE_SEC

    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec

    asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,
    )
    print("Model loaded.")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
```

**解释：**

1. **全局变量设置：** 将流式参数存为全局变量供 Flask 路由使用。
2. **模型加载 (`Qwen3ASRModel.LLM(...)`)：** 这是最关键的一步 — 使用 vLLM 后端加载 ASR 模型。`max_new_tokens=32` 表示每次生成最多 32 个 token（流式场景下每个 chunk 只需少量文本）。
3. **Flask 服务启动：** `threaded=True` 允许多线程处理并发请求，`use_reloader=False` 避免开发模式下重复加载模型。

---

## 第四部分：vLLM 后端初始化 — `Qwen3ASRModel.LLM()` 类方法

```python
# qwen_asr/inference/qwen3_asr.py (第 226-288 行)
@classmethod
def LLM(
    cls,
    model: str,
    forced_aligner: Optional[str] = None,
    forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
    max_inference_batch_size: int = -1,
    max_new_tokens: Optional[int] = 4096,
    **kwargs,
) -> "Qwen3ASRModel":
    try:
        from vllm import LLM as vLLM
        from vllm import SamplingParams
    except Exception as e:
        raise ImportError(
            "vLLM is not available. Install with: pip install qwen-asr[vllm]"
        ) from e

    llm = vLLM(model=model, **kwargs)

    processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
    sampling_params = SamplingParams(**({"temperature": 0.0, "max_tokens": max_new_tokens}))

    return cls(
        backend="vllm",
        model=llm,
        processor=processor,
        sampling_params=sampling_params,
        forced_aligner=None,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=None,
    )
```

**解释：**

1. **延迟导入 vLLM：** `from vllm import LLM` 放在方法内部，保证 vLLM 是可选依赖。
2. **创建 vLLM 引擎：** `vLLM(model=model, **kwargs)` 会加载模型权重到 GPU，`gpu_memory_utilization=0.8` 作为 kwargs 传入，表示 vLLM 最多使用 80% 的显存做 KV cache。
3. **加载 Processor：** `Qwen3ASRProcessor` 负责音频特征提取和文本编解码，`fix_mistral_regex=True` 修复 tokenizer 的正则匹配问题。
4. **采样参数：** `temperature=0.0` 表示贪心解码（确定性输出），`max_tokens=32` 限制单次推理生成的 token 数。

---

## 第五部分：会话管理

```python
# qwen_asr/cli/demo_streaming.py (第 38-64 行)
@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60

def _gc_sessions():
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception:
            pass
        SESSIONS.pop(sid, None)

def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s
```

**解释：**

- **Session 数据类：** 存储每个流式会话的 ASR 状态、创建时间和最后活跃时间。
- **SESSIONS 字典：** 以 UUID 为 key，存储所有活跃会话。
- **垃圾回收 (`_gc_sessions`)：** 每次获取会话时自动清理超过 10 分钟不活跃的会话。对于死掉的会话，先调用 `finish_streaming_transcribe` 释放模型资源。
- **`_get_session`：** 获取会话并更新最后活跃时间。

---

## 第六部分：API 端点 — `/api/start`

```python
# qwen_asr/cli/demo_streaming.py (第 417-427 行)
@app.post("/api/start")
def api_start():
    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})
```

**解释：** 客户端点击"开始"时调用此端点。流程：
1. 生成唯一会话 ID (UUID)
2. 初始化流式 ASR 状态（见下一部分详解）
3. 存入全局会话字典
4. 返回 `session_id` 给前端

---

## 第七部分：流式状态初始化 — `init_streaming_state()`

```python
# qwen_asr/inference/qwen3_asr.py (第 584-655 行)
def init_streaming_state(
    self,
    context: str = "",
    language: Optional[str] = None,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    chunk_size_sec: float = 2.0,
) -> ASRStreamingState:
    if self.backend != "vllm":
        raise ValueError("Streaming ASR is supported only for vLLM backend.")
    if chunk_size_sec is None or float(chunk_size_sec) <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

    chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))  # 1.0 * 16000 = 16000
    chunk_size_samples = max(1, chunk_size_samples)

    prompt_raw = self._build_text_prompt(context=context, force_language=None)

    return ASRStreamingState(
        unfixed_chunk_num=int(unfixed_chunk_num),  # 4
        unfixed_token_num=int(unfixed_token_num),  # 5
        chunk_size_sec=float(chunk_size_sec),       # 1.0
        chunk_size_samples=int(chunk_size_samples), # 16000
        chunk_id=0,
        buffer=np.zeros((0,), dtype=np.float32),    # 空缓冲区
        audio_accum=np.zeros((0,), dtype=np.float32), # 空累积音频
        prompt_raw=prompt_raw,                       # chat template 生成的基础 prompt
        context="",
        force_language=None,
        language="",
        text="",
        _raw_decoded="",
    )
```

**解释：**

- **SAMPLE_RATE = 16000：** 音频采样率固定为 16kHz。
- **chunk_size_samples = 16000：** 1 秒 × 16000 采样率 = 16000 个采样点。每积累 16000 个采样点触发一次模型推理。
- **buffer：** 暂存还不够一个完整 chunk 的零散音频数据。
- **audio_accum：** 从流开始至今的所有音频累积。每次推理都会把**全部**历史音频送入模型（不是只送当前 chunk）。
- **prompt_raw：** 由 chat template 生成的基础文本 prompt，格式类似 `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\naudio_placeholder<|im_end|>\n<|im_start|>assistant\n`。
- **_raw_decoded：** 累积的原始解码文本，用于构造"前缀引导"。

---

## 第八部分：ASRStreamingState 数据结构

```python
# qwen_asr/inference/qwen3_asr.py (第 77-128 行)
@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int      # 前 N 个 chunk 不使用前缀 (默认 4)
    unfixed_token_num: int      # 回退的 token 数量 (默认 5)
    chunk_size_sec: float       # chunk 大小（秒）
    chunk_size_samples: int     # chunk 大小（采样点数）

    chunk_id: int               # 当前 chunk 编号（0 起始）
    buffer: np.ndarray          # 未消费的缓冲音频
    audio_accum: np.ndarray     # 累积的全部音频

    prompt_raw: str             # 基础 prompt（chat template）
    context: str                # 上下文文本
    force_language: Optional[str]  # 强制语言（None 则自动检测）

    language: str               # 最新解析出的语言
    text: str                   # 最新解析出的文本
    _raw_decoded: str           # 内部累积的原始解码输出
```

**解释：** 这是整个流式 ASR 的核心状态对象。它是可变的（mutable），每次调用 `streaming_transcribe()` 都会原地修改。关键设计：
- **buffer vs audio_accum：** buffer 是暂存区（不够一个 chunk 的零头），audio_accum 是已经送入模型的完整历史音频。
- **_raw_decoded：** 存储模型的原始输出（包含 `language Chinese<asr_text>...` 格式），用于回退和前缀构造。
- **language/text：** 供外部读取的"干净"结果，由 `parse_asr_output()` 从 `_raw_decoded` 中解析得到。

---

## 第九部分：API 端点 — `/api/chunk`（核心流式推理）

```python
# qwen_asr/cli/demo_streaming.py (第 430-453 行)
@app.post("/api/chunk")
def api_chunk():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    asr.streaming_transcribe(wav, s.state)

    return jsonify(
        {
            "language": getattr(s.state, "language", "") or "",
            "text": getattr(s.state, "text", "") or "",
        }
    )
```

**解释：**

1. **音频接收：** 前端每 500ms 发送一段 float32 PCM 音频数据（`application/octet-stream` 格式）。
2. **数据校验：** 检查字节长度是否是 4 的倍数（float32 每个样本 4 字节）。
3. **调用 `streaming_transcribe`：** 将音频数据和会话状态传入（见下一部分详解）。
4. **返回结果：** 直接从 state 对象读取最新的 language 和 text 返回给前端。

---

## 第十部分：流式转录核心 — `streaming_transcribe()`

```python
# qwen_asr/inference/qwen3_asr.py (第 657-765 行)
def streaming_transcribe(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
    # 确保输入是 1D float32
    x = np.asarray(pcm16k)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.dtype == np.int16:
        x = (x.astype(np.float32) / 32768.0)
    else:
        x = x.astype(np.float32, copy=False)

    # 追加到缓冲区
    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)

    # 消费完整的 chunk
    while state.buffer.shape[0] >= state.chunk_size_samples:
        chunk = state.buffer[: state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples :]

        # 累积音频（从头开始重新喂入，无 padding）
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = chunk
        else:
            state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

        # 构造前缀：回退策略
        prefix = ""
        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            k = int(state.unfixed_token_num)
            while True:
                end_idx = max(0, len(cur_ids) - k)
                prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                if '\ufffd' not in prefix:
                    break
                else:
                    if end_idx == 0:
                        prefix = ""
                        break
                    k += 1

        prompt = state.prompt_raw + prefix

        # vLLM 推理
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}
        outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
        gen_text = outputs[0].outputs[0].text

        # 更新状态
        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt

        state.chunk_id += 1

    return state
```

**解释：** 这是整个流式 ASR 最核心的函数，逐步分析：

### 1. 音频预处理（第 704-713 行）
- 将输入转为 1D float32 数组，支持 int16 → float32 转换（除以 32768 归一化到 [-1, 1]）。

### 2. 缓冲与分块（第 715-722 行）
- 新音频追加到 `state.buffer`。
- 当缓冲区 >= `chunk_size_samples` (16000 个样本 = 1 秒) 时，取出一个完整 chunk 处理。
- **关键：** 每个 chunk 不是独立送入模型的，而是**累加到 `audio_accum`**，然后把从开头到当前的**全部音频**一起送入模型。这确保了模型有完整的上下文。

### 3. 前缀回退策略（第 730-746 行）— **流式 ASR 的精髓**

这是保证流式输出稳定性的核心机制：

- **前 4 个 chunk（chunk_id < 4）：** prefix = ""，即不给模型任何文本前缀引导。因为初始阶段音频太短，模型的输出不稳定，不适合作为引导。
- **第 5 个 chunk 起（chunk_id >= 4）：**
  - 取之前累积的原始解码文本 `_raw_decoded`
  - 用 tokenizer 编码为 token IDs
  - **回退最后 5 个 token**（去掉末尾可能不稳定的部分）
  - 将剩余 token 解码回文本作为 prefix
  - 检查是否包含 `\ufffd`（Unicode 替换字符，表示不完整的多字节字符），如果有就多回退一些

**为什么要回退？** 流式 ASR 中，每次推理只看到部分音频，模型对末尾部分的预测可能不准确。通过回退最后几个 token，让模型在新一轮推理时有机会"修正"之前的猜测。

### 4. vLLM 推理（第 750-754 行）
- 构造输入：`prompt`（基础 prompt + 前缀文本）+ `multi_modal_data`（全部累积音频）
- 调用 `self.model.generate()` 进行单次推理
- 提取生成文本 `gen_text`

### 5. 状态更新（第 756-763 行）
- `_raw_decoded = prefix + gen_text`：前缀 + 新生成文本 = 完整的累积解码结果
- `parse_asr_output()` 从原始文本中解析出语言和干净文本
- 更新 `state.language` 和 `state.text`
- chunk_id 自增

---

## 第十一部分：API 端点 — `/api/finish`

```python
# qwen_asr/cli/demo_streaming.py (第 456-469 行)
@app.post("/api/finish")
def api_finish():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    asr.finish_streaming_transcribe(s.state)
    out = {
        "language": getattr(s.state, "language", "") or "",
        "text": getattr(s.state, "text", "") or "",
    }
    SESSIONS.pop(session_id, None)
    return jsonify(out)
```

**解释：** 用户点击"停止"后调用。flush 缓冲区中剩余不满一个 chunk 的音频（尾音），做最后一次推理，然后清理会话。

---

## 第十二部分：尾音处理 — `finish_streaming_transcribe()`

```python
# qwen_asr/inference/qwen3_asr.py (第 767-829 行)
def finish_streaming_transcribe(self, state: ASRStreamingState) -> ASRStreamingState:
    if state.buffer is None or state.buffer.shape[0] == 0:
        return state

    tail = state.buffer
    state.buffer = np.zeros((0,), dtype=np.float32)

    # 尾音追加到累积音频
    if state.audio_accum.shape[0] == 0:
        state.audio_accum = tail
    else:
        state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

    # 同样的前缀回退策略
    prefix = ""
    if state.chunk_id < state.unfixed_chunk_num:
        prefix = ""
    else:
        cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
        end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
        prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])

    prompt = state.prompt_raw + prefix
    inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

    outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
    gen_text = outputs[0].outputs[0].text

    state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang
    state.text = txt

    state.chunk_id += 1
    return state
```

**解释：** 逻辑与 `streaming_transcribe()` 几乎一致，区别是：
- 不做 chunk 分块，直接把 buffer 中剩余的所有音频（不管多短）一次性追加
- 这确保了最后几百毫秒的语音也被模型处理，不会丢失末尾内容

---

## 第十三部分：前端 Web 界面（JavaScript 关键逻辑）

```javascript
// qwen_asr/cli/demo_streaming.py 内嵌 HTML (第 211-405 行)
const CHUNK_MS = 500;
const TARGET_SR = 16000;

// 麦克风采集
mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    video: false
});

audioCtx = new (window.AudioContext || window.webkitAudioContext)();
source = audioCtx.createMediaStreamSource(mediaStream);
processor = audioCtx.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
    if (!running) return;
    const input = e.inputBuffer.getChannelData(0);
    const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
    buf = concatFloat32(buf, resampled);
    if (!pushing) pump();
};

// pump: 每 500ms 发送一次音频
async function pump(){
    pushing = true;
    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000)); // 8000 个采样点

    while (running && buf.length >= chunkSamples){
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);

        const j = await apiPushChunk(chunk);
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
    }
    pushing = false;
}
```

**解释：**

1. **麦克风采集：** 使用 WebAudio API 获取单声道音频，开启回声消除、噪声抑制、自动增益。
2. **重采样：** 浏览器的采样率通常是 44100Hz 或 48000Hz，需要线性插值重采样到 16000Hz（模型要求）。
3. **分块发送：** 每 500ms 收集 `16000 × 0.5 = 8000` 个采样点，以 float32 二进制格式 POST 到 `/api/chunk`。
4. **实时更新：** 收到服务端返回的 language 和 text 后立即更新页面显示。

---

## 第十四部分：完整数据流示意

```
浏览器麦克风 (48kHz)
    │ resampleLinear → 16kHz
    ▼
前端 buffer (Float32Array)
    │ 每 500ms 取 8000 个采样点
    ▼
POST /api/chunk (application/octet-stream, 32000 bytes)
    │ np.frombuffer → float32 数组
    ▼
streaming_transcribe()
    │
    ├── state.buffer 追加新音频
    │
    ├── buffer >= 16000? (1秒的 chunk)
    │   ├── 是 → 取出 chunk, 追加到 audio_accum
    │   │       ├── chunk_id < 4?  → prefix = ""
    │   │       └── chunk_id >= 4? → prefix = 回退5个token后的文本
    │   │
    │   │   prompt = prompt_raw + prefix
    │   │   vLLM.generate(prompt, audio=全部累积音频)
    │   │       → "language Chinese<asr_text>你好世界"
    │   │       → parse → language="Chinese", text="你好世界"
    │   │
    │   └── 否 → 等待更多音频
    │
    ▼
返回 {"language": "Chinese", "text": "你好世界"}
    │
    ▼
前端更新显示
```

**注意：** 前端每 500ms 发送 0.5 秒音频，但后端 `chunk_size_sec=1.0`，所以每 **2 次** `/api/chunk` 请求才会触发一次模型推理。第 1 次请求只是把数据存入 buffer，第 2 次请求时 buffer 够 16000 个采样点了才会触发推理。

---

## 总结

| 组件 | 文件 | 作用 |
|------|------|------|
| CLI 入口 | `pyproject.toml` | 注册 `qwen-asr-demo-streaming` 命令 |
| 参数解析 + Flask 服务 | `qwen_asr/cli/demo_streaming.py` | Web 服务器、会话管理、3 个 API 端点 |
| 模型封装 + 流式推理 | `qwen_asr/inference/qwen3_asr.py` | `Qwen3ASRModel` 类，流式状态管理，前缀回退策略 |
| vLLM 模型实现 | `qwen_asr/core/vllm_backend/qwen3_asr.py` | 音频编码器 + 语言模型的 vLLM 注册 |
| 音频处理工具 | `qwen_asr/inference/utils.py` | 音频归一化、输出解析 |
| 前端界面 | 内嵌在 `demo_streaming.py` | 麦克风采集、重采样、分块发送 |
