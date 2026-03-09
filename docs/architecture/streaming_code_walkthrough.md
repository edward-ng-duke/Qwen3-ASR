# `qwen-asr-demo-streaming` 源码逐段解析

本文档对以下命令的完整执行链路进行源码级解析，按照**一段代码 → 一段解释**的格式组织。

```bash
qwen-asr-demo-streaming \
  --asr-model-path models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 6001
```

---

## 1. CLI 入口定义

```toml
# pyproject.toml

[project.scripts]
qwen-asr-demo = "qwen_asr.cli.demo:main"
qwen-asr-demo-streaming = "qwen_asr.cli.demo_streaming:main"
qwen-asr-serve = "qwen_asr.cli.serve:main"
```

**解释：** `pyproject.toml` 通过 `[project.scripts]` 注册了三个命令行工具。当你在终端执行 `qwen-asr-demo-streaming` 时，Python 的 entry point 机制会直接调用 `qwen_asr/cli/demo_streaming.py` 模块中的 `main()` 函数。这是整个程序的入口。

---

## 2. main() 函数 — 程序启动

```python
# qwen_asr/cli/demo_streaming.py

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

**解释：** `main()` 做了四件事：
1. 调用 `parse_args()` 解析命令行参数
2. 将流式相关参数（`unfixed_chunk_num`、`unfixed_token_num`、`chunk_size_sec`）存为全局变量，供后续 API 使用
3. 调用 `Qwen3ASRModel.LLM()` 用 vLLM 后端加载 ASR 模型（这是最耗时的步骤，会初始化 GPU、加载权重）
4. 启动 Flask Web 服务，`threaded=True` 表示每个请求用独立线程处理

注意 `max_new_tokens=32`：流式场景下每次只需生成少量 token（每 1 秒的音频只产生几个字），所以设得很小以加速推理。

---

## 3. 参数解析

```python
# qwen_asr/cli/demo_streaming.py

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

**解释：** 支持的参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-model-path` | `Qwen/Qwen3-ASR-1.7B` | 模型路径，可以是 HuggingFace repo 或本地目录 |
| `--host` | `0.0.0.0` | 绑定的网络地址，`0.0.0.0` 表示接受所有网卡的连接 |
| `--port` | `8000` | HTTP 端口（用户命令覆盖为 `6001`） |
| `--gpu-memory-utilization` | `0.8` | vLLM 预分配 GPU 显存的比例，`0.8` = 80% |
| `--unfixed-chunk-num` | `4` | 前 N 个 chunk 不使用前缀提示（从零开始识别） |
| `--unfixed-token-num` | `5` | 回滚最后 K 个 token 来减少边界抖动 |
| `--chunk-size-sec` | `1.0` | 每个推理 chunk 的时长（秒） |

---

## 4. 模型注册与 vLLM 初始化

```python
# qwen_asr/inference/qwen3_asr.py（模块顶层）

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except:
    pass
```

**解释：** Qwen3-ASR 是一个自定义模型架构，vLLM 默认不认识它。这段代码在模块导入时就将自定义模型类注册到 vLLM 的 `ModelRegistry` 中。这意味着**不能直接用 `vllm serve` 命令启动**，必须通过 `qwen-asr-*` 命令（它们会触发这个导入和注册）。`try/except` 保证了即使没装 vLLM 也不会报错。

---

## 5. Qwen3ASRModel.LLM() — vLLM 后端构造

```python
# qwen_asr/inference/qwen3_asr.py

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

    forced_aligner_model = None
    if forced_aligner is not None:
        forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
            forced_aligner, **(forced_aligner_kwargs or {})
        )

    return cls(
        backend="vllm",
        model=llm,
        processor=processor,
        sampling_params=sampling_params,
        forced_aligner=forced_aligner_model,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=None,
    )
```

**解释：** 这个类方法负责创建 vLLM 后端的 ASR 模型实例：

1. **延迟导入 vLLM**：`from vllm import LLM` 放在函数内部，使 vLLM 成为可选依赖
2. **创建 vLLM 引擎**：`vLLM(model=model, **kwargs)` 会加载模型权重到 GPU，初始化 KV cache。用户传入的 `gpu_memory_utilization=0.8` 就通过 `**kwargs` 传递给 vLLM
3. **加载 processor**：`Qwen3ASRProcessor` 负责文本 tokenize 和 chat template 渲染
4. **配置采样参数**：`temperature=0.0` 表示贪心解码（每次选概率最高的 token），`max_tokens=32`（由 `main()` 传入）
5. **封装返回**：将所有组件打包成 `Qwen3ASRModel` 实例，`backend="vllm"` 标记后端类型

---

## 6. Flask App 与全局状态

```python
# qwen_asr/cli/demo_streaming.py

app = Flask(__name__)

global asr
global UNFIXED_CHUNK_NUM
global UNFIXED_TOKEN_NUM
global CHUNK_SIZE_SEC

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60
```

**解释：**
- `app` 是 Flask 应用实例
- `asr` 是全局的 ASR 模型实例（在 `main()` 中赋值）
- `SESSIONS` 字典存放所有活跃的流式 session，key 是 session_id，value 是 Session 对象
- `SESSION_TTL_SEC = 600` 表示 session 超过 10 分钟没有活动就会被回收

---

## 7. Session 数据结构与 GC

```python
# qwen_asr/cli/demo_streaming.py

@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float


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
- `Session` 包含三个字段：`state`（ASR 流式状态对象）、`created_at`（创建时间）、`last_seen`（最后活跃时间）
- `_gc_sessions()` 是垃圾回收函数：遍历所有 session，超时的先调用 `finish_streaming_transcribe()` 释放资源，再从字典中移除
- `_get_session()` 在每次获取 session 时**顺便触发 GC**，并更新 `last_seen` 时间戳。这是一种"懒 GC"策略——不需要额外的定时器线程

---

## 8. API /api/start — 创建流式 Session

```python
# qwen_asr/cli/demo_streaming.py

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

**解释：** 当前端点击"开始"按钮时，会 POST 到这个接口。它做三件事：
1. 生成一个 UUID 作为 session_id
2. 调用 `asr.init_streaming_state()` 初始化流式识别状态
3. 将 session 存入全局字典，返回 session_id 给前端

---

## 9. init_streaming_state() — 初始化流式状态

```python
# qwen_asr/inference/qwen3_asr.py

def init_streaming_state(
    self,
    context: str = "",
    language: Optional[str] = None,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    chunk_size_sec: float = 2.0,
) -> ASRStreamingState:
    if self.backend != "vllm":
        raise ValueError("Streaming ASR is supported only for vLLM backend (backend='vllm').")
    if chunk_size_sec is None or float(chunk_size_sec) <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

    force_language = None
    if language is not None and str(language).strip() != "":
        ln = normalize_language_name(str(language))
        validate_language(ln)
        force_language = ln

    chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
    chunk_size_samples = max(1, chunk_size_samples)

    prompt_raw = self._build_text_prompt(context=context, force_language=force_language)

    return ASRStreamingState(
        unfixed_chunk_num=int(unfixed_chunk_num),
        unfixed_token_num=int(unfixed_token_num),
        chunk_size_sec=float(chunk_size_sec),
        chunk_size_samples=int(chunk_size_samples),
        chunk_id=0,
        buffer=np.zeros((0,), dtype=np.float32),
        audio_accum=np.zeros((0,), dtype=np.float32),
        prompt_raw=prompt_raw,
        context=context or "",
        force_language=force_language,
        language="",
        text="",
        _raw_decoded="",
    )
```

**解释：** 这个函数创建 `ASRStreamingState` 对象，其中关键初始化：

- **参数校验**：确认是 vLLM 后端，chunk 时长合法
- **语言处理**：如果指定了语言，规范化后存入 `force_language`，后续会在 prompt 中强制输出该语言
- **chunk 大小换算**：`chunk_size_sec=1.0` × `SAMPLE_RATE=16000` = `16000` 个采样点
- **构建基础 prompt**：`_build_text_prompt()` 用 chat template 生成 prompt 文本，类似 `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<audio>...<|im_end|>\n<|im_start|>assistant\n`
- **初始化空缓冲区**：`buffer`（待消费的新音频）和 `audio_accum`（历史累积音频）都是空的
- **初始化空输出**：`language=""`，`text=""`，`_raw_decoded=""`

---

## 10. ASRStreamingState — 流式状态字段详解

```python
# qwen_asr/inference/qwen3_asr.py

@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int        # 前 N 个 chunk 不用前缀提示
    unfixed_token_num: int        # 回滚最后 K 个 token
    chunk_size_sec: float         # 每个 chunk 的时长（秒）
    chunk_size_samples: int       # 每个 chunk 的采样点数

    chunk_id: int                 # 当前 chunk 编号（从 0 开始）
    buffer: np.ndarray            # 未消费的新音频缓冲区
    audio_accum: np.ndarray       # 从开始到当前的全部历史音频

    prompt_raw: str               # chat template 生成的基础 prompt
    context: str                  # 上下文字符串
    force_language: Optional[str] # 强制语言（None 表示自动检测）

    language: str                 # 最新识别出的语言
    text: str                     # 最新识别出的文本
    _raw_decoded: str             # 内部原始解码文本（用于回滚计算）
```

**解释：** 这是流式 ASR 的核心状态对象，贯穿整个 session 生命周期。重点理解：

- **`buffer` vs `audio_accum`**：`buffer` 是还没凑够一个 chunk 的零散音频；`audio_accum` 是从录音开始到现在的所有音频。每次推理都把 `audio_accum` 整体送入模型（这意味着推理耗时会随录音时长线性增长）
- **`_raw_decoded`**：存放模型的原始输出文本，用于 prefix rollback 计算。与 `text` 的区别是：`text` 经过了 `parse_asr_output()` 的解析和清洗，而 `_raw_decoded` 保留原始格式
- **`chunk_id`**：用于判断是否已经过了 `unfixed_chunk_num` 的阈值，决定是否启用 prefix 提示

---

## 11. 前端 — 录音与音频发送

```javascript
// qwen_asr/cli/demo_streaming.py（INDEX_HTML 中的 <script>）

const CHUNK_MS = 500;
const TARGET_SR = 16000;

// ... 获取麦克风
mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
    },
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

source.connect(processor);
processor.connect(audioCtx.destination);
```

**解释：** 前端的音频采集流程：

1. **获取麦克风**：通过 `getUserMedia()` 请求麦克风权限，指定单声道、开启回声消除和噪声抑制
2. **创建 AudioContext**：浏览器的音频处理环境（采样率通常是 44100Hz 或 48000Hz）
3. **ScriptProcessor**：每次从麦克风获取 4096 个采样点时触发回调
4. **重采样**：浏览器的原始采样率（如 48000Hz）通过线性插值降到 16000Hz（模型要求）
5. **累积缓冲**：重采样后的数据追加到 `buf` 中
6. **触发发送**：当 `buf` 中有足够数据且当前没有在发送时，调用 `pump()`

---

## 12. 前端 — 重采样函数

```javascript
function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++){
        const x = i / ratio;
        const x0 = Math.floor(x);
        const x1 = Math.min(x0 + 1, input.length - 1);
        const t = x - x0;
        out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
}
```

**解释：** 线性插值重采样。例如从 48000Hz → 16000Hz：
- `ratio = 16000/48000 = 1/3`，输出长度是输入的 1/3
- 对于输出的每个采样点 `i`，计算它在原始信号中的浮点位置 `x = i / ratio`
- 取 `x` 两侧的整数采样点 `x0` 和 `x1`，按距离做线性加权
- 这是一种简单但快速的重采样方法，对语音场景足够用

---

## 13. 前端 — pump() 发送循环

```javascript
async function pump(){
    if (pushing) return;
    pushing = true;

    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));  // 16000 * 0.5 = 8000

    try{
        while (running && buf.length >= chunkSamples){
            const chunk = buf.slice(0, chunkSamples);
            buf = buf.slice(chunkSamples);

            const j = await apiPushChunk(chunk);
            langEl.textContent = j.language || "—";
            textEl.textContent = j.text || "";
            if (running) setStatus("Listening… / 识别中…", "ok");
        }
    }catch(err){
        console.error(err);
        if (running) setStatus("Backend error / 后端错误: " + err.message, "err");
    }finally{
        pushing = false;
    }
}
```

**解释：** `pump()` 是前端的核心发送循环：
- **500ms 一个 chunk**：`CHUNK_MS=500`，`TARGET_SR=16000`，所以每个 chunk 是 8000 个采样点
- **循环消费缓冲区**：只要 `buf` 中有 ≥8000 个点，就切出一段发送
- **发送 chunk**：调用 `apiPushChunk()` 以 `application/octet-stream` 格式发送 float32 二进制数据
- **更新 UI**：收到后端返回的 `language` 和 `text` 后立即更新页面显示
- **互斥锁**：`pushing` 标志防止并发发送（因为 `onaudioprocess` 可能在 `pump()` 执行中再次触发）

注意：前端每 500ms 发一个 chunk，但后端的 `chunk_size_sec=1.0` 意味着需要攒两个前端 chunk 才触发一次模型推理。

---

## 14. 前端 — API 调用

```javascript
async function apiPushChunk(float32_16k){
    const r = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId), {
        method: "POST",
        headers: {"Content-Type":"application/octet-stream"},
        body: float32_16k.buffer
    });
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
}
```

**解释：** 关键细节：
- **Content-Type 是 `application/octet-stream`**：不是 JSON，而是原始二进制（float32 数组的底层 ArrayBuffer）
- **session_id 通过 URL 参数传递**
- **返回 JSON**：`{"language": "Chinese", "text": "你好世界"}`

---

## 15. API /api/chunk — 接收音频并推理

```python
# qwen_asr/cli/demo_streaming.py

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

**解释：** 这是流式识别的核心 API：

1. **获取 session**：从 URL 参数中取 session_id，通过 `_get_session()` 查找（同时触发 GC）
2. **验证格式**：确认 Content-Type 是 octet-stream，数据长度是 4 的倍数（float32 = 4 字节）
3. **解码音频**：`np.frombuffer()` 将二进制数据转为 numpy float32 数组
4. **调用推理**：`asr.streaming_transcribe(wav, s.state)` 把音频喂入模型，state 会被原地修改
5. **返回结果**：从 state 中读取最新的 `language` 和 `text` 返回给前端

---

## 16. streaming_transcribe() — 流式推理核心

```python
# qwen_asr/inference/qwen3_asr.py

def streaming_transcribe(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
    if self.backend != "vllm":
        raise ValueError("streaming_transcribe() is supported only for vLLM backend (backend='vllm').")

    # Ensure 1D mono
    x = np.asarray(pcm16k)
    if x.ndim != 1:
        x = x.reshape(-1)

    # Convert to float32 PCM in [-1, 1] if int16 provided
    if x.dtype == np.int16:
        x = (x.astype(np.float32) / 32768.0)
    else:
        x = x.astype(np.float32, copy=False)

    # Append to buffer
    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)
```

**解释：** 函数的前半部分做输入预处理：
- 确保数据是一维的 float32 数组
- 如果输入是 int16 格式（PCM 16-bit），除以 32768 转为 [-1, 1] 范围
- 将新音频追加到 `state.buffer` 中等待消费

---

## 17. streaming_transcribe() — Chunk 消费与 Prefix Rollback

```python
    # Consume full chunks
    while state.buffer.shape[0] >= state.chunk_size_samples:
        chunk = state.buffer[: state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples :]

        # Accumulate audio (re-feed from start, no padding)
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = chunk
        else:
            state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

        # Build prefix with rollback strategy
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
```

**解释：** 这是流式推理最核心的逻辑，分三步：

**步骤 1：切 chunk** — 从 buffer 中取出 `chunk_size_samples`（16000 个采样点 = 1 秒）的音频

**步骤 2：累积音频** — 将 chunk 追加到 `audio_accum`。注意每次推理都会把 `audio_accum`（从开始到现在的全部音频）整体送入模型，所以**推理耗时随录音时长线性增长**

**步骤 3：Prefix Rollback 策略** — 这是减少流式识别"边界抖动"的关键技巧：
- **前 N 个 chunk**（`chunk_id < unfixed_chunk_num=4`）：不使用前缀，让模型从零开始识别。因为初始阶段音频太短，前缀反而会误导模型
- **之后的 chunk**：将上一次的完整输出文本 `_raw_decoded` tokenize，**去掉最后 K 个 token**，剩余部分作为前缀。这样模型的输出会被引导为以前缀开头继续生成
- **为什么要回滚 K 个 token？** 因为流式音频在 chunk 边界处可能把一个字/词切断了，最后几个 token 的识别是不确定的。回滚后让模型重新决定这些 token，可以减少抖动
- **`\ufffd` 检查**：这是 Unicode 替换字符，表示 token 截断位置恰好在一个多字节字符的中间。遇到这种情况就多回滚一个 token，直到解码出干净的文本

---

## 18. streaming_transcribe() — vLLM 推理与输出更新

```python
        prompt = state.prompt_raw + prefix

        # vLLM input: single item
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

        outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
        gen_text = outputs[0].outputs[0].text

        # Accumulate raw decoded (then parse to lang/text)
        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text

        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt

        state.chunk_id += 1

    return state
```

**解释：**

1. **构建完整 prompt**：`prompt_raw`（chat template）+ `prefix`（前缀文本），拼接后模型会从 prefix 之后继续生成
2. **构建 vLLM 输入**：`multi_modal_data` 中的 `audio` 是一个列表，包含 `audio_accum`（全部历史音频）
3. **调用 vLLM 推理**：`model.generate()` 是同步阻塞的，返回生成的文本
4. **拼接结果**：`_raw_decoded = prefix + gen_text`，这就是当前对全部历史音频的完整识别结果
5. **解析输出**：`parse_asr_output()` 从原始文本中提取语言和文本
6. **递增 chunk_id**：计数器加一，用于下次判断是否使用 prefix

---

## 19. API /api/finish — 结束流式识别

```python
# qwen_asr/cli/demo_streaming.py

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

**解释：** 当用户点击"停止"按钮时：
1. 调用 `finish_streaming_transcribe()` 处理 buffer 中剩余的尾部音频
2. 返回最终的语言和文本
3. 从 SESSIONS 字典中移除这个 session

---

## 20. finish_streaming_transcribe() — 尾部音频处理

```python
# qwen_asr/inference/qwen3_asr.py

def finish_streaming_transcribe(self, state: ASRStreamingState) -> ASRStreamingState:
    if state.buffer is None or state.buffer.shape[0] == 0:
        return state

    tail = state.buffer
    state.buffer = np.zeros((0,), dtype=np.float32)

    # Append tail to accumulated audio
    if state.audio_accum.shape[0] == 0:
        state.audio_accum = tail
    else:
        state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

    # Prefix rollback strategy (same as per-chunk)
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

**解释：** 与 `streaming_transcribe()` 的区别：
- **不要求凑满一个 chunk**：把 buffer 中剩余的音频（不论多短）直接追加到 `audio_accum` 并送入模型
- **其他逻辑完全相同**：prefix rollback → 构建 prompt → vLLM 推理 → 解析结果
- 如果 buffer 为空（恰好在整 chunk 边界停止），直接返回不做推理

---

## 21. parse_asr_output() — 解析模型输出

```python
# qwen_asr/inference/utils.py

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "

def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        return "", s.strip()

    meta_lower = meta_part.lower()

    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        return "", t

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()
```

**解释：** 模型的原始输出格式为 `language Chinese<asr_text>你好世界`，这个函数负责解析：

1. **空值检查**：空输入返回 `("", "")`
2. **重复检测**：先调用 `detect_and_fix_repetitions()` 清理模型可能产生的重复输出
3. **强制语言模式**：如果用户指定了语言，模型输出就是纯文本，直接返回
4. **按 `<asr_text>` 标签分割**：前半部分是元信息（语言），后半部分是转写文本
5. **无标签情况**：整个输出当作纯文本返回，语言为空
6. **`language None` 特殊值**：模型认为是静音或空音频，返回 `("", "")`
7. **提取语言**：从元信息中找到 `language xxx` 行，规范化语言名称（首字母大写）

---

## 22. detect_and_fix_repetitions() — 模型重复输出修复

```python
# qwen_asr/inference/utils.py

def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return ''.join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text
```

**解释：** 语言模型有时会产生重复输出（如"的的的的的..."或"你好你好你好..."），这个函数做两层修复：

1. **`fix_char_repeats`**：单字符重复检测。如果同一个字符连续出现超过 `threshold=20` 次，只保留 1 个
2. **`fix_pattern_repeats`**：模式重复检测。检测长度 1~20 的子串，如果连续重复超过 20 次，只保留 1 次。例如 `"你好"` 重复 30 次 → 只保留 1 个 `"你好"`

这是一种简单但有效的后处理手段，防止模型"失控"输出大量重复内容。

---

## 23. _build_text_prompt() — Prompt 构建

```python
# qwen_asr/inference/qwen3_asr.py

def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
    msgs = self._build_messages(context=context, audio_payload="")
    base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}{'<asr_text>'}"
    return base

def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
    ]
```

**解释：** Prompt 构建分两步：

1. **构造 messages**：标准的 chat 格式，system 放上下文，user 放音频占位符
2. **apply_chat_template**：将 messages 渲染为模型需要的文本格式（加上 `<|im_start|>`/`<|im_end|>` 等特殊 token）
3. **强制语言**：如果指定了语言，在 prompt 末尾追加 `language Chinese<asr_text>`，这会引导模型直接输出该语言的转写文本，跳过语言检测步骤

---

## 完整执行流程图

```
用户执行命令
    │
    ▼
main() ─── parse_args() ─── Qwen3ASRModel.LLM() ─── app.run()
                                    │
                                    ├── vLLM(model=..., gpu_memory_utilization=0.8)
                                    ├── Qwen3ASRProcessor.from_pretrained()
                                    └── SamplingParams(temperature=0.0, max_tokens=32)

浏览器打开页面
    │
    ▼
GET / ─── 返回 INDEX_HTML（含 JS 录音逻辑）

用户点击"开始"
    │
    ▼
POST /api/start
    ├── uuid4() 生成 session_id
    ├── asr.init_streaming_state() → ASRStreamingState
    └── 返回 {"session_id": "..."}

浏览器开始录音（每 500ms）
    │
    ▼
POST /api/chunk（循环）
    ├── np.frombuffer() 解码 float32 音频
    ├── asr.streaming_transcribe(wav, state)
    │       ├── 追加到 buffer
    │       ├── buffer ≥ chunk_size_samples 时：
    │       │       ├── 追加到 audio_accum
    │       │       ├── Prefix Rollback（回滚最后 K 个 token）
    │       │       ├── vLLM generate（送入全部历史音频）
    │       │       ├── parse_asr_output() 解析语言和文本
    │       │       └── 更新 state.language / state.text
    │       └── 返回 state
    └── 返回 {"language": "...", "text": "..."}

用户点击"停止"
    │
    ▼
POST /api/finish
    ├── asr.finish_streaming_transcribe(state)
    │       ├── flush 剩余 buffer
    │       ├── 最后一次 vLLM 推理
    │       └── 更新最终结果
    ├── SESSIONS.pop(session_id)
    └── 返回最终 {"language": "...", "text": "..."}
```
