# Qwen3-ASR 统一推理服务开发计划

## 1. 核心思路

vLLM 每次推理都是无状态的。流式识别的本质是**累积音频 + 重新推理 + 前缀稳定化**，状态在服务端维护而非 vLLM 内部。因此可以用一个 FastAPI 服务统一暴露离线和实时两种 API，共享同一个 vLLM 模型实例。

### 现有代码分析

当前项目有三种独立的服务方式，各自加载模型，互不复用：

| 现有方式 | 入口 | 能力 | 局限 |
|---------|------|------|------|
| `qwen-asr-serve` | `cli/serve.py` | OpenAI 兼容 REST API（离线） | 不支持流式，只是 vLLM serve 的薄封装 |
| `qwen-asr-demo` | `cli/demo.py` | Gradio Web UI | 交互式界面，非生产 API |
| `qwen-asr-demo-streaming` | `cli/demo_streaming.py` | Flask + REST 流式 | 只有流式，不支持批量离线 |

**统一服务的目标**：一个进程、一次模型加载，同时提供离线批量和实时流式两种推理能力。

### 关键 API（来自 `qwen_asr` 包，无需重新实现）

```python
from qwen_asr import Qwen3ASRModel

# 加载模型（一次）
asr = Qwen3ASRModel.LLM(model="models/Qwen3-ASR-1.7B", gpu_memory_utilization=0.8)

# 离线批量转写
results = asr.transcribe(audio=[...], language=None, return_time_stamps=True)

# 流式转写
state = asr.init_streaming_state(unfixed_chunk_num=4, unfixed_token_num=5, chunk_size_sec=1.0)
asr.streaming_transcribe(audio_chunk, state)  # 每次送入一段音频，state.text 实时更新
asr.finish_streaming_transcribe(state)         # 结束，刷出尾部缓冲
```

## 2. 架构设计

```
┌──────────────────────────────────────────────────────┐
│              FastAPI 统一服务 (uvicorn)                │
│                                                      │
│  REST API                      WebSocket API         │
│  ─────────                     ──────────────        │
│  POST /v1/transcribe           WS /v1/stream         │
│  POST /v1/transcribe/upload                          │
│  POST /api/start               GET /health           │
│  POST /api/chunk               GET /v1/languages     │
│  POST /api/finish                                    │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              InferenceEngine                     │ │
│  │    asyncio.Lock 序列化 + asyncio.to_thread 异步  │ │
│  │    batch_sampling_params (max_tokens=4096)       │ │
│  │    streaming_sampling_params (max_tokens=32)     │ │
│  └────────────────────┬────────────────────────────┘ │
│                       │                              │
│  ┌────────────────────▼────────────────────────────┐ │
│  │         Qwen3ASRModel (vLLM backend)            │ │
│  │    .transcribe()                                │ │
│  │    .streaming_transcribe()                      │ │
│  │    .finish_streaming_transcribe()               │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │         SessionManager                          │ │
│  │    流式会话生命周期管理 + TTL 自动清理             │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 技术选型

- **FastAPI**：原生 async + WebSocket 支持，自动 OpenAPI 文档，且已是 vLLM 的依赖无需额外安装
- **包装 `Qwen3ASRModel`** 而非直接用 vLLM raw API：复用已有的 prompt 构建、音频归一化、长音频切分、输出解析、流式状态机，避免重复实现和维护

### 并发模型

`Qwen3ASRModel` 内部调用 `vllm.LLM.generate()` 是**同步阻塞**的，且**非线程安全**。方案：

```python
class InferenceEngine:
    def __init__(self, asr: Qwen3ASRModel):
        self._asr = asr
        self._lock = asyncio.Lock()          # 序列化 GPU 访问

    async def transcribe(self, ...):
        async with self._lock:               # 排队等锁
            self._asr.sampling_params = self._batch_params
            return await asyncio.to_thread(   # 不阻塞事件循环
                self._asr.transcribe, ...
            )

    async def streaming_transcribe(self, pcm16k, state):
        async with self._lock:
            self._asr.sampling_params = self._streaming_params
            return await asyncio.to_thread(
                self._asr.streaming_transcribe, pcm16k, state
            )
```

效果：
- 多个 WebSocket 连接和 REST 请求**同时存在**，推理**排队执行**
- 事件循环保持响应（接收新连接、WebSocket 消息、健康检查等）
- 批量请求内多个音频打包为一次 `transcribe()` 调用，vLLM 内部做 batch

### SamplingParams 切换

模型加载时用 `max_new_tokens=4096`（满足离线长音频需求）。流式推理每次只生成少量 token，因此在锁内临时切换 `sampling_params.max_tokens=32`。由于锁保证了互斥，这是安全的。

## 3. 文件结构

```
qwen_asr/
  service/                            # ← 新增 package
    __init__.py
    config.py                         # ServiceConfig (Pydantic Settings)
    engine.py                         # InferenceEngine (async Lock + thread offload)
    session_manager.py                # StreamingSession 生命周期管理
    audio_utils.py                    # 音频解码 (base64/upload/binary PCM)
    models.py                         # Pydantic 请求/响应 schema
    app.py                            # FastAPI app 工厂 + lifespan
    routes/
      __init__.py
      health.py                       # GET /health, GET /v1/languages
      batch.py                        # POST /v1/transcribe, /v1/transcribe/upload
      streaming.py                    # WS /v1/stream + REST /api/start|chunk|finish
  cli/
    serve_unified.py                  # ← 新增 CLI 入口

examples/
  example_unified_batch.py            # ← 新增客户端示例
  example_unified_streaming_ws.py     # ← 新增客户端示例

pyproject.toml                        # ← 修改：添加入口点
```

## 4. 各模块详细设计

### 4.1 config.py — 服务配置

```python
from typing import Optional
from pydantic_settings import BaseSettings

class ServiceConfig(BaseSettings):
    # 模型
    model_path: str                          # 模型路径或 HuggingFace repo id
    gpu_memory_utilization: float = 0.8
    max_new_tokens_batch: int = 4096         # 离线推理 max_tokens
    max_new_tokens_streaming: int = 32       # 流式推理 max_tokens
    max_inference_batch_size: int = 32

    # Forced Aligner（可选，提供时间戳能力）
    forced_aligner_path: Optional[str] = None
    forced_aligner_dtype: str = "bfloat16"
    forced_aligner_device: str = "cuda:0"

    # 流式默认参数
    default_unfixed_chunk_num: int = 4       # 前 N 个 chunk 不使用前缀
    default_unfixed_token_num: int = 5       # 前缀回退 token 数
    default_chunk_size_sec: float = 1.0      # chunk 时长（秒）

    # 会话管理
    session_ttl_sec: int = 600               # 10 分钟超时
    session_gc_interval_sec: int = 60        # GC 扫描间隔
    max_concurrent_sessions: int = 100

    # 服务器
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_prefix = "QWEN_ASR_"             # 支持环境变量覆盖
```

### 4.2 engine.py — 推理引擎

封装 `Qwen3ASRModel`，提供异步接口：

| 方法 | 说明 | 是否需要锁 |
|------|------|-----------|
| `async transcribe(audio, context, language, return_time_stamps)` | 离线批量转写 | 是 |
| `init_streaming_state(**kwargs)` | 初始化流式状态（纯 CPU 操作） | 否 |
| `async streaming_transcribe(pcm16k, state)` | 单次流式步进 | 是 |
| `async finish_streaming(state)` | 流式结束，刷出缓冲 | 是 |
| `get_supported_languages()` | 返回支持语言列表 | 否 |

关键实现细节：
- 在锁内切换 `sampling_params`（batch 用 4096 tokens，streaming 用 32 tokens）
- `asyncio.to_thread()` 将阻塞的 `generate()` 调用放到线程池

### 4.3 session_manager.py — 会话管理器

参照现有 `demo_streaming.py`（`cli/demo_streaming.py:38-73`）的 Session 模式，升级为异步版本：

```python
@dataclass
class StreamingSession:
    session_id: str
    state: ASRStreamingState     # 来自 qwen_asr.inference.qwen3_asr
    created_at: float
    last_activity: float

class SessionManager:
    async def create(state) -> str           # 创建会话，返回 session_id
    async def get(session_id) -> Session     # 获取会话（更新 last_activity）
    async def remove(session_id) -> Session  # 移除会话
    async def start_gc_loop()                # 启动后台 GC 任务
    async def stop_gc_loop()                 # 停止 GC
```

GC 逻辑：定期扫描超时会话 → 调用 `engine.finish_streaming(state)` 释放资源 → 从字典中移除。

### 4.4 audio_utils.py — 音频解码工具

| 函数 | 输入 | 输出 | 场景 |
|------|------|------|------|
| `decode_upload_file(UploadFile)` | FastAPI 文件上传 | `(np.ndarray, sr)` | `/v1/transcribe/upload` |
| `decode_base64_audio(str)` | base64 字符串 | `(np.ndarray, sr)` | `/v1/transcribe` JSON body |
| `decode_binary_pcm(bytes)` | 原始 float32 字节 | `np.ndarray` | 流式 `/api/chunk` |

内部统一使用 `soundfile.read()` 解码，输出 float32 numpy 数组。

### 4.5 models.py — Pydantic 请求/响应模型

```python
# --- 离线转写 ---

class TranscribeRequest(BaseModel):
    audio: list                              # URL 列表、base64 列表、或混合
    context: str | list[str] = ""
    language: Optional[str | list[Optional[str]]] = None
    return_time_stamps: bool = False

class TimestampItem(BaseModel):
    text: str
    start_time: float
    end_time: float

class TranscriptionResult(BaseModel):
    language: str
    text: str
    time_stamps: Optional[list[TimestampItem]] = None

class TranscribeResponse(BaseModel):
    results: list[TranscriptionResult]

# --- 流式 ---

class StreamStartRequest(BaseModel):
    language: Optional[str] = None
    context: str = ""
    unfixed_chunk_num: Optional[int] = None
    unfixed_token_num: Optional[int] = None
    chunk_size_sec: Optional[float] = None

class StreamStartResponse(BaseModel):
    session_id: str

class StreamChunkResponse(BaseModel):
    language: str
    text: str
    chunk_id: int
    is_final: bool = False

# --- 健康检查 ---

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    active_sessions: int
    supported_languages: list[str]
```

### 4.6 routes/health.py — 健康检查路由

| 端点 | 说明 |
|------|------|
| `GET /health` | 返回模型名称、活跃会话数、支持语言列表 |
| `GET /v1/languages` | 返回 52 种支持语言列表 |

### 4.7 routes/batch.py — 离线转写路由

**端点 1：`POST /v1/transcribe`**（JSON body）

接受 URL 或 base64 音频列表，批量转写。支持 context、language、return_time_stamps 参数。多个音频会打包为一次 `engine.transcribe()` 调用。

请求示例：
```json
{
  "audio": [
    "https://example.com/audio1.wav",
    "data:audio/wav;base64,UklGR..."
  ],
  "language": "Chinese",
  "return_time_stamps": false
}
```

响应示例：
```json
{
  "results": [
    {"language": "Chinese", "text": "你好世界", "time_stamps": null},
    {"language": "Chinese", "text": "测试音频", "time_stamps": null}
  ]
}
```

**端点 2：`POST /v1/transcribe/upload`**（multipart 文件上传）

接受单个音频文件上传，参数通过 Form 字段传递。

```bash
curl -X POST http://localhost:8000/v1/transcribe/upload \
  -F "file=@audio.wav" \
  -F "language=Chinese" \
  -F "return_time_stamps=false"
```

### 4.8 routes/streaming.py — 流式转写路由

提供两种流式接口：WebSocket（推荐）和 REST（兼容现有客户端）。

#### WebSocket 接口：`WS /v1/stream`

协议流程：

```
客户端                              服务端
  │                                    │
  │──── JSON {"type":"start", ...} ───→│  创建 session + state
  │←── JSON {"type":"started",    ────│
  │         "session_id":"xxx"}        │
  │                                    │
  │──── binary (float32 PCM 16kHz) ──→│  streaming_transcribe()
  │←── JSON {"type":"partial",    ────│
  │         "text":"你好", ...}        │
  │                                    │
  │──── binary ──────────────────────→│  streaming_transcribe()
  │←── JSON {"type":"partial",    ────│
  │         "text":"你好世界", ...}     │
  │          ...                       │
  │                                    │
  │──── JSON {"type":"finish"} ──────→│  finish_streaming_transcribe()
  │←── JSON {"type":"final",     ────│
  │         "text":"你好世界。", ...}   │
  │                                    │
  │←── WebSocket close ──────────────│  清理 session
```

start 消息可选参数：
```json
{
  "type": "start",
  "language": "Chinese",
  "context": "",
  "unfixed_chunk_num": 4,
  "unfixed_token_num": 5,
  "chunk_size_sec": 1.0
}
```

binary 帧格式：float32 little-endian PCM，16kHz 采样率，单声道。

异常处理：客户端断开时自动调用 `finish_streaming` 清理。

#### REST 兼容接口（兼容现有 `demo_streaming.py` 前端）

| 端点 | 方法 | body | 说明 |
|------|------|------|------|
| `POST /api/start` | JSON | `StreamStartRequest` | 创建会话，返回 `session_id` |
| `POST /api/chunk?session_id=xxx` | binary | float32 PCM bytes | 发送音频块，返回当前识别结果 |
| `POST /api/finish?session_id=xxx` | 无 | — | 结束会话，返回最终结果 |

### 4.9 app.py — FastAPI 应用工厂

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config

    # ── 启动 ──
    asr = Qwen3ASRModel.LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_inference_batch_size=config.max_inference_batch_size,
        max_new_tokens=config.max_new_tokens_batch,
        forced_aligner=config.forced_aligner_path,
        ...
    )
    app.state.engine = InferenceEngine(asr, streaming_max_tokens=config.max_new_tokens_streaming)
    app.state.session_manager = SessionManager(...)
    await app.state.session_manager.start_gc_loop()

    yield

    # ── 关闭 ──
    await app.state.session_manager.stop_gc_loop()
    # 清理所有存活会话

def create_app(config: ServiceConfig) -> FastAPI:
    app = FastAPI(title="Qwen3-ASR Unified Service", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.include_router(health.router)
    app.include_router(batch.router)
    app.include_router(streaming.router)
    return app
```

### 4.10 cli/serve_unified.py — CLI 入口

```python
def main():
    parser = argparse.ArgumentParser(prog="qwen-asr-serve-unified")
    parser.add_argument("--model", required=True)
    parser.add_argument("--forced-aligner", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens-batch", type=int, default=4096)
    parser.add_argument("--max-new-tokens-streaming", type=int, default=32)
    parser.add_argument("--session-ttl", type=int, default=600)
    parser.add_argument("--max-sessions", type=int, default=100)
    args = parser.parse_args()

    config = ServiceConfig(...)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, workers=1)  # 必须单 worker
```

启动命令：
```bash
qwen-asr-serve-unified --model models/Qwen3-ASR-1.7B --port 8000
```

## 5. API 总览

| 端点 | 方法 | 用途 |
|------|------|------|
| `GET /health` | GET | 健康检查、模型信息、活跃会话数 |
| `GET /v1/languages` | GET | 支持的 52 种语言列表 |
| `POST /v1/transcribe` | POST (JSON) | 离线批量转写 (URL/base64) |
| `POST /v1/transcribe/upload` | POST (multipart) | 离线文件上传转写 |
| `WS /v1/stream` | WebSocket | 实时流式转写（推荐） |
| `POST /api/start` | POST (JSON) | REST 流式：创建会话 |
| `POST /api/chunk` | POST (binary) | REST 流式：发送音频块 |
| `POST /api/finish` | POST | REST 流式：结束会话 |

## 6. 分步实施计划

### 第 1 步：基础设施层

创建 `qwen_asr/service/` 包，实现以下模块：

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包标记 |
| `config.py` | Pydantic Settings 配置模型 |
| `models.py` | 请求/响应 Pydantic schema |
| `audio_utils.py` | 音频解码工具函数 |
| `engine.py` | InferenceEngine 异步推理封装 |
| `session_manager.py` | 流式会话生命周期管理 |

### 第 2 步：路由层

| 文件 | 职责 |
|------|------|
| `routes/__init__.py` | 包标记 |
| `routes/health.py` | 健康检查和元数据端点 |
| `routes/batch.py` | 离线批量转写端点 |
| `routes/streaming.py` | WebSocket + REST 流式端点 |

### 第 3 步：应用组装

| 文件 | 职责 |
|------|------|
| `app.py` | FastAPI 工厂 + lifespan（模型加载/清理） |
| `cli/serve_unified.py` | CLI 入口（argparse + uvicorn） |

### 第 4 步：集成注册

修改 `pyproject.toml`：

```toml
[project.scripts]
qwen-asr-demo = "qwen_asr.cli.demo:main"
qwen-asr-demo-streaming = "qwen_asr.cli.demo_streaming:main"
qwen-asr-serve = "qwen_asr.cli.serve:main"
qwen-asr-serve-unified = "qwen_asr.cli.serve_unified:main"   # ← 新增
```

### 第 5 步：客户端示例

| 文件 | 说明 |
|------|------|
| `examples/example_unified_batch.py` | 用 requests/httpx 调用离线 API |
| `examples/example_unified_streaming_ws.py` | 用 websockets 库调用 WebSocket 流式 API |

## 7. 错误处理策略

| 场景 | HTTP 状态码 | 处理方式 |
|------|------------|---------|
| 请求参数校验失败 | 400/422 | Pydantic 自动校验 |
| 音频格式错误/解码失败 | 400 | 捕获异常，返回描述性错误 |
| 不支持的语言 | 422 | `validate_language()` 抛出 ValueError |
| 流式会话不存在 | 404 | session_id 找不到 |
| 并发会话数超限 | 503 | SessionManager 拒绝创建 |
| GPU/推理错误 | 500 | 捕获异常，返回错误信息 |
| WebSocket 异常断开 | — | 自动调用 `finish_streaming` 清理 |
| 会话 TTL 超时 | — | 后台 GC 自动清理并调用 `finish_streaming` |

## 8. 验证方式

1. **启动服务**
   ```bash
   qwen-asr-serve-unified --model models/Qwen3-ASR-1.7B --port 8000
   ```

2. **健康检查**
   ```bash
   curl http://localhost:8000/health
   ```

3. **离线转写测试**
   ```bash
   # 文件上传
   curl -X POST http://localhost:8000/v1/transcribe/upload \
     -F "file=@data/2news_30s.mp3"

   # JSON body (URL)
   curl -X POST http://localhost:8000/v1/transcribe \
     -H "Content-Type: application/json" \
     -d '{"audio": ["https://example.com/test.wav"]}'
   ```

4. **流式转写测试**
   ```bash
   python examples/example_unified_streaming_ws.py
   ```

5. **REST 流式兼容测试**
   ```bash
   # 创建会话
   curl -X POST http://localhost:8000/api/start

   # 发送音频块
   curl -X POST "http://localhost:8000/api/chunk?session_id=xxx" \
     -H "Content-Type: application/octet-stream" --data-binary @chunk.pcm

   # 结束会话
   curl -X POST "http://localhost:8000/api/finish?session_id=xxx"
   ```

6. **并发测试**：同时发起多个请求，验证锁机制和会话隔离

## 9. 需要创建/修改的完整文件列表

| 文件 | 操作 | 预估代码量 |
|------|------|-----------|
| `qwen_asr/service/__init__.py` | 新建 | ~5 行 |
| `qwen_asr/service/config.py` | 新建 | ~40 行 |
| `qwen_asr/service/models.py` | 新建 | ~60 行 |
| `qwen_asr/service/audio_utils.py` | 新建 | ~40 行 |
| `qwen_asr/service/engine.py` | 新建 | ~70 行 |
| `qwen_asr/service/session_manager.py` | 新建 | ~80 行 |
| `qwen_asr/service/routes/__init__.py` | 新建 | ~5 行 |
| `qwen_asr/service/routes/health.py` | 新建 | ~25 行 |
| `qwen_asr/service/routes/batch.py` | 新建 | ~70 行 |
| `qwen_asr/service/routes/streaming.py` | 新建 | ~150 行 |
| `qwen_asr/service/app.py` | 新建 | ~60 行 |
| `qwen_asr/cli/serve_unified.py` | 新建 | ~50 行 |
| `examples/example_unified_batch.py` | 新建 | ~40 行 |
| `examples/example_unified_streaming_ws.py` | 新建 | ~60 行 |
| `pyproject.toml` | 修改 | +1 行 |
| **合计** | | **~755 行** |

## 10. 注意事项

- **workers 必须为 1**：只有一个 GPU 模型实例，多 worker 会重复加载模型导致 OOM
- **流式会话内存**：每个会话累积完整音频在 `state.audio_accum` 中。100 个并发 10 分钟会话 ≈ 100 × 600s × 16000 × 4B = 3.84 GB CPU RAM。通过 `max_concurrent_sessions` 和 `session_ttl_sec` 控制上限
- **Prompt 一致性**：统一使用 `Qwen3ASRModel` 的 prompt 模板（含空 system message），与原始 `qwen-asr-serve` 的 vLLM 原生 serve 略有差异（后者无 system message），详见 `serve_vs_demo_diff_analysis.md`
- **推理延迟**：在高并发下，由于锁序列化了所有推理调用，离线长音频请求可能阻塞流式请求。未来可考虑优先级队列（流式请求优先）
- **依赖**：FastAPI 和 uvicorn 已是 vLLM 的传递依赖，无需额外安装。`pydantic-settings` 可能需要单独安装
