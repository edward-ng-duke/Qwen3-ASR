# Qwen3-ASR 1.7B Transformers 热词离线服务

这是与仓库原有 vLLM 部署完全隔离的实现。服务只加载
`Qwen/Qwen3-ASR-1.7B-hf`，调用 Transformers 原生
`apply_transcription_request(..., prompt=...)`；不加载 Forced Aligner。SSE/WS 提供的是 VAD/提交区间的
整数毫秒时间戳，不是词级强制对齐时间戳。

## 固定版本

- 基础镜像：`qwenllm/qwen3-asr:latest@sha256:fb75b775f089e06e5a1aaebffd421e37505cc630d50c86d889d95ffa45a7e16a`
- Transformers：`c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d`
- 模型：`Qwen/Qwen3-ASR-1.7B-hf@bcd2b5b7f32b480ab5790554cfa8347f246a14f3`
- Silero VAD：`6.2.1`，CPU ONNX
- 镜像：`mygpt-qwen3-asr-hotword:v1.2.2`

构建会联网下载这些固定资产。模型、VAD、依赖、测试音频和网页在成品镜像内，运行时设置
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，所有模型加载都使用 `local_files_only=True`。

## 构建与启动

```bash
cd /home/edward/research/Qwen3-ASR/transformers_hotword
./scripts/build.sh
./scripts/run.sh
```

服务映射为 `http://127.0.0.1:6002`。ASR 容器只连接 `internal` Docker 网络；一个不使用 GPU 的
窄 TCP 网关同时连接 internal/edge 网络，把宿主机 6002 的入站连接转发到 ASR 8000。网关不实现
任意目标转发，因此 ASR 没有借此访问外网的路径。两个服务复用同一镜像层，不额外复制 32GB。
启动前脚本会检查 6002，绝不会主动终止未知服务。

设置鉴权：

```bash
ASR_API_KEY='replace-me' ./scripts/run.sh
```

REST/SSE 使用 `Authorization: Bearer replace-me`。浏览器 WebSocket 无法添加该 Header，因此在首个
`start` JSON 的 `api_key` 字段发送；Key 只保存在测试页内存中，不写日志或 localStorage。

## API

普通文件识别：

```bash
curl -sS http://127.0.0.1:6002/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F language=zh \
  -F 'prompt=这是一场技术会议。' \
  -F 'hotwords=["Qwen3-ASR","Transformers"]'
```

返回字段为 `language/text/raw_text/duration/processing_ms/model/request_id`。

SSE 使用同样的 multipart 字段：

```bash
curl -N http://127.0.0.1:6002/v1/audio/transcriptions/stream \
  -F file=@sample.wav -F language=auto -F 'hotwords=["Quilter"]'
```

事件顺序是 `started`、零个或多个 `partial`、零个或多个 `segment_final`、`final`；失败后发送 `error`。
文件必须先上传完毕，之后才开始响应事件。上传内容流式写临时磁盘，不整体驻留内存。
无语音时不发送 `segment_final`，最终事件满足 `final=true` 且最终文本为空。每个 `segment_final`
包含单调、非重叠的 `start_ms/end_ms`。

WebSocket 地址为 `ws://127.0.0.1:6002/v1/audio/realtime`。第一帧：

```json
{
  "type": "start",
  "encoding": "pcm_f32le",
  "sample_rate": 48000,
  "channels": 1,
  "language": "auto",
  "prompt": "会议上下文",
  "hotwords": ["Qwen3-ASR", "Transformers"]
}
```

随后发送小端 `float32` 或 `int16` PCM 二进制帧；控制帧是 `{"type":"ping"}` 和
`{"type":"finish"}`。初始化后的语言、prompt、hotwords 不可修改。

WebSocket 是双向实时传输，但模型没有可复用的增量音频编码状态。每个新 VAD 语音段默认在
0.5 秒时尝试首个非空 preview，之后每 1 秒重解码当前窗口；分别可通过
`ASR_REALTIME_FIRST_UPDATE_SECONDS` 和 `ASR_REALTIME_UPDATE_SECONDS` 覆盖，且前者必须大于 0
并不大于后者。
用连续结果公共前缀产生永不撤回的 `confirmed`，变化部分放在 `preview`。默认窗口 10 秒、重叠
2 秒、静音 600ms 结束一句；连续语音达到窗口上限时提交一段并保留重叠音频。

## 健康、指标和约束

- `/health/live`：进程启动后立即成功。
- `/health/ready`：模型加载和预热后成功，并返回 HTTP/SSE/realtime/timestamp/hotwords/prompt 能力。
- `/v1/models`：模型 revision、设备、精度、Attention 后端和 compile 状态。
- `/metrics`：请求数、错误数、延迟、RTF、音频秒数、队列、活动实时会话和显存。
- 文件最大 8 GiB、音频最长 4 小时；实时会话最长 1 小时、空闲 5 分钟。
- 一个 GPU worker；实时优先，最多 4 个实时会话；单路立即执行，同长度桶的已排队任务最多合批 4 路。
- 带 prompt/hotwords 的短文件先通过主 Silero VAD。主 VAD 未检出语音时，服务会自动以
  `prompt=None` 对整段重试一次，避免漏掉真实短句；不会递归重试。
- 服务会拦截 prompt 原样回显和多热词列表回显。命中后会丢弃可疑结果，并对同一分段做一次
  无上下文重试。返回的是重试结果，可能包含模型自身的普通幻觉或 `嗯。`、`Hmm.`、`Ahem.` 等语气词。
- 英文识别使用非 ASCII prompt（例如中文会议背景）时，还会在片段结束后做一次无 prompt 复核。
  两次结果的规范化文本一致度低于默认阈值 0.45 时，服务丢弃带 prompt 结果并采用无上下文结果，
  用来阻断 prompt 被翻译或语义改写后回显。该路径在 SSE/WS 中会先抑制尚未复核的 partial；因此
  可能直到 `segment_final` 才出现文本。仅传 hotwords、不传 prompt 的请求不触发这次额外推理。
- WebSocket 在有上下文时同时使用阈值 0.5 的主 VAD 和阈值 0.1 的候选 VAD。主 VAD 区间带
  上下文识别，只有候选 VAD 命中的差集区间使用无上下文识别。主句段结束时会重置与其重叠且仍未
  闭合的候选状态，避免把主句前后的静音当成独立语音；候选独有段还必须不少于 250ms，且归一化
  RMS 不低于默认值 0.0015，才会进入模型。阈值可用 `ASR_CANDIDATE_MIN_RMS` 调整，设为 0 可禁用
  能量门槛。
- 指标 `asr_context_leaks_suppressed_total` 记录后置拦截；`asr_context_fallback_total` 和
  `asr_context_fallback_duration_seconds` 记录回退原因、结果和耗时；`asr_prompt_verification_total`
  和 `asr_prompt_verification_duration_seconds` 记录跨语言 prompt 复核结果与耗时。复核一致度阈值可用
  `ASR_PROMPT_VERIFICATION_MIN_AGREEMENT` 调整，必须在 0 和 1 之间；
  `asr_realtime_candidate_segments_total` 记录候选段提交或抑制原因。日志不记录上下文或输出正文。
- WebSocket 长累计假设发生内部修订时保持 confirmed 单调，禁止整段重复拼接。`preview` 仍会变化，
  只能用于显示；持久化或下游消费必须以 `confirmed` 为准。
- BF16 优先、FP16 回退、无量化。4090 应使用现成 FlashAttention 2；不支持时才回退 SDPA。
- RTX 4090 实测 eager 的单路延迟低于 `torch.compile`，因此最终配置默认 eager。可设置
  `ASR_TORCH_COMPILE=1` 复测 compile；编译或预热失败会显式记录并回退 eager。
- 默认日志不包含音频、prompt、hotwords、text、raw_text、confirmed、preview 或 delta。

## 验证

无需网络的容器内验收（容器用 `--network none`）：

```bash
./scripts/offline_smoke.sh
```

热词三组对照：

```bash
docker exec qwen3-asr-hotword \
  python /opt/qwen3-asr-hotword/scripts/validate_hotwords.py
```

公开验证音频的 URL、参考文本、许可证和 SHA256 见 `assets/manifest.json`。英文样本是官方模型卡的
Quilter 示例；中文样本是 Wikimedia Commons 的 CC0 北京话“维基百科，自由的百科全书”。

同卡性能脚本：

```bash
python scripts/benchmark.py \
  --url http://127.0.0.1:6002 \
  --model Qwen/Qwen3-ASR-1.7B-hf \
  --runs 30 --warmup 5 \
  ../data/2news_30s.mp3 ../data/什么是标准的上海话_30s.wav
```

脚本输出 p50、p95、RTF 和音频吞吐。vLLM 基线应在同一 GPU 上顺序启动后用相同文件、次数和语言
另跑一次；不要同时驻留两个 ASR 模型，以免显存竞争污染结果。

镜像内比较 Transformers dynamic/static KV cache：

```bash
python scripts/cache_benchmark.py --runs 10 \
  ../data/2news_30s.mp3 ../data/什么是标准的上海话_30s.wav
```

实时单路/四路验收：

```bash
python scripts/realtime_benchmark.py \
  --url http://127.0.0.1:6002 --sessions 4 --max-audio-seconds 8 \
  ../data/2news_30s.mp3 ../data/5_en.wav
```

脚本按真实时间发送 PCM，报告首结果延迟、更新间隔、`confirmed` 单调性和每路最终文本。
