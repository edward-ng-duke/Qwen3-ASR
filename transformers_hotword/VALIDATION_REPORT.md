# Qwen3-ASR Transformers 热词离线镜像验收报告

采样日期：2026-08-03 至 2026-08-05（Asia/Shanghai）  
验收设备：NVIDIA GeForce RTX 4090 24 GiB，Linux x86_64，CUDA 12.8

## 最终产物

- 镜像：`mygpt-qwen3-asr-hotword:v1.0`
- Image ID：`sha256:507211db2cc56b2efcf3faeda0261326886768ff88093df88fbbf9d05ad8bd35`
- 镜像大小：32,510,164,350 bytes（约 32.5 GB）
- 直接回滚镜像：`mygpt-qwen3-asr-hotword:rollback-0b0af2b7`，Image ID
  `sha256:0b0af2b7d7bae87ff0ac0d22c2e0d1df31685b2effa10842c7e363f6f366968c`
- 更早回滚镜像：`mygpt-qwen3-asr-hotword:rollback-08e6f458` 与
  `sha256:fae63ee022fc611a394c0fc5ab4b2da3dbecb301cb27682ac4fe793b2e09b86c`
  （均保留，未清理）
- 模型：`Qwen/Qwen3-ASR-1.7B-hf@bcd2b5b7f32b480ab5790554cfa8347f246a14f3`
- Transformers：`c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d`（包版本 `5.15.0.dev0`）
- PyTorch / CUDA / FlashAttention：`2.9.1+cu128` / `12.8` / `2.8.3`
- 4090 实际运行配置：BF16、FlashAttention 2、dynamic KV cache、eager
- 运行容器：`qwen3-asr-hotword` + `qwen3-asr-hotword-gateway`
- 宿主机地址：`http://127.0.0.1:6002`

模型 30 个文件、离线资产 3 个文件和 224 行 Python 环境快照均已写入 `/opt/build-info`；
模型和资产的 SHA256 全量复验通过。

## 功能与离线验收

| 项目 | 结果 |
|---|---|
| 单次 HTTP 转写 | 通过，普通/热词请求返回固定响应结构 |
| 文件 SSE | 通过，事件顺序为 `started → partial* → segment_final → final` |
| 实时 WebSocket | 通过，PCM 双向传输、partial、句尾提交和 final 均正常 |
| 离线 UI | 通过，HTML/JS/CSS 不含 CDN 或外部 URL |
| 参数限制和 422 | 通过，非法 hotwords 返回结构化错误 |
| 临时文件清理 | 通过，请求完成后临时目录无残留文件 |
| Prometheus 指标 | 通过 |
| `--network none` | 启动、ready、HTTP、SSE、WebSocket、UI 检查全部通过 |
| 长期运行外网隔离 | ASR 容器只连接 internal 网络；访问 Hugging Face 实测返回 `URLError` |
| 自动化测试 | 49 passed；ruff 和 Compose 配置检查通过 |
| GPU 并发 | 四路 HTTP 和四路实时均无 OOM、无串话 |

Compose 使用一个固定上游的窄 TCP 网关解决 Docker internal 网络无法直接发布宿主机端口的问题。
ASR 容器没有外网出口；网关只把宿主机 6002 转到固定的 `asr:8000`。

## 原生热词路径验证

服务合并出的 prompt 与 Processor 对 `input_ids` 反解出的 system message 完全一致：

- 英文：`Vocabulary: Quilter, apostle, gospel`
- 中文：`Vocabulary: 维基百科, 百科全书`

这证明请求实际进入了 `AutoProcessor.apply_transcription_request(..., prompt=...)` 的原生路径，
而不是转写后的字符串替换。

| 样本 | 无热词 | 正确热词 | 无关热词 |
|---|---|---|---|
| 官方 Quilter 英文样本 | 正确 | 正确 | 正确 |
| CC0 中文“维基百科”样本 | 正确 | 正确 | 正确 |

两个公开样本在无热词时已经能正确识别，因此这组样本没有表现出可量化增益；正确热词全部命中，
无关热词没有造成回归。结论仅是“原生 contextual prompt 生效且行为稳定”，不把热词宣称为强制替换。

### 上下文污染安全回归

已复现旧实现会在粉噪、白噪和纯音上逐字输出整张热词表或原样 prompt。第一版安全修复使用主
Silero VAD 和上下文回显后置过滤，堵住泄漏后又造成 4/100 真实短句误杀。本镜像保留两道防线，
但把原来的空结果改为“每个音频分段最多一次无上下文回退”：

1. HTTP/SSE 中，带上下文且不超过 30 秒的音频被主 VAD 拒绝后，以 `prompt=None` 对整段推理一次。
2. 带上下文的最终结果命中 prompt/热词表回显过滤后，丢弃可疑结果，并对同一分段以
   `prompt=None` 重试一次。
3. 有上下文的 WebSocket 同时运行阈值 0.5 的主 VAD 与阈值 0.1 的候选 VAD；主区间带上下文，
   候选独有且不少于 250ms 的差集区间无上下文识别。主区间后置过滤同样只回退一次。

原 Agent 的 100 条真实语音脚本复跑后，热词与 prompt 两组误杀均由 4% 降为 0/100。四条目标短句
恢复为“呢個物件。”“一個好。”“啥东西？”和 “Person.”；两条咳嗽在热词请求下返回
`Hmm.`/`Ahem.`。88 条热词回归的专名命中保持 44/48（92%），没有因回退削弱。

HTTP 16 组、WS 8 组噪声配置和 7 种对抗输入均未出现 prompt/热词表泄漏；SSE 的噪声及真实语音
中途事件也没有泄漏。无上下文回退会原样保留模型自己的普通输出，因此部分噪声可能返回 `嗯。`，
但不会返回本次请求的上下文。混合“噪声→真实语音→噪声→纯音”和 9 秒纯噪声 WS 也无吐表。

最终镜像将 CPU ONNX VAD 常驻并在 readiness 前预热。未触发回退的 7.032 秒中文样本 30 次实测
p50/p95 为 357/370ms，低于旧镜像报告的 366/381ms，没有性能回退。VAD 拒绝回退端到端
p50/p95 为 137/148ms；先做上下文推理再回退为 654/663ms。12 路同时带热词请求 12/12 成功，
与单路基准逐字一致，总墙钟 1.46 秒。

旧 WebSocket 在累计稳定文本超过 120 字符、后续假设发生小修改时会拼接两份完整假设。长英文加速
复现用例修复后 confirmed 全程单调，最终文本中目标开头只出现一次。最终镜像的 20 秒用例产生
19 次 partial、2 次 segment_final 和 1 次 final，没有错误或重复拼接。

### 英文跨语言 prompt 译文泄漏修复

在上一镜像 `08e6f458…` 上确认了一个独立缺陷：`language=en` 搭配中文 prompt 时，个别真实咳嗽段
会把 prompt 的英文译文当成转写结果。例如 prompt“这是一场技术讨论会。”会稳定输出
`This is a technical discussion meeting.`，同一样本 10/10 复现。它属于上下文语义回显，原有
字符串匹配只能识别中文原文，无法识别翻译或改写，因此是服务端安全过滤缺陷。

修复没有内置这句 prompt 或任何译文词典。英文识别使用含非 ASCII 字母的 prompt 时，片段结束后
额外运行一次 `prompt=None` 推理，并比较两次经过 NFKC、大小写和标点规范化后的字符一致度：

- 一致度不低于 0.45：保留带 prompt 结果，允许 `Astroic → Azure` 这类局部上下文纠正。
- 一致度低于 0.45：丢弃带 prompt 结果，采用无上下文结果。
- SSE/WS 在复核结束前抑制该路径的 partial，译文不会先从 preview 泄出。
- 仅传 hotwords 时不复核，也不增加推理；复核失败时不返回未经验证的上下文结果。

防过拟合复验结果：150 条新语音（含 25 条英文短句）为 0/150 误杀、0/150 泄漏；25 条真实
非语音为 0/25 上下文泄漏；五段咳嗽 × 四种热词规模 × 10 次为 0/200 泄漏；另换四种不同中文
prompt 的三段咳嗽也为 0 泄漏。原 4 条短句全部恢复，专名命中保持 44/48（92%）。正式镜像中
目标咳嗽连续 10 次均返回 `Ahem.`；SSE 只发送 `started → segment_final → final`，任何事件均无
中文 prompt 或英文译文。54.6 秒 WS 无热词/有热词两轮分别输出 330/332 字，confirmed 单调且无重复。

1.68 秒真实英文音频预热后各 30 次端到端实测：无上下文 p50/p95 为 173/188ms，仅 hotwords 为
179/188ms，跨文字 prompt 复核为 293/309ms。额外延迟只发生在该窄路径。12 路带热词并发
12/12 成功、0 串话，总墙钟 1.40 秒。

### 实时尾静音幻觉修复

从 `conference_ws` GridFS 按持久证据 ID `6a733b4b8cccd526b5d42566` 重取 7.9 秒原始音频；对应
segment 为 `6a733b4b8cccd526b5d42568`。音频只有 2.5–4.5 秒有人声，实际会议热词为
`顾程、张思成、吴泽、何佳玥`。上一镜像 `0b0af2b7…` 真实节奏直连 WS 时稳定产生三个最终段：

1. `我今天跟何佳玥聊了会儿天。`
2. `嗯。`
3. `你想干嘛呢？`

无热词对照只有第一句，证明额外文本不是客户端拼接。独立复算 Silero 状态后确认：主 VAD 区间为
2.850–4.638 秒；阈值 0.1 的候选 VAD 从 2.242 秒开始后直到 `finish` 都未结束。旧差集逻辑因此
额外提交 0.608 秒前缀（归一化 RMS 0.000580）和 3.262 秒尾静音（RMS 0.000330）。

本镜像保留候选阈值 0.1，以免简单升到 0.3 后重新漏掉极短弱语音，但增加两道提交约束：主 VAD
句段结束时重置并丢弃与其重叠的未闭合候选状态；候选独有段除不少于 250ms 外，归一化 RMS 还必须
不低于 0.0015。正式镜像用同一 GridFS 音频与四个真实热词复验，最终只剩真实第一句；开发镜像
连续 10 次为 10/10 完全一致。指标同时记录 `overlap_reset=1`、`low_energy=1`。

四条原短句 HTTP 回归全部非空，包含 `啥东西？` 与 `Person.`；54.6 秒长 WS 无热词/有热词均
confirmed 单调、无重复。四路真实时间 WS 为 0 错误，更新间隔 p95 1.140 秒。证据音频不进入仓库，
继续以 GridFS 对象作为持久证据源。

## 同卡性能

下表是相同 RTX 4090、相同音频和语言、顺序启动服务后的结果。单路各 30 次，四路各 20 次。
端到端时间包含上传、FFmpeg 解码、Processor 和模型推理。

| 服务 / 并发 | 音频 | p50 | p95 | p50 RTF | 音频吞吐 |
|---|---:|---:|---:|---:|---:|
| vLLM / 1 | 7.032s | 0.128s | 0.134s | 0.0182 | 54.55x |
| Transformers eager / 1 | 7.032s | 0.366s | 0.381s | 0.0520 | 19.21x |
| vLLM / 1 | 30s | 0.550s | 0.554s | 0.0183 | 54.38x |
| Transformers eager / 1 | 30s | 1.598s | 1.622s | 0.0533 | 18.77x |
| vLLM / 4 | 7.032s | 0.143s | 0.165s | 0.0204 | 186.81x |
| Transformers eager / 4 | 7.032s | 0.611s | 0.666s | 0.0868 | 45.15x |
| vLLM / 4 | 30s | 0.614s | 0.634s | 0.0205 | 194.25x |
| Transformers eager / 4 | 30s | 3.025s | 3.131s | 0.1008 | 39.43x |

最终镜像另做了 5 次短复验：7 秒 p50 0.376 秒，30 秒 p50 1.627 秒，与 30 次正式结果一致。

单路 Transformers p50 约为 vLLM 的 2.9 倍，四路差距更大，所以不能声称性能已经接近 vLLM。
这是当前原生 Transformers generate 路径的诚实实测结果；实现已经避免单路凑 batch 等待，并保留了
实时优先和“已同时排队才合批”的调度。

### 优化选择

| 配置 | 7 秒 p50 | 30 秒 p50 | 结论 |
|---|---:|---:|---|
| eager | 0.366s | 1.598s | 最快，最终采用 |
| `torch.compile` default | 0.493s | 2.271s | graph break/recompile，较慢 |
| dynamic KV cache（模型直测） | 0.307s | 1.542s | 最快，最终采用 |
| static KV cache（模型直测） | 0.570s | 2.943s | 自动编译和重编译开销明显 |

`torch.compile` 仍可通过 `ASR_TORCH_COMPILE=1` 复测，失败会明确回退 eager。
static/dynamic cache 直测各预热 2 次、运行 10 次，排除了 HTTP/FFmpeg 开销。

Transformers 进程在 HTTP 四路压测后的 `nvidia-smi` 驻留峰值为 6,040 MiB；旧 vLLM 基线因
`--gpu-memory-utilization 0.45` 约驻留 11.9 GiB。该数值不包含机器上原有的其他 GPU 服务。

## 实时性能

| 场景 | 首结果 p50 | 首结果 p95 | 更新间隔 p95 | 结果 |
|---|---:|---:|---:|---|
| 1 路、真实时间发送 | 1.300s | 1.300s | 1.056s | confirmed 单调，无错误 |
| 4 路、真实时间发送 | 1.322s | 1.323s | 1.160s | confirmed 单调，无错误、无串话 |

四路用中文新闻与英文语音交替发送；中文路只输出中文新闻，英文路只输出英文内容。单路在最终镜像
上再次复验得到首结果 1.198 秒、更新间隔 p95 1.049 秒。

短句回退版再次用四路、真实时间发送 7–8 秒音频复验：首结果 p50/p95 为 1.382/1.383 秒，更新间隔
p95 为 1.165 秒，四路均 confirmed 单调且无错误。54.6 秒长拼接音频在无热词/有热词两轮分别输出
330/332 字，confirmed 全程单调，重复子串检测为零。

## 上一镜像一小时观察

上一镜像 `08e6f458…` 切换后从 2026-08-04 22:22:00 至 23:22:14（Asia/Shanghai）连续观察
3,614 秒，每 30 秒
检查一次局域网 ready、ASR/网关健康状态、实际容器 Image ID、错误日志和 GPU。共 120 次采样：

- ready 成功 120/120。
- ASR 与网关 healthy 120/120，Image ID 始终为 `sha256:08e6f458e1ae...`，RestartCount 为 0。
- `ERROR`、`Traceback`、CUDA OOM、队列满和回退失败合计 0。
- GPU 总占用稳定在 19,131 MiB，未观察到显存爬升；采样时整机可用约 4,918 MiB。
- `blade-demo` 与 `sam3` 运行时长保持原值，未被修改或重启。

本次镜像 `0b0af2b7…` 于 2026-08-05 切换后，ASR 和网关均为 healthy、RestartCount 为 0，实际
容器 Image ID 与标签一致；局域网 ready、正式 HTTP/SSE 复现、Prometheus 指标和 internal 网络
隔离均通过。切换后 GPU 0 总占用 19,141 MiB、可用 4,908 MiB。上一镜像的一小时数据仅作为历史
稳定性记录，不冒充本次新镜像的一小时观察。

尾静音修复镜像 `507211db…` 切换后，ASR 和网关均为 healthy、RestartCount 为 0；局域网正式 WS
证据复验通过，未再返回静音幻觉文本。`0b0af2b7…` 已保留为直接回滚镜像。

## 已知边界

- Qwen3-ASR 本身不是原生流式编码器；WebSocket 每约 1 秒重识别当前工作窗口。
- 热词是 contextual bias，不保证百分之百命中或强制改写。
- 带上下文请求可能自动降级为无上下文识别；回退不会泄漏请求上下文，但可能保留无上下文模型自身的
  普通幻觉或 `嗯。`、`Hmm.`、`Ahem.` 等语气词。
- 英文识别搭配非 ASCII prompt 时每个分段会增加一次安全复核推理，SSE/WS 可能没有 partial、直接从
  `segment_final` 给出可信文本；仅 hotwords 路径不受影响。
- Transformers 的单路延迟已选用本机最快稳定配置，但仍明显慢于 vLLM。
- TLS 仍由外部反向代理承担。
