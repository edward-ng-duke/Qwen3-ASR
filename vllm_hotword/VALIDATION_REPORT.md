# Qwen3-ASR vLLM 热词兼容版验证报告

验证日期：2026-08-08（Asia/Shanghai）  
候选服务：`http://127.0.0.1:6001` / `http://10.0.0.93:6001`  
对照服务：`http://127.0.0.1:6002`  
最终镜像：`mygpt-qwen3-asr:v1.0`，`sha256:7c32d95b16d45ca3019faab90a34d8beb2e435226e422be695dd993d16594e0d`  
回滚镜像：`mygpt-qwen3-asr:rollback-92d2412e`

## 结论

新服务已替换原 6001 容器。原生无上下文请求保持 vLLM/OpenAI 路径，带
`prompt/hotwords` 的请求经过与 Transformers 版相同的 VAD、泄漏抑制和最多一次无上下文
回退。HTTP、SSE、WS、上下文安全、并发、离线资产和局域网访问均通过本轮验收。

识别质量与 Transformers 不保证逐字相同；150 条新 GridFS 语音中，无上下文逐字一致
111/150，带上下文逐字一致 117/150。两套服务均无新增空输出或上下文泄漏。按计划，质量文本
差异只记录，不作为自动回滚条件。

## 部署状态

- `mygpt-qwen3-asr-engine`：vLLM 0.26.0，引擎只加入 Docker internal 网络。
- `mygpt-qwen3-asr`：安全网关，发布 `6001:8000`，两个容器均使用同一最终 Image ID。
- 模型：`Qwen/Qwen3-ASR-1.7B`，revision
  `7278e1e70fe206f11671096ffdd38061171dd6e5`。
- 基础镜像：`vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`。
- GPU：RTX 4090；切换后全机驻留约 19.8/24.6 GB，现场剩余约 4.8 GB。新 vLLM 引擎进程
  驻留约 8.8 GB；该数值会随并发和机器上其他服务变化。
- `10.0.0.93:6001/health/ready` 与本机地址均返回 200。现场 `10.0.0.32:6001` 是其他
  地址，readiness 返回 404，不应写入本次 Agent 配置。

## 静态与接口测试

- `pytest -q vllm_hotword/tests transformers_hotword/tests`：58 passed。
- `ruff check vllm_hotword transformers_hotword/app transformers_hotword/tests`：通过。
- `docker compose ... config --quiet`：通过。
- API 矩阵：live、ready、models、原生旧模型别名、prompt、字符串热词、JSON 数组热词、
  prompt+hotwords、非法 JSON 422、上下文原生 stream 422、SSE 全部通过。
- SSE 事件为 `started -> partial -> segment_final -> final`，sequence 单调。
- `/v1/chat/completions` 透明代理到 vLLM，现场请求返回 200；`/v1/models` 保持原生 vLLM
  响应结构。
- 远程 URL JSON 请求返回 400 `multipart/form-data is required`，网关不下载外部音频。

## A/B 识别结果

### 公开小样本

英文 Quilter 样本和中文 CC0 样本分别以无热词、正确热词、无关热词请求两套服务，共六组。
6001 与 6002 为 6/6 规范化后逐字一致，CER 为 0。

### 150 条新 GridFS 语音

本轮从 `conference_ws.segments` 和 `audio_segments` 确定性重新选取，未使用上一轮 Agent 的
样本输出：

| 分组 | 数量 |
|---|---:|
| 粤语 | 13 |
| 中文极短 | 40 |
| 中文短句 | 30 |
| 英文短句 | 25 |
| 中文中长句 | 20 |
| 中文补充长句 | 22 |
| 合计 | 150 |

库中符合条件且带 GridFS 音频的粤语样本只有 13 条，因此以 22 条中文补充长句补齐 150 条。
两套服务分别运行无上下文和带 12 个热词，共 600 次请求：

- 所有请求成功。
- 无上下文逐字一致 111/150；带上下文逐字一致 117/150。
- 6001 新增空输出 0/150；6002 新增空输出 0/150。
- 6001 上下文泄漏 0/150；6002 上下文泄漏 0/150。
- 按库中存量 `raw_text/text` 粗算 mean CER：6001 为 1.006，6002 为 0.990。存量字段含历史
  标注和清洗口径差异，该聚合值不适合作为绝对准确率，只用于本轮同源参考。

### 并发

使用中英文公开音频交替执行 12 路 HTTP：6001 与 6002 均为 12/12 成功；同类别结果与单路
基准一致，未出现上下文串话、OOM 或死锁。

## 安全回归

- 25 条新的真实非语音 GridFS 片段（咳嗽、笑声、音乐、掌声），统一使用最危险组合
  `language=en + 中文 prompt + 12 个热词`：6001 和 6002 均为 25/25 成功、上下文泄漏 0。
- 中文 prompt `这是一场技术讨论会。` 配英文语言和目标咳嗽连续运行 10 次：两套服务均稳定
  输出 `Cough.`，未输出中文原文或英文译文。
- 白噪、粉噪、纯音配 none/prompt/hotwords/both 的 HTTP 矩阵：6001 为 0/12 泄漏；WS
  同组合为 0/12 泄漏。SSE 噪声路径没有把上下文放入稳定事件。
- 原七类对抗输入在构建候选上均未泄漏；最终镜像使用同一代码层，并额外通过上述真实非语音集、
  静音尾部持久证据和离线冒烟。
- GridFS file ID `6a733b4b8cccd526b5d42566` 仅在运行时提取，未提交音频。6001 与 6002
  最终均只输出 `我今天跟何佳玥聊了会儿天。`；事件为 partial、partial、segment_final、final，
  confirmed 单调，无静音尾部 26 字幻觉。
- 54.6 秒真实会议音频 WS：confirmed 全程单调，segment_final 没有整段重复，最终无 error。
- 所有上下文回退最多执行一次；指标中已暴露 context leak、fallback、prompt verification 和
  candidate VAD 抑制计数。

## 性能

同一 RTX 4090、相同 7 秒和 30 秒音频、相同语言、预热 5 次后各运行 30 次：

| 服务 | 音频 | p50 | p95 | p50 RTF | 吞吐 |
|---|---:|---:|---:|---:|---:|
| vLLM 6001 | 7s | 0.171s | 0.190s | 0.0244 | 40.19x |
| Transformers 6002 | 7s | 0.464s | 0.486s | 0.0663 | 14.98x |
| vLLM 6001 | 30s | 0.747s | 0.759s | 0.0249 | 40.08x |
| Transformers 6002 | 30s | 2.069s | 2.098s | 0.0690 | 14.48x |

本次音频上，vLLM 单路端到端约快 2.7 倍。该结果不能外推为所有语言、长度和上下文组合的固定
比例。

4 路 12 秒真实时间 WS：首结果 p50/p95 1.095/1.113 秒，更新间隔 p95 1.316 秒，4/4
confirmed 单调且无错误。8 路功能测试无 OOM、串话或错误，但更新间隔 p95 超过约 1.5 秒，故正式
默认上限按计划固定为 4，而不是 8。

## 离线验收

- 最终镜像以 `docker run --network none` 成功导入 vLLM 0.26.0、ONNX Runtime、实时 VAD，
  并确认 `/opt/models/Qwen3-ASR-1.7B/config.json` 存在。
- 引擎容器强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
  `HF_DATASETS_OFFLINE=1`，实测外网连接失败。
- 引擎只在 internal 网络上；只有网关发布 6001。网关只接受上传文件/PCM，不支持远程音频 URL，
  代码路径没有 Hugging Face、GitHub、CDN 或其他外网回退。
- 模型、Silero VAD、FFmpeg、Python 依赖和构建清单均在镜像内。

## 尚存边界

- 计划中原始“40 条普通识别集”和上一 Agent scratchpad 的精确 150 条文件未位于本仓或磁盘，
  因此本轮使用公开样本与重新从 GridFS 选取的 150 条新样本，避免伪造原始数据结果。
- 原生无上下文请求尽量透明代理。对 vLLM 不能直接接收的非 16 kHz PCM WAV，网关会先离线解码/
  重采样再调用无上下文引擎；这是为了兼容真实 GridFS 音频，响应仍保持所请求格式。
- 识别差异属于两个推理栈的实际行为。安全阻断项已通过，但专名效果和质量仍应随生产语料持续观察。
