# Qwen3-ASR v1.2.2 构建与候选验收记录

时间：2026-08-09 14:35（Asia/Shanghai）

## 镜像

| 镜像 | 完整 Image ID | 大小 | 构建时间 | OCI version | 最终构建耗时 |
|---|---|---:|---|---:|---:|
| `mygpt-qwen3-asr:v1.2.2` | `sha256:1587ba72b2ca3856ed33eafa68c6a404c8d18064a32526f9cba0cf096e2ed43c` | 23,754,658,330 B | 2026-08-09 14:27:37 +08:00 | `1.2.2` | 69.34 s |
| `mygpt-qwen3-asr-hotword:v1.2.2` | `sha256:86bdb84324d470f01f8f24ea0f54864a6caa189815befdeaf347bca49db687c5` | 32,510,240,118 B | 2026-08-09 14:27:41 +08:00 | `1.2.2` | 73.46 s |

判断运行版本时以完整 Image ID 为准，不只看可变 tag。

## 修复内容

- 新增 `ASR_REALTIME_FIRST_UPDATE_SECONDS=0.5`；每个主 VAD 段累计 0.5 秒时尝试首个非空 partial。
- 首帧为空或被上下文安全校验抑制时按 0.5 秒重试，成功后切换到 `ASR_REALTIME_UPDATE_SECONDS=1.0`。
- 新 VAD 段和 `segment_final` 后重置首帧状态；不足首帧周期的短语音允许直接 final。
- 保留 v1.2.1 的 PrefixStabilizer、两显示单位即时修订、两轮大幅修订确认和 final 防重复规则。
- 修复 10 秒窗口 `segment_final` 后新 stabilizer 误把累计开头作为 preview 重放的问题：精确后缀重叠被剔除，三显示单位以上的已确认开头重放被抑制。
- 新增不含正文的首个 partial 延迟、partial 推理结果和空结果指标。
- vLLM gateway 与 Transformers 服务共用同一 realtime、热词、prompt、SSE 和时间戳实现。

配置约束：

```yaml
ASR_REALTIME_FIRST_UPDATE_SECONDS: 0.5
ASR_REALTIME_UPDATE_SECONDS: 1.0
```

必须满足 `0 < FIRST_UPDATE_SECONDS <= UPDATE_SECONDS`。

## 测试与真实链路

- Transformers 共享实现测试：66 passed。
- vLLM gateway 契约测试：11 passed。
- 单会话真实 WS：首个 partial 1.174 s，后续更新 P95 1.049 s，confirmed 单调，error 0。
- 四并发真实 WS：首帧 P50 1.089 s、P95 1.149 s，后续更新 P95 1.078 s；4/4 confirmed 单调，capacity 降级和 error 均为 0。
- v1.2.1 四并发更新 P95 为 1.107 s，本次变化 -2.64%。
- Playwright 真实链路记录 36 个 partial；同一临时气泡清空、累计开头重放和 ASR internal error 均为 0。
- `/health/ready` 返回 backend `vllm`，HTTP、file SSE、realtime、timestamps、hotwords、prompt 能力均为 `true`。

真实 Transformers 6002 当前仍返回空响应。本次结论仅为“共享代码和 Transformers 契约测试通过”；现场 6002 验收待服务恢复。

## 6102 候选 gateway 完整启动命令

该 gateway 复用现有 `asr-engine`，不会启动第二份 GPU 模型：

```bash
docker create \
  --name qwen3-asr-v122-candidate \
  --network qwen3-asr-vllm-edge \
  -p 6102:8000 \
  -e ASR_PORT=8000 \
  -e ASR_ENGINE_URL=http://asr-engine:8000 \
  -e ASR_MODEL_ID=Qwen/Qwen3-ASR-1.7B \
  -e ASR_MAX_REALTIME_SESSIONS=4 \
  -e ASR_REALTIME_FIRST_UPDATE_SECONDS=0.5 \
  -e ASR_REALTIME_UPDATE_SECONDS=1.0 \
  -e ASR_FALLBACK_VAD_THRESHOLD=0.1 \
  -e ASR_CANDIDATE_MIN_RMS=0.0015 \
  -e ASR_PROMPT_VERIFICATION_MIN_AGREEMENT=0.45 \
  -e ASR_API_KEY= \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  --entrypoint python3 \
  mygpt-qwen3-asr:v1.2.2 \
  -m vllm_hotword.app.main

docker network connect qwen3-asr-vllm-offline qwen3-asr-v122-candidate
docker start qwen3-asr-v122-candidate
```

关键参数：

| 参数 | 作用 |
|---|---|
| `--network ...-edge` | gateway 对外提供 HTTP、SSE 和 WS。 |
| `docker network connect ...-offline` | 通过内部 Docker DNS 访问 `asr-engine`。 |
| `-p 6102:8000` | 独立候选端口，不占用 6001。 |
| `ASR_ENGINE_URL` | 复用的 vLLM OpenAI engine 地址。 |
| `ASR_MODEL_ID` | 对外模型标识，需与 engine served model 一致。 |
| `ASR_MAX_REALTIME_SESSIONS` | realtime WS 并发上限。 |
| `ASR_REALTIME_FIRST_UPDATE_SECONDS` | 每个新 VAD 段首次尝试 partial 的累计语音时长。 |
| `ASR_REALTIME_UPDATE_SECONDS` | 首个非空 partial 之后的更新周期。 |
| VAD/verification 参数 | 候选 VAD 阈值、最低能量和上下文一致性阈值。 |
| `ASR_API_KEY` | 可选 API 鉴权；空值表示当前内网候选不鉴权。 |
| 三个 `*_OFFLINE=1` | 禁止运行期联网下载模型或数据。 |
| `--entrypoint ... -m ...` | 只启动轻量 gateway，不在镜像中启动第二个 engine。 |

## 正式部署参考

正式 6001 gateway 使用同一命令，将容器名改为 `mygpt-qwen3-asr`、端口改为 `6001:8000`，并增加 `--restart unless-stopped`。GPU engine 的启动参数继续沿用 v1.2.1 发布记录，仅将镜像标签改为 v1.2.2；切换前必须记录当前容器完整 Image ID。

## 边界

- 当前 6001、6002、4179、6101、6279 容器均未替换、停止或重启。
- 6102 候选 gateway 按交付要求继续运行。
- 两个仓库已有未提交和未跟踪内容均保留，没有批量暂存或清理。
