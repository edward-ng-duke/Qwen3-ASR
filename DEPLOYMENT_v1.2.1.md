# Qwen3-ASR v1.2.1 构建与候选验收记录

时间：2026-08-09 12:24（Asia/Shanghai）

## 交付结论

本次仅构建镜像并在独立候选端口验收，没有替换、停止或重启当前 6001、6002、4179 容器。

| 镜像 | 完整 Image ID | 大小 | OCI version | 最终构建耗时 |
|---|---|---:|---:|---:|
| `mygpt-qwen3-asr:v1.2.1` | `sha256:200b78ff0d649cd3de35dd26857a0b2bb1991c5b2eb3d6700f2a5e0f9e0dee5c` | 23,754,629,106 B（约 23.8 GB） | `1.2.1` | 11.10 s |
| `mygpt-qwen3-asr-hotword:v1.2.1` | `sha256:eebfe20632c830b7f2718370afba603ca8ee14176a0364ef0657b9f27046a57c` | 32,510,232,522 B（约 32.5 GB） | `1.2.1` | 17.91 s |

判断部署版本时应核对 Image ID，不应只依赖可变 tag。

## 修复内容

- `confirmed` 只增长，不因模型修订而回滚。
- preview 末尾两个显示单位内的修订即时显示；大幅修订需要连续两轮兼容结果。
- 冲突假设不会把已显示 preview 瞬时清空。
- final 使用可信末尾锚点或最近兼容假设对齐；无锚点时保留 confirmed，不拼接完整累计假设。
- 极短 VAD 残段重放累计开头时，仅提取可信新增尾部，防止跨 `segment_final` 的 `raw_text` 重复。
- 主 VAD 优先，候选 VAD 减去主覆盖；时间范围在推理和文本状态变更前裁剪、校验。
- 空、过短或已覆盖的 finish 区间安全跳过，正常会话仍以 `final:true` 结束。

## 自动化与真实链路结果

- 共享 Qwen 单元/协议测试：`73 passed`。
- vLLM 与 Transformers 共用同一份稳定器、VAD 区间和 realtime 协议实现。
- 候选 vLLM gateway：`http://127.0.0.1:6101`，复用当前健康 engine，不占用第二份 GPU 模型。
- readiness：HTTP、file SSE、realtime、realtime timestamps、hotwords、prompt 均为 `true`。
- 候选 realtime 完成 3 次成功会话；验收后 `asr_active_realtime_sessions=0`。
- 日志无 `internal_error`、traceback 或 request failed；重叠残段按安全原因跳过。
- 浏览器最终样本中“那好，天津路”只出现一次，后续“困惑”等尾部保留。

真实 Transformers 6002 本次没有恢复或替换。结论仅为“共享代码与契约测试通过”；真实 6002 现场验收待服务恢复后执行。

## 候选 gateway 启动命令

以下命令复用现有 `asr-engine`，只启动独立 gateway；候选测试使用 6101，不影响 6001：

```bash
docker create \
  --name qwen3-asr-v121-candidate \
  --network qwen3-asr-vllm-edge \
  -p 6101:8000 \
  -e ASR_PORT=8000 \
  -e ASR_ENGINE_URL=http://asr-engine:8000 \
  -e ASR_MODEL_ID=Qwen/Qwen3-ASR-1.7B \
  -e ASR_MAX_REALTIME_SESSIONS=4 \
  -e ASR_FALLBACK_VAD_THRESHOLD=0.1 \
  -e ASR_CANDIDATE_MIN_RMS=0.0015 \
  -e ASR_PROMPT_VERIFICATION_MIN_AGREEMENT=0.45 \
  -e ASR_API_KEY= \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  --entrypoint python3 \
  mygpt-qwen3-asr:v1.2.1 \
  -m vllm_hotword.app.main

docker network connect qwen3-asr-vllm-offline qwen3-asr-v121-candidate
docker start qwen3-asr-v121-candidate
```

关键参数：

| 参数 | 作用 |
|---|---|
| `--network ...-edge` | 让 gateway 能发布 HTTP/WS 端口。 |
| `docker network connect ...-offline` | 让 gateway 通过 Docker DNS 访问无外网的 `asr-engine`。 |
| `-p 6101:8000` | 候选宿主机端口；正式部署使用 `6001:8000`。 |
| `ASR_ENGINE_URL` | vLLM OpenAI engine 的容器内地址。 |
| `ASR_MODEL_ID` | 对外模型标识，必须与 engine served model 一致。 |
| `ASR_MAX_REALTIME_SESSIONS` | realtime WS 并发上限。 |
| 三个 VAD/verification 参数 | 候选 VAD 阈值、最低能量及上下文结果复核阈值。 |
| `ASR_API_KEY` | 可选服务鉴权；空值表示内网无鉴权。 |
| 三个 `*_OFFLINE=1` | 运行期禁止联网下载模型或数据。 |
| `--entrypoint ... -m ...` | 同一镜像作为轻量 gateway 启动，不启动第二个 vLLM engine。 |

## 正式 vLLM 双容器启动参考

正式环境应先启动 GPU engine，再启动同时连接 edge/offline 网络的 gateway：

```bash
docker network create qwen3-asr-vllm-edge
docker network create --internal qwen3-asr-vllm-offline

docker run -d \
  --name mygpt-qwen3-asr-engine \
  --restart unless-stopped \
  --gpus 'device=0' \
  --shm-size 4g \
  --network qwen3-asr-vllm-offline \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  --entrypoint vllm \
  mygpt-qwen3-asr:v1.2.1 \
  serve /opt/models/Qwen3-ASR-1.7B \
  --served-model-name Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.30 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

gateway 使用上一节命令，把容器名改为 `mygpt-qwen3-asr`、端口改为 `6001:8000`，并增加 `--restart unless-stopped`。

其中 `--gpus 'device=0'` 绑定 GPU 0，`--shm-size 4g` 为 vLLM worker 提供共享内存，`--gpu-memory-utilization 0.30` 限制显存占比，`--max-model-len 4096` 限制模型上下文长度。创建已有网络时 Docker 会报“already exists”，现场部署应先检查网络存在性。

## 回滚边界

- 当前 6001 继续运行 `v1.2.0`，6002 与 4179 状态未变。
- 验收完成后候选 gateway 容器已删除；v1.2.1 镜像与测试报告保留。
- 切换到 v1.2.1 前记录当前容器 Image ID；回滚时以该 ID 重建容器。
- 不要把 6002 尚未完成的真实验收描述为已通过。
