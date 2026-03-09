# Qwen3-ASR 部署快速参考

## 概览

Qwen3-ASR 提供三个 CLI 命令，适用于不同场景：

| 命令 | 协议 | 后端 | 流式 | 适用场景 |
|------|------|------|------|----------|
| `qwen-asr-serve` | OpenAI `/v1/audio/transcriptions` | vLLM | 否 | **生产 API 服务（推荐）** |
| `qwen-asr-demo` | Gradio Web UI | Transformers / vLLM | 否 | 开发调试、带时间戳转录 |
| `qwen-asr-demo-streaming` | Flask REST + Web UI | vLLM | 支持 | 实时语音转文字演示 |

> 详细注意事项见 [vllm_deployment_notes.md](vllm_deployment_notes.md)，API 调用方法见 [usage.md](usage.md)。

---

## 1. qwen-asr-serve（生产推荐）

提供 OpenAI 兼容的 `/v1/audio/transcriptions` 接口，适合生产部署。

```bash
# 1.7B 模型
qwen-asr-serve models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.6 \
  --host 0.0.0.0 \
  --port 6001

# 0.6B 小模型（更快，精度稍低）
qwen-asr-serve models/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.3 \
  --host 0.0.0.0 \
  --port 6001
```

---

## 2. qwen-asr-demo（Gradio Web UI）

### Transformers 后端（基本）

```bash
qwen-asr-demo \
  --asr-checkpoint models/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 6001
```

### Transformers 后端 + Forced Aligner（带时间戳）

```bash
qwen-asr-demo \
  --asr-checkpoint models/Qwen3-ASR-1.7B \
  --aligner-checkpoint models/Qwen3-ForcedAligner-0.6B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":8,"max_new_tokens":256}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 6001
```

### vLLM 后端

```bash
qwen-asr-demo \
  --asr-checkpoint models/Qwen3-ASR-1.7B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization":0.8,"max_inference_batch_size":8,"max_new_tokens":2048}' \
  --ip 0.0.0.0 --port 6001
```

### vLLM 后端 + Forced Aligner（带时间戳）

```bash
qwen-asr-demo \
  --asr-checkpoint models/Qwen3-ASR-1.7B \
  --aligner-checkpoint models/Qwen3-ForcedAligner-0.6B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization":0.8,"max_inference_batch_size":8,"max_new_tokens":2048}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 6001
```

---

## 3. qwen-asr-demo-streaming（实时流式）

仅支持 vLLM 后端。浏览器打开后通过麦克风实时转录。

```bash
qwen-asr-demo-streaming \
  --asr-model-path models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 6001
```

---

## 4. Docker 部署

### 4.1 官方镜像

```bash
docker run -d --gpus device=0 --name qwen3-asr \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 6001:6001 \
  --shm-size=4gb \
  qwenllm/qwen3-asr:latest
```

### 4.2 自定义 vLLM 镜像

先构建镜像（基于 `docker/Dockerfile-vllm-qwen3-asr`，在 `vllm/vllm-openai:nightly` 基础上加 libsndfile1 + ffmpeg）：

```bash
docker build -t vllm-qwen3-asr:v1.0 -f docker/Dockerfile-vllm-qwen3-asr docker/
```

启动容器：

```bash
docker run -d \
  --name qwen3-asr-1.7b \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 7869:8000 \
  -v /path/to/models:/models:ro \
  vllm-qwen3-asr:v1.0 \
  --model /models/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1
```

> `--ipc=host` 必须加，否则 PyTorch 共享内存分配会失败。

### 4.3 生产部署示例

基于官方镜像 + `qwen-asr-serve`，带自动重启和时区同步：

```bash
docker run -d --gpus device=0 --name mygpt-qwen3-asr \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 6001:6001 \
  --network host \
  --add-host=host.docker.internal:host-gateway \
  --restart unless-stopped \
  --volume /etc/localtime:/etc/localtime:ro \
  --volume /etc/timezone:/etc/timezone:ro \
  --shm-size=4gb \
  mygpt-qwen3-asr:v1.0 \
  qwen-asr-serve models/Qwen3-ASR-1.7B \
    --gpu-memory-utilization 0.6 \
    --host 0.0.0.0 \
    --port 6001
```

---

## 5. GPU 显存参考

| 模型 | `--gpu-memory-utilization` 推荐值 | 说明 |
|------|----------------------------------|------|
| Qwen3-ASR-0.6B | `0.3` | 小模型，30% 显存即可 |
| Qwen3-ASR-1.7B | `0.5 ~ 0.8` | 按实际显存调整，16GB+ 显卡建议 0.6 |

---

## 6. 常用参数速查

### qwen-asr-serve

| 参数 | 说明 | 示例 |
|------|------|------|
| 第一个位置参数 | 模型路径 | `models/Qwen3-ASR-1.7B` |
| `--gpu-memory-utilization` | GPU 显存使用比例 | `0.6` |
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `6001` |

### qwen-asr-demo

| 参数 | 说明 | 示例 |
|------|------|------|
| `--asr-checkpoint` | ASR 模型路径 | `models/Qwen3-ASR-1.7B` |
| `--aligner-checkpoint` | Forced Aligner 模型路径（可选，启用时间戳） | `models/Qwen3-ForcedAligner-0.6B` |
| `--backend` | 推理后端 | `transformers` / `vllm` |
| `--cuda-visible-devices` | 使用的 GPU 编号 | `0` |
| `--backend-kwargs` | 后端参数（JSON） | `'{"gpu_memory_utilization":0.8}'` |
| `--aligner-kwargs` | Aligner 参数（JSON） | `'{"device_map":"cuda:0","dtype":"bfloat16"}'` |
| `--ip` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `6001` |

### qwen-asr-demo-streaming

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-model-path` | `Qwen/Qwen3-ASR-1.7B` | 模型路径 |
| `--gpu-memory-utilization` | `0.8` | GPU 显存使用比例 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--unfixed-chunk-num` | `4` | 前 N 个 chunk 不使用前缀引导 |
| `--unfixed-token-num` | `5` | 回退 token 数量，减少边界抖动 |
| `--chunk-size-sec` | `1.0` | 每个音频块长度（秒） |
