# 部署文档

## 1. 导入镜像

```bash
docker load < mygpt-qwen3-asr-v1.0.tar.gz
```

## 2. 启动服务

> 说明：`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 用于禁止 vLLM / huggingface_hub 在加载阶段访问
> huggingface.co。模型权重已经随镜像打包在 `/data/shared/Qwen3-ASR/models/` 下，无需联网。

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
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  mygpt-qwen3-asr:v1.0 \
  qwen-asr-serve /data/shared/Qwen3-ASR/models/Qwen3-ASR-1.7B \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 6001
```

## 3. 验证

```bash
docker logs -f mygpt-qwen3-asr
```

服务就绪后监听 `6001` 端口。
