#!/usr/bin/env bash
set -euo pipefail

name=qwen3-asr-hotword-offline-smoke
cleanup() { docker rm -f "${name}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker run -d \
  --name "${name}" \
  --network none \
  --gpus 'device=0' \
  --shm-size 4g \
  -e ASR_TORCH_COMPILE=0 \
  -e ASR_WARMUP=0 \
  mygpt-qwen3-asr-hotword:v1.2.2 >/dev/null

docker exec "${name}" python /opt/qwen3-asr-hotword/scripts/offline_client_smoke.py
