# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=qwenllm/qwen3-asr:latest@sha256:fb75b775f089e06e5a1aaebffd421e37505cc630d50c86d889d95ffa45a7e16a
FROM ${BASE_IMAGE}

ARG TRANSFORMERS_REVISION=c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d
ARG MODEL_ID=Qwen/Qwen3-ASR-1.7B-hf
ARG MODEL_REVISION=bcd2b5b7f32b480ab5790554cfa8347f246a14f3

ENV VIRTUAL_ENV=/opt/qwen3-asr-hotword/venv \
    PATH=/opt/qwen3-asr-hotword/venv/bin:${PATH} \
    PYTHONPATH=/opt/qwen3-asr-hotword \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/qwen3-asr-hotword/hf-home \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    ASR_MODEL_PATH=/opt/models/Qwen3-ASR-1.7B-hf \
    ASR_MODEL_ID=Qwen/Qwen3-ASR-1.7B-hf \
    ASR_MODEL_REVISION=bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
    ASR_TRANSFORMERS_REVISION=c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d \
    ASR_TORCH_COMPILE=0 \
    ASR_TEMP_DIR=/var/lib/qwen3-asr/tmp \
    TORCHINDUCTOR_CACHE_DIR=/var/lib/qwen3-asr/torchinductor

WORKDIR /opt/qwen3-asr-hotword

COPY requirements-runtime.lock ./requirements-runtime.lock

# Do not install or build torch/flash-attn here. The venv sees the prebuilt
# packages from the official base, while app dependencies override old Python-only packages.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv --system-site-packages "${VIRTUAL_ENV}"
RUN python -m pip install --upgrade pip==26.0.1 setuptools==82.0.0 wheel==0.46.3 \
    && python -m pip install --upgrade-strategy only-if-needed -r requirements-runtime.lock \
    && python -m pip install --upgrade-strategy only-if-needed \
       "https://github.com/huggingface/transformers/archive/${TRANSFORMERS_REVISION}.tar.gz" \
    && command -v ffmpeg \
    && command -v ffprobe \
    && python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'flash_attn', flash_attn.__version__)" \
    && python -c "from transformers import Qwen3ASRForConditionalGeneration; print(Qwen3ASRForConditionalGeneration.__name__)"

# Build may use the network; the resulting image never does. Temporarily override
# the offline flags only for the exact immutable model snapshot.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', revision='${MODEL_REVISION}', local_dir='/opt/models/Qwen3-ASR-1.7B-hf')"

COPY app ./app
COPY scripts ./scripts
COPY README.md ./README.md
COPY assets ./assets

RUN python scripts/fetch_validation_assets.py \
    && mkdir -p /var/lib/qwen3-asr/tmp /var/lib/qwen3-asr/torchinductor /opt/build-info \
    && python -m pip freeze | LC_ALL=C sort > /opt/build-info/pip-freeze.txt \
    && find /opt/models/Qwen3-ASR-1.7B-hf -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /opt/build-info/model-sha256.txt \
    && find /opt/qwen3-asr-hotword/assets -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > /opt/build-info/assets-sha256.txt \
    && printf '%s\n' \
       "base_image=qwenllm/qwen3-asr:latest@sha256:fb75b775f089e06e5a1aaebffd421e37505cc630d50c86d889d95ffa45a7e16a" \
       "transformers_revision=${TRANSFORMERS_REVISION}" \
       "model_id=${MODEL_ID}" \
       "model_revision=${MODEL_REVISION}" \
       "silero_vad=6.2.1" > /opt/build-info/build.txt \
    && python -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained('/opt/models/Qwen3-ASR-1.7B-hf', local_files_only=True); assert 'prompt' in __import__('inspect').signature(p.apply_transcription_request).parameters" \
    && rm -rf "${HF_HOME}" /root/.cache/pip /tmp/*

LABEL org.opencontainers.image.title="Qwen3-ASR Transformers Hotword Offline Service" \
      org.opencontainers.image.version="1.2.2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.qwen.model="Qwen/Qwen3-ASR-1.7B-hf" \
      ai.qwen.model.revision="bcd2b5b7f32b480ab5790554cfa8347f246a14f3" \
      ai.qwen.transformers.revision="c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)" || exit 1

CMD ["python", "-m", "app.main"]
