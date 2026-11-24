# syntax=docker/dockerfile:1.7
ARG MODEL_SOURCE_IMAGE=mygpt-qwen3-asr:rollback-92d2412e
ARG BASE_IMAGE=vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
FROM ${MODEL_SOURCE_IMAGE} AS model_source

FROM ${BASE_IMAGE}

USER root
ENV PYTHONPATH=/opt/qwen3-asr-vllm \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HF_HOME=/opt/qwen3-asr-vllm/hf-home \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    ASR_MODEL_PATH=/opt/models/Qwen3-ASR-1.7B \
    ASR_MODEL_ID=Qwen/Qwen3-ASR-1.7B \
    ASR_MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5 \
    ASR_TEMP_DIR=/var/lib/qwen3-asr/tmp

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qwen3-asr-vllm
COPY vllm_hotword/requirements-runtime.lock ./vllm_hotword/requirements-runtime.lock
RUN python3 -m pip install --no-cache-dir -r vllm_hotword/requirements-runtime.lock \
    && python3 -c "import vllm; assert vllm.__version__ == '0.26.0', vllm.__version__"

# Reuse the exact immutable snapshot embedded in the old production image.
COPY --from=model_source \
    /data/shared/Qwen3-ASR/models/Qwen3-ASR-1.7B \
    /opt/models/Qwen3-ASR-1.7B
COPY transformers_hotword ./transformers_hotword
COPY vllm_hotword ./vllm_hotword

RUN test "$(head -n 1 /opt/models/Qwen3-ASR-1.7B/.cache/huggingface/download/config.json.metadata)" = \
       "7278e1e70fe206f11671096ffdd38061171dd6e5" \
    && mkdir -p /var/lib/qwen3-asr/tmp /opt/build-info \
    && python3 -m pip freeze | LC_ALL=C sort > /opt/build-info/pip-freeze.txt \
    && find /opt/models/Qwen3-ASR-1.7B -type f -print0 | LC_ALL=C sort -z \
       | xargs -0 sha256sum > /opt/build-info/model-sha256.txt \
    && printf '%s\n' \
       'base_image=vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52' \
       'model_id=Qwen/Qwen3-ASR-1.7B' \
       'model_revision=7278e1e70fe206f11671096ffdd38061171dd6e5' \
       'silero_vad=6.2.1' > /opt/build-info/build.txt \
    && python3 -m compileall -q transformers_hotword vllm_hotword \
    && rm -rf /root/.cache/pip /tmp/*

LABEL org.opencontainers.image.title="Qwen3-ASR vLLM Hotword Offline Service" \
      org.opencontainers.image.version="1.2.2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.qwen.model="Qwen/Qwen3-ASR-1.7B" \
      ai.qwen.model.revision="7278e1e70fe206f11671096ffdd38061171dd6e5" \
      ai.qwen.vllm.version="0.26.0"

EXPOSE 8000
