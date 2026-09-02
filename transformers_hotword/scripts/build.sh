#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
expected='sha256:fb75b775f089e06e5a1aaebffd421e37505cc630d50c86d889d95ffa45a7e16a'
if ! docker image inspect qwenllm/qwen3-asr:latest \
  --format '{{join .RepoDigests "\n"}}' 2>/dev/null | grep -Fqx "qwenllm/qwen3-asr@${expected}"; then
  docker pull "qwenllm/qwen3-asr@${expected}"
  docker tag "qwenllm/qwen3-asr@${expected}" qwenllm/qwen3-asr:latest
fi
if ! docker image inspect qwenllm/qwen3-asr:latest \
  --format '{{join .RepoDigests "\n"}}' | grep -Fqx "qwenllm/qwen3-asr@${expected}"; then
  echo 'local qwenllm/qwen3-asr:latest does not match the verified digest' >&2
  exit 1
fi

# The registry digest is verified above. Passing the local tag prevents BuildKit
# from re-fetching 28 GB of already verified base-image metadata/layers.
docker build --pull=false --build-arg BASE_IMAGE=qwenllm/qwen3-asr:latest \
  --progress=plain -t mygpt-qwen3-asr-hotword:v1.2.2 .
docker image inspect mygpt-qwen3-asr-hotword:v1.2.2 --format '{{.Id}} {{.Size}}'
