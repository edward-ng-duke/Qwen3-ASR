# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Examples for Qwen3ASRModel Streaming Inference (vLLM backend).

Note:
  Requires vLLM extra:
    pip install qwen-asr[vllm]
"""

import io
import re

import numpy as np
import soundfile as sf
from openai import OpenAI


# ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
ASR_MODEL_PATH = "models/Qwen3-ASR-1.7B"
VLLM_BASE_URL = "http://localhost:6001/v1"
LOCAL_WAV = "/home/edward/research/Qwen3-ASR/data/什么是标准的上海话.wav"


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Simple resample to 16k if needed (uses linear interpolation; good enough for a test)."""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _numpy_to_wav_bytes(wav: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _parse_asr_output(raw: str) -> tuple[str, str]:
    """Parse raw model output into (language, text)."""
    if not raw:
        return "", ""
    s = raw.strip()
    if "<asr_text>" in s:
        meta, text = s.split("<asr_text>", 1)
        lang = ""
        m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
        if m and m.group(1).lower() != "none":
            lang = m.group(1)
        return lang, text.strip()
    return "", s


def _common_suffix_prefix_len(a: str, b: str) -> int:
    max_len = min(len(a), len(b))
    for i in range(max_len, 0, -1):
        if a[-i:] == b[:i]:
            return i
    return 0


def run_streaming_case(
    client: OpenAI,
    model: str,
    wav16k: np.ndarray,
    step_ms: int,
    window_ms: int = 20000,
) -> None:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))
    window = int(round(window_ms / 1000.0 * sr))

    pos = 0
    call_id = 0
    prev_text = ""
    full_text = ""
    language = ""
    while pos < wav16k.shape[0]:
        pos = min(pos + step, wav16k.shape[0])
        call_id += 1

        start = max(0, pos - window)
        chunk = wav16k[start:pos]
        wav_bytes = _numpy_to_wav_bytes(chunk, sr)
        transcription = client.audio.transcriptions.create(
            model=model,
            file=wav_bytes,
        )
        language, text = _parse_asr_output(transcription.text or "")

        if not prev_text:
            full_text = text
        else:
            overlap = _common_suffix_prefix_len(prev_text, text)
            full_text = full_text + text[overlap:]

        prev_text = text

    final_text = full_text.strip() or "(empty)"
    print(f"[final] lang={language or '?'} | text={final_text}")


def main() -> None:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    wav, sr = sf.read(LOCAL_WAV, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav16k = _resample_to_16k(wav, sr)

    for step_ms in [2000]:
        run_streaming_case(client, ASR_MODEL_PATH, wav16k, step_ms)


if __name__ == "__main__":
    main()
