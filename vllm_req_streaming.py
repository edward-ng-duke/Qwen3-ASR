"""
Pseudo-streaming ASR client for Qwen3-ASR vLLM server.

Simulates streaming by progressively sending accumulated audio chunks
to /v1/audio/transcriptions, modeled after example_qwen3_asr_vllm_streaming.py.

Includes client-side prefix stabilization to reduce earlier text from changing
as more audio arrives (similar to unfixed_chunk_num / unfixed_token_num in the
real streaming implementation).

Usage:
  python vllm_req_streaming.py                          # use default audio
  python vllm_req_streaming.py --audio data/test.mp3    # custom audio file
  python vllm_req_streaming.py --step-ms 1000           # single step size
  python vllm_req_streaming.py --base-url http://host:6001/v1
"""

import argparse
import io
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from openai import OpenAI


_ASR_TEXT_TAG = "<asr_text>"


def parse_asr_output(raw: str) -> tuple:
    """Parse raw model output into (language, text).

    Handles formats like:
      "language Chinese<asr_text>你好世界"  -> ("Chinese", "你好世界")
      "language English<asr_text>hello"     -> ("English", "hello")
      "hello world"                         -> ("", "hello world")
    """
    if not raw:
        return "", ""
    s = raw.strip()
    if _ASR_TEXT_TAG in s:
        meta, text = s.split(_ASR_TEXT_TAG, 1)
        lang = ""
        m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
        if m and m.group(1).lower() != "none":
            lang = m.group(1)
        return lang, text.strip()
    return "", s


def load_audio_16k(audio_path: str) -> np.ndarray:
    """Load audio file via ffmpeg, resample to 16kHz mono float32."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", audio_path,
            "-f", "wav", "-ar", "16000", "-ac", "1",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    wav, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32)


def numpy_to_wav_bytes(wav: np.ndarray, sr: int = 16000) -> bytes:
    """Convert numpy float32 array to in-memory WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def common_prefix_len(a: str, b: str) -> int:
    """Return length of common prefix between two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def run_streaming(
    client: OpenAI,
    model: str,
    wav16k: np.ndarray,
    step_ms: int,
    unfixed_chunk_num: int = 2,
) -> None:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))

    print(f"\n===== streaming step = {step_ms} ms =====")

    pos = 0
    call_id = 0
    stable_text = ""      # frozen prefix that won't change
    prev_text = ""         # previous full text for comparison

    while pos < wav16k.shape[0]:
        pos = min(pos + step, wav16k.shape[0])
        call_id += 1

        # Send accumulated audio from the beginning to current position
        accumulated = wav16k[:pos]
        wav_bytes = numpy_to_wav_bytes(accumulated, sr)

        transcription = client.audio.transcriptions.create(
            model=model,
            file=wav_bytes,
        )
        raw = transcription.text or ""
        language, text = parse_asr_output(raw)

        # Client-side prefix stabilization:
        # After unfixed_chunk_num calls, the common prefix between consecutive
        # outputs is considered stable and won't be overwritten.
        if call_id > unfixed_chunk_num and prev_text:
            cp = common_prefix_len(prev_text, text)
            # Only grow stable_text, never shrink it
            if cp > len(stable_text):
                stable_text = text[:cp]
            # If new text diverges before stable boundary, keep stable prefix
            if text[:len(stable_text)] != stable_text:
                text = stable_text + text[len(stable_text):]

        prev_text = text
        print(f"[call {call_id:03d}] ({pos / sr:.1f}s) language={language!r} text={text!r}")

    print(f"[final] language={language!r} text={text!r}")


def main():
    parser = argparse.ArgumentParser(description="Pseudo-streaming ASR via vLLM server")
    parser.add_argument("--audio", default="data/什么是标准的上海话_30s.wav", help="Audio file path")
    parser.add_argument("--base-url", default="http://localhost:6001/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="models/Qwen3-ASR-1.7B", help="Model name on server")
    parser.add_argument(
        "--step-ms",
        type=int,
        nargs="+",
        default=[2000],
        help="Streaming step size(s) in ms (e.g. --step-ms 500 1000 2000)",
    )
    parser.add_argument(
        "--unfixed-chunk-num",
        type=int,
        default=2,
        help="Number of initial chunks before enabling prefix stabilization (default: 2)",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    print(f"Loading audio: {args.audio}")
    wav16k = load_audio_16k(args.audio)
    print(f"Audio duration: {wav16k.shape[0] / 16000:.2f}s ({wav16k.shape[0]} samples)")

    for step_ms in args.step_ms:
        run_streaming(client, args.model, wav16k, step_ms, args.unfixed_chunk_num)


if __name__ == "__main__":
    main()
