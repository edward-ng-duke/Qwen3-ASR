#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def decode(path: Path) -> np.ndarray:
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, dtype="<f4").copy()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="/opt/models/Qwen3-ASR-1.7B-hf", type=Path
    )
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("audio", nargs="+", type=Path)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model.eval()
    decoded = [(path, decode(path)) for path in args.audio]
    report = {"runs": args.runs, "warmup": args.warmup, "results": []}

    for cache in ("dynamic", "static"):
        for path, audio in decoded:
            inputs = processor.apply_transcription_request(
                audio=audio, language=args.language, prompt=None
            ).to("cuda:0", dtype)

            def infer() -> None:
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        cache_implementation=cache,
                    )
                torch.cuda.synchronize()

            for _ in range(args.warmup):
                infer()
            values = []
            for _ in range(args.runs):
                started = time.perf_counter()
                infer()
                values.append(time.perf_counter() - started)
            report["results"].append(
                {
                    "cache": cache,
                    "audio": str(path),
                    "audio_seconds": audio.size / 16000,
                    "p50_seconds": statistics.median(values),
                    "p95_seconds": percentile(values, 0.95),
                    "rtf": statistics.median(values) / (audio.size / 16000),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
