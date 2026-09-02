#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.hotwords import merge_prompt  # noqa: E402


def transcribe(
    client: httpx.Client, url: str, path: Path, language: str, hotwords: list[str]
) -> dict:
    with path.open("rb") as audio:
        response = client.post(
            url.rstrip("/") + "/v1/audio/transcriptions",
            files={"file": (path.name, audio, "application/octet-stream")},
            data={
                "language": language,
                "hotwords": json.dumps(hotwords, ensure_ascii=False),
            },
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def processor_system_prompt(processor, language: str, hotwords: list[str]) -> str:
    merged = merge_prompt(None, hotwords)
    inputs = processor.apply_transcription_request(
        audio=np.zeros(16_000, dtype=np.float32),
        language=language,
        prompt=merged,
    )
    decoded = processor.tokenizer.decode(
        inputs["input_ids"][0], skip_special_tokens=False
    )
    marker = "<|im_start|>system\n"
    if marker not in decoded:
        raise RuntimeError("processor output did not contain a system message")
    return decoded.split(marker, 1)[1].split("<|im_end|>", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--asset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets"
    )
    args = parser.parse_args()
    manifest = json.loads(
        (args.asset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    client = httpx.Client()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        os.getenv("ASR_MODEL_PATH", "/opt/models/Qwen3-ASR-1.7B-hf"),
        local_files_only=True,
    )
    report = {"model": "Qwen/Qwen3-ASR-1.7B-hf", "samples": []}
    for sample in manifest["samples"]:
        path = args.asset_dir / sample["filename"]
        variants = {
            "none": [],
            "correct": sample["hotwords"],
            "unrelated": ["OpenTelemetry", "量子纠缠", "喜马拉雅山脉"],
        }
        outputs = {
            name: transcribe(client, args.url, path, sample["language"], words)
            for name, words in variants.items()
        }
        report["samples"].append(
            {
                "filename": sample["filename"],
                "reference": sample["reference"],
                "license": sample["license"],
                "source_page": sample["source_page"],
                "sha256": sample["sha256"],
                "outputs": {name: output["text"] for name, output in outputs.items()},
                "merged_hotword_prompt": merge_prompt(None, sample["hotwords"]),
                "processor_system_prompt": processor_system_prompt(
                    processor, sample["language"], sample["hotwords"]
                ),
                "correct_hotword_hits": {
                    word: word.casefold() in outputs["correct"]["text"].casefold()
                    for word in sample["hotwords"]
                },
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
