#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def upload(client, url: str, audio: Path, data: dict[str, str]):
    with audio.open("rb") as source:
        return client.post(
            url,
            data=data,
            files={"file": (audio.name, source, "application/octet-stream")},
        )


def sse_types(text: str) -> list[str]:
    return [line[7:] for line in text.splitlines() if line.startswith("event: ")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:6001")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    results = []
    with httpx.Client(timeout=600, trust_env=False) as client:
        for path in ("/health/live", "/health/ready", "/v1/models"):
            response = client.get(base + path)
            results.append({"case": path, "status": response.status_code})

        cases = [
            ("native", {"model": "models/Qwen3-ASR-1.7B"}, 200),
            ("prompt", {"prompt": "技术会议"}, 200),
            ("hotwords-string", {"hotwords": "DeepSeek, 智谱"}, 200),
            ("hotwords-array", {"hotwords": '["DeepSeek","智谱"]'}, 200),
            (
                "both",
                {"prompt": "技术会议", "hotwords": '["DeepSeek","智谱"]'},
                200,
            ),
            ("bad-json", {"hotwords": '["broken"'}, 422),
            ("unsafe-stream", {"hotwords": "Qwen", "stream": "true"}, 422),
        ]
        for name, data, expected in cases:
            response = upload(
                client, base + "/v1/audio/transcriptions", args.audio, data
            )
            results.append(
                {
                    "case": name,
                    "status": response.status_code,
                    "expected": expected,
                    "passed": response.status_code == expected,
                }
            )

        response = upload(
            client,
            base + "/v1/audio/transcriptions/stream",
            args.audio,
            {"hotwords": '["DeepSeek","智谱"]'},
        )
        events = sse_types(response.text)
        results.append(
            {
                "case": "safe-sse",
                "status": response.status_code,
                "events": events,
                "passed": bool(events)
                and events[0] == "started"
                and events[-1] == "final",
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(item.get("passed", item["status"] == 200) for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

