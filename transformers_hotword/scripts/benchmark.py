#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(output)


def request(
    client: httpx.Client, url: str, model: str, path: Path, language: str
) -> tuple[float, dict]:
    started = time.perf_counter()
    with path.open("rb") as audio:
        response = client.post(
            url.rstrip("/") + "/v1/audio/transcriptions",
            files={"file": (path.name, audio, "application/octet-stream")},
            data={"model": model, "language": language},
            timeout=300,
        )
    response.raise_for_status()
    return time.perf_counter() - started, response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("audio", nargs="+")
    args = parser.parse_args()
    client = httpx.Client(limits=httpx.Limits(max_connections=max(8, args.concurrency)))
    report = {
        "url": args.url,
        "model": args.model,
        "runs": args.runs,
        "concurrency": args.concurrency,
        "audio": [],
    }
    for filename in args.audio:
        path = Path(filename)
        audio_duration = duration(path)
        for _ in range(args.warmup):
            request(client, args.url, args.model, path, args.language)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [
                pool.submit(request, client, args.url, args.model, path, args.language)
                for _ in range(args.runs)
            ]
            values = [future.result() for future in futures]
        wall = time.perf_counter() - started
        latencies = [item[0] for item in values]
        report["audio"].append(
            {
                "path": str(path),
                "duration": audio_duration,
                "p50_seconds": statistics.median(latencies),
                "p95_seconds": percentile(latencies, 0.95),
                "p50_rtf": statistics.median(latencies) / audio_duration,
                "throughput_audio_x": audio_duration * len(values) / wall,
                "sample_response": values[-1][1],
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
