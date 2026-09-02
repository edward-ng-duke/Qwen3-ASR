#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import websockets


def decode(path: Path, max_seconds: float) -> bytes:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        str(max_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    return subprocess.check_output(command)


def websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/v1/audio/realtime"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


async def session(
    index: int,
    uri: str,
    path: Path,
    pcm: bytes,
    language: str,
    hotwords: list[str],
    prompt: str | None,
    frame_seconds: float,
    pace: bool,
) -> dict:
    frame_bytes = max(2, int(16000 * frame_seconds) * 2)
    events: list[tuple[float, dict]] = []
    async with websockets.connect(uri, max_size=4 * 1024 * 1024) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "start",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": language,
                    "prompt": prompt,
                    "hotwords": hotwords,
                }
            )
        )
        ready = json.loads(await socket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"session {index} did not become ready: {ready}")
        started = time.perf_counter()

        async def receive() -> None:
            async for message in socket:
                payload = json.loads(message)
                events.append((time.perf_counter() - started, payload))
                if payload.get("type") in {"final", "error"}:
                    return

        receiver = asyncio.create_task(receive())
        for offset in range(0, len(pcm), frame_bytes):
            await socket.send(pcm[offset : offset + frame_bytes])
            if pace:
                target = started + min(offset + frame_bytes, len(pcm)) / 32000
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
        await socket.send(json.dumps({"type": "finish"}))
        await asyncio.wait_for(receiver, timeout=300)

    inference_events = [
        (at, payload)
        for at, payload in events
        if payload.get("type") in {"partial", "segment_final"}
    ]
    confirmed = ""
    monotonic = True
    for _at, payload in events:
        current = payload.get("confirmed", "")
        if current and not current.startswith(confirmed):
            monotonic = False
        if current:
            confirmed = current
    final = next(
        (
            payload
            for _at, payload in reversed(events)
            if payload.get("type") == "final"
        ),
        {},
    )
    times = [at for at, _payload in inference_events]
    return {
        "session": index,
        "audio": str(path),
        "audio_seconds": len(pcm) / 32000,
        "event_types": [payload.get("type") for _at, payload in events],
        "events": [
            {"at_seconds": round(at, 6), "payload": payload}
            for at, payload in events
        ],
        "first_result_seconds": times[0] if times else None,
        "update_intervals_seconds": [
            current - previous for previous, current in zip(times, times[1:])
        ],
        "confirmed_monotonic": monotonic,
        "text": final.get("text", ""),
        "error": next(
            (payload.get("error") for _at, payload in events if payload.get("error")),
            None,
        ),
    }


async def run(args: argparse.Namespace) -> dict:
    paths = [Path(filename) for filename in args.audio]
    decoded = [decode(path, args.max_audio_seconds) for path in paths]
    uri = websocket_url(args.url)
    results = await asyncio.gather(
        *[
            session(
                index,
                uri,
                paths[index % len(paths)],
                decoded[index % len(decoded)],
                args.language,
                args.hotword,
                args.prompt,
                args.frame_seconds,
                not args.no_pace,
            )
            for index in range(args.sessions)
        ]
    )
    first_results = [
        item["first_result_seconds"]
        for item in results
        if item["first_result_seconds"] is not None
    ]
    intervals = [
        interval for item in results for interval in item["update_intervals_seconds"]
    ]
    return {
        "url": args.url,
        "sessions": args.sessions,
        "paced": not args.no_pace,
        "context": {
            "hotword_count": len(args.hotword),
            "prompt_present": bool(args.prompt),
        },
        "first_result_p50_seconds": statistics.median(first_results)
        if first_results
        else None,
        "first_result_p95_seconds": percentile(first_results, 0.95),
        "update_interval_p95_seconds": percentile(intervals, 0.95),
        "all_confirmed_monotonic": all(item["confirmed_monotonic"] for item in results),
        "errors": [item["error"] for item in results if item["error"]],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sessions", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--language", default="auto")
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--prompt")
    parser.add_argument("--frame-seconds", type=float, default=1.0)
    parser.add_argument("--max-audio-seconds", type=float, default=12.0)
    parser.add_argument("--no-pace", action="store_true")
    parser.add_argument("audio", nargs="+")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
