#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import wave

import numpy as np
from websockets.sync.client import connect


BASE = "http://127.0.0.1:8000"


def wait_ready() -> None:
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/health/ready", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            time.sleep(2)
    raise RuntimeError("service did not become ready in ten minutes")


def wav_data() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    return output.getvalue()


def pink_noise() -> np.ndarray:
    rng = np.random.default_rng(7)
    bins = 3 * 16000 // 2 + 1
    spectrum = rng.normal(size=bins) + 1j * rng.normal(size=bins)
    frequencies = np.arange(bins, dtype=np.float64)
    frequencies[0] = 1.0
    spectrum /= np.sqrt(frequencies)
    samples = np.fft.irfft(spectrum, n=3 * 16000).astype(np.float32)
    samples *= 0.05 / max(float(np.max(np.abs(samples))), 1e-12)
    return samples


def wav_from_float32(samples: np.ndarray) -> bytes:
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()


def multipart(path: str, audio: bytes | None = None) -> dict:
    boundary = "offline-smoke-boundary"
    audio = audio or wav_data()
    pieces = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\nen\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="hotwords"\r\n\r\n["Quilter"]\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
        audio,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    request = urllib.request.Request(
        BASE + path,
        data=b"".join(pieces),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = response.read().decode()
        if path.endswith("/stream"):
            assert "event: final" in body
            assert "Quilter" not in body
            return {"sse": "ok"}
        return json.loads(body)


def websocket_smoke(samples: np.ndarray | None = None) -> None:
    with connect("ws://127.0.0.1:8000/v1/audio/realtime", open_timeout=10) as websocket:
        websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "en",
                    "hotwords": ["Quilter"],
                }
            )
        )
        assert json.loads(websocket.recv())["type"] == "ready"
        payload = samples if samples is not None else np.zeros(16000, np.float32)
        websocket.send(np.asarray(payload, dtype="<f4").tobytes())
        websocket.send(json.dumps({"type": "finish"}))
        while True:
            event = json.loads(websocket.recv())
            assert "Quilter" not in event.get("text", "")
            if event["type"] == "final":
                if samples is not None:
                    assert event["text"] == ""
                break


def main() -> None:
    wait_ready()
    with urllib.request.urlopen(BASE + "/", timeout=5) as response:
        assert b"Qwen3-ASR" in response.read()
    print(json.dumps(multipart("/v1/audio/transcriptions"), ensure_ascii=False))
    print(json.dumps(multipart("/v1/audio/transcriptions/stream"), ensure_ascii=False))
    noise = pink_noise()
    noise_response = multipart("/v1/audio/transcriptions", wav_from_float32(noise))
    assert "Quilter" not in noise_response["text"]
    websocket_smoke(noise)
    print("offline smoke passed")


if __name__ == "__main__":
    main()
