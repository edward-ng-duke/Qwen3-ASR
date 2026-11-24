from __future__ import annotations

import time
import wave
from io import BytesIO

from fastapi.testclient import TestClient

from transformers_hotword.app.backend import InferenceResult
from transformers_hotword.app.runtime import Runtime as SharedRuntime

from vllm_hotword.app.config import Settings
from vllm_hotword.app.gateway import create_app


class FakeBackend:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.compile_enabled = False
        self.tasks = []
        self.stream_tasks = []
        self.context_text = "热词识别成功"
        self.fallback_text = "无上下文识别成功"

    def load(self):
        return None

    def transcribe_batch(self, tasks):
        self.tasks.extend(tasks)
        return [
            InferenceResult(
                task.language or "Chinese",
                self.context_text if task.prompt else self.fallback_text,
                self.context_text if task.prompt else self.fallback_text,
                10.0,
            )
            for task in tasks
        ]

    def transcribe_stream(self, task, callback):
        self.stream_tasks.append(task)
        result = self.transcribe_batch([task])[0]
        callback(result)
        return result

    def info(self):
        return {
            "id": self.settings.model_id,
            "revision": self.settings.model_revision,
            "backend": "fake-vllm",
        }


def wav_bytes(seconds: float = 1.0, sample_rate: int = 16_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * int(seconds * sample_rate))
    return output.getvalue()


def make_client(tmp_path):
    settings = Settings(temp_dir=tmp_path, warmup=False, torch_compile=False)
    runtime = SharedRuntime(settings, backend_factory=FakeBackend)
    runtime.audio._speech_ranges = lambda audio: [(0, audio.samples)]
    runtime.audio.warmup_vad = lambda: None
    client = TestClient(create_app(runtime))
    return client, runtime


def wait_ready(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/health/ready").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("fake gateway did not become ready")
