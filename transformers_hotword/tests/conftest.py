from __future__ import annotations

import time
import wave
from io import BytesIO

from fastapi.testclient import TestClient

from transformers_hotword.app.backend import InferenceResult
from transformers_hotword.app.config import Settings
from transformers_hotword.app.runtime import Runtime
from transformers_hotword.app.service import create_app


class FakeBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.compile_enabled = True
        self.attention_backend = "flash_attention_2"
        self.dtype_name = "bfloat16"
        self.device_name = "Fake RTX 4090"
        self.batch_sizes: list[int] = []
        self.tasks = []
        self.stream_tasks = []
        self.context_text = "热词识别成功"
        self.fallback_text = "无上下文识别成功"
        self.context_raw_text = "language Chinese<asr_text>热词识别成功"
        self.fallback_raw_text = "language Chinese<asr_text>无上下文识别成功"
        self.raise_without_context: Exception | None = None

    def load(self) -> None:
        return None

    def transcribe_batch(self, tasks):
        self.batch_sizes.append(len(tasks))
        self.tasks.extend(tasks)
        results = []
        for task in tasks:
            if task.prompt is None and self.raise_without_context is not None:
                raise self.raise_without_context
            contextual = task.prompt is not None
            results.append(
                InferenceResult(
                    language=task.language or "Chinese",
                    text=self.context_text if contextual else self.fallback_text,
                    raw_text=(
                        self.context_raw_text
                        if contextual
                        else self.fallback_raw_text
                    ),
                    processing_ms=12.5,
                )
            )
        return results

    def transcribe_stream(self, task, callback):
        self.stream_tasks.append(task)
        if task.prompt is None and self.raise_without_context is not None:
            raise self.raise_without_context
        contextual = task.prompt is not None
        text = self.context_text if contextual else self.fallback_text
        raw_text = self.context_raw_text if contextual else self.fallback_raw_text
        partial = text[: max(1, len(text) // 2)]
        callback(InferenceResult(task.language or "Chinese", partial, partial, 5.0))
        callback(
            InferenceResult(
                task.language or "Chinese", text, raw_text, 8.0
            )
        )
        return InferenceResult(
            task.language or "Chinese",
            text,
            raw_text,
            12.5,
        )

    def info(self):
        return {
            "id": self.settings.model_id,
            "revision": self.settings.model_revision,
            "transformers_revision": self.settings.transformers_revision,
            "device": self.device_name,
            "dtype": self.dtype_name,
            "attention_backend": self.attention_backend,
            "torch_compile": self.compile_enabled,
        }


def wav_bytes(seconds: float = 1.0, sample_rate: int = 16_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * int(seconds * sample_rate))
    return output.getvalue()


def make_client(tmp_path, **overrides):
    values = {
        "temp_dir": tmp_path,
        "torch_compile": False,
        "warmup": False,
        "max_audio_seconds": 4 * 3600,
    }
    values.update(overrides)
    settings = Settings(**values)
    runtime = Runtime(settings, backend_factory=FakeBackend)
    # API unit tests use synthetic PCM. Treat it as speech by default; tests
    # for the contextual pre-gate override this explicitly.
    runtime.audio._speech_ranges = lambda audio: [(0, audio.samples)]
    runtime.audio.warmup_vad = lambda: None
    client = TestClient(create_app(runtime))
    return client, runtime


def wait_ready(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/health/ready").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("fake runtime did not become ready")
