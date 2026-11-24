from __future__ import annotations

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import soundfile as sf

from transformers_hotword.app.backend import (
    InferenceResult,
    InferenceTask,
    PartialCallback,
    is_effective_silence,
)
from transformers_hotword.app.languages import LANGUAGE_CODE_TO_NAME

from .config import Settings

LOGGER = logging.getLogger("qwen3_asr_vllm.backend")
LANGUAGE_NAME_TO_CODE = {
    name.casefold(): code for code, name in LANGUAGE_CODE_TO_NAME.items()
}


class VLLMBackend:
    """Synchronous adapter used by the existing priority scheduler."""

    compile_enabled = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.engine_url,
            timeout=httpx.Timeout(settings.engine_timeout_seconds),
            trust_env=False,
        )
        self._engine_model = settings.model_id
        self._engine_info: dict[str, object] = {}

    def load(self) -> None:
        deadline = time.monotonic() + self.settings.engine_timeout_seconds
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                response = self._client.get("/v1/models", timeout=10)
                response.raise_for_status()
                payload = response.json()
                models = payload.get("data") or []
                if not models:
                    raise RuntimeError("vLLM returned an empty model list")
                ids = [str(item.get("id")) for item in models if item.get("id")]
                for candidate in ids:
                    if candidate in self.settings.model_aliases:
                        self._engine_model = candidate
                        break
                else:
                    self._engine_model = ids[0]
                self._engine_info = payload
                LOGGER.info("vLLM engine ready model=%s", self._engine_model)
                return
            except BaseException as exc:
                last_error = exc
                time.sleep(1)
        raise RuntimeError(f"vLLM engine did not become ready: {last_error}")

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _wav(audio: np.ndarray) -> bytes:
        target = io.BytesIO()
        sf.write(target, audio, 16_000, format="WAV", subtype="PCM_16")
        return target.getvalue()

    def _transcribe(self, task: InferenceTask) -> InferenceResult:
        if is_effective_silence(task.audio):
            return InferenceResult(task.language, "", "", 0.0)
        started = time.perf_counter()
        data: dict[str, str] = {"model": self._engine_model}
        if task.language:
            data["language"] = LANGUAGE_NAME_TO_CODE.get(
                task.language.casefold(), task.language
            )
        if task.prompt:
            data["prompt"] = task.prompt
        response = self._client.post(
            "/v1/audio/transcriptions",
            data=data,
            files={"file": ("audio.wav", self._wav(task.audio), "audio/wav")},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"vLLM transcription failed status={response.status_code}: "
                f"{response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            language = task.language
        else:
            text = str(payload.get("text") or "").strip()
            language = payload.get("language") or task.language
        elapsed_ms = (time.perf_counter() - started) * 1000
        return InferenceResult(language, text, text, elapsed_ms)

    def transcribe_batch(self, tasks: list[InferenceTask]) -> list[InferenceResult]:
        if len(tasks) <= 1:
            return [self._transcribe(task) for task in tasks]
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self._transcribe, tasks))

    def transcribe_stream(
        self, task: InferenceTask, callback: PartialCallback
    ) -> InferenceResult:
        # The safety layer must inspect a complete hypothesis before confirming it.
        # Emit one preview for API parity; realtime mode still updates once per window.
        result = self._transcribe(task)
        if result.text:
            callback(result)
        return result

    def info(self) -> dict[str, object]:
        return {
            "id": self.settings.model_id,
            "aliases": sorted(self.settings.model_aliases),
            "revision": self.settings.model_revision,
            "backend": "vllm",
            "vllm_version": os.getenv("ASR_VLLM_VERSION", "0.26.0"),
            "engine_model": self._engine_model,
        }
