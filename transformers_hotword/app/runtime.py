from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .audio import AudioPipeline
from .backend import TransformersBackend
from .config import Settings
from .metrics import COMPILE_ENABLED, READY
from .scheduler import InferenceScheduler

LOGGER = logging.getLogger("qwen3_asr.runtime")


class Runtime:
    def __init__(
        self,
        settings: Settings | None = None,
        backend_factory: Callable[
            [Settings], TransformersBackend
        ] = TransformersBackend,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.validate()
        self.backend = backend_factory(self.settings)
        self.audio = AudioPipeline(
            self.settings.temp_dir,
            self.settings.max_upload_bytes,
            self.settings.max_audio_seconds,
            self.settings.segment_seconds,
            self.settings.segment_overlap_seconds,
            self.settings.vad_threshold,
            self.settings.vad_min_speech_ms,
            self.settings.vad_min_silence_ms,
        )
        self.scheduler: InferenceScheduler | None = None
        self.ready = False
        self.load_error: str | None = None
        self._loader: asyncio.Task | None = None

    def start_background(self) -> None:
        self.audio.cleanup_stale()
        self._loader = asyncio.create_task(self._load(), name="qwen3-asr-model-loader")

    async def _load(self) -> None:
        try:
            await asyncio.to_thread(self.backend.load)
            await asyncio.to_thread(self.audio.warmup_vad)
            self.scheduler = InferenceScheduler(
                self.backend,
                self.settings.max_pending_requests,
                self.settings.max_batch_size,
            )
            self.scheduler.start()
            self.ready = True
            READY.set(1)
            COMPILE_ENABLED.set(1 if self.backend.compile_enabled else 0)
        except BaseException as exc:
            LOGGER.exception("model initialization failed")
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            READY.set(0)

    async def close(self) -> None:
        if self.scheduler is not None:
            await self.scheduler.close()
        if self._loader is not None and not self._loader.done():
            self._loader.cancel()

    def require_scheduler(self) -> InferenceScheduler:
        if not self.ready or self.scheduler is None:
            detail = self.load_error or "model is still loading"
            raise RuntimeError(detail)
        return self.scheduler
