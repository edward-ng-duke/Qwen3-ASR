from __future__ import annotations

import os
from dataclasses import dataclass

from transformers_hotword.app.config import Settings as BaseSettings


@dataclass(frozen=True, slots=True)
class Settings(BaseSettings):
    model_id: str = os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
    model_revision: str = os.getenv(
        "ASR_MODEL_REVISION", "7278e1e70fe206f11671096ffdd38061171dd6e5"
    )
    transformers_revision: str = "vLLM-0.26.0"
    max_realtime_sessions: int = int(
        os.getenv("ASR_MAX_REALTIME_SESSIONS", "4")
    )
    engine_url: str = os.getenv("ASR_ENGINE_URL", "http://asr-engine:8000")
    engine_timeout_seconds: float = float(
        os.getenv("ASR_ENGINE_TIMEOUT_SECONDS", "600")
    )

    @property
    def model_aliases(self) -> frozenset[str]:
        return frozenset(
            {"models/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B", self.model_id}
        )
