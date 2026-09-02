from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path = Path(
        os.getenv("ASR_MODEL_PATH", "/opt/models/Qwen3-ASR-1.7B-hf")
    )
    model_id: str = os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B-hf")
    model_revision: str = os.getenv(
        "ASR_MODEL_REVISION", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    )
    transformers_revision: str = os.getenv(
        "ASR_TRANSFORMERS_REVISION", "c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d"
    )
    host: str = os.getenv("ASR_HOST", "0.0.0.0")
    port: int = int(os.getenv("ASR_PORT", "8000"))
    device: str = os.getenv("ASR_DEVICE", "cuda:0")
    api_key: str = os.getenv("ASR_API_KEY", "")
    temp_dir: Path = Path(os.getenv("ASR_TEMP_DIR", "/var/lib/qwen3-asr/tmp"))
    attention_backend: str = os.getenv("ASR_ATTENTION_BACKEND", "auto")
    # Eager is the measured fastest single-stream configuration on the target
    # RTX 4090. Keep compile available as an opt-in experiment.
    torch_compile: bool = _bool_env("ASR_TORCH_COMPILE", False)
    torch_compile_mode: str = os.getenv("ASR_TORCH_COMPILE_MODE", "default")
    warmup: bool = _bool_env("ASR_WARMUP", True)
    warmup_seconds: tuple[int, ...] = (1, 10, 30)
    max_new_tokens: int = int(os.getenv("ASR_MAX_NEW_TOKENS", "512"))
    max_batch_size: int = int(os.getenv("ASR_MAX_BATCH_SIZE", "4"))
    max_pending_requests: int = int(os.getenv("ASR_MAX_PENDING_REQUESTS", "32"))
    max_realtime_sessions: int = int(os.getenv("ASR_MAX_REALTIME_SESSIONS", "4"))
    max_upload_bytes: int = int(os.getenv("ASR_MAX_UPLOAD_BYTES", str(8 * 1024**3)))
    max_audio_seconds: float = float(os.getenv("ASR_MAX_AUDIO_SECONDS", str(4 * 3600)))
    max_session_seconds: float = float(os.getenv("ASR_MAX_SESSION_SECONDS", "3600"))
    realtime_idle_seconds: float = float(os.getenv("ASR_REALTIME_IDLE_SECONDS", "300"))
    realtime_update_seconds: float = float(
        os.getenv("ASR_REALTIME_UPDATE_SECONDS", "1")
    )
    realtime_first_update_seconds: float = float(
        os.getenv("ASR_REALTIME_FIRST_UPDATE_SECONDS", "0.5")
    )
    realtime_window_seconds: float = float(
        os.getenv("ASR_REALTIME_WINDOW_SECONDS", "10")
    )
    realtime_overlap_seconds: float = float(
        os.getenv("ASR_REALTIME_OVERLAP_SECONDS", "2")
    )
    segment_seconds: float = float(os.getenv("ASR_SEGMENT_SECONDS", "30"))
    segment_overlap_seconds: float = float(
        os.getenv("ASR_SEGMENT_OVERLAP_SECONDS", "1")
    )
    vad_threshold: float = float(os.getenv("ASR_VAD_THRESHOLD", "0.5"))
    fallback_vad_threshold: float = float(
        os.getenv("ASR_FALLBACK_VAD_THRESHOLD", "0.1")
    )
    candidate_min_rms: float = float(
        os.getenv("ASR_CANDIDATE_MIN_RMS", "0.0015")
    )
    prompt_verification_min_agreement: float = float(
        os.getenv("ASR_PROMPT_VERIFICATION_MIN_AGREEMENT", "0.45")
    )
    vad_min_speech_ms: int = int(os.getenv("ASR_VAD_MIN_SPEECH_MS", "250"))
    vad_min_silence_ms: int = int(os.getenv("ASR_VAD_MIN_SILENCE_MS", "600"))
    log_level: str = os.getenv("ASR_LOG_LEVEL", "INFO")

    def validate(self) -> None:
        if self.attention_backend not in {"auto", "flash_attention_2", "sdpa"}:
            raise ValueError(
                "ASR_ATTENTION_BACKEND must be auto, flash_attention_2, or sdpa"
            )
        if self.torch_compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ValueError(
                "ASR_TORCH_COMPILE_MODE must be default, reduce-overhead, or max-autotune"
            )
        if not 1 <= self.max_batch_size <= 4:
            raise ValueError("ASR_MAX_BATCH_SIZE must be between 1 and 4")
        if not 0 < self.realtime_first_update_seconds <= self.realtime_update_seconds:
            raise ValueError(
                "ASR_REALTIME_FIRST_UPDATE_SECONDS must be greater than 0 and "
                "no greater than ASR_REALTIME_UPDATE_SECONDS"
            )
        if self.realtime_overlap_seconds >= self.realtime_window_seconds:
            raise ValueError("realtime overlap must be shorter than its window")
        if self.segment_overlap_seconds >= self.segment_seconds:
            raise ValueError("segment overlap must be shorter than its segment")
        if not 0 < self.fallback_vad_threshold < self.vad_threshold:
            raise ValueError(
                "ASR_FALLBACK_VAD_THRESHOLD must be greater than 0 and lower "
                "than ASR_VAD_THRESHOLD"
            )
        if not 0 <= self.candidate_min_rms < 1:
            raise ValueError("ASR_CANDIDATE_MIN_RMS must be between 0 and 1")
        if not 0 < self.prompt_verification_min_agreement < 1:
            raise ValueError(
                "ASR_PROMPT_VERIFICATION_MIN_AGREEMENT must be between 0 and 1"
            )
