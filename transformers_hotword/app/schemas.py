from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    language: str | None
    text: str
    raw_text: str
    duration: float
    processing_ms: float
    model: str
    request_id: str


class StreamEvent(BaseModel):
    type: Literal["started", "partial", "segment_final", "final", "error"]
    language: str | None = None
    confirmed: str = ""
    delta: str = ""
    preview: str = ""
    text: str = ""
    raw_text: str = ""
    final: bool = False
    sequence: int = 0
    duration: float = 0.0
    start_ms: int | None = None
    end_ms: int | None = None
    processing_ms: float = 0.0
    model: str
    request_id: str
    error: dict[str, str] | None = None


class RealtimeInit(BaseModel):
    type: Literal["start"] = "start"
    encoding: Literal["pcm_s16le", "pcm_f32le"]
    sample_rate: int = Field(ge=8000, le=96000)
    channels: Literal[1, 2] = 1
    language: str = "auto"
    prompt: str | None = None
    hotwords: str | list[str] = Field(default_factory=list)
    api_key: str | None = None


class RealtimeEvent(BaseModel):
    type: Literal["ready", "partial", "segment_final", "final", "pong", "error"]
    language: str | None = None
    confirmed: str = ""
    delta: str = ""
    preview: str = ""
    text: str = ""
    raw_text: str = ""
    final: bool = False
    sequence: int = 0
    duration: float = 0.0
    start_ms: int | None = None
    end_ms: int | None = None
    processing_ms: float = 0.0
    model: str
    request_id: str
    error: dict[str, str] | None = None
