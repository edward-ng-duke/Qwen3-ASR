from __future__ import annotations

import asyncio
import hmac
import importlib.util
import logging
from pathlib import Path
import time
from typing import Literal

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

import transformers_hotword.app.realtime as shared_realtime_module
import transformers_hotword.app.service as shared_service_module
from transformers_hotword.app.audio import AudioError
from transformers_hotword.app.backend import InferenceResult
from transformers_hotword.app.fallback import submit_no_context_fallback
from transformers_hotword.app.hotwords import (
    HotwordValidationError,
    is_context_leak,
)
from transformers_hotword.app.languages import resolve_language
from transformers_hotword.app.prompt_verification import verify_prompt_result
from transformers_hotword.app.scheduler import OFFLINE_PRIORITY, SchedulerFull
from transformers_hotword.app.stability import merge_text

from .context import make_context, parse_hotwords
from .runtime import Runtime

LOGGER = logging.getLogger("qwen3_asr_vllm.gateway")


def _isolated_module(name: str, source: str):
    """Load shared code with private globals so compatibility hooks cannot leak."""
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared module {source}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "transformers_hotword.app"
    spec.loader.exec_module(module)
    return module


realtime_module = _isolated_module(
    "transformers_hotword.app._vllm_realtime",
    str(Path(shared_realtime_module.__file__).resolve()),
)
service_module = _isolated_module(
    "transformers_hotword.app._vllm_service",
    str(Path(shared_service_module.__file__).resolve()),
)
ServiceError = service_module.ServiceError


class RealtimeInit(BaseModel):
    type: Literal["start"] = "start"
    encoding: Literal["pcm_s16le", "pcm_f32le"]
    sample_rate: int = Field(ge=8000, le=96000)
    channels: Literal[1, 2] = 1
    language: str = "auto"
    prompt: str | None = None
    hotwords: str | list[str] = Field(default_factory=list)
    api_key: str | None = None


def _install_compatibility_hooks() -> None:
    # Reuse the tested SSE/WS safety implementation while widening only the
    # hotwords input type. The Transformers deployment itself is not modified.
    service_module.parse_hotwords = parse_hotwords
    service_module.make_context = make_context
    service_module.websocket_asr = realtime_module.websocket_asr
    realtime_module.make_context = make_context
    realtime_module.RealtimeInit = RealtimeInit


def _authorized(request: Request, runtime: Runtime) -> None:
    expected = runtime.settings.api_key
    if not expected:
        return
    authorization = request.headers.get("authorization", "")
    supplied = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(supplied, expected):
        raise ServiceError(401, "unauthorized", "invalid API key")


def _field(form, name: str, default: str | None = None) -> str | None:
    value = form.get(name, default)
    if value is None or isinstance(value, UploadFile):
        return default
    return str(value)


def _is_true(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"1", "true", "yes", "on"}


async def _requires_decode_fallback(upload: UploadFile) -> bool:
    suffix = Path(upload.filename or "").suffix.casefold()
    if suffix != ".wav":
        # Native vLLM audio/transcriptions accepts a narrower set than the
        # shared FFmpeg pipeline (field validation differs by container build).
        # Decode every non-WAV format here so MP3/M4A/FLAC/OGG behave exactly
        # like the Transformers deployment.
        return True
    await upload.seek(0)
    header = await upload.read(28)
    await upload.seek(0)
    if len(header) < 28 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return True
    sample_rate = int.from_bytes(header[24:28], "little")
    return sample_rate != 16_000


def _upstream_headers(response: httpx.Response) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("content-type", "content-disposition", "cache-control"):
        if value := response.headers.get(name):
            result[name] = value
    return result


async def _proxy_multipart(request: Request, runtime: Runtime, form):
    data: dict[str, str | list[str]] = {}
    files: list[tuple[str, tuple[str, object, str]]] = []
    uploads: list[UploadFile] = []
    for name, value in form.multi_items():
        if isinstance(value, UploadFile):
            await value.seek(0)
            uploads.append(value)
            files.append(
                (
                    name,
                    (
                        value.filename or "audio.bin",
                        value.file,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
        elif name != "hotwords":
            if name == "language" and str(value).strip().casefold() in {"", "auto"}:
                continue
            normalized = (
                runtime.settings.model_id if name == "model" else str(value)
            )
            previous = data.get(name)
            if previous is None:
                data[name] = normalized
            elif isinstance(previous, list):
                previous.append(normalized)
            else:
                data[name] = [previous, normalized]
    if not uploads:
        raise ServiceError(400, "invalid_audio", "a multipart file upload is required")

    client = httpx.Client(
        base_url=runtime.settings.engine_url,
        timeout=httpx.Timeout(runtime.settings.engine_timeout_seconds),
        trust_env=False,
    )
    upstream_request = client.build_request(
        "POST", "/v1/audio/transcriptions", data=data, files=files
    )
    response = await asyncio.to_thread(client.send, upstream_request, stream=True)
    if response.status_code >= 400:
        status_code = response.status_code
        code_by_status = {
            400: "invalid_audio",
            401: "unauthorized",
            403: "forbidden",
            413: "invalid_audio",
            422: "invalid_parameters",
            429: "queue_full",
            503: "not_ready",
        }
        code = code_by_status.get(status_code, "upstream_error")
        await asyncio.to_thread(response.close)
        await asyncio.to_thread(client.close)
        for upload in uploads:
            await upload.close()
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": code, "message": "ASR request failed"}},
        )
    iterator = response.iter_raw()

    def next_chunk() -> bytes | None:
        try:
            return next(iterator)
        except StopIteration:
            return None

    async def body():
        try:
            while (chunk := await asyncio.to_thread(next_chunk)) is not None:
                yield chunk
        finally:
            await asyncio.to_thread(response.close)
            await asyncio.to_thread(client.close)
            for upload in uploads:
                await upload.close()

    return StreamingResponse(
        body(),
        status_code=response.status_code,
        headers=_upstream_headers(response),
        media_type=None,
    )


async def _safe_context_transcription(
    runtime: Runtime,
    upload: UploadFile,
    language: str | None,
    merged_prompt: str | None,
    context,
    request_id: str,
) -> tuple[str | None, str, float, float]:
    await upload.seek(0)
    try:
        audio = await runtime.audio.prepare_upload(upload)
    except AudioError as exc:
        status = 413 if "exceeds" in str(exc) else 400
        message = "audio exceeds service limits" if status == 413 else "invalid audio"
        raise ServiceError(status, "invalid_audio", message) from exc

    started = time.perf_counter()
    detected_language = language
    final_text = ""
    try:
        ranges, vad_rejected = await service_module._segment_plan(
            runtime, audio, context.active
        )
        scheduler = runtime.require_scheduler()
        for start, end in ranges:
            samples = audio.read(start, end)
            use_context = context.active and not vad_rejected
            task = service_module._task(
                samples,
                language,
                merged_prompt if use_context else None,
                OFFLINE_PRIORITY,
            )
            if vad_rejected:
                result = await submit_no_context_fallback(
                    scheduler,
                    task,
                    mode="offline",
                    reason="vad_reject",
                    request_id=request_id,
                )
            else:
                result = await scheduler.submit(task)

            if use_context and is_context_leak(result.text, context):
                service_module.CONTEXT_LEAKS.labels("offline").inc()
                LOGGER.warning(
                    "context echo suppressed request_id=%s mode=offline", request_id
                )
                fallback = await submit_no_context_fallback(
                    scheduler,
                    service_module._task(
                        samples, language, None, OFFLINE_PRIORITY
                    ),
                    mode="offline",
                    reason="context_echo",
                    request_id=request_id,
                )
                result = InferenceResult(
                    fallback.language,
                    fallback.text,
                    fallback.raw_text,
                    result.processing_ms + fallback.processing_ms,
                )
            elif use_context:
                result = await verify_prompt_result(
                    scheduler,
                    service_module._task(
                        samples, language, None, OFFLINE_PRIORITY
                    ),
                    result,
                    context,
                    requested_language=language,
                    min_agreement=runtime.settings.prompt_verification_min_agreement,
                    mode="offline",
                    request_id=request_id,
                )
            detected_language = result.language or detected_language
            final_text = merge_text(final_text, result.text)
        return (
            detected_language,
            final_text,
            audio.duration,
            (time.perf_counter() - started) * 1000,
        )
    except SchedulerFull as exc:
        raise ServiceError(429, "queue_full", "GPU inference queue is full") from exc
    finally:
        audio.cleanup()


def create_app(runtime: Runtime | None = None):
    _install_compatibility_hooks()
    runtime = runtime or Runtime()
    app = service_module.create_app(runtime)  # tested SSE/WS/health/metrics routes
    app.title = "Qwen3-ASR vLLM Hotword Gateway"
    app.version = "1.2.2"

    removed = {"/", "/v1/audio/transcriptions", "/v1/models"}
    app.router.routes = [
        route for route in app.router.routes if getattr(route, "path", None) not in removed
    ]

    @app.post("/v1/audio/transcriptions")
    async def native_transcription(request: Request):
        _authorized(request, runtime)
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type.casefold():
            raise ServiceError(
                400, "invalid_request", "multipart/form-data is required"
            )
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise ServiceError(400, "invalid_audio", "field 'file' must be an upload")

        model = _field(form, "model", runtime.settings.model_id)
        if model not in runtime.settings.model_aliases:
            raise ServiceError(422, "unsupported_model", f"unsupported model {model!r}")
        prompt = _field(form, "prompt")
        hotwords = _field(form, "hotwords")
        try:
            merged_prompt, context = make_context(prompt, hotwords)
        except HotwordValidationError as exc:
            raise ServiceError(422, "invalid_parameters", str(exc)) from exc

        requires_decode_fallback = await _requires_decode_fallback(upload)
        if not context.active and not requires_decode_fallback:
            return await _proxy_multipart(request, runtime, form)
        if context.active and _is_true(_field(form, "stream")):
            raise ServiceError(
                422,
                "unsafe_context_stream",
                "contextual streaming must use /v1/audio/transcriptions/stream",
            )

        response_format = (_field(form, "response_format", "json") or "json").lower()
        if response_format not in {"json", "verbose_json", "text"}:
            raise ServiceError(
                422,
                "unsupported_response_format",
                "contextual requests support json, verbose_json, or text",
            )
        try:
            language = resolve_language(_field(form, "language", "auto"))
        except ValueError as exc:
            raise ServiceError(422, "invalid_parameters", str(exc)) from exc

        request_id = request.headers.get("x-request-id", "") or __import__(
            "uuid"
        ).uuid4().hex
        detected, text, duration, processing_ms = await _safe_context_transcription(
            runtime,
            upload,
            language,
            merged_prompt,
            context,
            request_id,
        )
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": detected,
                    "duration": duration,
                    "text": text,
                    "segments": [],
                    "usage": {"type": "duration", "seconds": duration},
                }
            )
        response = JSONResponse({"text": text})
        response.headers["x-asr-processing-ms"] = f"{processing_ms:.3f}"
        return response

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy_vllm(path: str, request: Request):
        _authorized(request, runtime)
        body = await request.body()
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.casefold()
            not in {"host", "content-length", "connection", "authorization"}
        }
        async with httpx.AsyncClient(
            base_url=runtime.settings.engine_url,
            timeout=httpx.Timeout(runtime.settings.engine_timeout_seconds),
            trust_env=False,
        ) as client:
            response = await client.request(
                request.method,
                f"/{path}",
                params=request.query_params,
                headers=headers,
                content=body,
            )
        return Response(
            response.content,
            status_code=response.status_code,
            headers=_upstream_headers(response),
        )

    return app


app = create_app()
