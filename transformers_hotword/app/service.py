from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .audio import AudioError, PreparedAudio, SAMPLE_RATE
from .backend import InferenceResult, InferenceTask
from .fallback import submit_no_context_fallback
from .hotwords import (
    ContextGuard,
    HotwordValidationError,
    is_context_leak,
    looks_like_context_prefix,
    make_context,
    parse_hotwords,
)
from .languages import resolve_language
from .metrics import (
    AUDIO_SECONDS,
    CONTEXT_LEAKS,
    REQUESTS,
    REQUEST_LATENCY,
    RTF,
    update_gpu_metrics,
)
from .prompt_verification import (
    requires_prompt_verification,
    verify_prompt_result,
)
from .realtime import SessionLimiter, websocket_asr
from .runtime import Runtime
from .scheduler import OFFLINE_PRIORITY, SchedulerFull
from .schemas import StreamEvent, TranscriptionResponse
from .stability import merge_text

LOGGER = logging.getLogger("qwen3_asr.api")
STATIC_DIR = Path(__file__).parent / "static"


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def create_app(runtime: Runtime | None = None) -> FastAPI:
    runtime = runtime or Runtime()
    limiter = SessionLimiter(runtime.settings.max_realtime_sessions)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start_background()
        yield
        await runtime.close()

    app = FastAPI(
        title="Qwen3-ASR Transformers Hotword",
        version="1.2.2",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = runtime.settings.api_key
        if not expected:
            return
        supplied = ""
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        if not hmac.compare_digest(supplied, expected):
            raise ServiceError(401, "unauthorized", "invalid API key")

    @app.middleware("http")
    async def metadata_logging(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.error(
                "request failed request_id=%s path=%s error_type=UnhandledError",
                request_id,
                request.url.path,
            )
            raise
        response.headers["x-request-id"] = request_id
        LOGGER.info(
            "request request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        if runtime.ready:
            return {
                "status": "ready",
                **runtime.backend.info(),
                "capabilities": {
                    "http": True,
                    "file_sse": True,
                    "realtime": True,
                    "realtime_segment_timestamps": True,
                    "hotwords": True,
                    "prompt": True,
                },
            }
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading" if not runtime.load_error else "failed",
                "detail": None,
            },
        )

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        if not runtime.ready:
            raise ServiceError(503, "not_ready", "model is not ready")
        return {"object": "list", "data": [runtime.backend.info()]}

    @app.get("/metrics", dependencies=[Depends(authorize)])
    async def metrics():
        update_gpu_metrics()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def validate_fields(
        model: str | None,
        language: str | None,
        prompt: str | None,
        hotwords: str | list[str] | None,
    ):
        if model and model != runtime.settings.model_id:
            raise ServiceError(
                422, "unsupported_model", f"only {runtime.settings.model_id} is loaded"
            )
        try:
            resolved_language = resolve_language(language)
            merged_prompt, context = make_context(prompt, parse_hotwords(hotwords))
        except (ValueError, HotwordValidationError) as exc:
            raise ServiceError(422, "invalid_parameters", str(exc)) from exc
        return resolved_language, merged_prompt, context

    async def prepare(upload: UploadFile) -> PreparedAudio:
        if not runtime.ready:
            raise ServiceError(503, "not_ready", "model is not ready")
        try:
            return await runtime.audio.prepare_upload(upload)
        except AudioError as exc:
            status = 413 if "exceeds" in str(exc) else 400
            message = "audio exceeds service limits" if status == 413 else "invalid audio"
            raise ServiceError(status, "invalid_audio", message) from exc

    @app.post(
        "/v1/audio/transcriptions",
        response_model=TranscriptionResponse,
        dependencies=[Depends(authorize)],
    )
    async def transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default="auto"),
        prompt: str | None = Form(default=None),
        hotwords: str | None = Form(default="[]"),
    ):
        request_id = uuid.uuid4().hex
        request_started = time.perf_counter()
        resolved_language, merged_prompt, context = validate_fields(
            model, language, prompt, hotwords
        )
        audio = await prepare(file)
        status = "error"
        try:
            scheduler = runtime.require_scheduler()
            ranges, vad_rejected = await _segment_plan(
                runtime, audio, context.active
            )
            text = ""
            raw_parts: list[str] = []
            detected_language = resolved_language
            for start, end in ranges:
                segment_audio = audio.read(start, end)
                use_context = not vad_rejected
                task = _task(
                    segment_audio,
                    resolved_language,
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
                detected_language = result.language or detected_language
                if use_context and _context_leak_detected(
                    result.text, context, "offline", request_id
                ):
                    fallback = await submit_no_context_fallback(
                        scheduler,
                        _task(
                            segment_audio,
                            resolved_language,
                            None,
                            OFFLINE_PRIORITY,
                        ),
                        mode="offline",
                        reason="context_echo",
                        request_id=request_id,
                    )
                    result = InferenceResult(
                        language=fallback.language,
                        text=fallback.text,
                        raw_text=fallback.raw_text,
                        processing_ms=result.processing_ms + fallback.processing_ms,
                    )
                    detected_language = result.language or detected_language
                elif use_context:
                    result = await verify_prompt_result(
                        scheduler,
                        _task(
                            segment_audio,
                            resolved_language,
                            None,
                            OFFLINE_PRIORITY,
                        ),
                        result,
                        context,
                        requested_language=resolved_language,
                        min_agreement=(
                            runtime.settings.prompt_verification_min_agreement
                        ),
                        mode="offline",
                        request_id=request_id,
                    )
                    detected_language = result.language or detected_language
                text = merge_text(text, result.text)
                if result.raw_text:
                    raw_parts.append(result.raw_text)
            elapsed = time.perf_counter() - request_started
            status = "success"
            return TranscriptionResponse(
                language=detected_language,
                text=text,
                raw_text="\n".join(raw_parts),
                duration=audio.duration,
                processing_ms=elapsed * 1000,
                model=runtime.settings.model_id,
                request_id=request_id,
            )
        except SchedulerFull as exc:
            raise ServiceError(429, "queue_full", str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - request_started
            _record_request("offline", status, elapsed, audio.duration)
            audio.cleanup()

    @app.post("/v1/audio/transcriptions/stream", dependencies=[Depends(authorize)])
    async def transcription_stream(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default="auto"),
        prompt: str | None = Form(default=None),
        hotwords: str | None = Form(default="[]"),
    ):
        request_id = uuid.uuid4().hex
        resolved_language, merged_prompt, context = validate_fields(
            model, language, prompt, hotwords
        )
        audio = await prepare(file)

        async def events():
            sequence = 0
            request_started = time.perf_counter()
            status = "error"
            confirmed = ""
            detected_language = resolved_language
            raw_parts: list[str] = []
            previous_end_ms = 0
            try:
                sequence += 1
                yield _sse(
                    "started",
                    StreamEvent(
                        type="started",
                        duration=audio.duration,
                        sequence=sequence,
                        model=runtime.settings.model_id,
                        request_id=request_id,
                    ),
                )
                ranges, vad_rejected = await _segment_plan(
                    runtime, audio, context.active
                )
                scheduler = runtime.require_scheduler()
                loop = asyncio.get_running_loop()
                for start, end in ranges:
                    segment_audio = audio.read(start, end)
                    use_context = not vad_rejected
                    partials: asyncio.Queue[InferenceResult] = asyncio.Queue()

                    def callback(result: InferenceResult) -> None:
                        loop.call_soon_threadsafe(partials.put_nowait, result)

                    task = _task(
                        segment_audio,
                        resolved_language,
                        merged_prompt if use_context else None,
                        OFFLINE_PRIORITY,
                    )
                    if vad_rejected:
                        inference = asyncio.create_task(
                            submit_no_context_fallback(
                                scheduler,
                                task,
                                mode="sse",
                                reason="vad_reject",
                                request_id=request_id,
                                callback=callback,
                            )
                        )
                    else:
                        inference = asyncio.create_task(
                            scheduler.submit(task, callback)
                        )
                    previous_preview = ""
                    while not inference.done() or not partials.empty():
                        try:
                            partial = await asyncio.wait_for(
                                partials.get(), timeout=0.05
                            )
                        except asyncio.TimeoutError:
                            continue
                        if not partial.text or partial.text == previous_preview:
                            continue
                        if use_context and (
                            looks_like_context_prefix(partial.text, context)
                            or is_context_leak(partial.text, context)
                            or requires_prompt_verification(
                                resolved_language,
                                partial.language,
                                context.prompt,
                            )
                        ):
                            continue
                        delta = (
                            partial.text[len(previous_preview) :]
                            if partial.text.startswith(previous_preview)
                            else ""
                        )
                        previous_preview = partial.text
                        sequence += 1
                        yield _sse(
                            "partial",
                            StreamEvent(
                                type="partial",
                                language=partial.language or detected_language,
                                confirmed=confirmed,
                                delta=delta,
                                preview=partial.text,
                                text=merge_text(confirmed, partial.text),
                                raw_text=partial.raw_text,
                                final=False,
                                sequence=sequence,
                                duration=audio.duration,
                                processing_ms=partial.processing_ms,
                                model=runtime.settings.model_id,
                                request_id=request_id,
                            ),
                        )
                    result = await inference
                    if use_context and _context_leak_detected(
                        result.text, context, "sse", request_id
                    ):
                        fallback = await submit_no_context_fallback(
                            scheduler,
                            _task(
                                segment_audio,
                                resolved_language,
                                None,
                                OFFLINE_PRIORITY,
                            ),
                            mode="sse",
                            reason="context_echo",
                            request_id=request_id,
                        )
                        result = InferenceResult(
                            language=fallback.language,
                            text=fallback.text,
                            raw_text=fallback.raw_text,
                            processing_ms=(
                                result.processing_ms + fallback.processing_ms
                            ),
                        )
                    elif use_context:
                        result = await verify_prompt_result(
                            scheduler,
                            _task(
                                segment_audio,
                                resolved_language,
                                None,
                                OFFLINE_PRIORITY,
                            ),
                            result,
                            context,
                            requested_language=resolved_language,
                            min_agreement=(
                                runtime.settings.prompt_verification_min_agreement
                            ),
                            mode="sse",
                            request_id=request_id,
                        )
                    old_confirmed = confirmed
                    confirmed = merge_text(confirmed, result.text)
                    start_ms, end_ms = _segment_range_ms(
                        start,
                        end,
                        previous_end_ms=previous_end_ms,
                        total_samples=audio.samples,
                    )
                    previous_end_ms = end_ms
                    detected_language = result.language or detected_language
                    if result.raw_text:
                        raw_parts.append(result.raw_text)
                    sequence += 1
                    yield _sse(
                        "segment_final",
                        StreamEvent(
                            type="segment_final",
                            language=detected_language,
                            confirmed=confirmed,
                            delta=confirmed[len(old_confirmed) :]
                            if confirmed.startswith(old_confirmed)
                            else "",
                            text=confirmed,
                            raw_text=result.raw_text,
                            final=False,
                            sequence=sequence,
                            duration=audio.duration,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            processing_ms=result.processing_ms,
                            model=runtime.settings.model_id,
                            request_id=request_id,
                        ),
                    )
                elapsed = time.perf_counter() - request_started
                sequence += 1
                yield _sse(
                    "final",
                    StreamEvent(
                        type="final",
                        language=detected_language,
                        confirmed=confirmed,
                        text=confirmed,
                        raw_text="\n".join(raw_parts),
                        final=True,
                        sequence=sequence,
                        duration=audio.duration,
                        processing_ms=elapsed * 1000,
                        model=runtime.settings.model_id,
                        request_id=request_id,
                    ),
                )
                status = "success"
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                LOGGER.error(
                    "SSE request failed request_id=%s error_type=%s",
                    request_id,
                    type(exc).__name__,
                )
                sequence += 1
                yield _sse(
                    "error",
                    StreamEvent(
                        type="error",
                        final=True,
                        sequence=sequence,
                        duration=audio.duration,
                        model=runtime.settings.model_id,
                        request_id=request_id,
                        error={"code": "inference_failed", "message": "inference failed"},
                    ),
                )
            finally:
                elapsed = time.perf_counter() - request_started
                _record_request("sse", status, elapsed, audio.duration)
                audio.cleanup()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.websocket("/v1/audio/realtime")
    async def realtime(websocket: WebSocket):
        await websocket_asr(websocket, runtime, limiter)

    return app


async def _segment_plan(
    runtime: Runtime, audio: PreparedAudio, context_active: bool
) -> tuple[list[tuple[int, int]], bool]:
    ranges = await asyncio.to_thread(
        runtime.audio.segment_ranges, audio, context_active
    )
    is_short = audio.samples <= int(runtime.settings.segment_seconds * SAMPLE_RATE)
    vad_rejected = context_active and is_short and not ranges
    if vad_rejected:
        return [(0, audio.samples)], True
    return ranges, False


def _task(audio, language, prompt, priority) -> InferenceTask:
    duration = len(audio) / SAMPLE_RATE
    bucket = next(
        (boundary for boundary in (5, 10, 20, 30) if duration <= boundary), 30
    )
    return InferenceTask(audio, language, prompt, priority, bucket)


def _segment_range_ms(
    start: int,
    end: int,
    *,
    previous_end_ms: int,
    total_samples: int,
) -> tuple[int, int]:
    """Convert a decoded/VAD sample range to a monotonic source-audio range.

    Long inference chunks intentionally overlap.  The public range describes only
    the newly committed source interval, so its start is clipped to the preceding
    committed end.  These are approximate segment boundaries, not word alignment.
    """
    bounded_start = min(max(start, 0), total_samples)
    bounded_end = min(max(end, bounded_start), total_samples)
    raw_start_ms = round(bounded_start * 1000 / SAMPLE_RATE)
    raw_end_ms = round(bounded_end * 1000 / SAMPLE_RATE)
    start_ms = max(raw_start_ms, previous_end_ms)
    if raw_end_ms <= start_ms:
        raise ValueError(
            "segment range is empty after removing inference overlap: "
            f"start_ms={start_ms} end_ms={raw_end_ms}"
        )
    return start_ms, raw_end_ms


def _sse(event: str, payload: StreamEvent) -> str:
    data = json.dumps(payload.model_dump(), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _record_request(mode: str, status: str, elapsed: float, duration: float) -> None:
    REQUESTS.labels(mode, status).inc()
    REQUEST_LATENCY.labels(mode).observe(elapsed)
    if duration > 0:
        AUDIO_SECONDS.labels(mode).inc(duration)
        RTF.labels(mode).observe(elapsed / duration)


def _context_leak_detected(
    text: str, context: ContextGuard, mode: str, request_id: str
) -> bool:
    if not is_context_leak(text, context):
        return False
    CONTEXT_LEAKS.labels(mode).inc()
    LOGGER.warning("context echo suppressed request_id=%s mode=%s", request_id, mode)
    return True


app = create_app()
