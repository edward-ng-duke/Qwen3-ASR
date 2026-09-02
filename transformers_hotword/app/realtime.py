from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid

import numpy as np
from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from .audio import AudioError, PCMStreamDecoder, RealtimeVAD, SAMPLE_RATE
from .backend import InferenceResult, InferenceTask
from .fallback import submit_no_context_fallback
from .hotwords import (
    HotwordValidationError,
    is_context_leak,
    looks_like_context_prefix,
    make_context,
)
from .languages import resolve_language
from .metrics import (
    ACTIVE_REALTIME,
    AUDIO_SECONDS,
    CONTEXT_LEAKS,
    REALTIME_CANDIDATE_SEGMENTS,
    REALTIME_FIRST_PARTIAL_LATENCY,
    REALTIME_PARTIAL_INFERENCES,
    REQUESTS,
    REQUEST_LATENCY,
    RTF,
)
from .prompt_verification import (
    requires_prompt_verification,
    verify_prompt_result,
)
from .runtime import Runtime
from .scheduler import REALTIME_PRIORITY, SchedulerFull
from .schemas import RealtimeEvent, RealtimeInit
from .stability import (
    PrefixStabilizer,
    merge_final_segment,
    merge_text,
    normalize_preview_after_confirmed,
)

LOGGER = logging.getLogger("qwen3_asr.realtime")
MAX_WS_FRAME_BYTES = 1024 * 1024


class SessionLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            ACTIVE_REALTIME.set(self.active)
            return True

    async def release(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)
            ACTIVE_REALTIME.set(self.active)


class _TimelineBuffer:
    """Bounded PCM buffer addressed with absolute 16 kHz sample positions."""

    def __init__(self) -> None:
        self.samples = np.empty(0, dtype=np.float32)
        self.start = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.size:
            self.samples = np.concatenate((self.samples, samples))

    def read(self, start: int, end: int) -> np.ndarray:
        start = max(start, self.start)
        end = min(end, self.start + self.samples.size)
        if end <= start:
            return np.empty(0, dtype=np.float32)
        return self.samples[start - self.start : end - self.start].copy()

    def trim_before(self, position: int) -> None:
        position = min(max(position, self.start), self.start + self.samples.size)
        trim = position - self.start
        if trim:
            self.samples = self.samples[trim:]
            self.start = position


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _subtract_intervals(
    start: int, end: int, covered: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    residuals: list[tuple[int, int]] = []
    cursor = start
    for covered_start, covered_end in _merge_intervals(covered):
        if covered_end <= cursor or covered_start >= end:
            continue
        if covered_start > cursor:
            residuals.append((cursor, min(covered_start, end)))
        cursor = max(cursor, covered_end)
        if cursor >= end:
            break
    if cursor < end:
        residuals.append((cursor, end))
    return residuals


async def websocket_asr(
    websocket: WebSocket, runtime: Runtime, limiter: SessionLimiter
) -> None:
    request_id = uuid.uuid4().hex
    await websocket.accept()
    if not await limiter.acquire():
        await _send_error(
            websocket,
            runtime,
            request_id,
            "too_many_sessions",
            "realtime session limit reached",
        )
        await websocket.close(code=1013)
        return

    started = time.perf_counter()
    audio_seconds = 0.0
    status = "error"
    sequence = 0
    try:
        if not runtime.ready:
            await _send_error(
                websocket,
                runtime,
                request_id,
                "not_ready",
                "model is not ready",
            )
            await websocket.close(code=1013)
            return
        try:
            init_payload = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            init = RealtimeInit.model_validate(init_payload)
            language = resolve_language(init.language)
            prompt, context = make_context(init.prompt, init.hotwords)
        except (
            asyncio.TimeoutError,
            ValidationError,
            ValueError,
            HotwordValidationError,
        ):
            await _send_error(
                websocket, runtime, request_id, "invalid_init", "invalid start frame"
            )
            await websocket.close(code=1008)
            return

        header = websocket.headers.get("authorization", "")
        header_key = header[7:] if header.lower().startswith("bearer ") else ""
        supplied_key = init.api_key or header_key
        if runtime.settings.api_key and not hmac.compare_digest(
            supplied_key, runtime.settings.api_key
        ):
            await _send_error(
                websocket, runtime, request_id, "unauthorized", "invalid API key"
            )
            await websocket.close(code=4401)
            return

        decoder = PCMStreamDecoder(init.encoding, init.sample_rate, init.channels)
        primary_vad = RealtimeVAD(
            runtime.settings.vad_threshold, runtime.settings.vad_min_silence_ms
        )
        candidate_vad = (
            RealtimeVAD(
                runtime.settings.fallback_vad_threshold,
                runtime.settings.vad_min_silence_ms,
            )
            if context.active
            else None
        )
        scheduler = runtime.require_scheduler()
        model_id = runtime.settings.model_id
        timeline = _TimelineBuffer()
        total_samples = 0
        last_inference_sample = 0
        first_partial_visible = False
        primary_started_at: float | None = None
        primary_start: int | None = None
        primary_cursor: int | None = None
        candidate_start: int | None = None
        candidate_cursor: int | None = None
        primary_covered: list[tuple[int, int]] = []
        confirmed = ""
        emitted_text = ""
        raw_finals: list[str] = []
        detected_language: str | None = language
        processing_total_ms = 0.0
        previous_segment_end_sample = 0
        stabilizer = PrefixStabilizer(rollback_units=2)

        async def send_event(
            event_type: str,
            *,
            final: bool,
            segment: np.ndarray | None = None,
            use_context: bool = True,
            fallback_reason: str | None = None,
            start_sample: int | None = None,
            end_sample: int | None = None,
        ) -> bool:
            nonlocal confirmed
            nonlocal emitted_text
            nonlocal detected_language
            nonlocal sequence
            nonlocal processing_total_ms
            nonlocal previous_segment_end_sample
            nonlocal stabilizer

            start_ms: int | None = None
            end_ms: int | None = None
            if event_type == "segment_final":
                if start_sample is None or end_sample is None:
                    LOGGER.warning(
                        "realtime segment suppressed request_id=%s "
                        "reason=missing_boundaries",
                        request_id,
                    )
                    return False
                bounded_start = min(max(0, start_sample), total_samples)
                bounded_end = min(max(bounded_start, end_sample), total_samples)
                commit_start = max(bounded_start, previous_segment_end_sample)
                # Primary VAD has already enforced its own speech minimum. Only
                # a tail created by overlap clipping needs the explicit guard;
                # keeping unclipped short ranges also preserves low-latency
                # utterances and the existing protocol contract.
                minimum = (
                    int(runtime.settings.vad_min_speech_ms * SAMPLE_RATE / 1000)
                    if commit_start > bounded_start
                    else 1
                )
                if bounded_end - commit_start < minimum:
                    LOGGER.info(
                        "realtime segment suppressed request_id=%s "
                        "reason=non_overlapping_or_short duration_ms=%.1f",
                        request_id,
                        max(0, bounded_end - commit_start) * 1000 / SAMPLE_RATE,
                    )
                    return False
                segment = timeline.read(commit_start, bounded_end)
                if segment.size < minimum:
                    LOGGER.info(
                        "realtime segment suppressed request_id=%s "
                        "reason=timeline_range_unavailable duration_ms=%.1f",
                        request_id,
                        segment.size * 1000 / SAMPLE_RATE,
                    )
                    return False
                start_sample = commit_start
                end_sample = bounded_end
                start_ms = round(commit_start * 1000 / SAMPLE_RATE)
                end_ms = round(bounded_end * 1000 / SAMPLE_RATE)

            if segment is None or segment.size == 0:
                sequence += 1
                await websocket.send_json(
                    RealtimeEvent(
                        type=event_type,
                        language=detected_language,
                        confirmed=confirmed,
                        text=confirmed,
                        final=final,
                        sequence=sequence,
                        duration=total_samples / SAMPLE_RATE,
                        processing_ms=processing_total_ms,
                        model=model_id,
                        request_id=request_id,
                    ).model_dump()
                )
                return True

            task = InferenceTask(
                audio=np.asarray(segment, dtype=np.float32),
                language=language,
                prompt=prompt if use_context else None,
                priority=REALTIME_PRIORITY,
                bucket=_bucket(segment.size / SAMPLE_RATE),
            )
            if fallback_reason is not None:
                result = await submit_no_context_fallback(
                    scheduler,
                    task,
                    mode="realtime",
                    reason=fallback_reason,
                    request_id=request_id,
                )
            else:
                result = await scheduler.submit(task)
            event_processing_ms = result.processing_ms
            processing_total_ms += result.processing_ms
            detected_language = result.language or detected_language

            if (
                not final
                and use_context
                and (
                    looks_like_context_prefix(result.text, context)
                    or is_context_leak(result.text, context)
                    or requires_prompt_verification(
                        language, result.language, context.prompt
                    )
                )
            ):
                if event_type == "partial":
                    REALTIME_PARTIAL_INFERENCES.labels("suppressed").inc()
                return False

            if final and use_context and is_context_leak(result.text, context):
                CONTEXT_LEAKS.labels("realtime").inc()
                LOGGER.warning(
                    "context echo suppressed request_id=%s mode=realtime", request_id
                )
                fallback = await submit_no_context_fallback(
                    scheduler,
                    InferenceTask(
                        audio=np.asarray(segment, dtype=np.float32),
                        language=language,
                        prompt=None,
                        priority=REALTIME_PRIORITY,
                        bucket=_bucket(segment.size / SAMPLE_RATE),
                    ),
                    mode="realtime",
                    reason="context_echo",
                    request_id=request_id,
                )
                result = InferenceResult(
                    language=fallback.language,
                    text=fallback.text,
                    raw_text=fallback.raw_text,
                    processing_ms=result.processing_ms + fallback.processing_ms,
                )
                event_processing_ms = result.processing_ms
                processing_total_ms += fallback.processing_ms
                detected_language = result.language or detected_language
            elif final and use_context:
                contextual_processing_ms = result.processing_ms
                result = await verify_prompt_result(
                    scheduler,
                    InferenceTask(
                        audio=np.asarray(segment, dtype=np.float32),
                        language=language,
                        prompt=None,
                        priority=REALTIME_PRIORITY,
                        bucket=_bucket(segment.size / SAMPLE_RATE),
                    ),
                    result,
                    context,
                    requested_language=language,
                    min_agreement=(
                        runtime.settings.prompt_verification_min_agreement
                    ),
                    mode="realtime",
                    request_id=request_id,
                )
                event_processing_ms = result.processing_ms
                processing_total_ms += (
                    result.processing_ms - contextual_processing_ms
                )
                detected_language = result.language or detected_language

            if final:
                segment_text = stabilizer.finalize(result.text)
                previous_confirmed = confirmed
                confirmed = merge_final_segment(confirmed, segment_text)
                delta = (
                    confirmed[len(previous_confirmed) :]
                    if confirmed.startswith(previous_confirmed)
                    else ""
                )
                preview = ""
                full_text = confirmed
                event_confirmed = confirmed
                if result.raw_text:
                    raw_finals.append(result.raw_text)
            else:
                stable = stabilizer.update(result.text)
                event_confirmed = merge_text(confirmed, stable.confirmed)
                preview = normalize_preview_after_confirmed(
                    event_confirmed,
                    stable.preview,
                )
                full_text = merge_text(event_confirmed, preview)
                delta = (
                    full_text[len(emitted_text) :]
                    if full_text.startswith(emitted_text)
                    else ""
                )

            sequence += 1
            emitted_text = full_text
            if event_type == "segment_final":
                # Boundaries were validated and clipped before inference/state
                # mutation, so a benign VAD overlap cannot terminate finish.
                assert start_sample is not None and end_sample is not None
                previous_segment_end_sample = end_sample
            await websocket.send_json(
                RealtimeEvent(
                    type=event_type,
                    language=detected_language,
                    confirmed=event_confirmed,
                    delta=delta,
                    preview=preview,
                    text=full_text,
                    raw_text=result.raw_text,
                    final=False,
                    sequence=sequence,
                    duration=total_samples / SAMPLE_RATE,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    processing_ms=event_processing_ms,
                    model=model_id,
                    request_id=request_id,
                ).model_dump()
            )
            if final:
                stabilizer = PrefixStabilizer(rollback_units=2)
                emitted_text = confirmed
            visible = bool(event_confirmed or preview)
            if event_type == "partial":
                REALTIME_PARTIAL_INFERENCES.labels(
                    "visible" if visible else "empty"
                ).inc()
            return visible

        def add_primary_coverage(start: int, end: int) -> None:
            nonlocal primary_covered
            primary_covered = _merge_intervals(primary_covered + [(start, end)])

        async def send_primary(start: int, end: int) -> None:
            segment = timeline.read(start, end)
            if segment.size:
                await send_event(
                    "segment_final",
                    final=True,
                    segment=segment,
                    use_context=context.active,
                    start_sample=start,
                    end_sample=end,
                )

        async def send_candidate_residuals(start: int, end: int) -> None:
            coverage = list(primary_covered)
            if primary_start is not None:
                coverage.append((primary_start, end))
            minimum = int(runtime.settings.vad_min_speech_ms * SAMPLE_RATE / 1000)
            for residual_start, residual_end in _subtract_intervals(
                start, end, coverage
            ):
                if residual_end - residual_start < minimum:
                    REALTIME_CANDIDATE_SEGMENTS.labels("too_short").inc()
                    continue
                segment = timeline.read(residual_start, residual_end)
                if segment.size < minimum:
                    REALTIME_CANDIDATE_SEGMENTS.labels("too_short").inc()
                    continue
                rms = float(
                    np.sqrt(np.mean(np.square(segment, dtype=np.float64)))
                )
                if rms < runtime.settings.candidate_min_rms:
                    REALTIME_CANDIDATE_SEGMENTS.labels("low_energy").inc()
                    LOGGER.info(
                        "candidate residual suppressed request_id=%s "
                        "reason=low_energy duration_ms=%.1f rms=%.6f",
                        request_id,
                        segment.size * 1000 / SAMPLE_RATE,
                        rms,
                    )
                    continue
                REALTIME_CANDIDATE_SEGMENTS.labels("submitted").inc()
                await send_event(
                    "segment_final",
                    final=True,
                    segment=segment,
                    use_context=False,
                    fallback_reason="vad_reject",
                    start_sample=residual_start,
                    end_sample=residual_end,
                )

        async def process_vad_events(
            primary_events: list[dict[str, int]],
            candidate_events: list[dict[str, int]],
        ) -> None:
            nonlocal primary_start
            nonlocal primary_cursor
            nonlocal candidate_start
            nonlocal candidate_cursor
            nonlocal last_inference_sample
            nonlocal first_partial_visible
            nonlocal primary_started_at

            candidate_was_reset = False

            for event in primary_events:
                if "start" in event and primary_start is None:
                    start = min(max(0, event["start"]), total_samples)
                    primary_start = start
                    primary_cursor = start
                    last_inference_sample = start
                    first_partial_visible = False
                    primary_started_at = time.monotonic()
                if "end" in event and primary_start is not None:
                    end = min(max(primary_start, event["end"]), total_samples)
                    start = primary_cursor if primary_cursor is not None else primary_start
                    await send_primary(start, end)
                    add_primary_coverage(primary_start, end)
                    if candidate_vad is not None:
                        if candidate_start is not None:
                            REALTIME_CANDIDATE_SEGMENTS.labels(
                                "overlap_reset"
                            ).inc()
                        candidate_start = None
                        candidate_cursor = None
                        reset_candidate = getattr(candidate_vad, "reset", None)
                        if reset_candidate is not None:
                            reset_candidate()
                        candidate_was_reset = True
                    primary_start = None
                    primary_cursor = None
                    last_inference_sample = total_samples
                    first_partial_visible = False
                    primary_started_at = None

            for event in ([] if candidate_was_reset else candidate_events):
                if "start" in event and candidate_start is None:
                    start = min(max(0, event["start"]), total_samples)
                    candidate_start = start
                    candidate_cursor = start
                if "end" in event and candidate_start is not None:
                    end = min(max(candidate_start, event["end"]), total_samples)
                    start = (
                        candidate_cursor
                        if candidate_cursor is not None
                        else candidate_start
                    )
                    await send_candidate_residuals(start, end)
                    candidate_start = None
                    candidate_cursor = None

        async def flush_full_windows() -> None:
            nonlocal primary_cursor
            nonlocal candidate_cursor
            nonlocal last_inference_sample
            window = int(runtime.settings.realtime_window_seconds * SAMPLE_RATE)
            candidate_window = int(
                min(10.0, runtime.settings.realtime_window_seconds) * SAMPLE_RATE
            )
            overlap = int(runtime.settings.realtime_overlap_seconds * SAMPLE_RATE)
            advance = window - overlap
            candidate_advance = candidate_window - overlap

            while (
                primary_start is not None
                and primary_cursor is not None
                and total_samples - primary_cursor >= window
            ):
                end = primary_cursor + window
                await send_primary(primary_cursor, end)
                add_primary_coverage(primary_cursor, end)
                primary_cursor += advance
                last_inference_sample = total_samples

            while (
                candidate_start is not None
                and candidate_cursor is not None
                and total_samples - candidate_cursor >= candidate_window
            ):
                end = candidate_cursor + candidate_window
                await send_candidate_residuals(candidate_cursor, end)
                candidate_cursor += candidate_advance

        def trim_timeline() -> None:
            keep_from = max(0, total_samples - 2 * SAMPLE_RATE)
            if primary_cursor is not None:
                keep_from = min(keep_from, primary_cursor)
            if candidate_cursor is not None:
                keep_from = min(keep_from, candidate_cursor)
            timeline.trim_before(keep_from)
            cutoff = timeline.start
            primary_covered[:] = [
                (max(start, cutoff), end)
                for start, end in primary_covered
                if end > cutoff
            ]

        await websocket.send_json(
            RealtimeEvent(
                type="ready",
                final=False,
                model=model_id,
                request_id=request_id,
            ).model_dump()
        )

        should_finish = False
        while not should_finish:
            message = await asyncio.wait_for(
                websocket.receive(), timeout=runtime.settings.realtime_idle_seconds
            )
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            binary = message.get("bytes")
            text = message.get("text")
            if binary is not None:
                if len(binary) > MAX_WS_FRAME_BYTES:
                    sequence += 1
                    await _send_error(
                        websocket,
                        runtime,
                        request_id,
                        "frame_too_large",
                        "binary frame exceeds 1 MiB",
                        sequence=sequence,
                    )
                    await websocket.close(code=1009)
                    return
                samples = decoder.feed(binary)
                if samples.size == 0:
                    continue
                timeline.append(samples)
                total_samples += samples.size
                audio_seconds = total_samples / SAMPLE_RATE
                if audio_seconds > runtime.settings.max_session_seconds:
                    sequence += 1
                    await _send_error(
                        websocket,
                        runtime,
                        request_id,
                        "session_too_long",
                        "session exceeds one hour",
                        sequence=sequence,
                    )
                    await websocket.close(code=1008)
                    return

                primary_events = primary_vad.feed(samples)
                candidate_events = (
                    candidate_vad.feed(samples) if candidate_vad is not None else []
                )
                await process_vad_events(primary_events, candidate_events)
                await flush_full_windows()

                window = int(runtime.settings.realtime_window_seconds * SAMPLE_RATE)
                update_seconds = (
                    runtime.settings.realtime_update_seconds
                    if first_partial_visible
                    else runtime.settings.realtime_first_update_seconds
                )
                if (
                    primary_start is not None
                    and total_samples - last_inference_sample
                    >= update_seconds * SAMPLE_RATE
                ):
                    start = max(
                        primary_cursor if primary_cursor is not None else primary_start,
                        total_samples - window,
                    )
                    visible = await send_event(
                        "partial",
                        final=False,
                        segment=timeline.read(start, total_samples),
                        use_context=context.active,
                    )
                    last_inference_sample = total_samples
                    if visible and not first_partial_visible:
                        first_partial_visible = True
                        if primary_started_at is not None:
                            REALTIME_FIRST_PARTIAL_LATENCY.observe(
                                max(0.0, time.monotonic() - primary_started_at)
                            )
                trim_timeline()
            elif text is not None:
                try:
                    control = __import__("json").loads(text)
                except ValueError:
                    control = {}
                control_type = control.get("type")
                if control_type == "finish":
                    should_finish = True
                elif control_type == "ping":
                    sequence += 1
                    await websocket.send_json(
                        RealtimeEvent(
                            type="pong",
                            sequence=sequence,
                            duration=total_samples / SAMPLE_RATE,
                            processing_ms=processing_total_ms,
                            model=model_id,
                            request_id=request_id,
                        ).model_dump()
                    )
                else:
                    sequence += 1
                    await _send_error(
                        websocket,
                        runtime,
                        request_id,
                        "invalid_control",
                        "use ping or finish",
                        sequence=sequence,
                    )

        tail = decoder.finish()
        primary_finish_events: list[dict[str, int]] = []
        candidate_finish_events: list[dict[str, int]] = []
        if tail.size:
            timeline.append(tail)
            total_samples += tail.size
            primary_finish_events.extend(primary_vad.feed(tail))
            if candidate_vad is not None:
                candidate_finish_events.extend(candidate_vad.feed(tail))
        primary_finish_events.extend(primary_vad.finish())
        if candidate_vad is not None:
            candidate_finish_events.extend(candidate_vad.finish())
        await process_vad_events(primary_finish_events, candidate_finish_events)
        await flush_full_windows()

        if primary_start is not None:
            start = primary_cursor if primary_cursor is not None else primary_start
            await send_primary(start, total_samples)
            add_primary_coverage(primary_start, total_samples)
            primary_start = None
            primary_cursor = None
        if candidate_start is not None:
            start = (
                candidate_cursor if candidate_cursor is not None else candidate_start
            )
            await send_candidate_residuals(start, total_samples)
            candidate_start = None
            candidate_cursor = None

        sequence += 1
        await websocket.send_json(
            RealtimeEvent(
                type="final",
                language=detected_language,
                confirmed=confirmed,
                text=confirmed,
                raw_text="\n".join(raw_finals),
                final=True,
                sequence=sequence,
                duration=total_samples / SAMPLE_RATE,
                processing_ms=processing_total_ms,
                model=model_id,
                request_id=request_id,
            ).model_dump()
        )
        await websocket.close(code=1000)
        status = "success"
    except WebSocketDisconnect:
        status = "cancelled"
    except asyncio.TimeoutError:
        sequence += 1
        await _safe_error(
            websocket,
            runtime,
            request_id,
            "idle_timeout",
            "no data received for five minutes",
            1008,
            sequence=sequence,
        )
    except (AudioError, SchedulerFull, ValueError):
        sequence += 1
        await _safe_error(
            websocket,
            runtime,
            request_id,
            "invalid_request",
            "invalid realtime request",
            1008,
            sequence=sequence,
        )
    except Exception as exc:
        LOGGER.error(
            "realtime request failed request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )
        sequence += 1
        await _safe_error(
            websocket,
            runtime,
            request_id,
            "internal_error",
            "internal realtime error",
            1011,
            sequence=sequence,
        )
    finally:
        elapsed = time.perf_counter() - started
        REQUESTS.labels("realtime", status).inc()
        REQUEST_LATENCY.labels("realtime").observe(elapsed)
        if audio_seconds > 0:
            AUDIO_SECONDS.labels("realtime").inc(audio_seconds)
            RTF.labels("realtime").observe(elapsed / audio_seconds)
        await limiter.release()


async def _send_error(
    websocket: WebSocket,
    runtime: Runtime,
    request_id: str,
    code: str,
    message: str,
    *,
    sequence: int = 0,
) -> None:
    await websocket.send_json(
        RealtimeEvent(
            type="error",
            final=True,
            model=runtime.settings.model_id,
            request_id=request_id,
            sequence=sequence,
            error={"code": code, "message": message},
        ).model_dump()
    )


async def _safe_error(
    websocket: WebSocket,
    runtime: Runtime,
    request_id: str,
    code: str,
    message: str,
    close_code: int,
    *,
    sequence: int = 0,
) -> None:
    try:
        await _send_error(
            websocket,
            runtime,
            request_id,
            code,
            message,
            sequence=sequence,
        )
        await websocket.close(code=close_code)
    except Exception:
        pass


def _bucket(duration: float) -> int:
    for boundary in (5, 10, 20, 30):
        if duration <= boundary:
            return boundary
    return 30
