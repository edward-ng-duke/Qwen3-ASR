from __future__ import annotations

import logging
import time

from .backend import InferenceResult, InferenceTask, PartialCallback
from .metrics import CONTEXT_FALLBACK_DURATION, CONTEXT_FALLBACKS
from .scheduler import InferenceScheduler

LOGGER = logging.getLogger("qwen3_asr.fallback")
VALID_REASONS = {"vad_reject", "context_echo"}


async def submit_no_context_fallback(
    scheduler: InferenceScheduler,
    task: InferenceTask,
    *,
    mode: str,
    reason: str,
    request_id: str,
    callback: PartialCallback | None = None,
) -> InferenceResult:
    """Submit exactly one unguarded, no-context inference retry."""
    if reason not in VALID_REASONS:
        raise ValueError(f"unsupported fallback reason: {reason}")
    if task.prompt is not None:
        raise ValueError("fallback inference task must use prompt=None")

    started = time.perf_counter()
    try:
        result = await scheduler.submit(task, callback)
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        CONTEXT_FALLBACKS.labels(mode, reason, "error").inc()
        CONTEXT_FALLBACK_DURATION.labels(mode, reason).observe(elapsed)
        LOGGER.warning(
            "context fallback failed request_id=%s mode=%s reason=%s "
            "elapsed_ms=%.1f error_type=%s",
            request_id,
            mode,
            reason,
            elapsed * 1000,
            type(exc).__name__,
        )
        raise

    elapsed = time.perf_counter() - started
    outcome = "nonempty" if result.text else "empty"
    CONTEXT_FALLBACKS.labels(mode, reason, outcome).inc()
    CONTEXT_FALLBACK_DURATION.labels(mode, reason).observe(elapsed)
    LOGGER.info(
        "context fallback request_id=%s mode=%s reason=%s outcome=%s "
        "elapsed_ms=%.1f",
        request_id,
        mode,
        reason,
        outcome,
        elapsed * 1000,
    )
    return result
