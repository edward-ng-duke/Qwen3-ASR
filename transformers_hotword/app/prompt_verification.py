from __future__ import annotations

import logging
import time
import unicodedata
from difflib import SequenceMatcher

from .backend import InferenceResult, InferenceTask
from .hotwords import ContextGuard
from .metrics import (
    CONTEXT_FALLBACK_DURATION,
    CONTEXT_FALLBACKS,
    CONTEXT_LEAKS,
    PROMPT_VERIFICATION_DURATION,
    PROMPT_VERIFICATIONS,
)
from .scheduler import InferenceScheduler

LOGGER = logging.getLogger("qwen3_asr.prompt_verification")


def requires_prompt_verification(
    requested_language: str | None,
    detected_language: str | None,
    prompt: str | None,
) -> bool:
    """Return whether an English result used a prompt written in another script."""
    target_language = requested_language or detected_language
    if (target_language or "").casefold() != "english" or not prompt:
        return False
    return any(character.isalpha() and not character.isascii() for character in prompt)


def _compact_transcription(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(character for character in normalized if character.isalnum())


def transcription_agreement(contextual: str, no_context: str) -> float:
    """Character-level agreement robust to spacing, punctuation and local fixes."""
    left = _compact_transcription(contextual)
    right = _compact_transcription(no_context)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


async def verify_prompt_result(
    scheduler: InferenceScheduler,
    fallback_task: InferenceTask,
    contextual_result: InferenceResult,
    context: ContextGuard,
    *,
    requested_language: str | None,
    min_agreement: float,
    mode: str,
    request_id: str,
) -> InferenceResult:
    """Select a contextual result only when an unprompted pass supports it."""
    if not contextual_result.text or not requires_prompt_verification(
        requested_language, contextual_result.language, context.prompt
    ):
        return contextual_result
    if fallback_task.prompt is not None:
        raise ValueError("prompt verification task must use prompt=None")

    started = time.perf_counter()
    try:
        no_context_result = await scheduler.submit(fallback_task)
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        PROMPT_VERIFICATIONS.labels(mode, "error").inc()
        PROMPT_VERIFICATION_DURATION.labels(mode).observe(elapsed)
        LOGGER.warning(
            "prompt verification failed request_id=%s mode=%s elapsed_ms=%.1f "
            "error_type=%s",
            request_id,
            mode,
            elapsed * 1000,
            type(exc).__name__,
        )
        raise

    elapsed = time.perf_counter() - started
    agreement = transcription_agreement(
        contextual_result.text, no_context_result.text
    )
    use_no_context = agreement < min_agreement
    outcome = "used_no_context" if use_no_context else "kept_context"
    PROMPT_VERIFICATIONS.labels(mode, outcome).inc()
    PROMPT_VERIFICATION_DURATION.labels(mode).observe(elapsed)

    selected = no_context_result if use_no_context else contextual_result
    if use_no_context:
        CONTEXT_LEAKS.labels(mode).inc()
        fallback_outcome = "nonempty" if no_context_result.text else "empty"
        CONTEXT_FALLBACKS.labels(mode, "context_echo", fallback_outcome).inc()
        CONTEXT_FALLBACK_DURATION.labels(mode, "context_echo").observe(elapsed)
        LOGGER.warning(
            "semantic prompt echo suppressed request_id=%s mode=%s "
            "agreement=%.3f elapsed_ms=%.1f",
            request_id,
            mode,
            agreement,
            elapsed * 1000,
        )
    else:
        LOGGER.info(
            "prompt verification request_id=%s mode=%s outcome=kept_context "
            "agreement=%.3f elapsed_ms=%.1f",
            request_id,
            mode,
            agreement,
            elapsed * 1000,
        )

    return InferenceResult(
        language=selected.language,
        text=selected.text,
        raw_text=selected.raw_text,
        processing_ms=(
            contextual_result.processing_ms + no_context_result.processing_ms
        ),
    )
