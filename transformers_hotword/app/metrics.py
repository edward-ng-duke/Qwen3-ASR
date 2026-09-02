from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


REQUESTS = Counter("asr_requests_total", "ASR requests", ("mode", "status"))
REQUEST_LATENCY = Histogram(
    "asr_request_duration_seconds", "ASR wall latency", ("mode",)
)
RTF = Histogram(
    "asr_realtime_factor", "Processing seconds divided by audio seconds", ("mode",)
)
AUDIO_SECONDS = Counter("asr_audio_seconds_total", "Audio seconds processed", ("mode",))
QUEUE_DEPTH = Gauge("asr_queue_depth", "Pending GPU inference jobs", ("priority",))
ACTIVE_REALTIME = Gauge(
    "asr_active_realtime_sessions", "Open realtime WebSocket sessions"
)
GPU_MEMORY = Gauge("asr_gpu_memory_bytes", "CUDA memory", ("kind",))
READY = Gauge("asr_ready", "Whether the model is ready")
COMPILE_ENABLED = Gauge("asr_compile_enabled", "Whether torch.compile is active")
CONTEXT_LEAKS = Counter(
    "asr_context_leaks_suppressed_total",
    "Context echo outputs suppressed by the safety guard",
    ("mode",),
)
CONTEXT_FALLBACKS = Counter(
    "asr_context_fallback_total",
    "No-context retries used to protect contextual ASR requests",
    ("mode", "reason", "outcome"),
)
CONTEXT_FALLBACK_DURATION = Histogram(
    "asr_context_fallback_duration_seconds",
    "Wall time spent on no-context fallback inference",
    ("mode", "reason"),
)
PROMPT_VERIFICATIONS = Counter(
    "asr_prompt_verification_total",
    "Cross-language prompt verification decisions",
    ("mode", "outcome"),
)
PROMPT_VERIFICATION_DURATION = Histogram(
    "asr_prompt_verification_duration_seconds",
    "Wall time spent verifying cross-language prompt results",
    ("mode",),
)
REALTIME_CANDIDATE_SEGMENTS = Counter(
    "asr_realtime_candidate_segments_total",
    "Realtime candidate-VAD segment decisions",
    ("outcome",),
)
REALTIME_PARTIAL_INFERENCES = Counter(
    "asr_realtime_partial_inferences_total",
    "Realtime partial inference outcomes",
    ("outcome",),
)
REALTIME_FIRST_PARTIAL_LATENCY = Histogram(
    "asr_realtime_first_partial_latency_seconds",
    "Wall latency from primary VAD start to first non-empty partial",
)


def update_gpu_metrics() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            GPU_MEMORY.labels("used").set(total - free)
            GPU_MEMORY.labels("free").set(free)
            GPU_MEMORY.labels("total").set(total)
            GPU_MEMORY.labels("allocated").set(torch.cuda.memory_allocated())
            GPU_MEMORY.labels("reserved").set(torch.cuda.memory_reserved())
    except Exception:
        return
