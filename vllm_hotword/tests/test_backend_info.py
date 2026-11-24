from __future__ import annotations

from vllm_hotword.app.backend import VLLMBackend
from vllm_hotword.app.config import Settings


def test_backend_info_defaults_to_bundled_vllm(monkeypatch):
    monkeypatch.delenv("ASR_VLLM_VERSION", raising=False)
    backend = VLLMBackend(Settings())
    try:
        assert backend.info()["vllm_version"] == "0.26.0"
    finally:
        backend.close()


def test_backend_info_can_report_compat_engine_version(monkeypatch):
    monkeypatch.setenv("ASR_VLLM_VERSION", "0.14.0")
    backend = VLLMBackend(Settings())
    try:
        assert backend.info()["vllm_version"] == "0.14.0"
    finally:
        backend.close()
