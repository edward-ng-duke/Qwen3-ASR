from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse

from transformers_hotword.app.hotwords import HotwordValidationError

import vllm_hotword.app.gateway as gateway_module
from vllm_hotword.app.context import make_context, parse_hotwords
from vllm_hotword.app.backend import LANGUAGE_NAME_TO_CODE
from vllm_hotword.app.gateway import RealtimeInit

from .conftest import make_client, wait_ready, wav_bytes


def test_dual_hotword_formats_and_merge():
    assert parse_hotwords("DeepSeek, 智谱, DeepSeek") == ["DeepSeek", "智谱"]
    assert parse_hotwords('[" DeepSeek ", "智谱"]') == ["DeepSeek", "智谱"]
    assert parse_hotwords("Vocabulary: OpenAI，Qwen") == ["OpenAI", "Qwen"]
    prompt, guard = make_context("技术会议", "DeepSeek, 智谱")
    assert prompt == "技术会议\nVocabulary: DeepSeek, 智谱"
    assert guard.hotwords == ("DeepSeek", "智谱")
    assert LANGUAGE_NAME_TO_CODE["english"] == "en"
    assert LANGUAGE_NAME_TO_CODE["cantonese"] == "yue"


def test_invalid_json_and_limits():
    with pytest.raises(HotwordValidationError):
        parse_hotwords('["broken"')
    with pytest.raises(HotwordValidationError):
        parse_hotwords(["x" * 65])
    with pytest.raises(HotwordValidationError):
        make_context("x" * 4097, [])


def test_realtime_init_accepts_string_and_array():
    base = {"encoding": "pcm_s16le", "sample_rate": 16000}
    assert RealtimeInit.model_validate({**base, "hotwords": "Qwen, 智谱"}).hotwords
    assert RealtimeInit.model_validate({**base, "hotwords": ["Qwen"]}).hotwords


def test_native_context_request_accepts_both_model_aliases(tmp_path):
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        for model in ("models/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B"):
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
                data={
                    "model": model,
                    "language": "zh",
                    "prompt": "技术会议",
                    "hotwords": '["DeepSeek", "智谱"]',
                },
            )
            assert response.status_code == 200, response.text
            assert response.json() == {"text": "热词识别成功"}
        assert runtime.backend.tasks[-1].prompt == (
            "技术会议\nVocabulary: DeepSeek, 智谱"
        )


def test_native_context_stream_redirects_to_safe_endpoint(tmp_path):
    client, _runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": "Qwen", "stream": "true"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsafe_context_stream"


def test_no_context_request_uses_native_proxy(tmp_path, monkeypatch):
    called = []

    async def fake_proxy(_request, _runtime, form):
        called.append(tuple(form.keys()))
        return JSONResponse({"text": "native"})

    monkeypatch.setattr(gateway_module, "_proxy_multipart", fake_proxy)
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={"model": "models/Qwen3-ASR-1.7B", "language": "auto"},
        )
    assert response.json() == {"text": "native"}
    assert called
    assert runtime.backend.tasks == []


def test_native_proxy_normalizes_upstream_error_without_leaking_body(
    tmp_path, monkeypatch
):
    class FakeResponse:
        status_code = 503
        headers = {"content-type": "application/json"}

        def close(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, *args, **kwargs):
            return object()

        def send(self, *_args, **_kwargs):
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setattr(gateway_module.httpx, "Client", FakeClient)
    client, _runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={"language": "auto"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "not_ready", "message": "ASR request failed"}
    }


def test_48khz_wav_uses_decode_fallback_without_context(tmp_path):
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={
                "file": (
                    "sample.wav",
                    wav_bytes(sample_rate=48_000),
                    "audio/wav",
                )
            },
            data={"language": "zh"},
        )
    assert response.status_code == 200
    assert response.json() == {"text": "无上下文识别成功"}
    assert runtime.backend.tasks[-1].prompt is None


def test_non_wav_uses_shared_decode_path_without_context(tmp_path):
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.mp3", wav_bytes(), "audio/mpeg")},
            data={"language": "zh"},
        )
    assert response.status_code == 200
    assert response.json() == {"text": "无上下文识别成功"}
    assert runtime.backend.tasks[-1].prompt is None


def test_sse_plain_string_hotwords_has_monotonic_sequence(tmp_path):
    client, _runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions/stream",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={"model": "Qwen/Qwen3-ASR-1.7B", "hotwords": "Qwen, 智谱"},
        )
    assert response.status_code == 200
    payloads = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [item["sequence"] for item in payloads] == sorted(
        item["sequence"] for item in payloads
    )
    assert [item["type"] for item in payloads][0] == "started"
    assert [item["type"] for item in payloads][-1] == "final"
    segment = next(item for item in payloads if item["type"] == "segment_final")
    assert segment["start_ms"] == 0
    assert segment["end_ms"] == 1000
    assert all(
        item["start_ms"] is None and item["end_ms"] is None
        for item in payloads
        if item["type"] != "segment_final"
    )


def test_vad_reject_falls_back_without_context(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.audio._speech_ranges = lambda _audio: []
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("short.wav", wav_bytes(0.8), "audio/wav")},
            data={"hotwords": "Qwen"},
        )
    assert response.json() == {"text": "无上下文识别成功"}
    assert len(runtime.backend.tasks) == 1
    assert runtime.backend.tasks[0].prompt is None
