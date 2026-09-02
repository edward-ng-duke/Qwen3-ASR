from __future__ import annotations

import json
import time

import numpy as np

from transformers_hotword.app import realtime
from transformers_hotword.app.service import _segment_range_ms

from .conftest import make_client, wait_ready, wav_bytes


def test_health_model_and_offline_transcription(tmp_path):
    client, _runtime = make_client(tmp_path)
    with client:
        assert client.get("/health/live").status_code == 200
        wait_ready(client)
        model = client.get("/v1/models").json()["data"][0]
        assert model["attention_backend"] == "flash_attention_2"
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"language": "zh", "prompt": "测试", "hotwords": '["Qwen3-ASR"]'},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["language"] == "Chinese"
        assert payload["text"] == "热词识别成功"
        assert payload["duration"] == 1.0


def test_auth_and_dual_hotword_formats(tmp_path):
    client, _runtime = make_client(tmp_path, api_key="secret")
    with client:
        wait_ready(client)
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            ).status_code
            == 200
        )
        for hotwords in ("not-json", '["not-json"]'):
            response = client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer secret"},
                files={"file": ("short.wav", wav_bytes(), "audio/wav")},
                data={"hotwords": hotwords},
            )
            assert response.status_code == 200

        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer secret"},
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["broken"'},
        )
        assert response.status_code == 422


def _sse_payloads(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_contextual_http_falls_back_without_context_after_vad_reject(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.audio._speech_ranges = lambda _audio: []
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("noise.wav", wav_bytes(), "audio/wav")},
            data={"language": "zh", "hotwords": '["DeepSeek", "智谱"]'},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "无上下文识别成功"
        assert [task.prompt for task in runtime.backend.tasks] == [None]


def test_contextual_http_falls_back_once_after_context_echo(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "DeepSeek，智谱，英伟达。"
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"language": "zh", "hotwords": '["DeepSeek", "智谱", "英伟达"]'},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "无上下文识别成功"
        assert len(runtime.backend.tasks) == 2
        assert runtime.backend.tasks[0].prompt is not None
        assert runtime.backend.tasks[1].prompt is None


def test_contextual_http_normal_result_only_runs_once(tmp_path):
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["DeepSeek"]'},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "热词识别成功"
        assert len(runtime.backend.tasks) == 1
        assert runtime.backend.tasks[0].prompt is not None


def test_english_translated_prompt_echo_uses_no_context_result(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "This is a technical discussion meeting."
    runtime.backend.context_raw_text = runtime.backend.context_text
    runtime.backend.fallback_text = "Ahem."
    runtime.backend.fallback_raw_text = "Ahem."
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("cough.wav", wav_bytes(2.04), "audio/wav")},
            data={"language": "en", "prompt": "这是一场技术讨论会。"},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "Ahem."
        assert [task.prompt for task in runtime.backend.tasks] == [
            "这是一场技术讨论会。",
            None,
        ]


def test_english_prompt_keeps_context_when_no_context_result_agrees(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "We deployed Azure in production today."
    runtime.backend.fallback_text = "We deployed Astroic in production today."
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("speech.wav", wav_bytes(2.5), "audio/wav")},
            data={"language": "en", "prompt": "这是一次云平台部署会议。"},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "We deployed Azure in production today."
        assert len(runtime.backend.tasks) == 2


def test_english_hotwords_without_prompt_do_not_add_verification(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "DeepSeek was deployed."
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("speech.wav", wav_bytes(), "audio/wav")},
            data={"language": "en", "hotwords": '["DeepSeek"]'},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "DeepSeek was deployed."
        assert len(runtime.backend.tasks) == 1


def test_contextual_http_returns_empty_fallback_without_retrying_again(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.audio._speech_ranges = lambda _audio: []
    runtime.backend.fallback_text = ""
    runtime.backend.fallback_raw_text = ""
    with client:
        wait_ready(client)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["DeepSeek"]'},
        )
        assert response.status_code == 200
        assert response.json()["text"] == ""
        assert len(runtime.backend.tasks) == 1
        assert runtime.backend.tasks[0].prompt is None


def test_sse_event_order(tmp_path):
    client, _runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.stream(
            "POST",
            "/v1/audio/transcriptions/stream",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["热词"]'},
        ) as response:
            body = "".join(response.iter_text())
        event_names = [
            line.removeprefix("event: ")
            for line in body.splitlines()
            if line.startswith("event: ")
        ]
        assert event_names[0] == "started"
        assert "partial" in event_names
        assert "segment_final" in event_names
        assert event_names[-1] == "final"
        payloads = _sse_payloads(body)
        segment = next(item for item in payloads if item["type"] == "segment_final")
        assert (segment["start_ms"], segment["end_ms"]) == (0, 1000)
        assert all(
            item["start_ms"] is None and item["end_ms"] is None
            for item in payloads
            if item["type"] != "segment_final"
        )


def test_sse_no_speech_finishes_without_segment(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.audio._speech_ranges = lambda _audio: []
    with client:
        wait_ready(client)
        with client.stream(
            "POST",
            "/v1/audio/transcriptions/stream",
            files={"file": ("silence.wav", wav_bytes(31), "audio/wav")},
        ) as response:
            payloads = _sse_payloads("".join(response.iter_text()))

    assert [item["type"] for item in payloads] == ["started", "final"]
    assert payloads[-1]["final"] is True
    assert payloads[-1]["text"] == ""


def test_segment_range_ms_removes_long_chunk_overlap():
    first = _segment_range_ms(
        0,
        30 * 16_000,
        previous_end_ms=0,
        total_samples=58 * 16_000,
    )
    second = _segment_range_ms(
        28 * 16_000,
        58 * 16_000,
        previous_end_ms=first[1],
        total_samples=58 * 16_000,
    )
    assert first == (0, 30_000)
    assert second == (30_000, 58_000)


def test_segment_range_ms_preserves_vad_silence_and_clamps_to_audio():
    first = _segment_range_ms(
        2 * 16_000,
        5 * 16_000,
        previous_end_ms=0,
        total_samples=10 * 16_000,
    )
    second = _segment_range_ms(
        7 * 16_000,
        12 * 16_000,
        previous_end_ms=first[1],
        total_samples=10 * 16_000,
    )
    assert first == (2_000, 5_000)
    assert second == (7_000, 10_000)


def test_sse_vad_reject_streams_one_no_context_result(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.audio._speech_ranges = lambda _audio: []
    with client:
        wait_ready(client)
        with client.stream(
            "POST",
            "/v1/audio/transcriptions/stream",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["DeepSeek", "智谱"]'},
        ) as response:
            body = "".join(response.iter_text())
        payloads = _sse_payloads(body)
        assert [item["sequence"] for item in payloads] == list(
            range(1, len(payloads) + 1)
        )
        assert [item["type"] for item in payloads].count("segment_final") == 1
        assert payloads[-1]["text"] == "无上下文识别成功"
        assert [task.prompt for task in runtime.backend.stream_tasks] == [None]


def test_sse_context_echo_retries_without_replaying_partials(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "DeepSeek，智谱，英伟达。"
    with client:
        wait_ready(client)
        with client.stream(
            "POST",
            "/v1/audio/transcriptions/stream",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["DeepSeek", "智谱", "英伟达"]'},
        ) as response:
            body = "".join(response.iter_text())
        payloads = _sse_payloads(body)
        finals = [item for item in payloads if item["type"] == "segment_final"]
        assert len(finals) == 1
        assert finals[0]["text"] == "无上下文识别成功"
        assert all("DeepSeek，智谱，英伟达" not in item.get("confirmed", "") for item in payloads)
        assert len(runtime.backend.stream_tasks) == 1
        assert runtime.backend.stream_tasks[0].prompt is not None
        assert len(runtime.backend.tasks) == 1
        assert runtime.backend.tasks[0].prompt is None


def test_sse_english_prompt_translation_never_reaches_events(tmp_path):
    client, runtime = make_client(tmp_path)
    translated = "This is a technical discussion meeting."
    runtime.backend.context_text = translated
    runtime.backend.context_raw_text = translated
    runtime.backend.fallback_text = "Ahem."
    runtime.backend.fallback_raw_text = "Ahem."
    with client:
        wait_ready(client)
        with client.stream(
            "POST",
            "/v1/audio/transcriptions/stream",
            files={"file": ("cough.wav", wav_bytes(2.04), "audio/wav")},
            data={"language": "en", "prompt": "这是一场技术讨论会。"},
        ) as response:
            body = "".join(response.iter_text())
        payloads = _sse_payloads(body)
        assert translated not in body
        assert [item["type"] for item in payloads] == [
            "started",
            "segment_final",
            "final",
        ]
        assert payloads[-1]["text"] == "Ahem."
        assert len(runtime.backend.stream_tasks) == 1
        assert len(runtime.backend.tasks) == 1
        assert runtime.backend.tasks[0].prompt is None


def test_sse_fallback_error_emits_error_without_suppressed_text(tmp_path):
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "DeepSeek，智谱，英伟达。"
    runtime.backend.raise_without_context = RuntimeError("fallback failed")
    with client:
        wait_ready(client)
        with client.stream(
            "POST",
            "/v1/audio/transcriptions/stream",
            files={"file": ("short.wav", wav_bytes(), "audio/wav")},
            data={"hotwords": '["DeepSeek", "智谱", "英伟达"]'},
        ) as response:
            body = "".join(response.iter_text())
        payloads = _sse_payloads(body)
        assert payloads[-1]["type"] == "error"
        assert payloads[-1]["error"] == {
            "code": "inference_failed",
            "message": "inference failed",
        }
        assert "fallback failed" not in body
        assert "DeepSeek，智谱，英伟达" not in "".join(
            item.get("confirmed", "") for item in payloads
        )


def test_websocket_realtime(tmp_path, monkeypatch):
    class FakeVAD:
        def __init__(self, *_args):
            self.started = False

        def feed(self, samples):
            if not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", FakeVAD)
    client, _runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["热词"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(np.zeros(16000, dtype="<f4").tobytes())
            assert websocket.receive_json()["type"] == "partial"
            websocket.send_text(json.dumps({"type": "finish"}))
            assert websocket.receive_json()["type"] == "segment_final"
            final = websocket.receive_json()
            assert final["type"] == "final"
            assert final["final"] is True


def test_websocket_first_partial_uses_half_second_then_one_second(
    tmp_path, monkeypatch
):
    class FakeVAD:
        def __init__(self, *_args):
            self.started = False

        def feed(self, samples):
            if not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", FakeVAD)
    client, runtime = make_client(
        tmp_path,
        realtime_first_update_seconds=0.5,
        realtime_update_seconds=1.0,
    )
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                }
            )
            assert websocket.receive_json()["type"] == "ready"

            websocket.send_bytes(np.zeros(7840, dtype="<f4").tobytes())
            time.sleep(0.02)
            assert len(runtime.backend.tasks) == 0

            websocket.send_bytes(np.zeros(160, dtype="<f4").tobytes())
            assert websocket.receive_json()["type"] == "partial"
            first_count = len(runtime.backend.tasks)
            assert first_count == 1

            websocket.send_bytes(np.zeros(15840, dtype="<f4").tobytes())
            time.sleep(0.02)
            assert len(runtime.backend.tasks) == first_count

            websocket.send_bytes(np.zeros(160, dtype="<f4").tobytes())
            assert websocket.receive_json()["type"] == "partial"
            assert len(runtime.backend.tasks) == first_count + 1

            websocket.send_text(json.dumps({"type": "finish"}))
            assert websocket.receive_json()["type"] == "segment_final"
            assert websocket.receive_json()["type"] == "final"


def test_websocket_invalid_start_does_not_echo_input(tmp_path):
    client, _runtime = make_client(tmp_path)
    secret = "api_key=sk-ws-secret prompt=private-context"
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "invalid",
                    "sample_rate": 16000,
                    "channels": 1,
                    "prompt": secret,
                }
            )
            event = websocket.receive_json()

    assert event["error"] == {
        "code": "invalid_init",
        "message": "invalid start frame",
    }
    assert secret not in json.dumps(event)


def test_websocket_vad_sentence_end_and_reconnect(tmp_path, monkeypatch):
    class FakeVAD:
        def __init__(self, *_args):
            self.calls = 0

        def feed(self, samples):
            self.calls += 1
            if self.calls == 1:
                return [{"start": 0}]
            if self.calls == 2:
                return [{"end": samples.size * 2}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", FakeVAD)
    client, _runtime = make_client(tmp_path)
    init = {
        "type": "start",
        "encoding": "pcm_f32le",
        "sample_rate": 16000,
        "channels": 1,
        "language": "zh",
        "hotwords": ["热词"],
    }
    with client:
        wait_ready(client)
        for _attempt in range(2):
            with client.websocket_connect("/v1/audio/realtime") as websocket:
                websocket.send_json(init)
                assert websocket.receive_json()["type"] == "ready"
                websocket.send_bytes(np.ones(512, dtype="<f4").tobytes())
                websocket.send_bytes(np.ones(512, dtype="<f4").tobytes())
                assert websocket.receive_json()["type"] == "segment_final"
                websocket.send_text(json.dumps({"type": "finish"}))
                assert websocket.receive_json()["type"] == "final"


def test_websocket_noise_without_vad_speech_is_not_transcribed(tmp_path, monkeypatch):
    class NoSpeechVAD:
        def __init__(self, *_args):
            pass

        def feed(self, _samples):
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", NoSpeechVAD)
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["DeepSeek", "智谱"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(np.ones(16000, dtype="<f4").tobytes())
            websocket.send_text(json.dumps({"type": "finish"}))
            final = websocket.receive_json()
            assert final["type"] == "final"
            assert final["text"] == ""
            assert runtime.backend.batch_sizes == []


def test_websocket_candidate_vad_recovers_short_contextual_speech(
    tmp_path, monkeypatch
):
    class ThresholdVAD:
        def __init__(self, threshold, _min_silence_ms):
            self.is_candidate = threshold < 0.5
            self.started = False

        def feed(self, samples):
            if self.is_candidate and not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", ThresholdVAD)
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["DeepSeek"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(np.ones(16000, dtype="<f4").tobytes())
            websocket.send_text(json.dumps({"type": "finish"}))
            segment = websocket.receive_json()
            final = websocket.receive_json()
            assert segment["type"] == "segment_final"
            assert segment["text"] == "无上下文识别成功"
            assert final["type"] == "final"
            assert final["text"] == "无上下文识别成功"
            assert [task.prompt for task in runtime.backend.tasks] == [None]


def test_websocket_candidate_vad_rejects_low_energy_residual(
    tmp_path, monkeypatch
):
    class CandidateOnlyVAD:
        def __init__(self, threshold, _min_silence_ms):
            self.is_candidate = threshold < 0.5
            self.started = False

        def feed(self, samples):
            if self.is_candidate and not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

        def reset(self):
            self.started = False

    monkeypatch.setattr(realtime, "RealtimeVAD", CandidateOnlyVAD)
    client, runtime = make_client(tmp_path, candidate_min_rms=0.0015)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["何佳玥"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            quiet_noise = np.full(16000, 0.0005, dtype="<f4")
            websocket.send_bytes(quiet_noise.tobytes())
            websocket.send_text(json.dumps({"type": "finish"}))
            final = websocket.receive_json()
            assert final["type"] == "final"
            assert final["text"] == ""
            assert runtime.backend.tasks == []


def test_websocket_primary_end_resets_overlapping_candidate(
    tmp_path, monkeypatch
):
    class StuckCandidateVAD:
        candidate_reset_calls = 0

        def __init__(self, threshold, _min_silence_ms):
            self.is_primary = threshold >= 0.5
            self.calls = 0

        def feed(self, samples):
            self.calls += 1
            if not samples.size:
                return []
            if self.calls == 1:
                return [{"start": 0}]
            if self.is_primary and self.calls == 2:
                return [{"end": samples.size * 2}]
            return []

        def finish(self):
            return []

        def reset(self):
            if not self.is_primary:
                type(self).candidate_reset_calls += 1

    monkeypatch.setattr(realtime, "RealtimeVAD", StuckCandidateVAD)
    client, runtime = make_client(tmp_path, candidate_min_rms=0.0015)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["何佳玥"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            chunk = np.ones(512, dtype="<f4").tobytes()
            websocket.send_bytes(chunk)
            websocket.send_bytes(chunk)
            assert websocket.receive_json()["type"] == "segment_final"
            websocket.send_text(json.dumps({"type": "finish"}))
            assert websocket.receive_json()["type"] == "final"
            assert len(runtime.backend.tasks) == 1
            assert runtime.backend.tasks[0].prompt is not None
            assert StuckCandidateVAD.candidate_reset_calls == 1


def test_websocket_context_echo_falls_back_once_at_segment_final(
    tmp_path, monkeypatch
):
    class PrimaryOnlyVAD:
        def __init__(self, threshold, _min_silence_ms):
            self.is_primary = threshold >= 0.5
            self.started = False

        def feed(self, samples):
            if self.is_primary and not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", PrimaryOnlyVAD)
    client, runtime = make_client(tmp_path)
    runtime.backend.context_text = "DeepSeek，智谱，英伟达。"
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["DeepSeek", "智谱", "英伟达"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(np.ones(512, dtype="<f4").tobytes())
            websocket.send_text(json.dumps({"type": "finish"}))
            segment = websocket.receive_json()
            final = websocket.receive_json()
            assert segment["type"] == "segment_final"
            assert segment["text"] == "无上下文识别成功"
            assert final["text"] == "无上下文识别成功"
            assert [task.prompt for task in runtime.backend.tasks] == [
                "Vocabulary: DeepSeek, 智谱, 英伟达",
                None,
            ]


def test_websocket_english_prompt_translation_is_suppressed_until_verified(
    tmp_path, monkeypatch
):
    class PrimaryOnlyVAD:
        def __init__(self, threshold, _min_silence_ms):
            self.is_primary = threshold >= 0.5
            self.started = False

        def feed(self, samples):
            if self.is_primary and not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", PrimaryOnlyVAD)
    client, runtime = make_client(tmp_path)
    translated = "This is a technical discussion meeting."
    runtime.backend.context_text = translated
    runtime.backend.context_raw_text = translated
    runtime.backend.fallback_text = "Ahem."
    runtime.backend.fallback_raw_text = "Ahem."
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "en",
                    "prompt": "这是一场技术讨论会。",
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(np.ones(16000, dtype="<f4").tobytes())
            websocket.send_text(json.dumps({"type": "finish"}))
            segment = websocket.receive_json()
            final = websocket.receive_json()
            assert segment["type"] == "segment_final"
            assert translated not in json.dumps(segment)
            assert segment["text"] == "Ahem."
            assert final["type"] == "final"
            assert final["text"] == "Ahem."
            assert [task.prompt for task in runtime.backend.tasks] == [
                "这是一场技术讨论会。",
                "这是一场技术讨论会。",
                None,
            ]


def test_websocket_primary_candidate_overlap_is_not_submitted_twice(
    tmp_path, monkeypatch
):
    class OverlapVAD:
        def __init__(self, *_args):
            self.calls = 0

        def feed(self, samples):
            self.calls += 1
            if self.calls == 1:
                return [{"start": 0}]
            if self.calls == 2:
                return [{"end": samples.size * 2}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", OverlapVAD)
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["DeepSeek"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(np.ones(512, dtype="<f4").tobytes())
            websocket.send_bytes(np.ones(512, dtype="<f4").tobytes())
            assert websocket.receive_json()["type"] == "segment_final"
            websocket.send_text(json.dumps({"type": "finish"}))
            assert websocket.receive_json()["type"] == "final"
            assert len(runtime.backend.tasks) == 1
            assert runtime.backend.tasks[0].prompt is not None


def test_websocket_candidate_vad_ten_second_windows_flush_and_deduplicate(
    tmp_path, monkeypatch
):
    class CandidateOnlyVAD:
        def __init__(self, threshold, _min_silence_ms):
            self.is_candidate = threshold < 0.5
            self.started = False

        def feed(self, samples):
            if self.is_candidate and not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", CandidateOnlyVAD)
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["DeepSeek"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            chunk = np.ones(16000, dtype="<f4").tobytes()
            for _ in range(11):
                websocket.send_bytes(chunk)
            first = websocket.receive_json()
            assert first["type"] == "segment_final"
            websocket.send_text(json.dumps({"type": "finish"}))
            second = websocket.receive_json()
            final = websocket.receive_json()
            assert second["type"] == "segment_final"
            assert final["type"] == "final"
            assert final["text"] == "无上下文识别成功"
            assert [task.prompt for task in runtime.backend.tasks] == [None, None]


def test_websocket_finish_skips_short_overlap_tail_and_still_emits_final(
    tmp_path, monkeypatch
):
    class CandidateOnlyVAD:
        def __init__(self, threshold, _min_silence_ms):
            self.is_candidate = threshold < 0.5
            self.started = False

        def feed(self, samples):
            if self.is_candidate and not self.started and samples.size:
                self.started = True
                return [{"start": 0}]
            return []

        def finish(self):
            return []

    monkeypatch.setattr(realtime, "RealtimeVAD", CandidateOnlyVAD)
    client, runtime = make_client(tmp_path)
    with client:
        wait_ready(client)
        with client.websocket_connect("/v1/audio/realtime") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "language": "zh",
                    "hotwords": ["DeepSeek"],
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            one_second = np.ones(16000, dtype="<f4").tobytes()
            for _ in range(10):
                websocket.send_bytes(one_second)
            first = websocket.receive_json()
            assert first["type"] == "segment_final"
            assert (first["start_ms"], first["end_ms"]) == (0, 10000)

            # The 8.0-10.1s finish range only contains 100ms after clipping the
            # already committed 10s. It is safely suppressed rather than raising
            # the historical non-overlapping-range RuntimeError.
            websocket.send_bytes(np.ones(1600, dtype="<f4").tobytes())
            websocket.send_text(json.dumps({"type": "finish"}))
            final = websocket.receive_json()

            assert final["type"] == "final"
            assert final["final"] is True
            assert final.get("error") is None
            assert len(runtime.backend.tasks) == 1
