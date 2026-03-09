"""
Tests for realtime_demo_1.py: core logic unit tests + WebSocket integration tests.
"""
import asyncio
import json
import struct
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Ensure the repo root is on sys.path so we can import the demo module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_demo_1 import (
    ASRConfig,
    ASREngine,
    StreamingSession,
    _common_prefix_len,
    _parse_asr_output,
    _suffix_prefix_overlap,
    _to_wav,
    app,
    engine,
    config,
)


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def cfg():
    return ASRConfig(sample_rate=16000, unfixed_chunk_num=2, chunk_ms=1000)


@pytest.fixture
def session(cfg):
    return StreamingSession(config=cfg)


# ===========================================================================
# Part 1: Core pure function unit tests
# ===========================================================================
class TestParseAsrOutput:
    def test_empty_string(self):
        assert _parse_asr_output("") == ("", "")

    def test_none_input(self):
        assert _parse_asr_output(None) == ("", "")

    def test_no_tag(self):
        assert _parse_asr_output("hello world") == ("", "hello world")

    def test_with_language_and_tag(self):
        raw = "language zh<asr_text>你好世界"
        lang, text = _parse_asr_output(raw)
        assert lang == "zh"
        assert text == "你好世界"

    def test_language_none(self):
        raw = "language none<asr_text>hello"
        lang, text = _parse_asr_output(raw)
        assert lang == ""
        assert text == "hello"

    def test_language_case_insensitive(self):
        raw = "Language EN<asr_text>hello"
        lang, text = _parse_asr_output(raw)
        assert lang == "EN"

    def test_no_language_with_tag(self):
        raw = "<asr_text>just text"
        lang, text = _parse_asr_output(raw)
        assert lang == ""
        assert text == "just text"

    def test_whitespace_stripping(self):
        raw = "  language en <asr_text>  hello world  "
        lang, text = _parse_asr_output(raw)
        assert lang == "en"
        assert text == "hello world"


class TestCommonPrefixLen:
    def test_identical(self):
        assert _common_prefix_len("abc", "abc") == 3

    def test_partial(self):
        assert _common_prefix_len("abcdef", "abcxyz") == 3

    def test_no_overlap(self):
        assert _common_prefix_len("abc", "xyz") == 0

    def test_empty_first(self):
        assert _common_prefix_len("", "abc") == 0

    def test_empty_second(self):
        assert _common_prefix_len("abc", "") == 0

    def test_both_empty(self):
        assert _common_prefix_len("", "") == 0

    def test_one_is_prefix_of_other(self):
        assert _common_prefix_len("ab", "abcde") == 2

    def test_unicode(self):
        assert _common_prefix_len("你好世界", "你好朋友") == 2


class TestSuffixPrefixOverlap:
    def test_overlap(self):
        assert _suffix_prefix_overlap("abcde", "dexy") == 2

    def test_no_overlap(self):
        assert _suffix_prefix_overlap("abc", "xyz") == 0

    def test_full_overlap_a_suffix_equals_b(self):
        assert _suffix_prefix_overlap("xxab", "ab") == 2

    def test_empty_first(self):
        assert _suffix_prefix_overlap("", "abc") == 0

    def test_empty_second(self):
        assert _suffix_prefix_overlap("abc", "") == 0

    def test_identical(self):
        assert _suffix_prefix_overlap("abc", "abc") == 3

    def test_single_char_overlap(self):
        assert _suffix_prefix_overlap("hello", "o world") == 1

    def test_unicode_overlap(self):
        assert _suffix_prefix_overlap("你好世界", "世界你好") == 2


# ===========================================================================
# Part 2: StreamingSession unit tests
# ===========================================================================
class TestStreamingSessionAudio:
    def test_initial_state(self, session):
        assert session.audio_accum.shape[0] == 0
        assert session.audio_duration_sec == 0.0

    def test_append_single_chunk(self, session):
        chunk = np.ones(16000, dtype=np.float32)
        session.append_audio(chunk)
        assert session.audio_accum.shape[0] == 16000
        assert session.audio_duration_sec == pytest.approx(1.0)

    def test_append_multiple_chunks(self, session):
        for _ in range(5):
            session.append_audio(np.zeros(3200, dtype=np.float32))
        assert session.audio_accum.shape[0] == 16000
        assert session.audio_duration_sec == pytest.approx(1.0)

    def test_lazy_cache_invalidation(self, session):
        session.append_audio(np.ones(100, dtype=np.float32))
        _ = session.audio_accum  # populate cache
        assert session._audio_cache is not None
        session.append_audio(np.ones(100, dtype=np.float32))
        assert session._audio_cache is None  # invalidated
        result = session.audio_accum
        assert result.shape[0] == 200
        assert session._audio_cache is not None  # re-cached

    def test_audio_data_integrity(self, session):
        c1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c2 = np.array([4.0, 5.0], dtype=np.float32)
        session.append_audio(c1)
        session.append_audio(c2)
        np.testing.assert_array_equal(
            session.audio_accum, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        )


class TestSlidingWindow:
    def test_no_trim_within_window(self, cfg):
        session = StreamingSession(config=cfg, context_window_sec=2)
        # 1.5 seconds < 2 second window
        session.append_audio(np.zeros(24000, dtype=np.float32))
        assert session._audio_len == 24000

    def test_trim_exceeds_window(self, cfg):
        session = StreamingSession(config=cfg, context_window_sec=1)
        # Append 2 seconds of audio
        session.append_audio(np.zeros(32000, dtype=np.float32))
        # Should be trimmed to 1 second
        assert session._audio_len == 16000
        assert session.audio_accum.shape[0] == 16000

    def test_no_trim_when_unlimited(self, cfg):
        session = StreamingSession(config=cfg, context_window_sec=0)
        session.append_audio(np.zeros(160000, dtype=np.float32))
        assert session._audio_len == 160000

    def test_trim_resets_state(self, cfg):
        session = StreamingSession(config=cfg, context_window_sec=1)
        session.call_id = 5
        session.prev_text = "previous"
        session.stable_text = "stable"
        session.append_audio(np.zeros(32000, dtype=np.float32))
        assert session.call_id == 0
        assert session.prev_text == ""
        assert session.stable_text == ""

    def test_trim_sets_dedup_flag(self, cfg):
        session = StreamingSession(config=cfg, context_window_sec=1)
        session.prev_text = "some text"
        session.append_audio(np.zeros(32000, dtype=np.float32))
        assert session._needs_dedup is True
        assert session._pre_trim_text == "some text"

    def test_unresolved_dedup_commits_on_next_trim(self, cfg):
        session = StreamingSession(config=cfg, context_window_sec=1)
        session.prev_text = "first"
        # First trim
        session.append_audio(np.zeros(32000, dtype=np.float32))
        assert session._needs_dedup is True
        assert session._pre_trim_text == "first"

        session.prev_text = "second"
        # Second trim without resolving dedup → commits first
        session.append_audio(np.zeros(32000, dtype=np.float32))
        assert session.finalized_text == "first"
        assert session._pre_trim_text == "second"


# ===========================================================================
# Part 3: ASREngine._stabilize unit tests
# ===========================================================================
class TestStabilize:
    def make_engine(self):
        cfg = ASRConfig(unfixed_chunk_num=2)
        eng = ASREngine.__new__(ASREngine)
        eng.config = cfg
        return eng

    def test_no_lock_within_unfixed_chunks(self):
        eng = self.make_engine()
        cfg = ASRConfig(unfixed_chunk_num=2)
        session = StreamingSession(config=cfg)

        eng._stabilize(session, "hello world", "en")
        assert session.call_id == 1
        assert session.text == "hello world"
        assert session.stable_text == ""

        eng._stabilize(session, "hello worlds", "en")
        assert session.call_id == 2
        assert session.stable_text == ""

    def test_prefix_lock_after_unfixed_chunks(self):
        eng = self.make_engine()
        cfg = ASRConfig(unfixed_chunk_num=2)
        session = StreamingSession(config=cfg)

        eng._stabilize(session, "hello world", "en")
        eng._stabilize(session, "hello world!", "en")
        # call_id is now 2, prev_text = "hello world!"
        eng._stabilize(session, "hello world! How", "en")
        # call_id=3 > unfixed_chunk_num=2, should lock prefix
        assert session.call_id == 3
        assert len(session.stable_text) > 0
        assert session.text.startswith(session.stable_text)

    def test_stable_text_grows_not_shrinks(self):
        eng = self.make_engine()
        cfg = ASRConfig(unfixed_chunk_num=1)
        session = StreamingSession(config=cfg)

        eng._stabilize(session, "abc", "en")
        eng._stabilize(session, "abcdef", "en")
        st1 = session.stable_text
        eng._stabilize(session, "abcdefghi", "en")
        st2 = session.stable_text
        assert len(st2) >= len(st1)

    def test_language_update(self):
        eng = self.make_engine()
        cfg = ASRConfig()
        session = StreamingSession(config=cfg)

        eng._stabilize(session, "hello", "en")
        assert session.language == "en"
        eng._stabilize(session, "你好", "zh")
        assert session.language == "zh"

    def test_language_not_cleared_when_empty(self):
        eng = self.make_engine()
        cfg = ASRConfig()
        session = StreamingSession(config=cfg)

        eng._stabilize(session, "hello", "en")
        eng._stabilize(session, "hello world", "")
        assert session.language == "en"

    def test_dedup_after_trim(self):
        eng = self.make_engine()
        cfg = ASRConfig(unfixed_chunk_num=2)
        session = StreamingSession(config=cfg)

        # Simulate pre-trim text
        session._pre_trim_text = "hello world"
        session._needs_dedup = True

        # New transcription with overlap: "world" is suffix of pre_trim and prefix of new
        eng._stabilize(session, "world! How are you", "en")
        # Should commit "hello " (pre_trim minus overlap)
        assert session.finalized_text == "hello "
        assert session._needs_dedup is False
        assert "world! How are you" in session.text

    def test_dedup_no_overlap(self):
        eng = self.make_engine()
        cfg = ASRConfig(unfixed_chunk_num=2)
        session = StreamingSession(config=cfg)

        session._pre_trim_text = "hello"
        session._needs_dedup = True

        eng._stabilize(session, "completely different", "en")
        # No overlap → commit entire pre_trim_text
        assert session.finalized_text == "hello"


# ===========================================================================
# Part 4: WebSocket integration tests
# ===========================================================================
class TestToWav:
    def test_produces_wav_bytes(self):
        audio = np.zeros(16000, dtype=np.float32)
        result = _to_wav(audio)
        assert isinstance(result, bytes)
        assert len(result) > 44  # WAV header is 44 bytes
        assert result[:4] == b"RIFF"

    def test_correct_duration(self):
        import soundfile as sf
        import io
        audio = np.zeros(16000, dtype=np.float32)
        wav = _to_wav(audio, sr=16000)
        data, sr = sf.read(io.BytesIO(wav))
        assert sr == 16000
        assert len(data) == 16000


class TestWebSocketIntegration:
    """Integration tests using FastAPI TestClient with mocked ASREngine."""

    @pytest.fixture(autouse=True)
    def setup_mock_engine(self):
        """Mock the engine to avoid needing a real vLLM server."""
        import realtime_demo_1 as mod

        mock_engine = MagicMock(spec=ASREngine)
        mock_engine.transcribe = AsyncMock(
            return_value={"language": "en", "text": "hello world"}
        )
        mock_engine.close = AsyncMock()

        self._original_engine = mod.engine
        mod.engine = mock_engine
        self.mock_engine = mock_engine
        yield
        mod.engine = self._original_engine

    def get_client(self):
        from starlette.testclient import TestClient
        return TestClient(app)

    def make_audio_frame(self, n_samples=16000, value=0.0):
        """Create a valid float32 PCM binary frame."""
        audio = np.full(n_samples, value, dtype=np.float32)
        return audio.tobytes()

    def test_basic_flow(self):
        """Connect → config → audio → receive result → finish."""
        client = self.get_client()
        with client.websocket_connect("/ws/asr") as ws:
            # Send config
            ws.send_json({"type": "config", "context_sec": 10})

            # Send audio frame
            ws.send_bytes(self.make_audio_frame(16000))

            # Should receive a transcription result (from the infer task or finish)
            # Send finish to get final result
            ws.send_json({"type": "finish"})
            data = ws.receive_json()
            assert "text" in data
            assert data["text"] == "hello world"

    def test_invalid_audio_frame(self):
        """Non-multiple-of-4 bytes should return error."""
        client = self.get_client()
        with client.websocket_connect("/ws/asr") as ws:
            ws.send_bytes(b"\x00\x01\x02")  # 3 bytes, not multiple of 4
            data = ws.receive_json()
            assert "error" in data
            assert "float32" in data["error"]

    def test_nan_audio_does_not_crash(self):
        """Audio with NaN values should be sanitized, not crash."""
        client = self.get_client()
        with client.websocket_connect("/ws/asr") as ws:
            # Create frame with NaN
            audio = np.array([float("nan"), 0.5, float("inf"), -0.3], dtype=np.float32)
            ws.send_bytes(audio.tobytes())

            # Should not crash; send finish to verify
            ws.send_json({"type": "finish"})
            data = ws.receive_json()
            assert "text" in data or "error" not in data

    def test_finish_returns_final_transcription(self):
        """Finish should return the mock transcription."""
        client = self.get_client()
        with client.websocket_connect("/ws/asr") as ws:
            ws.send_bytes(self.make_audio_frame())
            ws.send_json({"type": "finish"})
            data = ws.receive_json()
            assert data["language"] == "en"
            assert data["text"] == "hello world"

    def test_config_updates_context_window(self):
        """Config message should update session's context_window_sec."""
        self.mock_engine.transcribe = AsyncMock(
            return_value={"language": "en", "text": "test"}
        )
        client = self.get_client()
        with client.websocket_connect("/ws/asr") as ws:
            ws.send_json({"type": "config", "context_sec": 5})
            ws.send_bytes(self.make_audio_frame())
            ws.send_json({"type": "finish"})
            data = ws.receive_json()
            assert "text" in data

    def test_empty_audio_finish(self):
        """Finish with no audio should return empty/default result."""
        # When no audio, transcribe returns current session state
        self.mock_engine.transcribe = AsyncMock(
            return_value={"language": "", "text": ""}
        )
        client = self.get_client()
        with client.websocket_connect("/ws/asr") as ws:
            ws.send_json({"type": "finish"})
            data = ws.receive_json()
            assert "text" in data

    def test_metrics_endpoint(self):
        """GET /metrics should return JSON with expected fields."""
        client = self.get_client()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_sessions" in data
        assert "total_sessions" in data
        assert "total_infer_calls" in data

    def test_index_page(self):
        """GET / should return HTML."""
        client = self.get_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Qwen3-ASR" in resp.text

    def test_audio_processor_js(self):
        """GET /audio-processor.js should return JS."""
        client = self.get_client()
        resp = client.get("/audio-processor.js")
        assert resp.status_code == 200
        assert "PCMProcessor" in resp.text
        assert "_resample" in resp.text  # Verify resampling was moved to worklet
