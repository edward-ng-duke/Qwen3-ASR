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
    _check_energy_vad,
    _common_prefix_len,
    _parse_asr_output,
    _remove_output_duplicates,
    _split_at_last_sentence_boundary,
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
        eng._tokenizer = None
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


# ===========================================================================
# Part 5: New anti-repetition feature tests
# ===========================================================================
class TestEnergyVAD:
    def test_silence_returns_false(self):
        chunk = np.zeros(16000, dtype=np.float32)
        assert _check_energy_vad(chunk, threshold=0.01) is False

    def test_speech_returns_true(self):
        chunk = np.full(16000, 0.1, dtype=np.float32)
        assert _check_energy_vad(chunk, threshold=0.01) is True

    def test_empty_chunk(self):
        chunk = np.array([], dtype=np.float32)
        assert _check_energy_vad(chunk) is False

    def test_threshold_boundary(self):
        # RMS of constant 0.01 signal = 0.01, should not exceed threshold=0.01
        chunk = np.full(16000, 0.01, dtype=np.float32)
        assert _check_energy_vad(chunk, threshold=0.01) is False
        assert _check_energy_vad(chunk, threshold=0.005) is True


class TestRemoveOutputDuplicates:
    def test_no_overlap(self):
        assert _remove_output_duplicates("xyz", "abc") == "xyz"

    def test_full_prefix_overlap(self):
        assert _remove_output_duplicates("world!", "hello world") == "!"

    def test_partial_overlap(self):
        # "你好世界" ends with "就是我认为" prefix "就是我认" (4 chars, meets min_match_len)
        assert _remove_output_duplicates("就是我认为好的", "前面就是我认") == "为好的"

    def test_empty_delta(self):
        assert _remove_output_duplicates("", "abc") == ""

    def test_empty_recent(self):
        assert _remove_output_duplicates("abc", "") == "abc"

    def test_short_match_ignored(self):
        # Overlap of 2 chars < min_match_len=4
        assert _remove_output_duplicates("ab123", "xxab") == "ab123"

    def test_exact_duplicate(self):
        assert _remove_output_duplicates("hello", "hello") == ""


class TestSplitAtLastSentenceBoundary:
    def test_chinese_period(self):
        complete, frag = _split_at_last_sentence_boundary("你好。世界")
        assert complete == "你好。"
        assert frag == "世界"

    def test_no_terminator(self):
        complete, frag = _split_at_last_sentence_boundary("hello world")
        assert complete == ""
        assert frag == "hello world"

    def test_ends_with_terminator(self):
        complete, frag = _split_at_last_sentence_boundary("hello world.")
        assert complete == "hello world."
        assert frag == ""

    def test_multiple_terminators(self):
        complete, frag = _split_at_last_sentence_boundary("a。b！c")
        assert complete == "a。b！"
        assert frag == "c"

    def test_empty_string(self):
        complete, frag = _split_at_last_sentence_boundary("")
        assert complete == ""
        assert frag == ""

    def test_question_mark(self):
        complete, frag = _split_at_last_sentence_boundary("你好？世界")
        assert complete == "你好？"
        assert frag == "世界"


class TestParseAsrOutputRepetition:
    def test_cleans_char_repetitions(self):
        # 25 repeated 好 should be cleaned (threshold=10)
        raw = "language zh<asr_text>你" + "好" * 25
        lang, text = _parse_asr_output(raw, repetition_threshold=10)
        assert text.count("好") < 25

    def test_cleans_pattern_repetitions(self):
        # "你好" repeated 25 times
        raw = "language zh<asr_text>" + "你好" * 25
        lang, text = _parse_asr_output(raw, repetition_threshold=10)
        assert text.count("你好") < 25

    def test_no_cleaning_below_threshold(self):
        raw = "language zh<asr_text>" + "你好" * 5
        lang, text = _parse_asr_output(raw, repetition_threshold=10)
        assert text == "你好" * 5

    def test_disabled_when_threshold_zero(self):
        raw = "language zh<asr_text>" + "好" * 25
        lang, text = _parse_asr_output(raw, repetition_threshold=0)
        assert text == "好" * 25


# ===========================================================================
# Part 6: E2E streaming simulation tests
# ===========================================================================
def _has_significant_repetition(text: str, min_pattern_len: int = 4, max_repeats: int = 3) -> bool:
    """Check if text contains any pattern repeated consecutively too many times."""
    n = len(text)
    for k in range(min_pattern_len, min(n // max_repeats + 1, 30)):
        for i in range(n - k * max_repeats + 1):
            pat = text[i:i + k]
            count = 1
            pos = i + k
            while pos + k <= n and text[pos:pos + k] == pat:
                count += 1
                pos += k
            if count >= max_repeats:
                return True
    return False


class TestStreamingE2E:
    """E2E tests: real ASREngine._stabilize + mocked vLLM API responses.

    These tests simulate realistic streaming ASR scenarios by:
    - Using the real stabilization, dedup, and delta computation pipeline
    - Mocking only the vLLM API layer (client.audio.transcriptions.create)
    - Feeding progressive transcription results as a real model would produce
    """

    def make_engine_and_session(self, context_sec=10, unfixed_chunk_num=4,
                                unfixed_token_num=0):
        cfg = ASRConfig(
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            repetition_threshold=10,
        )
        eng = ASREngine.__new__(ASREngine)
        eng.config = cfg
        eng._tokenizer = None
        eng._semaphore = asyncio.Semaphore(10)
        eng.client = MagicMock()
        session = StreamingSession(config=cfg, context_window_sec=context_sec)
        return eng, session

    def _setup_responses(self, eng, responses):
        """Configure mock to return a sequence of ASR transcription results."""
        mock_transcription = MagicMock()
        call_idx = [0]

        async def fake_create(**kwargs):
            idx = min(call_idx[0], len(responses) - 1)
            mock_transcription.text = responses[idx]
            call_idx[0] += 1
            return mock_transcription

        eng.client.audio.transcriptions.create = AsyncMock(side_effect=fake_create)
        return call_idx

    async def _run_streaming(self, eng, session, responses):
        """Helper: send chunks and collect deltas."""
        self._setup_responses(eng, responses)
        deltas = []
        for i in range(len(responses)):
            chunk = np.random.randn(16000).astype(np.float32) * 0.1
            session.append_audio(chunk)
            result = await eng.transcribe(session)
            if result.get("delta"):
                deltas.append(result["delta"])
        return deltas

    def test_progressive_chinese_no_repetition(self):
        """Scenario 1: Progressive Chinese transcription without window trim."""
        eng, session = self.make_engine_and_session(context_sec=0)  # unlimited

        responses = [
            "language Chinese<asr_text>页面有哪些",
            "language Chinese<asr_text>页面有哪些？嗯嗯",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度、质量评分是怎么样的？",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度、质量评分是怎么样的？就是我认为，",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度、质量评分是怎么样的？就是我认为，就是基本上你所有事儿都可以靠它干。",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度、质量评分是怎么样的？就是我认为，就是基本上你所有事儿都可以靠它干。你有什么事儿是不能靠它干的呢？",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度、质量评分是怎么样的？就是我认为，就是基本上你所有事儿都可以靠它干。你有什么事儿是不能靠它干的呢？我我就是你只要是",
            "language Chinese<asr_text>页面有哪些？嗯嗯，页面的加载速度、质量评分是怎么样的？就是我认为，就是基本上你所有事儿都可以靠它干。你有什么事儿是不能靠它干的呢？我我就是你只要是如果你的工作就是坐在办公室面前对这一台电脑，那这个事儿。",
        ]

        deltas = asyncio.run(self._run_streaming(eng, session, responses))
        joined = "".join(deltas)
        assert len(joined) > 0, "Should have produced some output"
        assert not _has_significant_repetition(joined), \
            f"Repetition detected in delta output: {joined[:200]}..."

    def test_sliding_window_no_repetition(self):
        """Scenario 2: Window trim triggers cross-window dedup."""
        eng, session = self.make_engine_and_session(context_sec=3)

        responses = [
            "language Chinese<asr_text>今天天气真好。",
            "language Chinese<asr_text>今天天气真好。我们去公园",
            "language Chinese<asr_text>今天天气真好。我们去公园散步吧。",
            # Window trims here (>3s audio), model re-transcribes retained audio:
            "language Chinese<asr_text>我们去公园散步吧。",
            "language Chinese<asr_text>我们去公园散步吧。路上看到了",
            "language Chinese<asr_text>我们去公园散步吧。路上看到了很多花。",
            # Another trim:
            "language Chinese<asr_text>很多花。",
            "language Chinese<asr_text>很多花。真是美丽的一天。",
        ]

        deltas = asyncio.run(self._run_streaming(eng, session, responses))
        joined = "".join(deltas)
        assert len(joined) > 0
        for sentence in ["今天天气真好", "我们去公园散步吧", "很多花"]:
            count = joined.count(sentence)
            assert count <= 1, \
                f"Sentence '{sentence}' appears {count} times in: {joined}"

    def test_model_repetition_cleaned(self):
        """Scenario 3: Model outputs contain pattern repetition (ASR degeneration)."""
        eng, session = self.make_engine_and_session(context_sec=0)

        responses = [
            "language Chinese<asr_text>你好",
            "language Chinese<asr_text>你好世界" + "你好世界" * 15,  # 16x repetition
            "language Chinese<asr_text>你好世界。今天真好",
        ]

        deltas = asyncio.run(self._run_streaming(eng, session, responses))
        joined = "".join(deltas)
        assert joined.count("你好世界") < 10, \
            f"Too many '你好世界' repetitions in: {joined}"

    def test_silence_produces_no_extra_deltas(self):
        """Scenario 4: Repeated identical response should not produce spurious deltas."""
        eng, session = self.make_engine_and_session(context_sec=0)

        responses = ["language Chinese<asr_text>你好"] * 10

        deltas = asyncio.run(self._run_streaming(eng, session, responses))
        joined = "".join(deltas)
        assert joined.count("你好") <= 2, \
            f"'你好' repeated too many times in deltas: {joined}"

    def test_long_session_output_bounded(self):
        """Scenario 5: Long session - output length proportional to input, not runaway."""
        eng, session = self.make_engine_and_session(context_sec=5)

        base_sentences = [
            "第一句话。", "第二句话。", "第三句话。", "第四句话。", "第五句话。",
            "第六句话。", "第七句话。", "第八句话。", "第九句话。", "第十句话。",
        ]
        responses = []
        window_text = ""
        for i in range(20):
            sent_idx = i % len(base_sentences)
            if i > 0 and i % 5 == 0:
                window_text = base_sentences[sent_idx]
            else:
                window_text += base_sentences[sent_idx]
            responses.append(f"language Chinese<asr_text>{window_text}")

        deltas = asyncio.run(self._run_streaming(eng, session, responses))
        joined = "".join(deltas)
        assert len(joined) < 500, \
            f"Output too long ({len(joined)} chars), possible runaway repetition"
        assert not _has_significant_repetition(joined)
