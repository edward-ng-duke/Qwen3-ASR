"""
E2E and unit tests for realtime_demo.py (Flask HTTP polling ASR demo).

Covers: pure helpers, session management, transcribe+stabilize logic,
Flask endpoint integration, and full session lifecycle flows.
"""
import io
import os
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import realtime_demo as mod
from realtime_demo import (
    Session,
    _gc_sessions,
    _get_session,
    app,
    common_prefix_len,
    numpy_to_wav_bytes,
    parse_asr_output,
    transcribe_and_stabilize,
)


# ===========================================================================
# Helpers
# ===========================================================================
def make_audio_bytes(n_samples=16000, value=0.0):
    """Create raw float32 PCM bytes."""
    return np.full(n_samples, value, dtype=np.float32).tobytes()


def _mock_transcription(text="language en<asr_text>hello world"):
    t = MagicMock()
    t.text = text
    return t


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def flask_client():
    """Provide a Flask test client with mocked vLLM client and clean state."""
    orig_client = mod.client
    orig_model = mod.MODEL
    orig_unfixed = mod.UNFIXED_CHUNK_NUM
    orig_max = mod.MAX_AUDIO_SAMPLES
    orig_chunk_ms = mod.CHUNK_MS
    orig_sessions = dict(mod.SESSIONS)

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = _mock_transcription()
    mock_client.base_url = "http://test:7869/v1"

    mod.client = mock_client
    mod.MODEL = "test-model"
    mod.UNFIXED_CHUNK_NUM = 2
    mod.MAX_AUDIO_SAMPLES = 600 * 16000
    mod.CHUNK_MS = 1000
    mod.SESSIONS.clear()

    app.config["TESTING"] = True
    with app.test_client() as tc:
        yield tc

    mod.client = orig_client
    mod.MODEL = orig_model
    mod.UNFIXED_CHUNK_NUM = orig_unfixed
    mod.MAX_AUDIO_SAMPLES = orig_max
    mod.CHUNK_MS = orig_chunk_ms
    mod.SESSIONS.clear()
    mod.SESSIONS.update(orig_sessions)


@pytest.fixture
def mock_client_ref(flask_client):
    """Return the mock OpenAI client for per-test reconfiguration."""
    return mod.client


@pytest.fixture
def start_session(flask_client):
    """Create a session via API and return session_id."""
    resp = flask_client.post("/api/start")
    return resp.get_json()["session_id"]


# ===========================================================================
# Part 1: Pure function unit tests
# ===========================================================================
class TestPureFunctions:
    # --- numpy_to_wav_bytes ---
    def test_wav_produces_riff_header(self):
        audio = np.zeros(16000, dtype=np.float32)
        wav = numpy_to_wav_bytes(audio)
        assert isinstance(wav, bytes)
        assert len(wav) > 44
        assert wav[:4] == b"RIFF"

    def test_wav_roundtrip(self):
        import soundfile as sf

        audio = np.random.randn(16000).astype(np.float32)
        wav = numpy_to_wav_bytes(audio, sr=16000)
        data, sr = sf.read(io.BytesIO(wav))
        assert sr == 16000
        assert len(data) == 16000

    # --- parse_asr_output ---
    def test_parse_empty(self):
        assert parse_asr_output("") == ("", "")

    def test_parse_none(self):
        assert parse_asr_output(None) == ("", "")

    def test_parse_no_tag(self):
        assert parse_asr_output("hello world") == ("", "hello world")

    def test_parse_with_language(self):
        lang, text = parse_asr_output("language zh<asr_text>你好世界")
        assert lang == "zh"
        assert text == "你好世界"

    def test_parse_language_none(self):
        lang, text = parse_asr_output("language none<asr_text>hello")
        assert lang == ""
        assert text == "hello"

    def test_parse_language_case_insensitive(self):
        lang, _ = parse_asr_output("Language EN<asr_text>hello")
        assert lang == "EN"

    def test_parse_no_language_with_tag(self):
        lang, text = parse_asr_output("<asr_text>just text")
        assert lang == ""
        assert text == "just text"

    def test_parse_whitespace_stripping(self):
        lang, text = parse_asr_output("  language en <asr_text>  hello world  ")
        assert lang == "en"
        assert text == "hello world"

    # --- common_prefix_len ---
    def test_prefix_identical(self):
        assert common_prefix_len("abc", "abc") == 3

    def test_prefix_partial(self):
        assert common_prefix_len("abcdef", "abcxyz") == 3

    def test_prefix_no_overlap(self):
        assert common_prefix_len("abc", "xyz") == 0

    def test_prefix_both_empty(self):
        assert common_prefix_len("", "") == 0

    def test_prefix_one_is_prefix(self):
        assert common_prefix_len("ab", "abcde") == 2

    def test_prefix_unicode(self):
        assert common_prefix_len("你好世界", "你好朋友") == 2


# ===========================================================================
# Part 2: Session management
# ===========================================================================
class TestSessionManagement:
    def setup_method(self):
        mod.SESSIONS.clear()

    def teardown_method(self):
        mod.SESSIONS.clear()

    def test_get_session_returns_none_for_missing(self):
        assert _get_session("nonexistent") is None

    def test_get_session_updates_last_seen(self):
        now = time.time()
        s = Session(created_at=now, last_seen=now - 100)
        mod.SESSIONS["abc"] = s
        result = _get_session("abc")
        assert result is s
        assert s.last_seen >= now

    def test_gc_removes_expired(self):
        now = time.time()
        mod.SESSIONS["old"] = Session(
            created_at=now - mod.SESSION_TTL_SEC - 10,
            last_seen=now - mod.SESSION_TTL_SEC - 10,
        )
        _gc_sessions()
        assert "old" not in mod.SESSIONS

    def test_gc_keeps_active(self):
        now = time.time()
        mod.SESSIONS["fresh"] = Session(created_at=now, last_seen=now)
        _gc_sessions()
        assert "fresh" in mod.SESSIONS

    def test_gc_triggered_by_get_session(self):
        now = time.time()
        mod.SESSIONS["expired"] = Session(
            created_at=now - mod.SESSION_TTL_SEC - 10,
            last_seen=now - mod.SESSION_TTL_SEC - 10,
        )
        mod.SESSIONS["valid"] = Session(created_at=now, last_seen=now)
        _get_session("valid")
        assert "expired" not in mod.SESSIONS
        assert "valid" in mod.SESSIONS


# ===========================================================================
# Part 3: transcribe_and_stabilize logic
# ===========================================================================
class TestTranscribeAndStabilize:
    @pytest.fixture(autouse=True)
    def setup_globals(self):
        self._orig_client = mod.client
        self._orig_model = mod.MODEL
        self._orig_unfixed = mod.UNFIXED_CHUNK_NUM

        self.mock_client = MagicMock()
        self.mock_client.audio.transcriptions.create.return_value = _mock_transcription()
        mod.client = self.mock_client
        mod.MODEL = "test-model"
        mod.UNFIXED_CHUNK_NUM = 2
        yield
        mod.client = self._orig_client
        mod.MODEL = self._orig_model
        mod.UNFIXED_CHUNK_NUM = self._orig_unfixed

    def _session(self, audio_samples=16000):
        now = time.time()
        s = Session(created_at=now, last_seen=now)
        if audio_samples > 0:
            s.audio_accum = np.zeros(audio_samples, dtype=np.float32)
        return s

    def test_empty_audio_returns_current_state(self):
        s = self._session(audio_samples=0)
        s.language = "en"
        s.text = "existing"
        result = transcribe_and_stabilize(s)
        assert result == {"language": "en", "text": "existing"}
        self.mock_client.audio.transcriptions.create.assert_not_called()

    def test_single_call_sets_call_id(self):
        s = self._session()
        result = transcribe_and_stabilize(s)
        assert s.call_id == 1
        assert s.stable_text == ""
        assert "text" in result

    def test_no_prefix_lock_within_unfixed_chunks(self):
        self.mock_client.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>hello"),
            _mock_transcription("language en<asr_text>hello world"),
        ]
        s = self._session()
        transcribe_and_stabilize(s)
        assert s.call_id == 1
        assert s.stable_text == ""
        transcribe_and_stabilize(s)
        assert s.call_id == 2
        assert s.stable_text == ""

    def test_prefix_lock_after_unfixed_chunks(self):
        self.mock_client.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>hello world"),
            _mock_transcription("language en<asr_text>hello world!"),
            _mock_transcription("language en<asr_text>hello world! How"),
        ]
        s = self._session()
        transcribe_and_stabilize(s)
        transcribe_and_stabilize(s)
        transcribe_and_stabilize(s)
        assert s.call_id == 3
        assert len(s.stable_text) > 0
        assert s.text.startswith(s.stable_text)

    def test_stable_text_grows_not_shrinks(self):
        mod.UNFIXED_CHUNK_NUM = 1
        self.mock_client.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>abc"),
            _mock_transcription("language en<asr_text>abcdef"),
            _mock_transcription("language en<asr_text>abcdefghi"),
        ]
        s = self._session()
        transcribe_and_stabilize(s)
        transcribe_and_stabilize(s)
        st1 = s.stable_text
        transcribe_and_stabilize(s)
        st2 = s.stable_text
        assert len(st2) >= len(st1)

    def test_stable_text_prefix_enforcement(self):
        mod.UNFIXED_CHUNK_NUM = 1
        self.mock_client.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>hello world"),
            _mock_transcription("language en<asr_text>hello world today"),
            # Third call: text diverges from stable prefix
            _mock_transcription("language en<asr_text>DIFFERENT text"),
        ]
        s = self._session()
        transcribe_and_stabilize(s)
        transcribe_and_stabilize(s)
        stable = s.stable_text
        assert len(stable) > 0
        transcribe_and_stabilize(s)
        # Text should still start with stable_text
        assert s.text.startswith(stable)

    def test_language_update(self):
        self.mock_client.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>hello"),
            _mock_transcription("language zh<asr_text>你好"),
        ]
        s = self._session()
        transcribe_and_stabilize(s)
        assert s.language == "en"
        transcribe_and_stabilize(s)
        assert s.language == "zh"

    def test_language_not_cleared_when_empty(self):
        self.mock_client.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>hello"),
            _mock_transcription("<asr_text>hello world"),
        ]
        s = self._session()
        transcribe_and_stabilize(s)
        assert s.language == "en"
        transcribe_and_stabilize(s)
        assert s.language == "en"

    def test_finalized_text_prepended(self):
        s = self._session()
        s.finalized_text = "Previously said: "
        result = transcribe_and_stabilize(s)
        assert s.text.startswith("Previously said: ")

    def test_api_error_returns_502_tuple(self):
        self.mock_client.audio.transcriptions.create.side_effect = Exception("connection refused")
        s = self._session()
        result = transcribe_and_stabilize(s)
        assert isinstance(result, tuple)
        assert result[1] == 502
        assert "error" in result[0]


# ===========================================================================
# Part 4: Flask endpoint integration tests
# ===========================================================================
class TestFlaskEndpoints:
    # --- GET / ---
    def test_index_returns_html(self, flask_client):
        resp = flask_client.get("/")
        assert resp.status_code == 200
        assert b"Qwen3-ASR" in resp.data

    def test_index_replaces_template_vars(self, flask_client):
        resp = flask_client.get("/")
        html = resp.data.decode()
        assert "test-model" in html
        assert str(mod.CHUNK_MS) in html

    # --- POST /api/start ---
    def test_start_returns_session_id(self, flask_client):
        resp = flask_client.post("/api/start")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32  # uuid4 hex

    def test_start_populates_sessions(self, flask_client):
        resp = flask_client.post("/api/start")
        sid = resp.get_json()["session_id"]
        assert sid in mod.SESSIONS

    # --- POST /api/chunk ---
    def test_chunk_missing_session_id(self, flask_client):
        resp = flask_client.post("/api/chunk")
        assert resp.status_code == 400
        assert "invalid session_id" in resp.get_json()["error"]

    def test_chunk_invalid_session_id(self, flask_client):
        resp = flask_client.post("/api/chunk?session_id=bogus")
        assert resp.status_code == 400

    def test_chunk_empty_body(self, flask_client, start_session):
        resp = flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=b"",
            content_type="application/octet-stream",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "language" in data

    def test_chunk_bad_byte_length(self, flask_client, start_session):
        resp = flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=b"\x00\x01\x02",
            content_type="application/octet-stream",
        )
        assert resp.status_code == 400
        assert "float32" in resp.get_json()["error"]

    def test_chunk_accumulates_audio(self, flask_client, start_session):
        audio = make_audio_bytes(16000)
        flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=audio,
            content_type="application/octet-stream",
        )
        s = mod.SESSIONS[start_session]
        assert s.audio_accum.shape[0] == 16000

    def test_chunk_returns_transcription(self, flask_client, start_session):
        resp = flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["language"] == "en"
        assert data["text"] == "hello world"

    def test_chunk_context_sec_param(self, flask_client, start_session):
        flask_client.post(
            f"/api/chunk?session_id={start_session}&context_sec=5",
            data=make_audio_bytes(100),
            content_type="application/octet-stream",
        )
        s = mod.SESSIONS[start_session]
        assert s.context_window_sec == 5

    def test_chunk_sliding_window_trims_audio(self, flask_client, start_session, mock_client_ref):
        # Send 2 seconds with context_sec=1
        flask_client.post(
            f"/api/chunk?session_id={start_session}&context_sec=1",
            data=make_audio_bytes(32000),
            content_type="application/octet-stream",
        )
        s = mod.SESSIONS[start_session]
        assert s.audio_accum.shape[0] == 16000

    def test_chunk_sliding_window_resets_stabilization(self, flask_client, start_session, mock_client_ref):
        s = mod.SESSIONS[start_session]
        s.call_id = 5
        s.prev_text = "some text"
        s.stable_text = "some"
        flask_client.post(
            f"/api/chunk?session_id={start_session}&context_sec=1",
            data=make_audio_bytes(32000),
            content_type="application/octet-stream",
        )
        # After trim, call_id resets to 0, then transcribe_and_stabilize increments to 1
        assert s.call_id == 1
        assert s.stable_text == ""

    def test_chunk_sliding_window_commits_stable_text(self, flask_client, start_session, mock_client_ref):
        s = mod.SESSIONS[start_session]
        s.stable_text = "committed "
        flask_client.post(
            f"/api/chunk?session_id={start_session}&context_sec=1",
            data=make_audio_bytes(32000),
            content_type="application/octet-stream",
        )
        assert "committed " in s.finalized_text

    def test_chunk_max_audio_exceeded(self, flask_client, start_session, mock_client_ref):
        orig = mod.MAX_AUDIO_SAMPLES
        mod.MAX_AUDIO_SAMPLES = 16000 * 2  # 2 seconds
        try:
            resp = flask_client.post(
                f"/api/chunk?session_id={start_session}&context_sec=0",
                data=make_audio_bytes(16000 * 3),
                content_type="application/octet-stream",
            )
            assert resp.status_code == 400
            assert "max audio duration exceeded" in resp.get_json()["error"]
        finally:
            mod.MAX_AUDIO_SAMPLES = orig

    def test_chunk_vlm_error_returns_502(self, flask_client, start_session, mock_client_ref):
        mock_client_ref.audio.transcriptions.create.side_effect = Exception("timeout")
        resp = flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        assert resp.status_code == 502
        assert "error" in resp.get_json()

    # --- POST /api/finish ---
    def test_finish_missing_session_id(self, flask_client):
        resp = flask_client.post("/api/finish")
        assert resp.status_code == 400

    def test_finish_returns_transcription(self, flask_client, start_session):
        # Add audio so there's something to transcribe
        flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        resp = flask_client.post(f"/api/finish?session_id={start_session}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "text" in data

    def test_finish_removes_session(self, flask_client, start_session):
        flask_client.post(f"/api/finish?session_id={start_session}")
        assert start_session not in mod.SESSIONS

    def test_finish_empty_audio(self, flask_client, start_session, mock_client_ref):
        mock_client_ref.audio.transcriptions.create.return_value = _mock_transcription(
            "language en<asr_text>"
        )
        resp = flask_client.post(f"/api/finish?session_id={start_session}")
        assert resp.status_code == 200

    def test_finish_vlm_error_returns_502(self, flask_client, start_session, mock_client_ref):
        # Add audio first so transcribe_and_stabilize actually calls the client
        flask_client.post(
            f"/api/chunk?session_id={start_session}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        mock_client_ref.audio.transcriptions.create.side_effect = Exception("fail")
        resp = flask_client.post(f"/api/finish?session_id={start_session}")
        assert resp.status_code == 502


# ===========================================================================
# Part 5: End-to-end flow tests
# ===========================================================================
class TestEndToEndFlow:
    def test_full_session_lifecycle(self, flask_client, mock_client_ref):
        mock_client_ref.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>hello"),
            _mock_transcription("language en<asr_text>hello world"),
            _mock_transcription("language en<asr_text>hello world today"),
            _mock_transcription("language en<asr_text>hello world today is great"),
        ]

        # Start
        resp = flask_client.post("/api/start")
        sid = resp.get_json()["session_id"]

        # Send 3 chunks
        for _ in range(3):
            resp = flask_client.post(
                f"/api/chunk?session_id={sid}&context_sec=0",
                data=make_audio_bytes(16000),
                content_type="application/octet-stream",
            )
            assert resp.status_code == 200

        # Finish
        resp = flask_client.post(f"/api/finish?session_id={sid}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "text" in data
        assert len(data["text"]) > 0
        assert sid not in mod.SESSIONS

    def test_multi_session_isolation(self, flask_client, mock_client_ref):
        mock_client_ref.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>session one"),
            _mock_transcription("language zh<asr_text>第二个会话"),
        ]

        # Start two sessions
        r1 = flask_client.post("/api/start")
        sid1 = r1.get_json()["session_id"]
        r2 = flask_client.post("/api/start")
        sid2 = r2.get_json()["session_id"]
        assert sid1 != sid2

        # Send audio to session 1
        resp1 = flask_client.post(
            f"/api/chunk?session_id={sid1}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        d1 = resp1.get_json()
        assert d1["text"] == "session one"

        # Send audio to session 2
        resp2 = flask_client.post(
            f"/api/chunk?session_id={sid2}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        d2 = resp2.get_json()
        assert d2["text"] == "第二个会话"
        assert d2["language"] == "zh"

    def test_progressive_stabilization(self, flask_client, mock_client_ref):
        mod.UNFIXED_CHUNK_NUM = 2
        mock_client_ref.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>the quick"),
            _mock_transcription("language en<asr_text>the quick brown"),
            _mock_transcription("language en<asr_text>the quick brown fox"),
        ]

        resp = flask_client.post("/api/start")
        sid = resp.get_json()["session_id"]

        for _ in range(3):
            flask_client.post(
                f"/api/chunk?session_id={sid}&context_sec=0",
                data=make_audio_bytes(16000),
                content_type="application/octet-stream",
            )

        s = mod.SESSIONS[sid]
        # After 3 calls (> UNFIXED_CHUNK_NUM=2), stable_text should be locked
        assert len(s.stable_text) > 0
        assert s.text.startswith(s.stable_text)

    def test_sliding_window_with_finalization(self, flask_client, mock_client_ref):
        mock_client_ref.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>first part"),
            _mock_transcription("language en<asr_text>first part of speech"),
            _mock_transcription("language en<asr_text>second part"),
        ]

        resp = flask_client.post("/api/start")
        sid = resp.get_json()["session_id"]

        # Chunk 1: 0.5s audio, context_sec=1 → no trim
        flask_client.post(
            f"/api/chunk?session_id={sid}&context_sec=1",
            data=make_audio_bytes(8000),
            content_type="application/octet-stream",
        )

        s = mod.SESSIONS[sid]
        s.stable_text = "first part"

        # Chunk 2: send enough audio to exceed 1s → triggers trim
        flask_client.post(
            f"/api/chunk?session_id={sid}&context_sec=1",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )

        # stable_text should have been committed to finalized_text
        assert "first part" in s.finalized_text

    def test_new_session_after_finish(self, flask_client, mock_client_ref):
        mock_client_ref.audio.transcriptions.create.side_effect = [
            _mock_transcription("language en<asr_text>first session"),
            _mock_transcription("language en<asr_text>first session final"),  # finish also calls transcribe
            _mock_transcription("language en<asr_text>second session"),
        ]

        # Session 1
        r = flask_client.post("/api/start")
        sid1 = r.get_json()["session_id"]
        flask_client.post(
            f"/api/chunk?session_id={sid1}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        flask_client.post(f"/api/finish?session_id={sid1}")
        assert sid1 not in mod.SESSIONS

        # Session 2 - completely fresh
        r = flask_client.post("/api/start")
        sid2 = r.get_json()["session_id"]
        assert sid2 != sid1
        s = mod.SESSIONS[sid2]
        assert s.audio_accum.shape[0] == 0
        assert s.call_id == 0
        assert s.stable_text == ""

        resp = flask_client.post(
            f"/api/chunk?session_id={sid2}",
            data=make_audio_bytes(16000),
            content_type="application/octet-stream",
        )
        assert resp.get_json()["text"] == "second session"
