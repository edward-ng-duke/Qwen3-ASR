"""
Real-time streaming ASR demo via WebSocket + FastAPI.

Improved version of realtime_demo.py:
- Flask → FastAPI + uvicorn (async)
- HTTP polling → WebSocket (bidirectional streaming)
- ScriptProcessorNode → AudioWorkletProcessor
- Sync OpenAI → AsyncOpenAI with semaphore throttling
- WebSocket connection = Session (no global dict)

Usage:
  python realtime_demo_1.py
  python realtime_demo_1.py --base-url http://localhost:7869/v1 --port 6860
  python realtime_demo_1.py --ssl-certfile cert.pem --ssl-keyfile key.pem

Open: http://localhost:6860 (or https:// with SSL)
"""

import argparse
import asyncio
import io
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
import websockets
from openai import AsyncOpenAI

from qwen_asr.inference.utils import detect_and_fix_repetitions

logger = logging.getLogger("asr")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ASRConfig:
    base_url: str = "http://localhost:7869/v1"
    model: str = "/models/Qwen3-ASR-1.7B"
    host: str = "0.0.0.0"
    port: int = 6860
    unfixed_chunk_num: int = 4
    unfixed_token_num: int = 5
    chunk_ms: int = 1000
    max_audio_sec: int = 600
    sample_rate: int = 16000
    max_concurrent_infer: int = 10
    tokenizer_path: str = ""
    vad_url: str = "ws://localhost:7064"
    repetition_threshold: int = 10
    energy_vad_threshold: float = 0.01

    @property
    def max_audio_samples(self) -> int:
        return self.max_audio_sec * self.sample_rate


# ---------------------------------------------------------------------------
# Session (one per WebSocket connection, no global dict needed)
# ---------------------------------------------------------------------------
SENTENCE_TERMINATORS = set("。！？.!?\n")

@dataclass
class StreamingSession:
    config: ASRConfig
    call_id: int = 0
    stable_text: str = ""
    prev_text: str = ""
    language: str = ""
    text: str = ""
    finalized_text: str = ""
    context_window_sec: int = 10
    _pre_trim_text: str = ""
    _needs_dedup: bool = False
    _last_sent_len: int = 0
    _current_window_text: str = ""
    _recent_confirmed: str = ""
    _vad_speech_active: bool = False
    _vad_ever_had_speech: bool = False
    _vad_chunks_since_speech_end: int = 0
    _energy_speech_active: bool = False

    def __post_init__(self):
        self._audio_chunks: list[np.ndarray] = []
        self._audio_len: int = 0
        self._audio_cache: np.ndarray | None = None

    @property
    def audio_accum(self) -> np.ndarray:
        if self._audio_cache is None:
            if not self._audio_chunks:
                return np.array([], dtype=np.float32)
            self._audio_cache = np.concatenate(self._audio_chunks)
        return self._audio_cache

    @property
    def audio_duration_sec(self) -> float:
        return self._audio_len / self.config.sample_rate

    def append_audio(self, chunk: np.ndarray) -> None:
        self._audio_chunks.append(chunk)
        self._audio_len += chunk.shape[0]
        self._audio_cache = None  # invalidate
        self._apply_sliding_window()

    def _set_audio(self, audio: np.ndarray) -> None:
        self._audio_chunks = [audio]
        self._audio_len = audio.shape[0]
        self._audio_cache = audio

    def _apply_sliding_window(self) -> None:
        if self.context_window_sec <= 0:
            return
        max_samples = self.context_window_sec * self.config.sample_rate
        if self._audio_len <= max_samples:
            return

        # If previous trim's dedup wasn't resolved yet, commit it now
        if self._needs_dedup and self._pre_trim_text:
            self.finalized_text += self._pre_trim_text
            self._pre_trim_text = ""

        # Sentence-aware: commit complete sentences directly, only
        # the trailing fragment needs cross-window dedup
        current_text = self.prev_text or self.stable_text or ""
        complete, fragment = _split_at_last_sentence_boundary(current_text)
        if complete:
            self.finalized_text += complete
        self._pre_trim_text = fragment
        self._needs_dedup = bool(self._pre_trim_text)

        self._set_audio(self.audio_accum[-max_samples:])
        self.call_id = 0
        self.prev_text = ""
        self.stable_text = ""


# ---------------------------------------------------------------------------
# ASR helpers
# ---------------------------------------------------------------------------
def _check_energy_vad(chunk: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if audio chunk RMS exceeds silence threshold."""
    if chunk.shape[0] == 0:
        return False
    return float(np.sqrt(np.mean(chunk ** 2))) > threshold


_ASR_TEXT_TAG = "<asr_text>"


def _parse_asr_output(raw: str, repetition_threshold: int = 10) -> tuple:
    if not raw:
        return "", ""
    s = raw.strip()
    if repetition_threshold > 0:
        s = detect_and_fix_repetitions(s, threshold=repetition_threshold)
    if _ASR_TEXT_TAG in s:
        meta, text = s.split(_ASR_TEXT_TAG, 1)
        lang = ""
        m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
        if m and m.group(1).lower() != "none":
            lang = m.group(1)
        return lang, text.strip()
    return "", s


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _suffix_prefix_overlap(a: str, b: str) -> int:
    """Find length of longest suffix of a that equals a prefix of b."""
    max_overlap = min(len(a), len(b))
    for length in range(max_overlap, 0, -1):
        if a[-length:] == b[:length]:
            return length
    return 0


def _remove_output_duplicates(delta: str, recent: str, min_match_len: int = 4) -> str:
    """Remove prefix of delta if it duplicates a suffix of recently confirmed text."""
    if not delta or not recent:
        return delta
    max_check = min(len(delta), len(recent))
    for length in range(max_check, min_match_len - 1, -1):
        if recent.endswith(delta[:length]):
            return delta[length:]
    return delta


def _split_at_last_sentence_boundary(text: str) -> tuple[str, str]:
    """Split text into (complete_sentences, trailing_fragment).

    Returns text up to and including the last sentence terminator,
    and the remaining fragment. If no terminator found, returns ("", text).
    """
    for i in range(len(text) - 1, -1, -1):
        if text[i] in SENTENCE_TERMINATORS:
            return text[:i + 1], text[i + 1:]
    return "", text


# ---------------------------------------------------------------------------
# ASR Engine (async vLLM calls + semaphore throttling)
# ---------------------------------------------------------------------------
class ASREngine:
    def __init__(self, config: ASRConfig):
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
        self._semaphore = asyncio.Semaphore(config.max_concurrent_infer)
        self._tokenizer = None
        if config.unfixed_token_num > 0:
            tokenizer_path = config.tokenizer_path or config.model
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, trust_remote_code=True
                )
                logger.info(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(
                    f"Could not load tokenizer from {tokenizer_path}: {e}. "
                    f"Using char-based fallback for unfixed_token_num."
                )

    def _rollback_tokens(self, text: str, num_tokens: int) -> str:
        """Roll back num_tokens from the end of text."""
        if not text or num_tokens <= 0:
            return text
        if self._tokenizer is not None:
            token_ids = self._tokenizer.encode(text)
            k = num_tokens
            while True:
                end_idx = max(0, len(token_ids) - k)
                if end_idx == 0:
                    return ""
                rolled = self._tokenizer.decode(token_ids[:end_idx])
                if '\ufffd' not in rolled:
                    return rolled
                k += 1
        else:
            rollback_chars = num_tokens * 2
            if rollback_chars >= len(text):
                return ""
            return text[:-rollback_chars]

    async def transcribe(self, session: StreamingSession) -> dict:
        if session.audio_accum.shape[0] == 0:
            return {"language": session.language, "delta": "", "preview": "", "text": session.text}

        wav_bytes = _to_wav(session.audio_accum)
        async with self._semaphore:
            try:
                transcription = await self.client.audio.transcriptions.create(
                    model=self.config.model,
                    file=wav_bytes,
                )
                raw = transcription.text or ""
            except Exception as e:
                logger.error(f"vLLM API error: {e}")
                return {"error": f"vLLM API error: {e}"}

        language, text = _parse_asr_output(raw, self.config.repetition_threshold)
        self._stabilize(session, text, language)

        # Compute incremental output
        confirmed = session.finalized_text + session.stable_text
        delta = confirmed[session._last_sent_len:]
        # Output-level dedup guard: remove prefix of delta that duplicates recent output
        if delta:
            delta = _remove_output_duplicates(delta, session._recent_confirmed)
        if delta:
            session._recent_confirmed = (session._recent_confirmed + delta)[-200:]
        window_text = session._current_window_text
        preview = window_text[len(session.stable_text):] if len(window_text) > len(session.stable_text) else ""
        session._last_sent_len = len(confirmed)

        return {
            "language": session.language,
            "delta": delta,
            "preview": preview,
            "text": session.text,
        }

    def _stabilize(self, session: StreamingSession, text: str, language: str):
        session.call_id += 1
        if session.call_id > self.config.unfixed_chunk_num and session.prev_text:
            cp = _common_prefix_len(session.prev_text, text)
            if cp > len(session.stable_text):
                candidate = text[:cp]
                if self.config.unfixed_token_num > 0:
                    candidate = self._rollback_tokens(
                        candidate, self.config.unfixed_token_num
                    )
                if len(candidate) > len(session.stable_text):
                    session.stable_text = candidate
            if not text.startswith(session.stable_text):
                text = session.stable_text + text[len(session.stable_text):]

        # Resolve dedup after first post-trim transcription
        if session._needs_dedup and session._pre_trim_text:
            overlap = _suffix_prefix_overlap(session._pre_trim_text, text)
            committed = session._pre_trim_text
            if overlap > 0:
                committed = session._pre_trim_text[:-overlap]
            session.finalized_text += committed
            session._needs_dedup = False
            session._pre_trim_text = ""

        session.prev_text = text
        if language:
            session.language = language
        session.text = session.finalized_text + text
        session._current_window_text = text

    async def close(self):
        await self.client.close()


def _to_wav(audio: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    active_sessions: int = 0
    total_sessions: int = 0
    total_infer_calls: int = 0
    total_infer_errors: int = 0
    infer_latency_sum: float = 0.0
    audio_duration_processed: float = 0.0


metrics = Metrics()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
config = ASRConfig()
engine: ASREngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ASREngine(config)
    logger.info(f"ASR engine ready: {config.base_url} model={config.model}")
    yield
    await engine.close()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = INDEX_HTML
    html = html.replace("__CHUNK_MS__", str(config.chunk_ms))
    html = html.replace("__BASE_URL__", config.base_url)
    html = html.replace("__MODEL__", config.model)
    return html


@app.get("/audio-processor.js")
async def audio_processor_js():
    return Response(content=AUDIO_WORKLET_JS, media_type="application/javascript")


@app.get("/metrics")
async def get_metrics():
    avg_latency = (
        metrics.infer_latency_sum / max(metrics.total_infer_calls, 1) * 1000
    )
    return JSONResponse({
        "active_sessions": metrics.active_sessions,
        "total_sessions": metrics.total_sessions,
        "total_infer_calls": metrics.total_infer_calls,
        "total_infer_errors": metrics.total_infer_errors,
        "avg_infer_latency_ms": round(avg_latency, 1),
        "audio_hours_processed": round(metrics.audio_duration_processed / 3600, 3),
    })


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    """
    WebSocket protocol:
    - Client sends: binary frames (float32 PCM) or JSON control messages
    - Server sends: JSON { language, text } or { error }

    JSON control messages:
      { "type": "config", "context_sec": 10 }
      { "type": "finish" }
    """
    await ws.accept()
    session = StreamingSession(config=config)
    infer_task: asyncio.Task | None = None

    last_infer_time = 0.0
    min_infer_interval = config.chunk_ms / 1000.0
    last_activity = time.monotonic()
    timeout_sec = 300  # 5 min idle timeout

    metrics.active_sessions += 1
    metrics.total_sessions += 1

    # Connect to external VAD service
    vad_ws = None
    if not config.vad_url:
        logger.info(f"External VAD disabled, using energy VAD (threshold={config.energy_vad_threshold})")
    if config.vad_url:
        try:
            vad_ws = await websockets.connect(
                f"{config.vad_url}/vad_realtime_ws",
                open_timeout=5,
            )
            await vad_ws.send(json.dumps({
                "type": "init",
                "sample_rate": config.sample_rate,
                "chunk_size_ms": config.chunk_ms,
            }))
            init_resp = json.loads(await vad_ws.recv())
            if init_resp.get("status") != "initialized":
                logger.warning("VAD init failed, falling back to energy VAD")
                await vad_ws.close()
                vad_ws = None
            else:
                logger.info(f"External VAD connected: {config.vad_url}")
        except Exception as e:
            logger.warning(f"External VAD unavailable ({e}), falling back to energy VAD (threshold={config.energy_vad_threshold})")
            vad_ws = None

    timeout_event = asyncio.Event()

    async def watchdog():
        while True:
            await asyncio.sleep(30)
            if time.monotonic() - last_activity > timeout_sec:
                logger.info("Session timeout, signaling close")
                timeout_event.set()
                return

    watchdog_task = asyncio.create_task(watchdog())

    try:
        while True:
            if timeout_event.is_set():
                try:
                    await ws.send_json({"error": "session timeout"})
                except Exception:
                    pass
                break

            try:
                message = await asyncio.wait_for(ws.receive(), timeout=60)
            except asyncio.TimeoutError:
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
                continue

            last_activity = time.monotonic()

            if message["type"] == "websocket.disconnect":
                break

            # Binary frame = audio data
            if "bytes" in message and message["bytes"]:
                raw = message["bytes"]
                if len(raw) % 4 != 0:
                    await ws.send_json(
                        {"error": "float32 bytes length not multiple of 4"}
                    )
                    continue

                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                if not np.all(np.isfinite(chunk)):
                    logger.warning("Audio chunk contains NaN/Inf, sanitizing")
                    chunk = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                chunk = np.clip(chunk, -1.0, 1.0)
                session.append_audio(chunk)

                # VAD: forward audio to external VAD service
                if vad_ws:
                    try:
                        int16_bytes = (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                        await vad_ws.send(int16_bytes)
                        vad_resp = json.loads(await vad_ws.recv())
                        if vad_resp.get("status") == "detected":
                            for seg in vad_resp.get("segments", []):
                                if isinstance(seg, list) and len(seg) >= 2:
                                    if seg[1] == -1:
                                        session._vad_speech_active = True
                                        session._vad_ever_had_speech = True
                                        session._vad_chunks_since_speech_end = 0
                                    else:
                                        session._vad_speech_active = False
                                        session._vad_ever_had_speech = True
                                        session._vad_chunks_since_speech_end = 0
                        elif not session._vad_speech_active:
                            session._vad_chunks_since_speech_end += 1
                    except Exception as e:
                        logger.warning(f"VAD error: {e}, disabling VAD for this session")
                        vad_ws = None

                # Check max duration (unlimited mode)
                if (
                    session.context_window_sec <= 0
                    and session.audio_accum.shape[0] > config.max_audio_samples
                ):
                    await ws.send_json({"error": "max audio duration exceeded"})
                    break

                # Inference throttling: only infer when previous done + interval elapsed
                now = time.monotonic()
                if vad_ws:
                    should_infer = (
                        session._vad_speech_active
                        or (session._vad_ever_had_speech
                            and session._vad_chunks_since_speech_end <= 3)
                    )
                else:
                    # No external VAD → use energy-based fallback
                    is_speech = _check_energy_vad(chunk, config.energy_vad_threshold)
                    if is_speech:
                        session._energy_speech_active = True
                        session._vad_chunks_since_speech_end = 0
                    else:
                        if session._energy_speech_active:
                            session._energy_speech_active = False
                            session._vad_chunks_since_speech_end = 0
                        else:
                            session._vad_chunks_since_speech_end += 1
                    should_infer = (
                        session._energy_speech_active
                        or session._vad_chunks_since_speech_end <= 3
                    )
                if should_infer and (infer_task is None or infer_task.done()) and (
                    now - last_infer_time >= min_infer_interval
                ):
                    last_infer_time = now
                    infer_task = asyncio.create_task(
                        _do_infer_and_send(engine, session, ws)
                    )

            # Text frame = JSON control message
            elif "text" in message and message["text"]:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = ctrl.get("type", "")

                if msg_type == "config":
                    session.context_window_sec = int(ctrl.get("context_sec", 10))

                elif msg_type == "finish":
                    # Cancel any in-flight inference to avoid concurrent session mutation
                    if infer_task and not infer_task.done():
                        infer_task.cancel()
                        try:
                            await infer_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    # Final transcription
                    result = await engine.transcribe(session)
                    await ws.send_json(result)
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        metrics.active_sessions -= 1
        watchdog_task.cancel()
        if infer_task and not infer_task.done():
            infer_task.cancel()
        if vad_ws:
            try:
                await vad_ws.send(json.dumps({"type": "finish"}))
                await asyncio.wait_for(vad_ws.recv(), timeout=2)
            except Exception:
                pass
            await vad_ws.close()


async def _do_infer_and_send(
    eng: ASREngine, session: StreamingSession, ws: WebSocket
):
    t0 = time.monotonic()
    try:
        result = await eng.transcribe(session)
        metrics.total_infer_calls += 1
        elapsed = time.monotonic() - t0
        metrics.infer_latency_sum += elapsed
        metrics.audio_duration_processed += session.audio_duration_sec
        rtf = elapsed / max(session.audio_duration_sec, 0.001)
        logger.debug(
            f"infer latency={elapsed*1000:.0f}ms audio={session.audio_duration_sec:.1f}s rtf={rtf:.3f}"
        )
        await ws.send_json(result)
    except Exception as e:
        metrics.total_infer_errors += 1
        logger.error(f"Infer/send error: {e}")


# ---------------------------------------------------------------------------
# AudioWorklet processor JS (served at /audio-processor.js)
# ---------------------------------------------------------------------------
AUDIO_WORKLET_JS = r"""
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._bufferSize = 0;
    this._targetSamples = 16000;
    this._running = true;
    this._sourceSr = 16000;
    this._targetSr = 16000;

    this.port.onmessage = (e) => {
      if (e.data.type === 'stop') this._running = false;
      if (e.data.type === 'config') {
        this._targetSamples = e.data.targetSamples || 16000;
        this._sourceSr = e.data.sourceSr || 16000;
        this._targetSr = e.data.targetSr || 16000;
      }
    };
  }

  process(inputs) {
    if (!this._running) return false;
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];
    this._buffer.push(new Float32Array(channelData));
    this._bufferSize += channelData.length;

    if (this._bufferSize >= this._targetSamples) {
      let merged = this._mergeBuffers();
      if (this._sourceSr !== this._targetSr) {
        merged = this._resample(merged, this._sourceSr, this._targetSr);
      }
      this.port.postMessage({ type: 'audio', samples: merged }, [merged.buffer]);
      this._buffer = [];
      this._bufferSize = 0;
    }

    return true;
  }

  _resample(input, srcSr, dstSr) {
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  _mergeBuffers() {
    const out = new Float32Array(this._bufferSize);
    let offset = 0;
    for (const buf of this._buffer) {
      out.set(buf, offset);
      offset += buf.length;
    }
    return out;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
"""


# ---------------------------------------------------------------------------
# Frontend HTML (embedded, same UI style as realtime_demo.py)
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qwen3-ASR Realtime (WebSocket)</title>
  <style>
    :root{
      --bg:#f8fafc; --card:#ffffff; --muted:#5b6472; --text:#0f172a;
      --border:#e5e7eb; --ok:#059669; --warn:#d97706; --danger:#e11d48;
      --accent:#2563eb;
    }
    html,body{height:100%;margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,"Noto Sans";background:var(--bg);color:var(--text);}
    .wrap{height:100vh;padding:16px;box-sizing:border-box;display:flex;}
    .card{width:100%;height:100%;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px;box-sizing:border-box;box-shadow:0 10px 30px rgba(0,0,0,.06);display:flex;flex-direction:column;gap:12px;min-height:0;}
    h1{font-size:18px;margin:0;letter-spacing:.2px;display:flex;align-items:center;gap:8px;}
    h1 .tag{font-size:11px;padding:3px 8px;border-radius:999px;background:rgba(37,99,235,.1);color:var(--accent);font-weight:600;}
    .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
    button{border:1px solid var(--border);border-radius:10px;padding:10px 18px;cursor:pointer;color:var(--text);background:#f8fafc;transition:transform .05s,background .15s,border-color .15s;font-weight:700;font-size:14px;}
    button:hover{background:#f1f5f9;border-color:#cbd5e1;}
    button:active{transform:translateY(1px);}
    button.primary{border-color:rgba(5,150,105,.35);background:rgba(5,150,105,.10);color:#065f46;}
    button.danger{border-color:rgba(225,29,72,.35);background:rgba(225,29,72,.10);color:#9f1239;}
    button.ctx{font-size:12px;padding:6px 12px;border-radius:8px;}
    button.ctx.active{border-color:rgba(37,99,235,.4);background:rgba(37,99,235,.10);color:var(--accent);}
    button:disabled{opacity:.45;cursor:not-allowed;}
    .pill{font-size:12px;padding:5px 10px;border-radius:999px;border:1px solid var(--border);color:var(--muted);background:#f8fafc;user-select:none;}
    .pill.ok{color:#065f46;border-color:rgba(5,150,105,.35);background:rgba(5,150,105,.10);}
    .pill.warn{color:#92400e;border-color:rgba(217,119,6,.35);background:rgba(217,119,6,.10);}
    .pill.err{color:#9f1239;border-color:rgba(225,29,72,.35);background:rgba(225,29,72,.10);}
    .panel{border:1px solid var(--border);border-radius:12px;background:#fff;padding:12px;}
    .panel.textpanel{flex:1;display:flex;flex-direction:column;min-height:0;}
    .label{color:var(--muted);font-size:12px;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New";}
    #text{flex:1;min-height:0;white-space:pre-wrap;line-height:1.7;font-size:16px;padding:14px;border-radius:10px;border:1px solid var(--border);background:#f8fafc;overflow:auto;}
    .info{font-size:12px;color:var(--muted);margin-top:auto;padding-top:8px;border-top:1px solid var(--border);}
    .info code{background:#f1f5f9;padding:2px 5px;border-radius:4px;font-size:11px;}
    a{color:var(--accent);text-decoration:none;cursor:pointer;}
    #timer{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums;}
    .preview{color:var(--muted);opacity:0.5;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Qwen3-ASR Realtime <span class="tag">WebSocket</span></h1>
      <div class="row">
        <button id="btnStart" class="primary">Start</button>
        <button id="btnStop" class="danger" disabled>Stop</button>
        <span id="status" class="pill warn">Idle</span>
        <span id="timer"></span>
        <span style="margin-left:auto;display:flex;gap:4px;align-items:center;">
          <span class="label" style="margin:0;font-size:11px;">Window</span>
          <button class="ctx" data-ctx="3">3s</button>
          <button class="ctx" data-ctx="5">5s</button>
          <button class="ctx active" data-ctx="10">10s</button>
          <button class="ctx" data-ctx="0">&infin;</button>
        </span>
        <a id="btnClear" class="mono">Clear</a>
      </div>
      <div class="panel">
        <div class="label">Language</div>
        <div id="lang" class="mono">&mdash;</div>
      </div>
      <div class="panel textpanel">
        <div class="label">Transcription</div>
        <div id="text"></div>
      </div>
      <div class="info">
        Chunk interval: <code>__CHUNK_MS__ms</code> &middot;
        vLLM: <code>__BASE_URL__</code> &middot;
        Model: <code>__MODEL__</code> &middot;
        Protocol: <code>WebSocket</code>
      </div>
    </div>
  </div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const btnStart = $("btnStart");
  const btnStop  = $("btnStop");
  const btnClear = $("btnClear");
  const statusEl = $("status");
  const langEl   = $("lang");
  const textEl   = $("text");
  const timerEl  = $("timer");

  const CHUNK_MS  = __CHUNK_MS__;
  const TARGET_SR = 16000;

  let contextSec = 10;
  const ctxBtns = document.querySelectorAll("button.ctx");
  ctxBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      ctxBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      contextSec = parseInt(btn.dataset.ctx, 10);
      // Send config update to server via WebSocket
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "config", context_sec: contextSec }));
      }
    });
  });

  let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let mediaStream = null;
  let running = false;
  let startTime = 0;
  let timerInterval = null;
  let finishResolver = null;

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }
  function lockUI(on) {
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }
  function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m.toString().padStart(2, "0") + ":" + s.toString().padStart(2, "0");
  }

  btnClear.onclick = () => { textEl.innerHTML = ""; langEl.textContent = "\u2014"; };

  function connectWS() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(proto + "//" + location.host + "/ws/asr");
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        console.log("WebSocket connected");
        ws.send(JSON.stringify({ type: "config", context_sec: contextSec }));
        resolve();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "ping") return;
          if (data.error) {
            setStatus("Error: " + data.error, "err");
            if (finishResolver) { finishResolver(); finishResolver = null; }
            return;
          }
          if (data.language) langEl.textContent = data.language;
          // Incremental display: append confirmed delta, show preview temporarily
          if (data.delta) {
            const oldPreview = textEl.querySelector(".preview");
            if (oldPreview) oldPreview.remove();
            textEl.appendChild(document.createTextNode(data.delta));
          }
          {
            const oldPreview = textEl.querySelector(".preview");
            if (oldPreview) oldPreview.remove();
            if (data.preview) {
              const ps = document.createElement("span");
              ps.className = "preview";
              ps.textContent = data.preview;
              textEl.appendChild(ps);
            }
          }
          textEl.scrollTop = textEl.scrollHeight;
          if (finishResolver) { finishResolver(); finishResolver = null; }
        } catch (e) {
          console.error("Parse error:", e);
        }
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        if (finishResolver) { finishResolver(); finishResolver = null; }
        if (running) {
          setStatus("Reconnecting\u2026", "warn");
          textEl.innerHTML = "[Reconnected \u2014 previous context lost]\n";
          langEl.textContent = "\u2014";
          setTimeout(() => { connectWS().catch(() => {}); }, 1000);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        reject(err);
      };
    });
  }

  function waitForFinalResult(timeoutMs) {
    timeoutMs = timeoutMs || 15000;
    return new Promise((resolve) => {
      finishResolver = resolve;
      setTimeout(() => {
        if (finishResolver) { finishResolver = null; resolve(); }
      }, timeoutMs);
    });
  }

  async function startAudio() {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: TARGET_SR,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
    const actualSR = audioCtx.sampleRate;

    // Load AudioWorklet
    await audioCtx.audioWorklet.addModule("/audio-processor.js");

    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");

    // Configure chunk size and resampling (resampling done inside worklet)
    const targetSamples = Math.round(actualSR * (CHUNK_MS / 1000));
    workletNode.port.postMessage({
      type: "config",
      targetSamples: targetSamples,
      sourceSr: actualSR,
      targetSr: TARGET_SR,
    });

    workletNode.port.onmessage = (e) => {
      if (e.data.type === "audio" && ws && ws.readyState === WebSocket.OPEN) {
        // Backpressure: drop chunk if send buffer exceeds 256KB
        if (ws.bufferedAmount > 256 * 1024) {
          console.warn("WebSocket backpressure: dropping audio chunk");
          return;
        }
        ws.send(e.data.samples.buffer);
      }
    };

    source.connect(workletNode);
    workletNode.connect(audioCtx.destination);
  }

  function cleanup() {
    if (workletNode) {
      workletNode.port.postMessage({ type: "stop" });
      workletNode.disconnect();
      workletNode = null;
    }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  }

  btnStart.onclick = async () => {
    if (running) return;
    textEl.innerHTML = "";
    langEl.textContent = "\u2014";

    try {
      setStatus("Starting\u2026", "warn");
      lockUI(true);

      await connectWS();
      await startAudio();

      running = true;
      startTime = Date.now();
      timerEl.textContent = fmtTime(0);
      timerInterval = setInterval(() => {
        timerEl.textContent = fmtTime((Date.now() - startTime) / 1000);
      }, 1000);
      setStatus("Listening\u2026", "ok");
    } catch (err) {
      console.error(err);
      setStatus("Failed: " + err.message, "err");
      lockUI(false);
      running = false;
      cleanup();
      if (ws) { ws.close(); ws = null; }
    }
  };

  btnStop.onclick = async () => {
    if (!running) return;
    running = false;
    setStatus("Finishing\u2026", "warn");
    lockUI(false);

    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "finish" }));
        await waitForFinalResult();
      }
    } catch (err) {
      console.error(err);
      setStatus("Finish error: " + err.message, "err");
    } finally {
      cleanup();
      if (ws) { ws.close(1000, "user_stop"); ws = null; }
      setStatus("Stopped", "");
    }
  };
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time streaming ASR demo via WebSocket + FastAPI"
    )
    p.add_argument(
        "--base-url", default="http://localhost:7869/v1", help="vLLM server base URL"
    )
    p.add_argument(
        "--model", default="/models/Qwen3-ASR-1.7B", help="Model name on vLLM server"
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=6860, help="Bind port")
    p.add_argument(
        "--unfixed-chunk-num",
        type=int,
        default=4,
        help="Chunks before prefix stabilization",
    )
    p.add_argument(
        "--unfixed-token-num",
        type=int,
        default=5,
        help="Token rollback count for stabilization boundary (0 to disable)",
    )
    p.add_argument(
        "--chunk-ms", type=int, default=1000, help="Frontend chunk interval in ms"
    )
    p.add_argument(
        "--tokenizer",
        default="",
        help="Tokenizer path for token-level rollback (default: try model path, then char-based fallback)",
    )
    p.add_argument(
        "--max-audio-sec",
        type=int,
        default=600,
        help="Max audio duration per session (seconds)",
    )
    p.add_argument(
        "--vad-url",
        default="ws://localhost:7064",
        help="VAD WebSocket server URL (empty string to disable)",
    )
    p.add_argument(
        "--energy-vad-threshold",
        type=float,
        default=0.01,
        help="RMS threshold for energy-based VAD fallback (0 to disable)",
    )
    p.add_argument(
        "--repetition-threshold",
        type=int,
        default=10,
        help="Pattern repeat count threshold for repetition detection (0 to disable)",
    )
    p.add_argument("--ssl-certfile", default=None, help="SSL cert file")
    p.add_argument("--ssl-keyfile", default=None, help="SSL key file")
    return p.parse_args()


def main():
    args = parse_args()

    global config
    config = ASRConfig(
        base_url=args.base_url,
        model=args.model,
        host=args.host,
        port=args.port,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_ms=args.chunk_ms,
        max_audio_sec=args.max_audio_sec,
        tokenizer_path=args.tokenizer,
        vad_url=args.vad_url,
        repetition_threshold=args.repetition_threshold,
        energy_vad_threshold=args.energy_vad_threshold,
    )

    certfile = args.ssl_certfile
    keyfile = args.ssl_keyfile

    # Auto-detect cert.pem / key.pem in current directory
    if not certfile and not keyfile:
        if os.path.isfile("cert.pem") and os.path.isfile("key.pem"):
            certfile = "cert.pem"
            keyfile = "key.pem"
            logger.info("Auto-detected cert.pem and key.pem, enabling HTTPS")

    ssl_kwargs = {}
    if certfile and keyfile:
        if not os.path.isfile(certfile):
            logger.warning(
                f"SSL cert not found: {certfile}. "
                "Running without SSL. Microphone will only work on localhost. "
                "Generate self-signed cert: "
                "openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes"
            )
        else:
            ssl_kwargs = {
                "ssl_certfile": certfile,
                "ssl_keyfile": keyfile,
            }

    proto = "https" if ssl_kwargs else "http"
    print(f"vLLM server: {args.base_url}")
    print(f"Model:       {args.model}")
    print(f"Chunk:       {args.chunk_ms}ms")
    print(f"Stabilize:   unfixed_chunk={config.unfixed_chunk_num}, unfixed_token={config.unfixed_token_num}")
    print(f"VAD:         {config.vad_url or 'disabled'}")
    print(f"Listening:   {proto}://{args.host}:{args.port}")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        ws_max_size=1024 * 1024,  # 1MB per WS frame
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
