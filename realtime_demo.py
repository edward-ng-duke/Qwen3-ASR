"""
Real-time streaming ASR demo via vLLM API.

Connects to an already-running vLLM Docker container and provides a web UI
with real-time microphone capture and progressive transcription.

Usage:
  python realtime_demo.py
  python realtime_demo.py --base-url http://localhost:7869/v1 --port 7860
  python realtime_demo.py --ssl-certfile cert.pem --ssl-keyfile key.pem

Open: http://localhost:7860 (or https:// with SSL)
"""

import argparse
import io
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request
from openai import OpenAI

# ---------------------------------------------------------------------------
# Globals (set in main)
# ---------------------------------------------------------------------------
client: OpenAI = None
MODEL: str = ""
UNFIXED_CHUNK_NUM: int = 2
MAX_AUDIO_SAMPLES: int = 600 * 16000
CHUNK_MS: int = 1000

app = Flask(__name__)

SESSIONS: Dict[str, "Session"] = {}
SESSION_TTL_SEC = 10 * 60

_ASR_TEXT_TAG = "<asr_text>"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
@dataclass
class Session:
    created_at: float
    last_seen: float
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    call_id: int = 0
    stable_text: str = ""
    prev_text: str = ""
    language: str = ""
    text: str = ""
    finalized_text: str = ""
    context_window_sec: int = 0  # 0 = unlimited


def _gc_sessions():
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


# ---------------------------------------------------------------------------
# Helpers (adapted from vllm_req_streaming.py)
# ---------------------------------------------------------------------------
def numpy_to_wav_bytes(wav: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def parse_asr_output(raw: str) -> tuple:
    if not raw:
        return "", ""
    s = raw.strip()
    if _ASR_TEXT_TAG in s:
        meta, text = s.split(_ASR_TEXT_TAG, 1)
        lang = ""
        m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
        if m and m.group(1).lower() != "none":
            lang = m.group(1)
        return lang, text.strip()
    return "", s


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def transcribe_and_stabilize(session: Session) -> dict:
    """Call vLLM API with accumulated audio and apply prefix stabilization."""
    if session.audio_accum.shape[0] == 0:
        return {"language": session.language, "text": session.text}

    wav_bytes = numpy_to_wav_bytes(session.audio_accum)

    try:
        transcription = client.audio.transcriptions.create(
            model=MODEL,
            file=wav_bytes,
        )
        raw = transcription.text or ""
    except Exception as e:
        return {"error": f"vLLM API error: {e}"}, 502

    language, text = parse_asr_output(raw)

    session.call_id += 1
    if session.call_id > UNFIXED_CHUNK_NUM and session.prev_text:
        cp = common_prefix_len(session.prev_text, text)
        if cp > len(session.stable_text):
            session.stable_text = text[:cp]
        if text[: len(session.stable_text)] != session.stable_text:
            text = session.stable_text + text[len(session.stable_text) :]

    session.prev_text = text
    if language:
        session.language = language
    session.text = session.finalized_text + text

    return {"language": session.language, "text": session.text}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qwen3-ASR Realtime</title>
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Qwen3-ASR Realtime <span class="tag">vLLM</span></h1>
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
        Model: <code>__MODEL__</code>
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
    });
  });

  let audioCtx = null, processor = null, source = null, mediaStream = null;
  let sessionId = null, running = false;
  let buf = new Float32Array(0);
  let pushing = false;
  let startTime = 0;
  let timerInterval = null;

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }
  function lockUI(on) {
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }
  function concatF32(a, b) {
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0); out.set(b, a.length);
    return out;
  }
  function resample(input, srcSr, dstSr) {
    if (srcSr === dstSr) return input;
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
  function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m.toString().padStart(2, "0") + ":" + s.toString().padStart(2, "0");
  }

  async function apiStart() {
    const r = await fetch("/api/start", {method: "POST"});
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    sessionId = j.session_id;
  }
  async function apiChunk(f32) {
    const r = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId) + "&context_sec=" + contextSec, {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream"},
      body: f32.buffer
    });
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
  }
  async function apiFinish() {
    const r = await fetch("/api/finish?session_id=" + encodeURIComponent(sessionId), {method: "POST"});
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  btnClear.onclick = () => { textEl.textContent = ""; langEl.textContent = "\u2014"; };

  async function stopAudio() {
    try {
      if (processor) { processor.disconnect(); processor.onaudioprocess = null; }
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    } catch(e) {}
    processor = null; source = null; audioCtx = null; mediaStream = null;
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  }

  btnStart.onclick = async () => {
    if (running) return;
    textEl.textContent = "";
    langEl.textContent = "\u2014";
    buf = new Float32Array(0);

    try {
      setStatus("Starting\u2026", "warn");
      lockUI(true);
      await apiStart();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true},
        video: false
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = resample(input, audioCtx.sampleRate, TARGET_SR);
        buf = concatF32(buf, resampled);
        if (!pushing) pump();
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      running = true;
      startTime = Date.now();
      timerEl.textContent = fmtTime(0);
      timerInterval = setInterval(() => {
        timerEl.textContent = fmtTime((Date.now() - startTime) / 1000);
      }, 1000);
      setStatus("Listening\u2026", "ok");

    } catch(err) {
      console.error(err);
      setStatus("Failed: " + err.message, "err");
      lockUI(false);
      running = false;
      sessionId = null;
      await stopAudio();
    }
  };

  async function pump() {
    if (pushing) return;
    pushing = true;
    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));
    try {
      while (running && buf.length >= chunkSamples) {
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);
        const j = await apiChunk(chunk);
        if (j.error) { setStatus("Error: " + j.error, "err"); break; }
        langEl.textContent = j.language || "\u2014";
        textEl.textContent = j.text || "";
        // auto-scroll
        textEl.scrollTop = textEl.scrollHeight;
      }
    } catch(err) {
      console.error(err);
      if (running) setStatus("Error: " + err.message, "err");
    } finally {
      pushing = false;
    }
  }

  btnStop.onclick = async () => {
    if (!running) return;
    running = false;
    setStatus("Finishing\u2026", "warn");
    lockUI(false);
    await stopAudio();
    try {
      if (sessionId) {
        const j = await apiFinish();
        langEl.textContent = j.language || "\u2014";
        textEl.textContent = j.text || "";
      }
      setStatus("Stopped", "");
    } catch(err) {
      console.error(err);
      setStatus("Finish error: " + err.message, "err");
    } finally {
      sessionId = null;
      buf = new Float32Array(0);
      pushing = false;
    }
  };
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    html = INDEX_HTML
    html = html.replace("__CHUNK_MS__", str(CHUNK_MS))
    html = html.replace("__BASE_URL__", str(client.base_url) if client else "")
    html = html.replace("__MODEL__", MODEL)
    return Response(html, mimetype="text/html; charset=utf-8")


@app.post("/api/start")
def api_start():
    session_id = uuid.uuid4().hex
    now = time.time()
    SESSIONS[session_id] = Session(created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})


@app.post("/api/chunk")
def api_chunk():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    raw = request.get_data(cache=False)
    if len(raw) == 0:
        return jsonify({"language": s.language, "text": s.text})
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    chunk = np.frombuffer(raw, dtype=np.float32)
    s.audio_accum = np.concatenate([s.audio_accum, chunk])

    # Update context window from frontend (can change mid-session)
    ctx_sec = int(request.args.get("context_sec", "0"))
    s.context_window_sec = ctx_sec

    # Sliding window: trim audio and finalize stable text
    if s.context_window_sec > 0:
        max_samples = s.context_window_sec * 16000
        if s.audio_accum.shape[0] > max_samples:
            # Commit current stable text before trimming
            if s.stable_text:
                s.finalized_text += s.stable_text
            s.audio_accum = s.audio_accum[-max_samples:]
            # Reset stabilization state for new window
            s.call_id = 0
            s.prev_text = ""
            s.stable_text = ""
    elif s.audio_accum.shape[0] > MAX_AUDIO_SAMPLES:
        return jsonify({"error": "max audio duration exceeded"}), 400

    result = transcribe_and_stabilize(s)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@app.post("/api/finish")
def api_finish():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    result = transcribe_and_stabilize(s)
    SESSIONS.pop(session_id, None)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time streaming ASR demo via vLLM API")
    p.add_argument("--base-url", default="http://localhost:7869/v1", help="vLLM server base URL")
    p.add_argument("--model", default="/models/Qwen3-ASR-1.7B", help="Model name on vLLM server")
    p.add_argument("--host", default="0.0.0.0", help="Flask bind host")
    p.add_argument("--port", type=int, default=7860, help="Flask bind port")
    p.add_argument("--unfixed-chunk-num", type=int, default=2, help="Chunks before prefix stabilization")
    p.add_argument("--chunk-ms", type=int, default=1000, help="Frontend chunk interval in ms")
    p.add_argument("--max-audio-sec", type=int, default=600, help="Max audio duration per session (seconds)")
    p.add_argument("--ssl-certfile", default="cert.pem", help="SSL cert file (needed for mic on non-localhost)")
    p.add_argument("--ssl-keyfile", default="key.pem", help="SSL key file")
    return p.parse_args()


def main():
    args = parse_args()

    global client, MODEL, UNFIXED_CHUNK_NUM, MAX_AUDIO_SAMPLES, CHUNK_MS

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    MODEL = args.model
    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    MAX_AUDIO_SAMPLES = args.max_audio_sec * 16000
    CHUNK_MS = args.chunk_ms

    ssl_context = None
    if args.ssl_certfile and args.ssl_keyfile:
        ssl_context = (args.ssl_certfile, args.ssl_keyfile)

    print(f"vLLM server: {args.base_url}")
    print(f"Model:       {args.model}")
    print(f"Chunk:       {args.chunk_ms}ms")
    print(f"Listening:   {'https' if ssl_context else 'http'}://{args.host}:{args.port}")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
        ssl_context=ssl_context,
    )


if __name__ == "__main__":
    main()
