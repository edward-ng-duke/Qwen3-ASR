"""
E2E browser test: starts the realtime demo server, opens it in headless Chromium
via Playwright, injects a real audio file as if from a microphone, and checks
the transcription output for excessive repetition.

Requirements:
  pip install playwright
  playwright install chromium

Usage:
  python tests/test_e2e_browser.py [--audio data/什么是标准的上海话_30s.wav] [--duration 20]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wait_for_server(url, timeout=30):
    """Wait until server responds."""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def load_audio_file(path, target_sr=16000):
    """Load audio file and resample to target_sr, return float32 mono."""
    import soundfile as sf
    try:
        import librosa
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
    except ImportError:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def has_significant_repetition(text, min_pattern_len=4, max_repeats=3):
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
                return True, pat, count
    return False, "", 0


def run_test(audio_path, duration_sec=20, server_port=16860, base_url="http://localhost:7869/v1",
             model="/models/Qwen3-ASR-1.7B", headless=True):
    from playwright.sync_api import sync_playwright

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio = load_audio_file(audio_path)
    audio_duration = len(audio) / 16000
    print(f"  Duration: {audio_duration:.1f}s, samples: {len(audio)}")

    # Limit to duration_sec
    max_samples = int(duration_sec * 16000)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"  Trimmed to {duration_sec}s ({max_samples} samples)")

    # Start demo server (run from /tmp to avoid cert.pem auto-detection → no SSL)
    server_url = f"http://localhost:{server_port}"
    print(f"Starting demo server on port {server_port}...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "realtime_demo_1.py")
    server_proc = subprocess.Popen(
        [
            sys.executable, script,
            "--base-url", base_url,
            "--model", model,
            "--port", str(server_port),
            "--vad-url", "",  # disable external VAD, use energy VAD
            "--chunk-ms", "1000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/tmp",  # no cert.pem here → plain HTTP
        env={**os.environ, "PYTHONPATH": repo_root},
    )

    try:
        if not wait_for_server(server_url, timeout=15):
            out = server_proc.stdout.read(4096).decode(errors="replace")
            print(f"Server failed to start. Output:\n{out}")
            return False

        print(f"Server ready at {server_url}")

        # Prepare audio chunks (1 second each, as float32 bytes)
        chunk_size = 16000  # 1 second at 16kHz
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunks.append(chunk)

        print(f"  Prepared {len(chunks)} audio chunks (1s each)")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                permissions=["microphone"],
            )
            page = context.new_page()

            # Collect console logs
            console_logs = []
            page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

            print(f"Opening {server_url}...")
            page.goto(server_url, wait_until="networkidle")

            # Verify page loaded
            assert page.query_selector("#btnStart"), "Start button not found"
            print("  Page loaded OK")

            # Instead of trying to fake getUserMedia (complex), we'll directly
            # use the WebSocket API to inject audio and collect results.
            # This is more reliable than faking the microphone.
            print("Connecting WebSocket and injecting audio...")

            results = page.evaluate("""
                async (chunksBase64) => {
                    const results = [];
                    const TARGET_SR = 16000;

                    // Connect WebSocket
                    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const ws = new WebSocket(proto + '//' + location.host + '/ws/asr');
                    ws.binaryType = 'arraybuffer';

                    await new Promise((resolve, reject) => {
                        ws.onopen = resolve;
                        ws.onerror = reject;
                        setTimeout(() => reject(new Error('WS timeout')), 5000);
                    });

                    // Send config
                    ws.send(JSON.stringify({ type: 'config', context_sec: 10 }));

                    // Listen for responses
                    ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type !== 'ping') {
                                results.push(data);
                            }
                        } catch (e) {}
                    };

                    // Send audio chunks with delay
                    for (let i = 0; i < chunksBase64.length; i++) {
                        const binary = Uint8Array.from(atob(chunksBase64[i]), c => c.charCodeAt(0));
                        const float32 = new Float32Array(binary.buffer);
                        ws.send(float32.buffer);
                        // Wait for chunk interval
                        await new Promise(r => setTimeout(r, 200));
                    }

                    // Wait a bit for final inference
                    await new Promise(r => setTimeout(r, 3000));

                    // Send finish
                    ws.send(JSON.stringify({ type: 'finish' }));

                    // Wait for final response
                    await new Promise(r => setTimeout(r, 3000));

                    ws.close();
                    return results;
                }
            """, [
                # Convert each float32 chunk to base64
                __import__("base64").b64encode(chunk.tobytes()).decode()
                for chunk in chunks
            ])

            browser.close()

        # Analyze results
        print(f"\n{'='*60}")
        print(f"Received {len(results)} responses from server")

        deltas = []
        full_text = ""
        for r in results:
            if isinstance(r, dict):
                if r.get("delta"):
                    deltas.append(r["delta"])
                if r.get("text"):
                    full_text = r["text"]
                if r.get("error"):
                    print(f"  ERROR: {r['error']}")

        joined_deltas = "".join(deltas)

        print(f"\n--- Delta output ({len(joined_deltas)} chars) ---")
        print(joined_deltas[:500])
        if len(joined_deltas) > 500:
            print(f"... [{len(joined_deltas) - 500} more chars]")

        print(f"\n--- Final text ({len(full_text)} chars) ---")
        print(full_text[:500])
        if len(full_text) > 500:
            print(f"... [{len(full_text) - 500} more chars]")

        # Check for repetition
        print(f"\n{'='*60}")
        print("Repetition analysis:")

        found, pat, count = has_significant_repetition(joined_deltas)
        if found:
            print(f"  FAIL: Pattern '{pat}' repeated {count}x in deltas")
        else:
            print(f"  PASS: No significant repetition in deltas")

        found2, pat2, count2 = has_significant_repetition(full_text)
        if found2:
            print(f"  FAIL: Pattern '{pat2}' repeated {count2}x in final text")
        else:
            print(f"  PASS: No significant repetition in final text")

        # Check delta/text ratio (proxy for repetition)
        if full_text:
            ratio = len(joined_deltas) / len(full_text) if len(full_text) > 0 else 0
            print(f"  Delta/Text ratio: {ratio:.2f} (should be ~1.0, >1.5 is suspicious)")
            if ratio > 2.0:
                print(f"  WARN: Delta output much longer than final text - likely repetition in deltas")

        success = not found and not found2
        print(f"\n{'='*60}")
        print(f"Result: {'PASS' if success else 'FAIL'}")
        return success

    finally:
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="E2E browser test for realtime ASR demo")
    parser.add_argument("--audio", default="data/什么是标准的上海话_30s.wav",
                        help="Audio file to use as input")
    parser.add_argument("--duration", type=int, default=20,
                        help="Max audio duration in seconds")
    parser.add_argument("--port", type=int, default=16860,
                        help="Port for test server")
    parser.add_argument("--base-url", default="http://localhost:7869/v1",
                        help="vLLM backend URL")
    parser.add_argument("--model", default="/models/Qwen3-ASR-1.7B")
    parser.add_argument("--headed", action="store_true",
                        help="Run browser in headed mode (visible)")
    args = parser.parse_args()

    success = run_test(
        audio_path=args.audio,
        duration_sec=args.duration,
        server_port=args.port,
        base_url=args.base_url,
        model=args.model,
        headless=not args.headed,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
