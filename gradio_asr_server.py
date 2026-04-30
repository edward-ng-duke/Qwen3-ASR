"""
Standalone Gradio ASR server.

- ASR transcription via vLLM OpenAI-compatible API (external service)
- Timestamp alignment via transformers-loaded Qwen3-ForcedAligner (local)
- Gradio UI for upload-and-transcribe with optional timestamp visualization

Usage:
  python gradio_asr_server.py
  python gradio_asr_server.py --base-url http://localhost:6001/v1 --model /models/Qwen3-ASR-1.7B
  python gradio_asr_server.py --aligner-checkpoint models/Qwen3-ForcedAligner-0.6B
"""

import argparse
import base64
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from openai import OpenAI
from scipy.io.wavfile import write as wav_write

from qwen_asr import Qwen3ForcedAligner


# ---------------------------------------------------------------------------
# Audio helpers (from demo.py)
# ---------------------------------------------------------------------------
def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    if isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, int):
            return _normalize_audio(a1), int(a0)
        if isinstance(a1, int):
            return _normalize_audio(a0), int(a1)
    return None


def _parse_audio_any(audio: Any) -> Tuple[np.ndarray, int]:
    if audio is None:
        raise ValueError("Audio is required.")
    at = _audio_to_tuple(audio)
    if at is not None:
        return at
    raise ValueError("Unsupported audio input format.")


def _to_wav(audio: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ASR output parsing (inlined from utils.py to avoid heavy deps)
# ---------------------------------------------------------------------------
def _detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i : i + count])
                i += count
        return "".join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        if n < thresh * 2:
            return s
        i = 0
        result = []
        found = False
        while i <= n - thresh * 2:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i : i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx : start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index : end_index + k] == pattern:
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def _normalize_language_name(language: str) -> str:
    s = str(language).strip()
    if not s:
        return ""
    return s[:1].upper() + s[1:].lower()


def _parse_asr_output(raw: str) -> Tuple[str, str]:
    if not raw:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""
    s = _detect_and_fix_repetitions(s)

    if _ASR_TEXT_TAG not in s:
        return "", s

    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta_part.lower():
        t = text_part.strip()
        return ("", t) if t else ("", "")

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX) :].strip()
            if val:
                lang = _normalize_language_name(val)
            break

    return lang, text_part.strip()


# ---------------------------------------------------------------------------
# Timestamp visualization HTML (from demo.py)
# ---------------------------------------------------------------------------
def _make_timestamp_html(audio_upload: Any, timestamps: Any) -> str:
    at = _audio_to_tuple(audio_upload)
    if at is None:
        raise ValueError("Audio input is required for visualization.")
    audio, sr = at

    if not timestamps:
        return "<div style='color:#666'>No timestamps to visualize.</div>"
    if not isinstance(timestamps, list):
        raise ValueError("Timestamps must be a list (JSON array).")

    html_content = """
    <style>
        .word-alignment-container { display: flex; flex-wrap: wrap; gap: 10px; }
        .word-box {
            border: 1px solid #ddd; border-radius: 8px; padding: 10px;
            background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            text-align: center;
        }
        .word-text { font-size: 18px; font-weight: 700; margin-bottom: 5px; }
        .word-time { font-size: 12px; color: #666; margin-bottom: 8px; }
        .word-audio audio { width: 140px; height: 30px; }
        details { border: 1px solid #ddd; border-radius: 6px; padding: 10px; background-color: #f7f7f7; }
        summary { font-weight: 700; cursor: pointer; }
    </style>
    <details open>
        <summary>Timestamps Visualization</summary>
        <div class="word-alignment-container" style="margin-top: 14px;">
    """

    for item in timestamps:
        if not isinstance(item, dict):
            continue
        word = str(item.get("text", "") or "")
        start = item.get("start_time", None)
        end = item.get("end_time", None)
        if start is None or end is None:
            continue
        start, end = float(start), float(end)
        if end <= start:
            continue

        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        if end_sample <= start_sample:
            continue

        seg = audio[start_sample:end_sample]
        seg_i16 = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)
        mem = io.BytesIO()
        wav_write(mem, sr, seg_i16)
        mem.seek(0)
        b64 = base64.b64encode(mem.read()).decode("utf-8")
        audio_src = f"data:audio/wav;base64,{b64}"

        html_content += f"""
        <div class="word-box">
            <div class="word-text">{word}</div>
            <div class="word-time">{start} - {end} s</div>
            <div class="word-audio">
                <audio controls preload="none" src="{audio_src}"></audio>
            </div>
        </div>
        """

    html_content += "</div></details>"
    return html_content


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _title_case_display(s: str) -> str:
    s = (s or "").strip().replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


# ---------------------------------------------------------------------------
# Gradio demo builder
# ---------------------------------------------------------------------------
def build_demo(
    client: OpenAI,
    model: str,
    aligner: Optional[Qwen3ForcedAligner],
    base_url: str,
    aligner_ckpt: Optional[str],
) -> gr.Blocks:
    has_aligner = aligner is not None

    # Get language list from aligner if available
    if has_aligner:
        supported_langs_raw = aligner.get_supported_languages() or []
    else:
        supported_langs_raw = []

    lang_choices_disp, lang_map = _build_choices_and_map(supported_langs_raw)
    lang_choices = ["Auto"] + lang_choices_disp

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 ASR Server
**ASR Backend:** `vLLM API @ {base_url}`
**Model:** `{model}`
**Forced Aligner:** `{aligner_ckpt if aligner_ckpt else "(none)"}`
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                audio_in = gr.Audio(label="Audio Input", type="numpy")
                lang_in = gr.Dropdown(
                    label="Language",
                    choices=lang_choices,
                    value="Auto",
                    interactive=True,
                )
                if has_aligner:
                    ts_in = gr.Checkbox(
                        label="Return Timestamps",
                        value=True,
                    )
                else:
                    ts_in = gr.State(False)
                btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                out_lang = gr.Textbox(label="Detected Language", lines=1)
                out_text = gr.Textbox(label="Result Text", lines=12)

            if has_aligner:
                with gr.Column(scale=3):
                    out_ts = gr.JSON(label="Timestamps")
                    viz_btn = gr.Button("Visualize Timestamps", variant="secondary")
            else:
                with gr.Column(scale=3):
                    out_ts = gr.State(None)
                    viz_btn = gr.State(None)

        if has_aligner:
            with gr.Row():
                out_ts_html = gr.HTML(label="Timestamps Visualization")
        else:
            out_ts_html = gr.State("")

        def run(audio_upload: Any, lang_disp: str, return_ts: bool):
            wav, sr = _parse_audio_any(audio_upload)
            wav_bytes = _to_wav(wav, sr)

            # Call vLLM ASR via OpenAI API
            transcription = client.audio.transcriptions.create(
                model=model,
                file=wav_bytes,
            )
            raw = transcription.text or ""
            language, text = _parse_asr_output(raw)

            # Override language if user selected one
            if lang_disp and lang_disp != "Auto":
                language = lang_map.get(lang_disp, lang_disp)

            return_ts = bool(return_ts) and has_aligner

            if has_aligner:
                ts_payload = None
                if return_ts and text:
                    results = aligner.align(
                        audio=(wav, sr),
                        text=text,
                        language=language or "Chinese",
                    )
                    if results:
                        ts_payload = [
                            dict(
                                text=item.text,
                                start_time=item.start_time,
                                end_time=item.end_time,
                            )
                            for item in results[0]
                        ]
                return (
                    language or "",
                    text or "",
                    gr.update(value=ts_payload) if return_ts else gr.update(value=None),
                    gr.update(value=""),
                )
            else:
                return (
                    language or "",
                    text or "",
                )

        def visualize(audio_upload: Any, timestamps_json: Any):
            return _make_timestamp_html(audio_upload, timestamps_json)

        if has_aligner:
            btn.click(
                run,
                inputs=[audio_in, lang_in, ts_in],
                outputs=[out_lang, out_text, out_ts, out_ts_html],
            )
            viz_btn.click(
                visualize,
                inputs=[audio_in, out_ts],
                outputs=[out_ts_html],
            )
        else:
            btn.click(
                run,
                inputs=[audio_in, lang_in, ts_in],
                outputs=[out_lang, out_text],
            )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}")


def _parse_json_dict(s: Optional[str], *, name: str) -> Dict[str, Any]:
    if s is None or not str(s).strip():
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a JSON object (dict).")
    return obj


def _coerce_special_types(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "dtype" and isinstance(v, str):
            out[k] = _dtype_from_str(v)
        else:
            out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone Gradio ASR server (vLLM API + Transformers Aligner)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--base-url", default="http://localhost:6001/v1", help="vLLM API base URL")
    p.add_argument("--model", default="/models/Qwen3-ASR-1.7B", help="Model name on vLLM server")
    p.add_argument(
        "--aligner-checkpoint",
        default="models/Qwen3-ForcedAligner-0.6B",
        help="Qwen3-ForcedAligner checkpoint path (set empty to disable)",
    )
    p.add_argument(
        "--aligner-kwargs",
        default=None,
        help='JSON dict for aligner kwargs.\nExample: \'{"dtype":"bfloat16","device_map":"cuda:0"}\'',
    )
    p.add_argument("--ip", default="0.0.0.0", help="Gradio bind IP")
    p.add_argument("--port", type=int, default=7860, help="Gradio port")
    p.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Create public Gradio link",
    )
    p.add_argument("--concurrency", type=int, default=16, help="Gradio queue concurrency")
    return p


def main():
    args = build_parser().parse_args()

    # OpenAI client for vLLM
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    # Load aligner
    aligner = None
    aligner_ckpt = (args.aligner_checkpoint or "").strip()
    if aligner_ckpt:
        aligner_kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
        user_kwargs = _parse_json_dict(args.aligner_kwargs, name="--aligner-kwargs")
        aligner_kwargs.update(user_kwargs)
        aligner_kwargs = _coerce_special_types(aligner_kwargs)
        print(f"Loading aligner: {aligner_ckpt} with {aligner_kwargs}")
        aligner = Qwen3ForcedAligner.from_pretrained(aligner_ckpt, **aligner_kwargs)
        print("Aligner loaded.")
    else:
        aligner_ckpt = None

    demo = build_demo(client, args.model, aligner, args.base_url, aligner_ckpt)

    print(f"vLLM API:  {args.base_url}")
    print(f"Model:     {args.model}")
    print(f"Aligner:   {aligner_ckpt or '(disabled)'}")
    print(f"Listening: http://{args.ip}:{args.port}")

    demo.queue(default_concurrency_limit=args.concurrency).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
