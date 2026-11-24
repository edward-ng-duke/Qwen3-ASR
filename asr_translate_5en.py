#!/usr/bin/env python3
# coding=utf-8
"""对 data/5_en.wav 做 ASR（Qwen3-ASR），再用 LLM 翻译成中文并打印。

运行前设置 LLM_API_KEY，用于访问翻译模型。

流程：
  1. ffmpeg 把音频转成 16kHz 单声道 WAV
  2. 调用 Qwen3-ASR HTTP API 得到英文转写
  3. 调用 OpenAI 兼容 LLM 把英文译成中文
"""

import os
import re
import subprocess
import sys

from openai import OpenAI

from qwen_asr.inference.utils import parse_asr_output

# 长音频时 vLLM 会分段返回，每段都带一个 "language English<asr_text>" 标记，
# parse_asr_output 只能剥掉第一个，这里清掉文本中间残留的内联标记。
_INLINE_MARKER = re.compile(r"\s*language\s+\w+\s*<asr_text>\s*", re.IGNORECASE)

# ---- 硬编码配置 ----
AUDIO_PATH = "data/5_en.wav"
OUTPUT_PATH = "data/5_en_asr_zh.txt"

ASR_BASE_URL = "http://10.0.0.32:6001/v1"
ASR_MODEL = "models/Qwen3-ASR-1.7B"

LLM_BASE_URL = "http://10.0.0.94:5000/v1"
LLM_MODEL = "qwen36-27b-fp8"


def to_wav_16k_mono(path: str) -> bytes:
    """用 ffmpeg 把任意音频转成 16kHz 单声道 WAV 字节流。"""
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return result.stdout


def transcribe(wav_bytes: bytes) -> str:
    """调用 Qwen3-ASR，返回去掉前缀后的纯英文转写。"""
    client = OpenAI(base_url=ASR_BASE_URL, api_key="EMPTY")
    resp = client.audio.transcriptions.create(
        model=ASR_MODEL,
        file=("audio.wav", wav_bytes, "audio/wav"),
        language="en",
    )
    _, text = parse_asr_output(resp.text)
    text = _INLINE_MARKER.sub(" ", text).strip()
    return text


def translate_to_chinese(text: str) -> str:
    """调用 LLM 把英文翻译成中文，只返回译文。"""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=os.environ["LLM_API_KEY"])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是专业的翻译。把用户给出的英文准确、通顺地翻译成中文，只输出中文译文，不要添加解释或原文。",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main() -> int:
    print(f"[1/3] 转码音频: {AUDIO_PATH}")
    try:
        wav_bytes = to_wav_16k_mono(AUDIO_PATH)
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg，请先安装。", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"错误: ffmpeg 转码失败。\n{stderr}", file=sys.stderr)
        return 1

    print("[2/3] ASR 识别中（整段约 5 分钟音频，请稍候）...")
    try:
        en_text = transcribe(wav_bytes)
    except Exception as e:  # noqa: BLE001  网络/服务异常统一提示
        print(f"错误: ASR 请求失败: {e}", file=sys.stderr)
        return 1

    if not en_text:
        print("错误: ASR 返回为空。", file=sys.stderr)
        return 1

    print("\n===== 英文转写 (ASR) =====")
    print(en_text)

    print("\n[3/3] LLM 翻译成中文...")
    try:
        zh_text = translate_to_chinese(en_text)
    except Exception as e:  # noqa: BLE001
        print(f"错误: 翻译请求失败: {e}", file=sys.stderr)
        return 1

    print("\n===== 中文翻译 (LLM) =====")
    print(zh_text)

    out_path = OUTPUT_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("===== 英文转写 (ASR) =====\n")
        f.write(en_text + "\n\n")
        f.write("===== 中文翻译 (LLM) =====\n")
        f.write(zh_text + "\n")
    print(f"\n[已保存] 结果写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
