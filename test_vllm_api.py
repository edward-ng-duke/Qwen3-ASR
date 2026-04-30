"""
vLLM ASR API 测试脚本 —— 语言参数 & 热词测试

使用方法:
  python test_vllm_api.py --base-url http://localhost:7869/v1

注意：请在下方 TEST_CASES 中填写你自己的音频文件路径。
"""

import argparse
import subprocess
import sys
import time

from openai import OpenAI


def load_audio_as_wav(audio_path: str) -> bytes:
    """使用 ffmpeg 将音频转换为 16kHz 单声道 WAV 格式"""
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ffmpeg 错误: {result.stderr.decode()}", file=sys.stderr)
        return b""
    return result.stdout


def transcribe(client: OpenAI, model: str, audio_bytes: bytes,
               language: str | None = None, prompt: str | None = None) -> str:
    """调用 vLLM ASR API"""
    kwargs = {"model": model, "file": audio_bytes}
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    t0 = time.time()
    resp = client.audio.transcriptions.create(**kwargs)
    elapsed = time.time() - t0
    return resp.text, elapsed


# ============================================================
# 测试案例配置
# ============================================================

# ---- 语言测试 ----
# 请填写音频路径，留空则跳过该组测试
LANG_TEST_AUDIO_1 = ""  # <-- 填写第一个音频路径，例如 "data/上海话.wav"
LANG_TEST_AUDIO_1_DESC = "音频1"  # <-- 描述，例如 "上海话30秒"

LANG_TEST_AUDIO_2 = ""  # <-- 填写第二个音频路径，例如 "data/news.mp3"
LANG_TEST_AUDIO_2_DESC = "音频2"  # <-- 描述，例如 "中文新闻30秒"

# ---- 热词测试 ----
HOTWORD_TEST_AUDIO = ""  # <-- 填写音频路径
HOTWORD_TEST_AUDIO_DESC = "热词测试音频"  # <-- 描述
HOTWORDS = [
    # (描述, 热词字符串)
    ("无热词", None),
    ("热词A", "词1 词2 词3"),  # <-- 修改为你想测试的热词
    ("无关热词", "Apple Google Tesla"),  # <-- 故意无关的热词，观察干扰
]


def print_divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_language_tests(client: OpenAI, model: str, audio_path: str, desc: str):
    """对同一音频，分别用不同语言参数测试"""
    if not audio_path:
        return

    print_divider(f"语言测试: {desc} ({audio_path})")
    audio = load_audio_as_wav(audio_path)
    if not audio:
        print("  ⚠ 音频加载失败，跳过")
        return

    languages = [
        ("自动检测", None),
        ("中文 zh", "zh"),
        ("英文 en", "en"),
        ("日语 ja", "ja"),
    ]

    for lang_desc, lang_code in languages:
        try:
            text, elapsed = transcribe(client, model, audio, language=lang_code)
            print(f"\n  [{lang_desc}] ({elapsed:.1f}s)")
            print(f"  结果: {text}")
        except Exception as e:
            print(f"\n  [{lang_desc}] 错误: {e}")


def run_hotword_tests(client: OpenAI, model: str, audio_path: str, desc: str,
                      hotword_cases: list):
    """对同一音频，分别用不同热词测试"""
    if not audio_path:
        return

    print_divider(f"热词测试: {desc} ({audio_path})")
    audio = load_audio_as_wav(audio_path)
    if not audio:
        print("  ⚠ 音频加载失败，跳过")
        return

    for hw_desc, hw_prompt in hotword_cases:
        try:
            text, elapsed = transcribe(client, model, audio, prompt=hw_prompt)
            prompt_str = hw_prompt or "(无)"
            print(f"\n  [{hw_desc}] prompt=\"{prompt_str}\" ({elapsed:.1f}s)")
            print(f"  结果: {text}")
        except Exception as e:
            print(f"\n  [{hw_desc}] 错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="vLLM ASR API 语言 & 热词测试")
    parser.add_argument("--base-url", default="http://localhost:6001/v1")
    parser.add_argument("--model", default="/models/Qwen3-ASR-1.7B")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    # 检查服务
    try:
        models = client.models.list()
        print(f"服务连接成功: {args.base_url}")
        print(f"可用模型: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"无法连接服务 {args.base_url}: {e}")
        sys.exit(1)

    # 如果没有配置任何音频路径，用内置示例提示
    has_any = any([LANG_TEST_AUDIO_1, LANG_TEST_AUDIO_2, HOTWORD_TEST_AUDIO])
    if not has_any:
        print("\n⚠ 未配置任何音频路径！")
        print("请编辑脚本顶部的以下变量：")
        print("  LANG_TEST_AUDIO_1  - 语言测试音频1")
        print("  LANG_TEST_AUDIO_2  - 语言测试音频2")
        print("  HOTWORD_TEST_AUDIO - 热词测试音频")
        print("  HOTWORDS           - 热词测试案例")
        sys.exit(0)

    # 语言测试
    run_language_tests(client, args.model, LANG_TEST_AUDIO_1, LANG_TEST_AUDIO_1_DESC)
    run_language_tests(client, args.model, LANG_TEST_AUDIO_2, LANG_TEST_AUDIO_2_DESC)

    # 热词测试
    run_hotword_tests(client, args.model, HOTWORD_TEST_AUDIO, HOTWORD_TEST_AUDIO_DESC, HOTWORDS)

    print(f"\n{'='*60}")
    print("  测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
