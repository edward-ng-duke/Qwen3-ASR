"""
vLLM ASR 服务调用示例（带语言参数和热词）

使用方法:
  # 启动 vLLM 服务
  qwen-asr-serve models/Qwen3-ASR-1.7B --gpu-memory-utilization 0.6 --host 0.0.0.0 --port 6001

  # 运行示例
  python example_vllm_with_language.py --audio data/test.mp3 --language zh
  python example_vllm_with_language.py --audio data/test.mp3  # 自动检测语言
  python example_vllm_with_language.py --audio data/test.mp3 --prompt "交易 停滞"  # 使用热词

支持的语言代码 (ISO 639-1):
  zh - 中文       en - 英语       yue - 粤语      ja - 日语
  ko - 韩语       de - 德语       fr - 法语       es - 西班牙语
  pt - 葡萄牙语   ru - 俄语       ar - 阿拉伯语   it - 意大利语
  id - 印尼语     th - 泰语       vi - 越南语     tr - 土耳其语
"""

import argparse
import subprocess
import sys

from openai import OpenAI


def load_audio_as_wav(audio_path: str) -> bytes:
    """使用 ffmpeg 将音频转换为 16kHz 单声道 WAV 格式"""
    result = subprocess.run(
        [
            "ffmpeg", "-i", audio_path,
            "-f", "wav", "-ar", "16000", "-ac", "1",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def transcribe(
    client: OpenAI,
    model: str,
    audio_bytes: bytes,
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """调用 vLLM 服务进行语音转录

    Args:
        client: OpenAI 客户端
        model: 模型名称
        audio_bytes: WAV 格式音频数据
        language: 语言代码 (ISO 639-1)，None 表示自动检测
        prompt: 热词/上下文提示，用于引导模型识别特定词汇

    Returns:
        转录文本
    """
    kwargs = {
        "model": model,
        "file": audio_bytes,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    transcription = client.audio.transcriptions.create(**kwargs)
    return transcription.text


def main():
    parser = argparse.ArgumentParser(
        description="vLLM ASR 服务调用示例（带语言参数）"
    )
    parser.add_argument(
        "--audio",
        default="data/什么是标准的上海话_30s.wav",
        help="音频文件路径",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:6001/v1",
        help="vLLM 服务地址",
    )
    parser.add_argument(
        "--model",
        default="models/Qwen3-ASR-1.7B",
        help="模型名称",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="语言代码 (如 zh, en, ja)，不指定则自动检测",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="热词/上下文提示，用于引导模型识别特定词汇 (如 '交易 停滞 量化交易')",
    )
    args = parser.parse_args()

    # 初始化客户端
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    # 加载音频
    print(f"加载音频: {args.audio}")
    audio_bytes = load_audio_as_wav(args.audio)

    # 转录
    lang_info = args.language if args.language else "自动检测"
    print(f"语言: {lang_info}")
    if args.prompt:
        print(f"热词: {args.prompt}")
    print("转录中...")

    text = transcribe(client, args.model, audio_bytes, args.language, args.prompt)

    print(f"\n转录结果:\n{text}")


if __name__ == "__main__":
    main()
