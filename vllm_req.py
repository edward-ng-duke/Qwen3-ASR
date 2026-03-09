import subprocess
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:6001/v1",
    api_key="EMPTY"
)

audio_path = "data/2news_30s.mp3"
audio_path = "data/粤语新闻_30s.mp3"
audio_path = "data/什么是标准的上海话_30s.wav"

# Convert mp3 to wav using ffmpeg (pipe to stdout)
audio_file = subprocess.run(
    ["ffmpeg", "-i", audio_path, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
    capture_output=True, check=True,
).stdout

transcription = client.audio.transcriptions.create(
    model="models/Qwen3-ASR-1.7B",
    file=audio_file,
)

print(transcription.text)