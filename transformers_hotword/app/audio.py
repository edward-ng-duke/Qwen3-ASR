from __future__ import annotations

import asyncio
import math
import queue
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(ValueError):
    pass


@dataclass(slots=True)
class PreparedAudio:
    source_path: Path
    pcm_path: Path
    samples: int
    duration: float

    def read(self, start: int = 0, end: int | None = None) -> np.ndarray:
        mapped = np.memmap(self.pcm_path, mode="r", dtype="<f4")
        try:
            return np.asarray(mapped[start:end], dtype=np.float32).copy()
        finally:
            del mapped

    def cleanup(self) -> None:
        for path in (self.source_path, self.pcm_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class AudioPipeline:
    def __init__(
        self,
        temp_dir: Path,
        max_upload_bytes: int,
        max_audio_seconds: float,
        segment_seconds: float,
        overlap_seconds: float,
        vad_threshold: float,
        vad_min_speech_ms: int,
        vad_min_silence_ms: int,
    ) -> None:
        self.temp_dir = temp_dir
        self.max_upload_bytes = max_upload_bytes
        self.max_audio_seconds = max_audio_seconds
        self.segment_seconds = segment_seconds
        self.overlap_seconds = overlap_seconds
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = vad_min_speech_ms
        self.vad_min_silence_ms = vad_min_silence_ms
        self._vad_pool: queue.LifoQueue = queue.LifoQueue(maxsize=4)
        self._vad_pool_lock = threading.Lock()
        self._vad_models_created = 0

    @contextmanager
    def _vad_model(self):
        try:
            model = self._vad_pool.get_nowait()
        except queue.Empty:
            model = None
            with self._vad_pool_lock:
                if self._vad_models_created < self._vad_pool.maxsize:
                    from silero_vad import load_silero_vad

                    model = load_silero_vad(onnx=True, opset_version=16)
                    self._vad_models_created += 1
            if model is None:
                model = self._vad_pool.get()
        try:
            model.reset_states()
            yield model
        finally:
            model.reset_states()
            self._vad_pool.put(model)

    def warmup_vad(self) -> None:
        with self._vad_model():
            pass

    def cleanup_stale(self, older_than_seconds: float = 24 * 3600) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - older_than_seconds
        for path in self.temp_dir.glob("qwen3-asr-*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    async def prepare_upload(self, upload) -> PreparedAudio:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload.filename or "audio.bin").suffix[:12]
        source = tempfile.NamedTemporaryFile(
            dir=self.temp_dir, prefix="qwen3-asr-upload-", suffix=suffix, delete=False
        )
        source_path = Path(source.name)
        size = 0
        try:
            with source:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise AudioError(
                            f"upload exceeds {self.max_upload_bytes} bytes"
                        )
                    source.write(chunk)
            return await asyncio.to_thread(self.prepare_path, source_path)
        except Exception:
            source_path.unlink(missing_ok=True)
            raise

    def prepare_path(self, source_path: Path) -> PreparedAudio:
        probed_duration = self._probe_duration(source_path)
        if probed_duration is not None and probed_duration > self.max_audio_seconds:
            raise AudioError(f"audio exceeds {self.max_audio_seconds:g} seconds")

        handle = tempfile.NamedTemporaryFile(
            dir=self.temp_dir, prefix="qwen3-asr-pcm-", suffix=".f32le", delete=False
        )
        pcm_path = Path(handle.name)
        handle.close()
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            str(pcm_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=False)
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
                raise AudioError(f"ffmpeg could not decode audio: {detail.strip()}")
            size = pcm_path.stat().st_size
            if size % 4:
                raise AudioError("decoded PCM has an invalid length")
            samples = size // 4
            duration = samples / SAMPLE_RATE
            if samples == 0:
                raise AudioError("audio is empty")
            if duration > self.max_audio_seconds + 0.1:
                raise AudioError(f"audio exceeds {self.max_audio_seconds:g} seconds")
            return PreparedAudio(source_path, pcm_path, samples, duration)
        except Exception:
            pcm_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=30
        )
        if completed.returncode != 0:
            return None
        try:
            value = float(completed.stdout.strip())
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def segment_ranges(
        self, audio: PreparedAudio, require_speech: bool = False
    ) -> list[tuple[int, int]]:
        max_samples = int(self.segment_seconds * SAMPLE_RATE)
        if audio.samples <= max_samples:
            if require_speech and not self._speech_ranges(audio):
                return []
            return [(0, audio.samples)]

        speech_ranges = self._speech_ranges(audio)
        if not speech_ranges:
            return []

        grouped: list[tuple[int, int]] = []
        current_start, current_end = speech_ranges[0]
        for start, end in speech_ranges[1:]:
            if end - current_start <= max_samples:
                current_end = end
            else:
                grouped.extend(
                    self._split_range(current_start, current_end, max_samples)
                )
                current_start, current_end = start, end
        grouped.extend(self._split_range(current_start, current_end, max_samples))
        return grouped

    def _split_range(
        self, start: int, end: int, max_samples: int
    ) -> list[tuple[int, int]]:
        overlap = int(self.overlap_seconds * SAMPLE_RATE)
        result: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_samples, end)
            result.append((cursor, chunk_end))
            if chunk_end >= end:
                break
            cursor = max(cursor + 1, chunk_end - overlap)
        return result

    def _speech_ranges(self, audio: PreparedAudio) -> list[tuple[int, int]]:
        import torch
        from silero_vad import VADIterator

        mapped = np.memmap(audio.pcm_path, mode="r", dtype="<f4")
        pending_start: int | None = None
        ranges: list[tuple[int, int]] = []
        try:
            with self._vad_model() as model:
                iterator = VADIterator(
                    model,
                    threshold=self.vad_threshold,
                    sampling_rate=SAMPLE_RATE,
                    min_silence_duration_ms=self.vad_min_silence_ms,
                    speech_pad_ms=30,
                )
                for offset in range(0, audio.samples, 512):
                    block = np.array(
                        mapped[offset : offset + 512], dtype=np.float32, copy=True
                    )
                    if block.size < 512:
                        block = np.pad(block, (0, 512 - block.size))
                    event = iterator(torch.from_numpy(block))
                    if not event:
                        continue
                    if "start" in event:
                        pending_start = min(int(event["start"]), audio.samples)
                    elif "end" in event and pending_start is not None:
                        end = min(int(event["end"]), audio.samples)
                        if (
                            end - pending_start
                            >= self.vad_min_speech_ms * SAMPLE_RATE / 1000
                        ):
                            ranges.append((pending_start, end))
                        pending_start = None
            if pending_start is not None:
                ranges.append((pending_start, audio.samples))
        finally:
            del mapped
        return ranges


class PCMStreamDecoder:
    def __init__(self, encoding: str, sample_rate: int, channels: int) -> None:
        if encoding not in {"pcm_s16le", "pcm_f32le"}:
            raise AudioError("unsupported PCM encoding")
        if not 8000 <= sample_rate <= 96000:
            raise AudioError("sample_rate must be between 8000 and 96000")
        if channels not in {1, 2}:
            raise AudioError("channels must be 1 or 2")
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_bytes = 2 if encoding == "pcm_s16le" else 4
        self._bytes = bytearray()
        self._resampler = None
        if sample_rate != SAMPLE_RATE:
            import soxr

            self._resampler = soxr.ResampleStream(
                sample_rate, SAMPLE_RATE, 1, dtype="float32"
            )

    def feed(self, data: bytes) -> np.ndarray:
        self._bytes.extend(data)
        frame_bytes = self.sample_bytes * self.channels
        usable = len(self._bytes) - (len(self._bytes) % frame_bytes)
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        raw = bytes(self._bytes[:usable])
        del self._bytes[:usable]
        dtype = "<i2" if self.encoding == "pcm_s16le" else "<f4"
        samples = np.frombuffer(raw, dtype=dtype)
        if self.encoding == "pcm_s16le":
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32, copy=False)
        if self.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1, dtype=np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(samples, -1.0, 1.0, out=samples)
        if self._resampler is not None:
            samples = self._resampler.resample_chunk(samples, last=False)
        return np.asarray(samples, dtype=np.float32)

    def finish(self) -> np.ndarray:
        if self._bytes:
            raise AudioError("final PCM frame ends with a partial sample")
        if self._resampler is None:
            return np.empty(0, dtype=np.float32)
        return np.asarray(
            self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True),
            dtype=np.float32,
        )


class RealtimeVAD:
    def __init__(self, threshold: float, min_silence_ms: int) -> None:
        import torch
        from silero_vad import VADIterator, load_silero_vad

        self._torch = torch
        self._iterator = VADIterator(
            load_silero_vad(onnx=True, opset_version=16),
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=30,
        )
        self._pending = np.empty(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> list[dict[str, int]]:
        if samples.size:
            self._pending = np.concatenate((self._pending, samples))
        events: list[dict[str, int]] = []
        while self._pending.size >= 512:
            block = self._pending[:512]
            self._pending = self._pending[512:]
            event = self._iterator(self._torch.from_numpy(block))
            if event:
                events.append({key: int(value) for key, value in event.items()})
        return events

    def finish(self) -> list[dict[str, int]]:
        if self._pending.size == 0:
            return []
        block = np.pad(self._pending, (0, 512 - self._pending.size))
        self._pending = np.empty(0, dtype=np.float32)
        event = self._iterator(self._torch.from_numpy(block))
        return [{key: int(value) for key, value in event.items()}] if event else []

    def reset(self) -> None:
        """Forget an open speech hypothesis at a trusted primary-VAD boundary."""
        self._iterator.reset_states()
        self._pending = np.empty(0, dtype=np.float32)
