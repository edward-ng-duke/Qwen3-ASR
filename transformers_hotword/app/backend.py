from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Settings

LOGGER = logging.getLogger("qwen3_asr.backend")


@dataclass(slots=True)
class InferenceTask:
    audio: np.ndarray
    language: str | None
    prompt: str | None
    priority: int
    bucket: int


@dataclass(slots=True)
class InferenceResult:
    language: str | None
    text: str
    raw_text: str
    processing_ms: float


PartialCallback = Callable[[InferenceResult], None]


def is_effective_silence(audio: np.ndarray) -> bool:
    return audio.size == 0 or float(np.max(np.abs(audio))) < 1e-5


class TransformersBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.processor = None
        self.torch = None
        self.dtype = None
        self.dtype_name = "unknown"
        self.attention_backend = "unknown"
        self.compile_enabled = False
        self.device_name = "unknown"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-ASR hotword image requires an NVIDIA CUDA GPU")
        self.dtype, self.dtype_name = self._select_dtype(torch)
        self.device_name = torch.cuda.get_device_name(
            torch.device(self.settings.device)
        )
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

        requested_attention = self._select_attention_backend()
        LOGGER.info(
            "loading model id=%s revision=%s device=%s dtype=%s attention=%s",
            self.settings.model_id,
            self.settings.model_revision,
            self.settings.device,
            self.dtype_name,
            requested_attention,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.settings.model_path,
            local_files_only=True,
        )
        try:
            self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
                self.settings.model_path,
                local_files_only=True,
                dtype=self.dtype,
                attn_implementation=requested_attention,
            ).to(self.settings.device)
            self.attention_backend = requested_attention
        except Exception:
            if (
                requested_attention != "flash_attention_2"
                or self.settings.attention_backend != "auto"
            ):
                raise
            LOGGER.exception("FlashAttention 2 load failed; retrying with SDPA")
            self.model = None
            torch.cuda.empty_cache()
            self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
                self.settings.model_path,
                local_files_only=True,
                dtype=self.dtype,
                attn_implementation="sdpa",
            ).to(self.settings.device)
            self.attention_backend = "sdpa"
        self.model.eval()

        self._configure_forward(self.model.forward)

    @staticmethod
    def _select_dtype(torch_module):
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16, "bfloat16"
        return torch_module.float16, "float16"

    def _configure_forward(self, original_forward) -> None:
        assert self.model is not None and self.torch is not None
        if self.settings.torch_compile:
            try:
                compile_kwargs = {"fullgraph": False}
                if self.settings.torch_compile_mode != "default":
                    compile_kwargs["mode"] = self.settings.torch_compile_mode
                self.model.forward = self.torch.compile(
                    original_forward, **compile_kwargs
                )
                self.compile_enabled = True
                if self.settings.warmup:
                    self._warmup()
            except Exception:
                LOGGER.exception("torch.compile warmup failed; using eager forward")
                self.model.forward = original_forward
                self.torch.cuda.empty_cache()
                self.compile_enabled = False
                if self.settings.warmup:
                    self._warmup()
        elif self.settings.warmup:
            self._warmup()

    def _select_attention_backend(self) -> str:
        if self.settings.attention_backend != "auto":
            return self.settings.attention_backend
        assert self.torch is not None
        major, _minor = self.torch.cuda.get_device_capability(self.settings.device)
        if major >= 8:
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except Exception:
                LOGGER.exception("prebuilt flash_attn is unavailable; selecting SDPA")
        return "sdpa"

    def _warmup(self) -> None:
        assert self.model is not None and self.torch is not None
        LOGGER.info("starting warmup buckets=%s", self.settings.warmup_seconds)
        for seconds in self.settings.warmup_seconds:
            audio = np.zeros(seconds * 16_000, dtype=np.float32)
            task = InferenceTask(audio, "English", None, priority=10, bucket=seconds)
            self._generate_batch([task], max_new_tokens=8)
        self.torch.cuda.synchronize(self.settings.device)
        LOGGER.info("warmup completed compile=%s", self.compile_enabled)

    def transcribe_batch(self, tasks: list[InferenceTask]) -> list[InferenceResult]:
        active = [
            (index, task)
            for index, task in enumerate(tasks)
            if not is_effective_silence(task.audio)
        ]
        results = [InferenceResult(task.language, "", "", 0.0) for task in tasks]
        if active:
            generated = self._generate_batch(
                [task for _index, task in active],
                max_new_tokens=self.settings.max_new_tokens,
            )
            for (index, _task), result in zip(active, generated):
                results[index] = result
        return results

    def _generate_batch(
        self, tasks: list[InferenceTask], max_new_tokens: int
    ) -> list[InferenceResult]:
        assert (
            self.model is not None
            and self.processor is not None
            and self.torch is not None
        )
        started = time.perf_counter()
        inputs = self.processor.apply_transcription_request(
            audio=[task.audio for task in tasks],
            language=[task.language for task in tasks],
            prompt=[task.prompt for task in tasks],
        ).to(self.settings.device, self.dtype)
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        self.torch.cuda.synchronize(self.settings.device)
        processing_ms = (time.perf_counter() - started) * 1000
        generated = output_ids[:, input_length:]
        return [
            self._decode(generated[index], task.language, processing_ms)
            for index, task in enumerate(tasks)
        ]

    def transcribe_stream(
        self, task: InferenceTask, callback: PartialCallback
    ) -> InferenceResult:
        from transformers import TextIteratorStreamer

        if is_effective_silence(task.audio):
            return InferenceResult(task.language, "", "", 0.0)

        assert (
            self.model is not None
            and self.processor is not None
            and self.torch is not None
        )
        inputs = self.processor.apply_transcription_request(
            audio=task.audio,
            language=task.language,
            prompt=task.prompt,
        ).to(self.settings.device, self.dtype)
        input_length = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120,
        )
        state: dict[str, object] = {}
        started = time.perf_counter()

        def generate() -> None:
            try:
                with self.torch.inference_mode():
                    state["output"] = self.model.generate(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=self.settings.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )
            except BaseException as exc:  # propagated to the scheduler thread
                state["error"] = exc

        worker = threading.Thread(
            target=generate, name="qwen3-asr-generate", daemon=True
        )
        worker.start()
        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            parsed = self._parse_text(accumulated, task.language)
            callback(
                InferenceResult(
                    language=parsed[0],
                    text=parsed[1],
                    raw_text=accumulated,
                    processing_ms=(time.perf_counter() - started) * 1000,
                )
            )
        worker.join()
        if "error" in state:
            raise state["error"]  # type: ignore[misc]
        self.torch.cuda.synchronize(self.settings.device)
        output_ids = state["output"]
        generated = output_ids[0, input_length:]  # type: ignore[index]
        return self._decode(
            generated, task.language, (time.perf_counter() - started) * 1000
        )

    def _decode(
        self, token_ids, forced_language: str | None, processing_ms: float
    ) -> InferenceResult:
        assert self.processor is not None
        raw_text = self.processor.decode(token_ids, skip_special_tokens=False)
        clean_raw = self.processor.decode(token_ids, skip_special_tokens=True)
        language, text = self._parse_text(clean_raw, forced_language)
        return InferenceResult(language, text, raw_text, processing_ms)

    def _parse_text(
        self, decoded: str, forced_language: str | None
    ) -> tuple[str | None, str]:
        assert self.processor is not None
        parsed = self.processor.parse_output(decoded)
        if isinstance(parsed, list):
            parsed = parsed[0]
        language = parsed.get("language") or forced_language
        text = str(parsed.get("transcription") or "").strip()
        return language, text

    def info(self) -> dict[str, object]:
        return {
            "id": self.settings.model_id,
            "revision": self.settings.model_revision,
            "transformers_revision": self.settings.transformers_revision,
            "device": self.device_name,
            "dtype": self.dtype_name,
            "attention_backend": self.attention_backend,
            "torch_compile": self.compile_enabled,
            "torch_compile_mode": self.settings.torch_compile_mode,
        }
