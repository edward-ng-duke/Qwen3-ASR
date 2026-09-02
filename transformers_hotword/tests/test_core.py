from __future__ import annotations

import asyncio

import numpy as np
import pytest

from transformers_hotword.app.audio import (
    AudioPipeline,
    PCMStreamDecoder,
    PreparedAudio,
    SAMPLE_RATE,
)
from transformers_hotword.app.backend import (
    InferenceTask,
    TransformersBackend,
    is_effective_silence,
)
from transformers_hotword.app.config import Settings
from transformers_hotword.app.fallback import submit_no_context_fallback
from transformers_hotword.app.hotwords import (
    ContextGuard,
    HotwordValidationError,
    is_context_leak,
    looks_like_context_prefix,
    merge_prompt,
    parse_hotwords,
)
from transformers_hotword.app.languages import resolve_language
from transformers_hotword.app.prompt_verification import (
    requires_prompt_verification,
    transcription_agreement,
)
from transformers_hotword.app.scheduler import (
    OFFLINE_PRIORITY,
    InferenceScheduler,
    SchedulerFull,
)
from transformers_hotword.app.realtime import _merge_intervals, _subtract_intervals
from transformers_hotword.app.stability import (
    PrefixStabilizer,
    merge_final_segment,
    merge_revised_cumulative,
    merge_text,
    normalize_preview_after_confirmed,
)

from .conftest import FakeBackend


def test_realtime_first_update_interval_validation():
    Settings(
        realtime_first_update_seconds=0.5,
        realtime_update_seconds=1.0,
    ).validate()
    with pytest.raises(ValueError, match="ASR_REALTIME_FIRST_UPDATE_SECONDS"):
        Settings(
            realtime_first_update_seconds=0,
            realtime_update_seconds=1.0,
        ).validate()
    with pytest.raises(ValueError, match="ASR_REALTIME_FIRST_UPDATE_SECONDS"):
        Settings(
            realtime_first_update_seconds=1.1,
            realtime_update_seconds=1.0,
        ).validate()


def test_hotword_merge_is_ordered_trimmed_and_deduplicated():
    words = parse_hotwords('[" Qwen3-ASR ", "张三", "Qwen3-ASR"]')
    assert (
        merge_prompt("会议上下文", words) == "会议上下文\nVocabulary: Qwen3-ASR, 张三"
    )


def test_hotword_limits():
    assert parse_hotwords("not-json") == ["not-json"]
    with pytest.raises(HotwordValidationError):
        parse_hotwords('["not-json"')
    with pytest.raises(HotwordValidationError):
        merge_prompt("x" * 4097, [])
    with pytest.raises(HotwordValidationError):
        merge_prompt(None, ["x" * 65])


def test_context_echo_detection_does_not_reject_one_normal_hotword():
    guard = ContextGuard(
        "这是一场技术会议",
        (
            "商飞",
            "英伟达",
            "DeepSeek",
            "智谱",
            "声纹识别",
            "日产日清",
        ),
    )
    assert looks_like_context_prefix("商飞，英伟达", guard)
    assert is_context_leak("商飞，英伟达，DeepSeek，智谱。", guard)
    assert is_context_leak("正常开场。这是一场技术会议。", guard)
    assert not is_context_leak("我们使用 DeepSeek 完成了部署。", guard)
    assert not is_context_leak("商飞与英伟达正在开展正常业务合作。", guard)
    assert not looks_like_context_prefix("我们使用 DeepSeek 完成了部署。", guard)


def test_cross_language_prompt_verification_is_not_phrase_specific():
    assert requires_prompt_verification("English", None, "这是一场技术讨论会。")
    assert requires_prompt_verification("English", None, "هذا اجتماع تقني")
    assert requires_prompt_verification(None, "English", "技術会議です")
    assert not requires_prompt_verification("English", None, "Technical meeting")
    assert not requires_prompt_verification("Chinese", None, "这是一场技术讨论会。")
    assert not requires_prompt_verification("English", None, None)


def test_prompt_verification_agreement_tolerates_local_corrections():
    assert transcription_agreement(
        "We deployed Azure in production today.",
        "We deployed Astroic in production today.",
    ) > 0.7
    assert transcription_agreement(
        "This is a technical discussion meeting.", "Ahem."
    ) < 0.2


def test_language_resolution():
    assert resolve_language("auto") is None
    assert resolve_language("zh") == "Chinese"
    assert resolve_language("english") == "English"
    with pytest.raises(ValueError):
        resolve_language("xx")


def test_stabilizer_never_retracts_confirmed_text():
    stabilizer = PrefixStabilizer(rollback_units=2)
    stabilizer.update("今天上海天气很好")
    second = stabilizer.update("今天上海天气很好啊")
    third = stabilizer.update("今天上海天气很不错")
    assert second.confirmed
    assert third.confirmed.startswith(second.confirmed)
    assert merge_text("你好世界", "世界和平") == "你好世界和平"


def test_long_revised_hypothesis_is_not_duplicated():
    base = " ".join(f"word{index}" for index in range(35))
    stabilizer = PrefixStabilizer(rollback_units=2)
    stabilizer.update(base)
    stable = stabilizer.update(base + " first ending")
    assert len(stable.confirmed) > 120
    revised = base.replace("word5", "changed", 1) + " corrected ending"
    update = stabilizer.update(revised)
    final = stabilizer.finalize(revised)
    assert update.text.count("word0") == 1
    assert final.count("word0") == 1
    assert len(final) < len(base) + len(revised)
    assert merge_text(base, base) == base
    assert merge_revised_cumulative(base, revised).count("word0") == 1


def test_stabilizer_holds_visible_preview_when_hypothesis_rewrites_confirmed():
    stabilizer = PrefixStabilizer(rollback_units=2)
    hypotheses = [
        "那好，天津路。",
        "那好，天津路。我刚才说，我现在我我我在看了你这个产品。",
        "那好，天津路。老实说，我现在我我我在看了你这个产品以后啊。",
        "那好，天津路。老实说，我现在我我我在看了你这个产品以后啊，我就有些困惑。",
        "那好，天津路。老实说，我现在我我我在看了你这个产品以后啊，我就有些困惑，我我讨教啊。",
    ]
    for hypothesis in hypotheses:
        visible = stabilizer.update(hypothesis)

    before_revision = visible.text
    confirmed_before_revision = visible.confirmed
    conflicting = (
        "那好，天津路。好了，说我现在我我我在看了你这个产品以后啊，"
        "嗯嗯，我就有些。"
    )
    first = stabilizer.update(conflicting)
    second = stabilizer.update(conflicting)

    assert first.text == before_revision
    assert second.text == before_revision
    assert first.preview
    assert second.confirmed == confirmed_before_revision


def test_stabilizer_allows_two_unit_tail_revision_immediately():
    stabilizer = PrefixStabilizer(rollback_units=2)
    stabilizer.confirmed = "Hello world, "
    stabilizer.previous = "Hello world, alpha beta"
    stabilizer.displayed_text = stabilizer.previous

    revised = stabilizer.update("Hello world, gamma delta")

    assert revised.text == "Hello world, gamma delta"
    assert revised.preview == "gamma delta"


def test_stabilizer_requires_two_compatible_frames_for_large_preview_revision():
    stabilizer = PrefixStabilizer(rollback_units=2)
    stabilizer.update("今天北京天气很好")
    stable = stabilizer.update("今天北京天气很好啊")
    original = stable.text
    confirmed = stable.confirmed

    first = stabilizer.update(confirmed + "上海正在下大雨")
    second = stabilizer.update(confirmed + "上海正在下大雨呢")

    assert first.text == original
    assert second.text == confirmed + "上海正在下大雨呢"


def test_incompatible_large_revisions_restart_confirmation():
    stabilizer = PrefixStabilizer(rollback_units=2)
    stabilizer.update("今天北京天气很好")
    stable = stabilizer.update("今天北京天气很好啊")
    original = stable.text
    confirmed = stable.confirmed

    first = stabilizer.update(confirmed + "上海正在下大雨")
    second = stabilizer.update(confirmed + "广州气温明显上升")

    assert first.text == original
    assert second.text == original


def test_captured_revised_final_uses_terminal_anchor_without_duplication():
    confirmed = (
        "那好，天津路。老实说，我现在我我我在看了你这个产品以后啊，"
        "我就有些"
    )
    revised = (
        "那好，天津路。好了，说我现在我我我在看了你这个产品以后啊，"
        "嗯嗯，我就有些困惑，我我讨教啊，嗯嗯，假定，嗯，我是金信委。"
    )

    merged = merge_revised_cumulative(confirmed, revised)

    assert merged.startswith(confirmed)
    assert merged.count("那好，天津路") == 1
    assert merged.endswith("困惑，我我讨教啊，嗯嗯，假定，嗯，我是金信委。")


def test_revised_final_without_trustworthy_anchor_keeps_confirmed():
    confirmed = "已经确认的完整前缀内容"
    unrelated = "完全不同且无法安全对齐的新假设"

    assert merge_revised_cumulative(confirmed, unrelated) == confirmed


def test_final_segment_removes_captured_cumulative_replay():
    confirmed = (
        "那好，天津路。老实说，我现在我我我在看了你这个产品以后啊，"
        "我就有些"
    )
    replay = (
        "那好，天津路。好了，说我现在我我我在看了你这个产品以后啊，"
        "嗯嗯，我就有些困惑，我我讨教啊，嗯嗯，假定，嗯，我是金信委。"
    )

    merged = merge_final_segment(confirmed, replay)

    assert merged.startswith(confirmed)
    assert merged.count("那好，天津路") == 1
    assert "困惑" in merged


def test_preview_suppresses_revised_cumulative_replay_after_segment_final():
    confirmed = "那好，天津路。老实说，我现在我我我在看了你这个产品以"
    replayed_preview = (
        "那好，天津路。好了，说我现在我我我在看了你这个产品以后啊，"
        "嗯嗯，我就有些困惑，我要我我讨教啊。"
    )

    assert normalize_preview_after_confirmed(confirmed, replayed_preview) == ""
    assert normalize_preview_after_confirmed(
        confirmed, "后啊，我就有些困惑。"
    ) == "后啊，我就有些困惑。"


def test_preview_removes_exact_confirmed_suffix_overlap():
    assert normalize_preview_after_confirmed(
        "今天上海天气很好",
        "天气很好，我们出去走走。",
    ) == "，我们出去走走。"


def test_final_segment_keeps_independent_tail_without_anchor():
    assert merge_final_segment("第一段。", "完全独立的下一段。") == (
        "第一段。完全独立的下一段。"
    )


def test_pcm_decoder_handles_split_frames():
    decoder = PCMStreamDecoder("pcm_s16le", 16_000, 1)
    assert decoder.feed(b"\x00").size == 0
    assert decoder.feed(b"\x00\x00\x00").shape == (2,)
    assert decoder.finish().size == 0


def test_effective_digital_silence_gate():
    assert is_effective_silence(np.zeros(16000, dtype=np.float32))
    assert not is_effective_silence(np.array([0.0, 0.01], dtype=np.float32))


@pytest.mark.parametrize(
    ("supported", "expected"), [(True, "bfloat16"), (False, "float16")]
)
def test_dtype_selection(supported, expected):
    class FakeCuda:
        @staticmethod
        def is_bf16_supported():
            return supported

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = object()
        float16 = object()

    _dtype, name = TransformersBackend._select_dtype(FakeTorch)
    assert name == expected


def test_attention_auto_falls_back_to_sdpa_on_turing(tmp_path):
    class FakeCuda:
        @staticmethod
        def get_device_capability(_device):
            return (7, 5)

    class FakeTorch:
        cuda = FakeCuda()

    backend = TransformersBackend(
        Settings(temp_dir=tmp_path, torch_compile=False, warmup=False)
    )
    backend.torch = FakeTorch()
    assert backend._select_attention_backend() == "sdpa"


def test_compile_failure_restores_eager_and_warms_up(tmp_path, monkeypatch):
    original_forward = object()

    class FakeCuda:
        empty_cache_calls = 0

        @classmethod
        def empty_cache(cls):
            cls.empty_cache_calls += 1

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def compile(_forward, **_kwargs):
            raise RuntimeError("compile unavailable")

    class FakeModel:
        forward = original_forward

    settings = Settings(temp_dir=tmp_path, torch_compile=True, warmup=True)
    backend = TransformersBackend(settings)
    backend.torch = FakeTorch()
    backend.model = FakeModel()
    warmups = []
    monkeypatch.setattr(backend, "_warmup", lambda: warmups.append(True))
    backend._configure_forward(original_forward)
    assert backend.model.forward is original_forward
    assert backend.compile_enabled is False
    assert warmups == [True]
    assert FakeCuda.empty_cache_calls == 1


def test_compile_success_records_active_mode(tmp_path, monkeypatch):
    original_forward = object()
    compiled_forward = object()
    received = {}

    class FakeTorch:
        class cuda:
            @staticmethod
            def empty_cache():
                return None

        @staticmethod
        def compile(forward, **kwargs):
            received.update(kwargs)
            assert forward is original_forward
            return compiled_forward

    class FakeModel:
        forward = original_forward

    settings = Settings(
        temp_dir=tmp_path,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        warmup=True,
    )
    backend = TransformersBackend(settings)
    backend.torch = FakeTorch()
    backend.model = FakeModel()
    warmups = []
    monkeypatch.setattr(backend, "_warmup", lambda: warmups.append(True))
    backend._configure_forward(original_forward)
    assert backend.model.forward is compiled_forward
    assert backend.compile_enabled is True
    assert received == {"fullgraph": False, "mode": "reduce-overhead"}
    assert warmups == [True]


def test_four_hour_sparse_file_segments_without_loading_it_all(tmp_path, monkeypatch):
    pcm = tmp_path / "four-hours.f32le"
    samples = 4 * 3600 * SAMPLE_RATE
    with pcm.open("wb") as handle:
        handle.truncate(samples * 4)
    source = tmp_path / "source.wav"
    source.touch()
    prepared = PreparedAudio(source, pcm, samples, 4 * 3600)
    pipeline = AudioPipeline(tmp_path, 8 * 1024**3, 4 * 3600, 30, 1, 0.5, 250, 600)
    monkeypatch.setattr(pipeline, "_speech_ranges", lambda _audio: [(0, samples)])
    ranges = pipeline.segment_ranges(prepared)
    assert ranges[0] == (0, 30 * SAMPLE_RATE)
    assert ranges[-1][1] == samples
    assert all(end - start <= 30 * SAMPLE_RATE for start, end in ranges)


def test_contextual_short_audio_requires_vad_speech(tmp_path, monkeypatch):
    pcm = tmp_path / "noise.f32le"
    np.ones(SAMPLE_RATE, dtype="<f4").tofile(pcm)
    source = tmp_path / "noise.wav"
    source.touch()
    prepared = PreparedAudio(source, pcm, SAMPLE_RATE, 1.0)
    pipeline = AudioPipeline(tmp_path, 1024**3, 60, 30, 1, 0.5, 250, 600)
    monkeypatch.setattr(pipeline, "_speech_ranges", lambda _audio: [])
    assert pipeline.segment_ranges(prepared, require_speech=True) == []
    assert pipeline.segment_ranges(prepared, require_speech=False) == [(0, SAMPLE_RATE)]


def test_fallback_vad_threshold_must_be_lower_than_primary(tmp_path):
    Settings(
        temp_dir=tmp_path,
        vad_threshold=0.5,
        fallback_vad_threshold=0.1,
    ).validate()
    with pytest.raises(ValueError, match="ASR_FALLBACK_VAD_THRESHOLD"):
        Settings(
            temp_dir=tmp_path,
            vad_threshold=0.5,
            fallback_vad_threshold=0.5,
        ).validate()


def test_candidate_min_rms_validation(tmp_path):
    Settings(
        temp_dir=tmp_path,
        torch_compile=False,
        warmup=False,
        candidate_min_rms=0.0015,
    ).validate()
    with pytest.raises(ValueError):
        Settings(
            temp_dir=tmp_path,
            torch_compile=False,
            warmup=False,
            candidate_min_rms=-0.001,
        ).validate()
    with pytest.raises(ValueError):
        Settings(
            temp_dir=tmp_path,
            torch_compile=False,
            warmup=False,
            candidate_min_rms=1.0,
        ).validate()
    with pytest.raises(ValueError, match="ASR_FALLBACK_VAD_THRESHOLD"):
        Settings(
            temp_dir=tmp_path,
            vad_threshold=0.5,
            fallback_vad_threshold=0,
        ).validate()


def test_candidate_vad_interval_difference_prefers_primary_coverage():
    covered = _merge_intervals([(200, 500), (450, 700), (800, 900)])
    assert covered == [(200, 700), (800, 900)]
    assert _subtract_intervals(100, 1000, covered) == [
        (100, 200),
        (700, 800),
        (900, 1000),
    ]


def test_prepared_audio_cleanup_removes_both_temporary_files(tmp_path):
    source = tmp_path / "source.wav"
    pcm = tmp_path / "decoded.f32le"
    source.touch()
    pcm.touch()
    PreparedAudio(source, pcm, 0, 0).cleanup()
    assert not source.exists()
    assert not pcm.exists()


def test_scheduler_batches_already_queued_jobs(tmp_path):
    async def run():
        backend = FakeBackend(
            Settings(temp_dir=tmp_path, torch_compile=False, warmup=False)
        )
        scheduler = InferenceScheduler(backend, max_pending=32, max_batch_size=4)
        scheduler.start()
        tasks = [
            asyncio.create_task(
                scheduler.submit(
                    InferenceTask(
                        np.zeros(1600, np.float32), None, None, OFFLINE_PRIORITY, 5
                    )
                )
            )
            for _ in range(4)
        ]
        await asyncio.gather(*tasks)
        await scheduler.close()
        assert max(backend.batch_sizes) == 4

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", [RuntimeError("inference failed"), SchedulerFull("queue full")]
)
def test_no_context_fallback_propagates_errors(failure):
    class FailingScheduler:
        async def submit(self, _task, _callback=None):
            raise failure

    async def run():
        task = InferenceTask(
            np.ones(1600, np.float32), None, None, OFFLINE_PRIORITY, 5
        )
        with pytest.raises(type(failure), match=str(failure)):
            await submit_no_context_fallback(
                FailingScheduler(),
                task,
                mode="offline",
                reason="vad_reject",
                request_id="test-error",
            )

    asyncio.run(run())


def test_no_context_fallback_propagates_cancellation():
    class BlockingScheduler:
        async def submit(self, _task, _callback=None):
            await asyncio.Event().wait()

    async def run():
        task = InferenceTask(
            np.ones(1600, np.float32), None, None, OFFLINE_PRIORITY, 5
        )
        pending = asyncio.create_task(
            submit_no_context_fallback(
                BlockingScheduler(),
                task,
                mode="sse",
                reason="context_echo",
                request_id="test-cancel",
            )
        )
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(run())
