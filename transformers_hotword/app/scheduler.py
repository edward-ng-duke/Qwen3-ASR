from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass

from .backend import (
    InferenceResult,
    InferenceTask,
    PartialCallback,
    TransformersBackend,
)
from .metrics import QUEUE_DEPTH

REALTIME_PRIORITY = 0
OFFLINE_PRIORITY = 10


class SchedulerFull(RuntimeError):
    pass


@dataclass(slots=True)
class _Job:
    task: InferenceTask
    future: asyncio.Future[InferenceResult]
    callback: PartialCallback | None


class InferenceScheduler:
    def __init__(
        self, backend: TransformersBackend, max_pending: int, max_batch_size: int
    ) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.queue: asyncio.PriorityQueue[tuple[int, int, _Job]] = (
            asyncio.PriorityQueue(max_pending)
        )
        self._sequence = itertools.count()
        self._worker: asyncio.Task | None = None
        self._counts = {REALTIME_PRIORITY: 0, OFFLINE_PRIORITY: 0}

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(), name="qwen3-asr-gpu-scheduler"
            )

    async def close(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def submit(
        self, task: InferenceTask, callback: PartialCallback | None = None
    ) -> InferenceResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        job = _Job(task, future, callback)
        try:
            self.queue.put_nowait((task.priority, next(self._sequence), job))
        except asyncio.QueueFull as exc:
            raise SchedulerFull("GPU inference queue is full") from exc
        self._counts[task.priority] = self._counts.get(task.priority, 0) + 1
        self._update_depth(task.priority)
        return await future

    async def _run(self) -> None:
        while True:
            first_item = await self.queue.get()
            items = [first_item]
            first_job = first_item[2]
            self._decrement(first_job.task.priority)
            deferred: list[tuple[int, int, _Job]] = []
            if first_job.callback is None:
                while len(items) < self.max_batch_size:
                    try:
                        candidate = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    job = candidate[2]
                    self._decrement(job.task.priority)
                    compatible = (
                        job.callback is None
                        and job.task.priority == first_job.task.priority
                        and job.task.bucket == first_job.task.bucket
                    )
                    if compatible:
                        items.append(candidate)
                    else:
                        deferred.append(candidate)
                for item in deferred:
                    await self.queue.put(item)
                    self._counts[item[2].task.priority] = (
                        self._counts.get(item[2].task.priority, 0) + 1
                    )
                    self._update_depth(item[2].task.priority)
                    self.queue.task_done()
            jobs = [item[2] for item in items]
            try:
                if first_job.callback is not None:
                    result = await asyncio.to_thread(
                        self.backend.transcribe_stream,
                        first_job.task,
                        first_job.callback,
                    )
                    results = [result]
                else:
                    results = await asyncio.to_thread(
                        self.backend.transcribe_batch, [job.task for job in jobs]
                    )
                for job, result in zip(jobs, results):
                    if not job.future.cancelled():
                        job.future.set_result(result)
            except BaseException as exc:
                for job in jobs:
                    if not job.future.cancelled():
                        job.future.set_exception(exc)
            finally:
                for _item in items:
                    self.queue.task_done()

    def _decrement(self, priority: int) -> None:
        self._counts[priority] = max(0, self._counts.get(priority, 0) - 1)
        self._update_depth(priority)

    def _update_depth(self, priority: int) -> None:
        label = "realtime" if priority == REALTIME_PRIORITY else "offline"
        QUEUE_DEPTH.labels(label).set(self._counts.get(priority, 0))
