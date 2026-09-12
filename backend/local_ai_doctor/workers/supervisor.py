"""Async parent-side supervision for one crash-isolated model process."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import queue
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..domain.models import ModelDescriptor
from .runtime import worker_main


class WorkerFailure(RuntimeError):
    def __init__(self, error: Mapping[str, Any]) -> None:
        self.error = dict(error)
        super().__init__(str(error.get("message", "model worker failed")))


class ModelWorkerSupervisor:
    """Own a spawned worker; only serializable values cross this boundary."""

    def __init__(self, *, queue_limit: int = 256) -> None:
        self._context = mp.get_context("spawn")
        self._queue_limit = max(1, queue_limit)
        self._commands: Any
        self._output: Any
        self._cancel_event: Any
        self._process: Any
        self._build_worker()
        self._restart_lock = asyncio.Lock()
        self._pump_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._reply_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._run_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._loaded: dict[str, Any] | None = None
        self._loaded_request: dict[str, Any] | None = None
        self._run_lock = asyncio.Lock()
        self._active_run_id: str | None = None
        self._cancelled_run_ids: set[str] = set()
        self._stopping = False

    def _build_worker(self) -> None:
        self._poisoned = False
        self._commands = self._context.Queue(maxsize=self._queue_limit)
        self._output = self._context.Queue(maxsize=max(16, self._queue_limit * 4))
        self._cancel_event = self._context.Event()
        self._process = self._context.Process(
            target=worker_main,
            args=(self._commands, self._output, self._cancel_event),
            name="local-ai-doctor-model-worker",
            daemon=True,
        )

    @property
    def loaded(self) -> dict[str, Any] | None:
        return dict(self._loaded) if self._loaded else None

    @property
    def alive(self) -> bool:
        try:
            return bool(self._process.is_alive())
        except (AssertionError, ValueError):
            return False

    @property
    def available(self) -> bool:
        return self.alive and not getattr(self, "_poisoned", False)

    @staticmethod
    def _poisoned_error() -> dict[str, str]:
        return {
            "code": "model_worker_recycle_failed",
            "message": "timed-out model worker could not be terminated safely",
            "hint": "Restart the application before submitting another inference request.",
        }

    async def start(self) -> None:
        async with self._restart_lock:
            if self._stopping:
                raise WorkerFailure(
                    {
                        "code": "model_worker_stopping",
                        "message": "model worker is shutting down",
                        "hint": "Wait for application shutdown to complete before retrying.",
                    }
                )
            if getattr(self, "_poisoned", False) and self.alive:
                raise WorkerFailure(self._poisoned_error())
            if self.alive:
                return
            if self._process.pid is not None:
                await self._retire_dead_worker()
                self._build_worker()
            self._process.start()
            output = self._output
            process = self._process
            self._pump_task = asyncio.create_task(self._pump(output), name="model-worker-events")
            self._monitor_task = asyncio.create_task(
                self._monitor(process), name="model-worker-monitor"
            )

    async def _retire_dead_worker(self) -> None:
        process = self._process
        with contextlib.suppress(AssertionError, ValueError):
            if process.pid is not None and not process.is_alive():
                await asyncio.to_thread(process.join, 1.0)
        with contextlib.suppress(Exception):
            self._output.put_nowait(None)
        current_task = asyncio.current_task()
        tasks = tuple(
            task
            for task in (self._pump_task, self._monitor_task)
            if task is not None and task is not current_task
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pump_task = None
        self._monitor_task = None
        for channel in (self._commands, self._output):
            with contextlib.suppress(Exception):
                channel.cancel_join_thread()
                channel.close()

    async def _recycle_timed_out_worker(
        self,
        failed_process: Any,
        error: Mapping[str, Any],
    ) -> None:
        """Terminate a worker whose IPC state is no longer trustworthy.

        A timeout can leave the child computing indefinitely or leave a late
        reply in its output queue. Reusing either the process or its queues can
        misattribute that reply to later work, so replacement is atomic and the
        next operation starts from fresh IPC objects.
        """

        async with self._restart_lock:
            if failed_process is not self._process or self._stopping:
                return

            reply = {"kind": "reply", "ok": False, "error": dict(error)}
            for future in tuple(self._reply_futures.values()):
                if not future.done():
                    future.set_result(reply)
            for run_id, run_queue in tuple(self._run_queues.items()):
                terminal = {
                    "kind": "run_event",
                    "run_id": run_id,
                    "event_type": "error",
                    "payload": dict(error),
                }
                if run_queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        run_queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    run_queue.put_nowait(terminal)

            self._loaded = None
            self._loaded_request = None
            with contextlib.suppress(Exception):
                self._cancel_event.set()
            if self.alive:
                with contextlib.suppress(AssertionError, OSError, ValueError):
                    failed_process.terminate()
                await asyncio.to_thread(failed_process.join, 1.0)
            if self.alive:
                with contextlib.suppress(AssertionError, OSError, ValueError):
                    failed_process.kill()
                await asyncio.to_thread(failed_process.join, 1.0)
            if self.alive:
                self._poisoned = True
                raise WorkerFailure(self._poisoned_error())
            await self._retire_dead_worker()
            self._build_worker()

    async def _pump(self, output: Any) -> None:
        while True:
            item = await asyncio.to_thread(output.get)
            if item is None:
                return
            kind = item.get("kind")
            if kind == "reply":
                future = self._reply_futures.get(str(item.get("request_id")))
                if future is not None and not future.done():
                    future.set_result(item)
            elif kind == "run_event":
                run_queue = self._run_queues.get(str(item.get("run_id")))
                if run_queue is not None:
                    await run_queue.put(item)
            # Diagnostics intentionally remain private to the supervisor. The
            # structured error reply is safe for the API; tracebacks may contain paths.

    async def _monitor(self, process: Any) -> None:
        while not self._stopping:
            await asyncio.sleep(0.5)
            if process is not self._process:
                return
            if process.is_alive():
                continue
            self._loaded = None
            self._loaded_request = None
            error = {
                "code": "model_worker_exited",
                "message": f"model worker exited unexpectedly with code {process.exitcode}",
                "hint": "Inspect system memory, model integrity, and backend compatibility before retrying.",
            }
            reply = {"kind": "reply", "ok": False, "error": error}
            for future in tuple(self._reply_futures.values()):
                if not future.done():
                    future.set_result(reply)
            for run_id, run_queue in tuple(self._run_queues.items()):
                terminal = {
                    "kind": "run_event",
                    "run_id": run_id,
                    "event_type": "error",
                    "payload": error,
                }
                if run_queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        run_queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    run_queue.put_nowait(terminal)
            return

    async def _request(self, command: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
        if getattr(self, "_poisoned", False) or not self.alive:
            await self.start()
        process = self._process
        request_id = str(command.get("request_id") or uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._reply_futures[request_id] = future
        payload = dict(command)
        payload["request_id"] = request_id
        deadline = loop.time() + timeout_seconds
        try:
            await asyncio.to_thread(self._commands.put, payload, True, timeout_seconds)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            reply = await asyncio.wait_for(future, timeout=remaining)
        except (OSError, TimeoutError, ValueError, queue.Full) as exc:
            error = {
                "code": "model_worker_timeout",
                "message": f"model worker did not complete {command.get('op')} within {timeout_seconds:g} seconds",
                "hint": "Reduce the workload or increase the corresponding worker timeout.",
            }
            await self._recycle_timed_out_worker(process, error)
            raise WorkerFailure(error) from exc
        finally:
            self._reply_futures.pop(request_id, None)
        if not reply.get("ok"):
            raise WorkerFailure(reply.get("error", {"message": "model worker operation failed"}))
        return dict(reply.get("payload", {}))

    async def load_model(
        self,
        descriptor: ModelDescriptor,
        runtime: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        metadata = getattr(descriptor, "metadata", {}) or {}
        model = {
            "id": descriptor.id,
            "display_name": descriptor.display_name,
            "path": str(descriptor.path),
            "task": descriptor.task.value,
            "model_type": descriptor.model_type,
            "effective_context_limit": descriptor.effective_context_limit,
            "reasoning_delimiters": getattr(descriptor, "reasoning_delimiters", None),
            "fingerprint": descriptor.fingerprint.value,
            "embedding_pooling": metadata.get("pooling"),
            "joint_embedding_space": metadata.get("joint_embedding_space", False),
        }
        requested = {"model": model, "runtime": dict(runtime)}
        if (
            self._loaded is not None
            and self._loaded_request == requested
            and self._process.is_alive()
        ):
            return dict(self._loaded)

        # The worker unloads its current model before attempting a new load. Once
        # that request starts, the previous cache entry can no longer describe the
        # worker reliably, including when loading fails or times out.
        self._loaded = None
        self._loaded_request = None
        result = await self._request({"op": "load", **requested}, timeout_seconds)
        mismatches = {
            key: {"requested": expected, "loaded": result.get(key)}
            for key, expected in {
                "model_id": model["id"],
                "task": model["task"],
                "device": requested["runtime"].get("device"),
                "dtype": requested["runtime"].get("dtype"),
            }.items()
            if expected is not None and result.get(key) != expected
        }
        if mismatches:
            raise WorkerFailure(
                {
                    "code": "model_worker_state_mismatch",
                    "message": "model worker loaded a different model or runtime selection",
                    "hint": "Unload the model and retry with a clean worker process.",
                    "details": mismatches,
                }
            )
        self._loaded = result
        self._loaded_request = requested
        return dict(result)

    async def unload(self, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        result = await self._request({"op": "unload"}, timeout_seconds)
        self._loaded = None
        self._loaded_request = None
        return result

    async def generate(
        self,
        *,
        run_id: str,
        messages: list[dict[str, str]],
        sampling: Mapping[str, Any],
        effective_seed: int,
        instrumentation: str,
        deterministic_reference_mode: bool,
        max_prompt_tokens: int,
        reserved_output_tokens: int,
        timeout_seconds: float,
    ) -> AsyncIterator[dict[str, Any]]:
        if getattr(self, "_poisoned", False) or not self.alive:
            await self.start()
        async with self._run_lock:
            if getattr(self, "_poisoned", False) or not self.alive:
                await self.start()
            self._active_run_id = run_id
            request_id = str(uuid.uuid4())
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            run_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
            self._reply_futures[request_id] = future
            self._run_queues[run_id] = run_queue
            self._cancel_event.clear()
            process = self._process
            deadline = loop.time() + timeout_seconds
            if run_id in self._cancelled_run_ids:
                self._cancel_event.set()
            try:
                await asyncio.to_thread(
                    self._commands.put,
                    {
                        "op": "generate",
                        "request_id": request_id,
                        "run_id": run_id,
                        "messages": messages,
                        "sampling": dict(sampling),
                        "effective_seed": effective_seed,
                        "instrumentation": instrumentation,
                        "deterministic_reference_mode": deterministic_reference_mode,
                        "max_prompt_tokens": max_prompt_tokens,
                        "reserved_output_tokens": reserved_output_tokens,
                    },
                    True,
                    timeout_seconds,
                )
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError
                    event = await asyncio.wait_for(run_queue.get(), timeout=remaining)
                    yield event
                    if event.get("event_type") in {"completed", "cancelled", "error"}:
                        break
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                reply = await asyncio.wait_for(future, timeout=min(5.0, remaining))
                if not reply.get("ok") and event.get("event_type") != "error":
                    raise WorkerFailure(reply.get("error", {"message": "generation failed"}))
            except (OSError, TimeoutError, ValueError, queue.Full) as exc:
                self._cancel_event.set()
                error = {
                    "code": "inference_timeout",
                    "message": f"generation exceeded the {timeout_seconds:g}-second timeout",
                    "hint": "Reduce the output budget or increase workers.inference_timeout_seconds.",
                }
                await self._recycle_timed_out_worker(process, error)
                raise WorkerFailure(error) from exc
            finally:
                self._reply_futures.pop(request_id, None)
                self._run_queues.pop(run_id, None)
                self._cancelled_run_ids.discard(run_id)
                if self._active_run_id == run_id:
                    self._active_run_id = None

    async def embed(
        self,
        *,
        inputs: list[dict[str, Any]],
        dimensions: int | None,
        normalize: bool,
        batch_size: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return await self._request(
            {
                "op": "embed",
                "inputs": inputs,
                "dimensions": dimensions,
                "normalize": normalize,
                "batch_size": batch_size,
            },
            timeout_seconds,
        )

    async def score_prompt(self, text: str, *, timeout_seconds: float) -> dict[str, Any]:
        return await self._request(
            {"op": "score_prompt", "text": text},
            timeout_seconds,
        )

    def cancel(self, run_id: str | None = None) -> None:
        """Cancel one run without allowing a queued run to interrupt another."""

        if run_id is None:
            self._cancel_event.set()
            return
        self._cancelled_run_ids.add(run_id)
        if self._active_run_id == run_id:
            self._cancel_event.set()

    def forget_run(self, run_id: str) -> None:
        """Discard a cancellation queued before generation reached the worker."""

        if self._active_run_id != run_id:
            self._cancelled_run_ids.discard(run_id)

    async def close(self, *, grace_seconds: float = 15.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._cancel_event.set()
        if self.alive:
            with contextlib.suppress(WorkerFailure):
                await self._request({"op": "shutdown"}, grace_seconds)
            await asyncio.to_thread(self._process.join, grace_seconds)
            if self.alive:
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 5.0)
            if self.alive:
                self._process.kill()
                await asyncio.to_thread(self._process.join, 5.0)
        await self._retire_dead_worker()
        self._loaded = None
        self._loaded_request = None
