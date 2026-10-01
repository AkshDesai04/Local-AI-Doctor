"""Async parent-side supervision for one crash-isolated model process."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import queue
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..domain.capabilities import Capability, CapabilityState
from ..domain.models import ModelDescriptor, ModelTask, TrustDecision
from .runtime import worker_main


class WorkerFailure(RuntimeError):
    def __init__(self, error: Mapping[str, Any]) -> None:
        self.error = dict(error)
        super().__init__(str(error.get("message", "model worker failed")))


def worker_model_payload(descriptor: ModelDescriptor) -> dict[str, Any]:
    """The bounded model description the worker loads; never includes weights."""

    metadata = getattr(descriptor, "metadata", {}) or {}
    return {
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
        "trust_remote_code": descriptor.trust_decision is TrustDecision.REVIEWED_BUNDLED_CODE,
        # Chat media the worker must prepare with the checkpoint's processor.
        "media_modalities": sorted(
            modality
            for modality, capability in (
                ("image", Capability.VISION),
                ("video", Capability.VIDEO),
            )
            if descriptor.task is ModelTask.TEXT_GENERATION
            and descriptor.capabilities.support(capability).state is not CapabilityState.UNSUPPORTED
        ),
    }


class ModelWorkerSupervisor:
    """Own a spawned worker; only serializable values cross this boundary."""

    def __init__(
        self,
        *,
        queue_limit: int = 256,
        startup_timeout_seconds: float = 120.0,
    ) -> None:
        self._context = mp.get_context("spawn")
        self._queue_limit = max(1, queue_limit)
        self._startup_timeout_seconds = max(0.1, float(startup_timeout_seconds))
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
        # Mirrors the worker's residents, most recently used last. It is cleared
        # whenever the worker process is replaced, because its models die with it.
        self._resident: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._active_run_ids: set[str] = set()
        self._cancelled_run_ids: set[str] = set()
        self._ledger: dict[str, Any] | None = None
        self._ledger_at: float | None = None
        self._stopping = False

    def _build_worker(self) -> None:
        self._ready = False
        self._ready_event: asyncio.Event | None = None
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
        """The most recently used resident, or None."""

        if not self._resident:
            return None
        return dict(next(reversed(self._resident.values())))

    @property
    def resident(self) -> list[dict[str, Any]]:
        """Every resident model, most recently used last."""

        return [dict(entry) for entry in self._resident.values()]

    @property
    def ledger(self) -> dict[str, Any] | None:
        return dict(self._ledger) if self._ledger is not None else None

    @property
    def ledger_age_seconds(self) -> float | None:
        return None if self._ledger_at is None else time.monotonic() - self._ledger_at

    def touch(self, model_key: str) -> None:
        """Mark a resident as just used, so least-recently-used eviction skips it."""

        entry = self._resident.get(model_key)
        if entry is not None:
            entry["last_used_at"] = datetime.now(UTC).isoformat()
            self._resident.move_to_end(model_key)

    def _forget_residents(self) -> None:
        self._resident.clear()
        self._ledger = None
        self._ledger_at = None

    def _note_ledger(self, payload: Any) -> None:
        if isinstance(payload, Mapping) and isinstance(payload.get("ledger"), Mapping):
            self._ledger = dict(payload["ledger"])
            self._ledger_at = time.monotonic()

    @property
    def alive(self) -> bool:
        try:
            return bool(self._process.is_alive())
        except (AssertionError, ValueError):
            return False

    @property
    def available(self) -> bool:
        return (
            self.alive
            and bool(getattr(self, "_ready", False))
            and not getattr(self, "_poisoned", False)
        )

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
            if self.available:
                return
            if self.alive:
                with contextlib.suppress(AssertionError, OSError, ValueError):
                    self._process.terminate()
                await asyncio.to_thread(self._process.join, 5.0)
                if self.alive:
                    with contextlib.suppress(AssertionError, OSError, ValueError):
                        self._process.kill()
                    await asyncio.to_thread(self._process.join, 5.0)
                await self._retire_dead_worker()
                self._build_worker()
            if self._process.pid is not None:
                await self._retire_dead_worker()
                self._build_worker()
            ready_event = asyncio.Event()
            self._ready_event = ready_event
            self._forget_residents()
            self._process.start()
            output = self._output
            process = self._process
            self._pump_task = asyncio.create_task(self._pump(output), name="model-worker-events")
            self._monitor_task = asyncio.create_task(
                self._monitor(process), name="model-worker-monitor"
            )
            try:
                await asyncio.wait_for(
                    ready_event.wait(),
                    timeout=self._startup_timeout_seconds,
                )
            except TimeoutError as exc:
                error = {
                    "code": "model_worker_startup_timeout",
                    "message": (
                        "model worker did not finish startup within "
                        f"{self._startup_timeout_seconds:g} seconds"
                    ),
                    "hint": "Inspect packaged dependencies and increase workers.startup_timeout_seconds only when startup is expected to be slower.",
                }
                if self.alive:
                    with contextlib.suppress(AssertionError, OSError, ValueError):
                        process.terminate()
                    await asyncio.to_thread(process.join, 5.0)
                if self.alive:
                    with contextlib.suppress(AssertionError, OSError, ValueError):
                        process.kill()
                    await asyncio.to_thread(process.join, 5.0)
                await self._retire_dead_worker()
                self._build_worker()
                raise WorkerFailure(error) from exc
            if not self.available:
                error = {
                    "code": "model_worker_startup_failed",
                    "message": "model worker exited before completing its startup handshake",
                    "hint": "Inspect packaged dependencies and the private backend log.",
                }
                await self._retire_dead_worker()
                self._build_worker()
                raise WorkerFailure(error)

    async def _retire_dead_worker(self) -> None:
        self._ready = False
        ready_event = getattr(self, "_ready_event", None)
        if ready_event is not None:
            ready_event.set()
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

            self._forget_residents()
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
            self._note_ledger(item.get("payload"))
            if kind == "reply":
                future = self._reply_futures.get(str(item.get("request_id")))
                if future is not None and not future.done():
                    future.set_result(item)
            elif kind == "run_event":
                run_queue = self._run_queues.get(str(item.get("run_id")))
                if run_queue is not None:
                    await run_queue.put(item)
            elif kind == "ready":
                self._ready = True
                ready_event = getattr(self, "_ready_event", None)
                if ready_event is not None:
                    ready_event.set()
            # Diagnostics intentionally remain private to the supervisor. The
            # structured error reply is safe for the API; tracebacks may contain paths.

    async def _monitor(self, process: Any) -> None:
        while not self._stopping:
            await asyncio.sleep(0.5)
            if process is not self._process:
                return
            if process.is_alive():
                continue
            self._ready = False
            ready_event = getattr(self, "_ready_event", None)
            if ready_event is not None:
                ready_event.set()
            self._forget_residents()
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
        model_key: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        model = worker_model_payload(descriptor)
        requested = {"model": model, "runtime": dict(runtime)}
        # The worker replaces this key when the request starts, so its entry cannot
        # describe the worker any more, including when loading fails or times out.
        self._resident.pop(model_key, None)
        result = await self._request(
            {"op": "load", "model_key": model_key, **requested}, timeout_seconds
        )
        mismatches = {
            key: {"requested": expected, "loaded": result.get(key)}
            for key, expected in {
                "model_id": model["id"],
                "task": model["task"],
                "device": requested["runtime"].get("device"),
                "dtype": requested["runtime"].get("dtype"),
                "model_key": model_key,
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
        entry = {
            **result,
            "model_key": model_key,
            "display_name": descriptor.display_name,
            "last_used_at": datetime.now(UTC).isoformat(),
        }
        self._resident[model_key] = entry
        return dict(entry)

    async def flush(
        self,
        *,
        model_key: str,
        staging_dir: str,
        source_dir: str,
        derivation: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Have the worker save one quantized resident into an empty staging folder."""

        return await self._request(
            {
                "op": "flush",
                "model_key": model_key,
                "staging_dir": staging_dir,
                "source_dir": source_dir,
                "derivation": dict(derivation),
            },
            timeout_seconds,
        )

    async def unload(
        self, *, model_key: str | None = None, timeout_seconds: float = 30.0
    ) -> dict[str, Any]:
        """Unload one resident, or every resident when no key is given."""

        command: dict[str, Any] = {"op": "unload"}
        if model_key is not None:
            command["model_key"] = model_key
        try:
            result = await self._request(command, timeout_seconds)
        except WorkerFailure as exc:
            if model_key is not None and exc.error.get("code") == "model_not_resident":
                self._resident.pop(model_key, None)
            raise
        if model_key is None:
            self._resident.clear()
        for key in result.get("unloaded_model_keys") or ():
            self._resident.pop(str(key), None)
        return result

    async def generate(
        self,
        *,
        run_id: str,
        messages: list[dict[str, Any]],
        sampling: Mapping[str, Any],
        effective_seed: int,
        instrumentation: str,
        deterministic_reference_mode: bool,
        max_prompt_tokens: int,
        reserved_output_tokens: int,
        timeout_seconds: float,
        forced_prefix_token_ids: Sequence[int] = (),
        reasoning: bool | None = None,
        model_key: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if getattr(self, "_poisoned", False) or not self.alive:
            await self.start()
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        run_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self._reply_futures[request_id] = future
        self._run_queues[run_id] = run_queue
        if not self._active_run_ids:
            # The shared event means "cancel everything"; it is only reset while no
            # other run is in flight, so it can never swallow another run's stop.
            self._cancel_event.clear()
        self._active_run_ids.add(run_id)
        process = self._process
        deadline = loop.time() + timeout_seconds
        # True once the worker is known to be finished with this run, either
        # because a terminal event arrived or because a timeout replaced it.
        settled = False
        command: dict[str, Any] = {
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
            "forced_prefix_token_ids": list(forced_prefix_token_ids),
            "reasoning": reasoning,
        }
        if model_key is not None:
            command["model_key"] = model_key
        try:
            await asyncio.to_thread(self._commands.put, command, True, timeout_seconds)
            if run_id in self._cancelled_run_ids:
                # Commands are FIFO, so the worker has created the session by the
                # time it reads this cancel.
                self._send_cancel(run_id)
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                event = await asyncio.wait_for(run_queue.get(), timeout=remaining)
                settled = event.get("event_type") in {"completed", "cancelled", "error"}
                yield event
                if settled:
                    break
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            reply = await asyncio.wait_for(future, timeout=min(5.0, remaining))
            if not reply.get("ok") and event.get("event_type") != "error":
                raise WorkerFailure(reply.get("error", {"message": "generation failed"}))
        except (OSError, TimeoutError, ValueError, queue.Full) as exc:
            settled = True
            self._cancel_event.set()
            error = {
                "code": "inference_timeout",
                "message": f"generation exceeded the {timeout_seconds:g}-second timeout",
                "hint": "Reduce the output budget or increase workers.inference_timeout_seconds.",
            }
            await self._recycle_timed_out_worker(process, error)
            raise WorkerFailure(error) from exc
        finally:
            if not settled and process is self._process:
                # The consumer abandoned the stream (closed it, raised, or was
                # cancelled) before a terminal event, so the worker would keep
                # generating an orphaned run. Stop that run only, and wait briefly
                # for its reply so the worker is done with it before this returns.
                self._send_cancel(run_id)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(future, timeout=5.0)
            self._reply_futures.pop(request_id, None)
            self._run_queues.pop(run_id, None)
            self._cancelled_run_ids.discard(run_id)
            self._active_run_ids.discard(run_id)

    async def embed(
        self,
        *,
        inputs: list[dict[str, Any]],
        dimensions: int | None,
        normalize: bool,
        batch_size: int,
        timeout_seconds: float,
        model_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            {
                "op": "embed",
                "model_key": model_key,
                "inputs": inputs,
                "dimensions": dimensions,
                "normalize": normalize,
                "batch_size": batch_size,
            },
            timeout_seconds,
        )

    async def score_prompt(
        self, text: str, *, timeout_seconds: float, model_key: str | None = None
    ) -> dict[str, Any]:
        return await self._request(
            {"op": "score_prompt", "model_key": model_key, "text": text},
            timeout_seconds,
        )

    async def analyze_influence(
        self, command: Mapping[str, Any], *, timeout_seconds: float
    ) -> dict[str, Any]:
        """Re-run one persisted prefix in the worker and weigh its positions."""

        return await self._request({**command, "op": "analyze_influence"}, timeout_seconds)

    def cancel(self, run_id: str | None = None) -> None:
        """Cancel one run without touching any other; no run ID cancels everything."""

        if run_id is None:
            self._cancel_event.set()
            return
        self._cancelled_run_ids.add(run_id)
        if run_id in self._active_run_ids:
            self._send_cancel(run_id)

    def _send_cancel(self, run_id: str) -> None:
        try:
            self._commands.put_nowait({"op": "cancel", "run_id": run_id})
        except (AssertionError, OSError, ValueError, queue.Full):
            # A full or closed command queue must still stop the run; the shared
            # event stops every session, which is the safe side of the trade.
            self._cancel_event.set()

    def forget_run(self, run_id: str) -> None:
        """Discard a cancellation queued before generation reached the worker."""

        if run_id not in self._active_run_ids:
            self._cancelled_run_ids.discard(run_id)

    async def memory_status(self, *, timeout_seconds: float) -> dict[str, Any]:
        """Ask the worker for its memory ledger.

        A timeout recycles the worker, so callers never send this while an admitted
        job owns the worker.
        """

        result = await self._request({"op": "memory_status"}, timeout_seconds)
        self._note_ledger(result)
        return dict(result.get("ledger") or {})

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
        self._forget_residents()
