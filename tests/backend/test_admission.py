from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_ai_doctor.domain.models import TrustDecision
from local_ai_doctor.errors import LimitExceededError, WorkerBusyError
from local_ai_doctor.services.runs import RunManager
from local_ai_doctor.workers.admission import SingleWorkerAdmission
from local_ai_doctor.workers.supervisor import ModelWorkerSupervisor, WorkerFailure


def test_worker_availability_requires_completed_startup_handshake() -> None:
    supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
    supervisor._process = SimpleNamespace(is_alive=lambda: True)
    supervisor._poisoned = False
    supervisor._ready = False

    assert supervisor.available is False
    supervisor._ready = True
    assert supervisor.available is True


def test_queue_limit_counts_waiters_beyond_single_active_slot() -> None:
    admission = SingleWorkerAdmission(queue_limit=1)
    active = admission.reserve("active", "generation")
    queued = admission.reserve("queued", "embedding")

    with pytest.raises(LimitExceededError) as caught:
        admission.reserve("overflow", "prompt_score")

    assert caught.value.http_status == 429
    assert caught.value.to_dict()["code"] == "limit_exceeded"
    assert caught.value.details == {
        "queue_limit": 1,
        "queued_inference_count": 1,
        "active": False,
        "operation_kind": "prompt_score",
    }
    active.release()
    queued.release()


def test_inference_leases_serialize_load_and_operation_as_one_unit() -> None:
    async def scenario() -> None:
        admission = SingleWorkerAdmission(queue_limit=1)
        first = admission.reserve("first", "generation")
        second = admission.reserve("second", "embedding")
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        order: list[str] = []

        async def run_first() -> None:
            async with first:
                order.extend(("load:first", "infer:first"))
                first_entered.set()
                await release_first.wait()

        async def run_second() -> None:
            async with second:
                order.extend(("load:second", "infer:second"))

        first_task = asyncio.create_task(run_first())
        await first_entered.wait()
        second_task = asyncio.create_task(run_second())
        await asyncio.sleep(0)
        assert order == ["load:first", "infer:first"]

        release_first.set()
        await asyncio.gather(first_task, second_task)
        assert order == ["load:first", "infer:first", "load:second", "infer:second"]

    asyncio.run(scenario())


def test_lifecycle_mutation_is_rejected_while_inference_is_admitted() -> None:
    async def scenario() -> None:
        admission = SingleWorkerAdmission(queue_limit=1)
        reservation = admission.reserve("run", "generation")

        with pytest.raises(WorkerBusyError) as caught:
            async with admission.lifecycle("unload"):
                pytest.fail("busy lifecycle operation must not run")

        assert caught.value.http_status == 409
        assert caught.value.details["admitted_inference_count"] == 1
        reservation.release()
        async with admission.lifecycle("unload"):
            assert admission.snapshot().active_lifecycle_operation == "unload"

    asyncio.run(scenario())


def test_cancelling_waiter_frees_capacity_without_releasing_active_lease() -> None:
    async def scenario() -> None:
        admission = SingleWorkerAdmission(queue_limit=1)
        active = admission.reserve("active", "generation")
        await active.acquire()
        waiting = admission.reserve("waiting", "generation")

        assert waiting.cancel_if_waiting() is True
        assert waiting.cancel_if_waiting() is False
        replacement = admission.reserve("replacement", "embedding")
        snapshot = admission.snapshot()
        assert snapshot.active_inference_id == "active"
        assert snapshot.queued_inference_count == 1

        active.release()
        replacement.release()

    asyncio.run(scenario())


def test_run_manager_close_cancels_waiters_before_active_run_releases() -> None:
    async def scenario() -> None:
        admission = SingleWorkerAdmission(queue_limit=2)
        active = admission.reserve("active", "generation")
        waiting_one = admission.reserve("waiting-one", "generation")
        waiting_two = admission.reserve("waiting-two", "generation")
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        load_order: list[str] = []

        async def operation(reservation: Any, run_id: str) -> None:
            try:
                await reservation.acquire()
                load_order.append(run_id)
                if run_id == "active":
                    active_started.set()
                    await release_active.wait()
            finally:
                reservation.release()

        active_task = asyncio.create_task(operation(active, "active"))
        await active_started.wait()
        waiting_one_task = asyncio.create_task(operation(waiting_one, "waiting-one"))
        waiting_two_task = asyncio.create_task(operation(waiting_two, "waiting-two"))
        await asyncio.sleep(0)

        class Repository:
            def __init__(self) -> None:
                self.runs = {
                    "active": {"id": "active", "message_id": "active-message"},
                    "waiting-one": {
                        "id": "waiting-one",
                        "message_id": "waiting-one-message",
                    },
                    "waiting-two": {
                        "id": "waiting-two",
                        "message_id": "waiting-two-message",
                    },
                }

            async def get_run(self, run_id: str) -> dict[str, Any] | None:
                return self.runs.get(run_id)

            async def update_message(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            async def update_run(self, run_id: str, **changes: Any) -> None:
                self.runs[run_id].update(changes)

        class Worker:
            def cancel(self, run_id: str) -> None:
                if run_id == "active":
                    release_active.set()

            def forget_run(self, _run_id: str) -> None:
                return None

        class Events:
            async def publish(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        manager = RunManager.__new__(RunManager)
        manager.repository = Repository()
        manager.worker = Worker()
        manager.events = Events()
        manager._tasks = {
            "active": active_task,
            "waiting-one": waiting_one_task,
            "waiting-two": waiting_two_task,
        }
        manager._reservations = {
            "active": active,
            "waiting-one": waiting_one,
            "waiting-two": waiting_two,
        }

        await asyncio.wait_for(manager.close(), timeout=1.0)

        assert load_order == ["active"]
        assert manager.repository.runs["waiting-one"]["status"] == "cancelled"
        assert manager.repository.runs["waiting-two"]["status"] == "cancelled"
        assert waiting_one_task.cancelled()
        assert waiting_two_task.cancelled()

    asyncio.run(scenario())


def test_supervisor_run_scoped_cancel_does_not_interrupt_a_different_run() -> None:
    supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
    supervisor._cancel_event = SimpleNamespace(set=lambda: setattr(supervisor, "event_set", True))
    supervisor._cancelled_run_ids = set()
    supervisor._active_run_id = "active"
    supervisor.event_set = False

    supervisor.cancel("waiting")
    assert supervisor.event_set is False
    assert supervisor._cancelled_run_ids == {"waiting"}

    supervisor.cancel("active")
    assert supervisor.event_set is True
    assert supervisor._cancelled_run_ids == {"waiting", "active"}


def test_supervisor_load_cache_is_bound_to_model_identity_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
        supervisor._loaded = None
        supervisor._loaded_request = None
        supervisor._process = SimpleNamespace(is_alive=lambda: True)
        requests: list[dict[str, Any]] = []

        async def fake_request(
            command: Mapping[str, Any], timeout_seconds: float
        ) -> dict[str, Any]:
            assert timeout_seconds == 30.0
            request = dict(command)
            requests.append(request)
            model = request["model"]
            runtime = request["runtime"]
            return {
                "model_id": model["id"],
                "task": model["task"],
                "device": runtime["device"],
                "dtype": runtime["dtype"],
            }

        monkeypatch.setattr(supervisor, "_request", fake_request)
        first_descriptor = SimpleNamespace(
            id="stable-model-id",
            display_name="Model",
            path=Path("model"),
            task=SimpleNamespace(value="text_generation"),
            model_type="test",
            effective_context_limit=1024,
            fingerprint=SimpleNamespace(value="a" * 64),
            trust_decision=TrustDecision.BUILTIN_ONLY,
        )
        runtime = {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "cpu_threads": 4,
            "attention_backend": "sdpa",
        }

        await supervisor.load_model(first_descriptor, runtime, timeout_seconds=30.0)
        await supervisor.load_model(first_descriptor, dict(runtime), timeout_seconds=30.0)
        assert len(requests) == 1

        await supervisor.load_model(
            first_descriptor,
            {**runtime, "dtype": "float32"},
            timeout_seconds=30.0,
        )
        assert len(requests) == 2

        changed_checkpoint = SimpleNamespace(
            **{
                **vars(first_descriptor),
                "fingerprint": SimpleNamespace(value="b" * 64),
            }
        )
        await supervisor.load_model(
            changed_checkpoint,
            {**runtime, "dtype": "float32"},
            timeout_seconds=30.0,
        )
        assert len(requests) == 3
        assert requests[-1]["model"]["fingerprint"] == "b" * 64

    asyncio.run(scenario())


def test_supervisor_request_timeout_recycles_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = SimpleNamespace(is_alive=lambda: True)
        supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
        supervisor._process = process
        supervisor._commands = SimpleNamespace(put=lambda *_args: None)
        supervisor._reply_futures = {}
        observed: list[tuple[object, dict[str, Any]]] = []

        async def recycle(failed_process: object, error: Mapping[str, Any]) -> None:
            observed.append((failed_process, dict(error)))

        monkeypatch.setattr(supervisor, "_recycle_timed_out_worker", recycle)

        with pytest.raises(WorkerFailure) as caught:
            await supervisor._request({"op": "embed"}, timeout_seconds=0.001)

        assert caught.value.error["code"] == "model_worker_timeout"
        assert observed == [(process, caught.value.error)]
        assert supervisor._reply_futures == {}

    asyncio.run(scenario())


def test_supervisor_timeout_replacement_is_atomic_and_ignores_stale_recyclers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        class FakeProcess:
            pid = 123

            def __init__(self) -> None:
                self.running = True
                self.terminated = False

            def is_alive(self) -> bool:
                return self.running

            def terminate(self) -> None:
                self.terminated = True
                self.running = False

            def kill(self) -> None:
                self.running = False

            def join(self, _timeout: float) -> None:
                return None

        failed = FakeProcess()
        replacement = SimpleNamespace(pid=None, is_alive=lambda: False)
        supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
        supervisor._restart_lock = asyncio.Lock()
        supervisor._stopping = False
        supervisor._process = failed
        supervisor._loaded = {"model_id": "old"}
        supervisor._loaded_request = {"model": "old"}
        supervisor._cancel_event = SimpleNamespace(set=lambda: None)
        pending: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        run_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        run_queue.put_nowait({"event_type": "stale"})
        supervisor._reply_futures = {"pending": pending}
        supervisor._run_queues = {"run": run_queue}
        retired = False

        async def retire() -> None:
            nonlocal retired
            retired = True

        def build() -> None:
            supervisor._process = replacement

        monkeypatch.setattr(supervisor, "_retire_dead_worker", retire)
        monkeypatch.setattr(supervisor, "_build_worker", build)
        error = {"code": "inference_timeout", "message": "timed out"}

        await supervisor._recycle_timed_out_worker(failed, error)

        assert failed.terminated is True
        assert retired is True
        assert supervisor._process is replacement
        assert supervisor._loaded is None
        assert supervisor._loaded_request is None
        assert pending.result()["error"] == error
        assert run_queue.get_nowait()["event_type"] == "error"

        # A second timeout handler from the old process must not kill the new one.
        await supervisor._recycle_timed_out_worker(failed, error)
        assert supervisor._process is replacement

    asyncio.run(scenario())


def test_generation_timeout_recycles_worker_and_cleans_registered_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = SimpleNamespace(is_alive=lambda: True)
        cancel_event = SimpleNamespace(clear=lambda: None, set=lambda: None)
        supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
        supervisor._process = process
        supervisor._commands = SimpleNamespace(put=lambda *_args: None)
        supervisor._cancel_event = cancel_event
        supervisor._cancelled_run_ids = set()
        supervisor._run_lock = asyncio.Lock()
        supervisor._active_run_id = None
        supervisor._reply_futures = {}
        supervisor._run_queues = {}
        observed: list[tuple[object, dict[str, Any]]] = []

        async def recycle(failed_process: object, error: Mapping[str, Any]) -> None:
            observed.append((failed_process, dict(error)))

        monkeypatch.setattr(supervisor, "_recycle_timed_out_worker", recycle)

        with pytest.raises(WorkerFailure) as caught:
            async for _event in supervisor.generate(
                run_id="timed-out-run",
                messages=[{"role": "user", "content": "hello"}],
                sampling={},
                effective_seed=0,
                instrumentation="off",
                deterministic_reference_mode=True,
                max_prompt_tokens=32,
                reserved_output_tokens=8,
                timeout_seconds=0.001,
            ):
                pass

        assert caught.value.error["code"] == "inference_timeout"
        assert observed == [(process, caught.value.error)]
        assert supervisor._reply_futures == {}
        assert supervisor._run_queues == {}
        assert supervisor._active_run_id is None

    asyncio.run(scenario())


def _streaming_supervisor() -> tuple[ModelWorkerSupervisor, list[str | None]]:
    """A supervisor whose fake worker acknowledges a cancel request at once."""

    supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
    cancel_requests: list[str | None] = []

    class CancelEvent:
        @staticmethod
        def clear() -> None:
            return None

        @staticmethod
        def set() -> None:
            cancel_requests.append(supervisor._active_run_id)
            for future in supervisor._reply_futures.values():
                future.set_result({"ok": True})

    supervisor._process = SimpleNamespace(is_alive=lambda: True)
    supervisor._commands = SimpleNamespace(put=lambda *_args: None)
    supervisor._cancel_event = CancelEvent()
    supervisor._cancelled_run_ids = set()
    supervisor._run_lock = asyncio.Lock()
    supervisor._active_run_id = None
    supervisor._reply_futures = {}
    supervisor._run_queues = {}
    return supervisor, cancel_requests


async def _open_stream_with_one_event(
    supervisor: ModelWorkerSupervisor, event_type: str
) -> AsyncGenerator[dict[str, Any], None]:
    loop = asyncio.get_running_loop()

    def worker_receives_command(*_args: object) -> None:
        # Runs in a helper thread, like the real queue put; the fake worker answers
        # with one event on the loop thread.
        loop.call_soon_threadsafe(
            supervisor._run_queues["run"].put_nowait, {"event_type": event_type, "payload": {}}
        )

    supervisor._commands = SimpleNamespace(put=worker_receives_command)
    stream = supervisor.generate(
        run_id="run",
        messages=[{"role": "user", "content": "hello"}],
        sampling={},
        effective_seed=0,
        instrumentation="off",
        deterministic_reference_mode=False,
        max_prompt_tokens=32,
        reserved_output_tokens=8,
        timeout_seconds=30.0,
    )
    assert (await anext(stream))["event_type"] == event_type
    return stream


@pytest.mark.parametrize("abandon", ["close", "consumer_error"])
def test_abandoned_generation_stream_cancels_the_worker(abandon: str) -> None:
    async def scenario() -> None:
        supervisor, cancel_requests = _streaming_supervisor()
        stream = await _open_stream_with_one_event(supervisor, "token")
        assert cancel_requests == []

        if abandon == "close":
            await stream.aclose()
        else:
            with pytest.raises(RuntimeError, match="consumer failed"):
                await stream.athrow(RuntimeError("consumer failed"))

        assert cancel_requests == ["run"]
        assert supervisor._reply_futures == {}
        assert supervisor._run_queues == {}
        assert supervisor._active_run_id is None
        assert not supervisor._run_lock.locked()

    asyncio.run(scenario())


def test_generation_stream_closed_after_terminal_event_does_not_cancel() -> None:
    async def scenario() -> None:
        supervisor, cancel_requests = _streaming_supervisor()
        stream = await _open_stream_with_one_event(supervisor, "completed")

        await stream.aclose()

        assert cancel_requests == []
        assert supervisor._active_run_id is None

    asyncio.run(scenario())


def test_supervisor_fails_closed_when_a_timed_out_process_cannot_be_killed() -> None:
    async def scenario() -> None:
        class ImmortalProcess:
            pid = 123

            @staticmethod
            def is_alive() -> bool:
                return True

            @staticmethod
            def terminate() -> None:
                return None

            @staticmethod
            def kill() -> None:
                return None

            @staticmethod
            def join(_timeout: float) -> None:
                return None

        process = ImmortalProcess()
        supervisor = ModelWorkerSupervisor.__new__(ModelWorkerSupervisor)
        supervisor._restart_lock = asyncio.Lock()
        supervisor._stopping = False
        supervisor._poisoned = False
        supervisor._process = process
        supervisor._loaded = {"model_id": "unsafe"}
        supervisor._loaded_request = {"model": "unsafe"}
        supervisor._cancel_event = SimpleNamespace(set=lambda: None)
        supervisor._reply_futures = {}
        supervisor._run_queues = {}

        with pytest.raises(WorkerFailure) as caught:
            await supervisor._recycle_timed_out_worker(
                process, {"code": "inference_timeout", "message": "timed out"}
            )

        assert caught.value.error["code"] == "model_worker_recycle_failed"
        assert supervisor.available is False
        assert supervisor._loaded is None
        with pytest.raises(WorkerFailure) as retry:
            await supervisor.start()
        assert retry.value.error["code"] == "model_worker_recycle_failed"

    asyncio.run(scenario())
