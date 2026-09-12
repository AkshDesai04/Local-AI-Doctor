"""Bounded FIFO admission and exclusive leases for the single model worker."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ..errors import LimitExceededError, WorkerBusyError


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    active_inference_id: str | None
    active_lifecycle_operation: str | None
    queued_inference_count: int
    queue_limit: int


class InferenceReservation:
    """One admitted inference operation, either waiting or exclusively active."""

    def __init__(self, admission: SingleWorkerAdmission, inference_id: str, kind: str) -> None:
        self._admission = admission
        self.inference_id = inference_id
        self.kind = kind
        self._entered = False

    @property
    def active(self) -> bool:
        return self._admission.is_active(self.inference_id)

    async def acquire(self) -> None:
        if self._entered:
            raise RuntimeError("inference reservation is already active")
        await self._admission._activate(self)
        self._entered = True

    async def __aenter__(self) -> InferenceReservation:
        await self.acquire()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()

    def cancel_if_waiting(self) -> bool:
        return self._admission._cancel_if_waiting(self.inference_id)

    def release(self) -> None:
        self._admission._release(self.inference_id, entered=self._entered)
        self._entered = False


class SingleWorkerAdmission:
    """Own admission capacity and the worker's exclusive execution lease.

    Capacity is one active inference plus ``queue_limit`` waiting inferences.
    Reservations are created without awaiting, so an overflowing HTTP request
    receives a structured 429 before chat/run state is mutated.
    """

    def __init__(self, queue_limit: int) -> None:
        if queue_limit < 0:
            raise ValueError("queue_limit must be non-negative")
        self._queue_limit = queue_limit
        self._execution_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._states: dict[str, str] = {}
        self._active_inference_id: str | None = None
        self._active_lifecycle_operation: str | None = None

    def _waiting_count_locked(self) -> int:
        queued = sum(state == "queued" for state in self._states.values())
        if self._active_inference_id is None and self._active_lifecycle_operation is None:
            # One admitted operation owns the immediately runnable slot even if
            # its task has not reached acquire() yet.
            queued = max(0, queued - 1)
        return queued

    def snapshot(self) -> AdmissionSnapshot:
        with self._state_lock:
            return AdmissionSnapshot(
                active_inference_id=self._active_inference_id,
                active_lifecycle_operation=self._active_lifecycle_operation,
                queued_inference_count=self._waiting_count_locked(),
                queue_limit=self._queue_limit,
            )

    def is_active(self, inference_id: str) -> bool:
        with self._state_lock:
            return self._active_inference_id == inference_id

    def reserve(self, inference_id: str, kind: str) -> InferenceReservation:
        with self._state_lock:
            if inference_id in self._states:
                raise WorkerBusyError(
                    "the inference operation already has a worker reservation",
                    details={"inference_id": inference_id},
                )
            # While a lifecycle operation owns the worker, every inference is
            # waiting; otherwise capacity includes the one runnable operation.
            capacity = self._queue_limit + (
                0 if self._active_lifecycle_operation is not None else 1
            )
            if len(self._states) >= capacity:
                raise LimitExceededError(
                    "the local inference queue is full",
                    hint="Wait for an active run to finish or increase runtime.queue_limit.",
                    details={
                        "queue_limit": self._queue_limit,
                        "queued_inference_count": self._waiting_count_locked(),
                        "active": self._active_inference_id is not None
                        or self._active_lifecycle_operation is not None,
                        "operation_kind": kind,
                    },
                )
            self._states[inference_id] = "queued"
        return InferenceReservation(self, inference_id, kind)

    async def _activate(self, reservation: InferenceReservation) -> None:
        try:
            await self._execution_lock.acquire()
        except BaseException:
            reservation.release()
            raise
        with self._state_lock:
            state = self._states.get(reservation.inference_id)
            if state != "queued":
                self._execution_lock.release()
                raise asyncio.CancelledError
            self._states[reservation.inference_id] = "active"
            self._active_inference_id = reservation.inference_id

    def _cancel_if_waiting(self, inference_id: str) -> bool:
        with self._state_lock:
            if self._states.get(inference_id) != "queued":
                return False
            del self._states[inference_id]
            return True

    def _release(self, inference_id: str, *, entered: bool) -> None:
        release_execution = False
        with self._state_lock:
            state = self._states.pop(inference_id, None)
            if state == "active" and self._active_inference_id == inference_id:
                self._active_inference_id = None
                release_execution = True
        if entered and release_execution and self._execution_lock.locked():
            self._execution_lock.release()

    @asynccontextmanager
    async def lifecycle(self, operation: str) -> AsyncIterator[None]:
        """Reject lifecycle mutation when inference is admitted or active."""

        with self._state_lock:
            if self._states or self._active_lifecycle_operation is not None:
                raise WorkerBusyError(
                    f"cannot {operation} a model while the single worker is busy",
                    hint="Wait for queued and active inference operations to finish.",
                    details={
                        "operation": operation,
                        "active_inference_id": self._active_inference_id,
                        "admitted_inference_count": len(self._states),
                        "active_lifecycle_operation": self._active_lifecycle_operation,
                    },
                )
            self._active_lifecycle_operation = operation
        acquired = False
        try:
            await self._execution_lock.acquire()
            acquired = True
            yield
        finally:
            if acquired and self._execution_lock.locked():
                self._execution_lock.release()
            with self._state_lock:
                if self._active_lifecycle_operation == operation:
                    self._active_lifecycle_operation = None
