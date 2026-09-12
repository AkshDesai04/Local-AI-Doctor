"""Versioned, replayable run events with bounded subscriber backpressure."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = 1
TERMINAL_TYPES = frozenset({"completed", "cancelled", "error"})


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    run_id: str
    sequence: int = Field(gt=0)
    type: str
    monotonic_ns: int = Field(ge=0)
    server_time: str
    payload: dict[str, Any] = Field(default_factory=dict)


PersistEvent = Callable[[Mapping[str, Any]], Awaitable[None]]
ReplayEvents = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


@dataclass(eq=False, slots=True)
class _Subscriber:
    queue: asyncio.Queue[RunEvent]
    last_delivered: int


class EventBroker:
    """Fan out run events and retain durable sequence-based replay semantics."""

    def __init__(
        self,
        *,
        persist: PersistEvent | None = None,
        replay: ReplayEvents | None = None,
        subscriber_queue_size: int = 256,
    ) -> None:
        self._persist = persist
        self._replay = replay
        self._subscriber_queue_size = subscriber_queue_size
        self._sequences: dict[str, int] = defaultdict(int)
        self._subscribers: dict[str, set[_Subscriber]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._publish_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def restore_sequence(self, run_id: str, sequence: int) -> None:
        async with self._lock:
            self._sequences[run_id] = max(self._sequences[run_id], sequence)

    async def publish(
        self, run_id: str, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> RunEvent:
        async with self._publish_locks[run_id]:
            async with self._lock:
                sequence = self._sequences[run_id] + 1
                self._sequences[run_id] = sequence
                event = RunEvent(
                    run_id=run_id,
                    sequence=sequence,
                    type=event_type,
                    monotonic_ns=time.monotonic_ns(),
                    server_time=datetime.now(UTC).isoformat(),
                    payload=dict(payload or {}),
                )
                subscribers = tuple(self._subscribers.get(run_id, ()))

            persistence_error: Exception | None = None
            if self._persist is not None:
                try:
                    await self._persist(event.model_dump(mode="json"))
                except Exception as exc:
                    # Live subscribers must still receive a terminal state when
                    # optional telemetry persistence is unhealthy. The caller is
                    # notified after fanout so it can downgrade the durable run.
                    persistence_error = exc

            for subscriber in subscribers:
                try:
                    subscriber.queue.put_nowait(event)
                except asyncio.QueueFull:
                    # A slow browser can reconnect from the last event actually
                    # delivered, not merely the last one queued for it.
                    while not subscriber.queue.empty():
                        subscriber.queue.get_nowait()
                    subscriber.queue.put_nowait(
                        RunEvent(
                            run_id=run_id,
                            sequence=sequence,
                            type="resync_required",
                            monotonic_ns=time.monotonic_ns(),
                            server_time=datetime.now(UTC).isoformat(),
                            payload={"resume_after": subscriber.last_delivered},
                        )
                    )
            if persistence_error is not None:
                raise persistence_error
        return event

    async def subscribe(self, run_id: str, after_sequence: int = 0) -> AsyncIterator[RunEvent]:
        subscriber = _Subscriber(
            queue=asyncio.Queue(self._subscriber_queue_size),
            last_delivered=max(0, after_sequence),
        )
        async with self._lock:
            self._subscribers[run_id].add(subscriber)
        try:
            # Register before replay so an event committed during the replay
            # window is also queued. Sequence filtering removes duplicates.
            if self._replay is not None:
                for raw in await self._replay(run_id, max(0, after_sequence)):
                    payload = raw.get("payload")
                    if payload is None:
                        payload = raw.get("payload_json", {})
                    event = RunEvent(
                        version=raw.get("version", PROTOCOL_VERSION),
                        run_id=run_id,
                        sequence=raw["sequence"],
                        type=raw["type"],
                        monotonic_ns=raw["monotonic_ns"],
                        server_time=raw.get("created_at", datetime.now(UTC).isoformat()),
                        payload=payload,
                    )
                    await self.restore_sequence(run_id, event.sequence)
                    after_sequence = event.sequence
                    subscriber.last_delivered = event.sequence
                    yield event
                    if event.type in TERMINAL_TYPES:
                        return
            while True:
                event = await subscriber.queue.get()
                if event.sequence <= after_sequence and event.type != "resync_required":
                    continue
                if event.type != "resync_required":
                    after_sequence = event.sequence
                    subscriber.last_delivered = event.sequence
                yield event
                if event.type in TERMINAL_TYPES:
                    return
        finally:
            async with self._lock:
                self._subscribers[run_id].discard(subscriber)
                if not self._subscribers[run_id]:
                    self._subscribers.pop(run_id, None)
