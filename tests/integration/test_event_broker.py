from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_ai_doctor.services.events import EventBroker


@pytest.mark.asyncio
async def test_events_have_monotonic_sequences_and_replay_from_an_offset() -> None:
    persisted: list[dict[str, Any]] = []

    async def persist(event: dict[str, Any]) -> None:
        persisted.append(dict(event))

    async def replay(_run_id: str, after: int) -> list[dict[str, Any]]:
        return [event for event in persisted if event["sequence"] > after]

    publisher = EventBroker(persist=persist)
    await publisher.publish("run-1", "created", {"stage": "queued"})
    await publisher.publish("run-1", "token", {"token_index": 0})
    await publisher.publish("run-1", "completed", {"finish_reason": "length"})

    assert [event["sequence"] for event in persisted] == [1, 2, 3]
    assert {event["version"] for event in persisted} == {1}
    assert all(event["monotonic_ns"] >= 0 for event in persisted)

    reconnected = EventBroker(persist=persist, replay=replay)
    replayed = [event async for event in reconnected.subscribe("run-1", after_sequence=1)]
    assert [(event.sequence, event.type) for event in replayed] == [
        (2, "token"),
        (3, "completed"),
    ]

    next_event = await reconnected.publish("run-1", "warning", {"message": "after replay"})
    assert next_event.sequence == 4


@pytest.mark.asyncio
async def test_slow_subscriber_gets_resync_marker_when_its_queue_overflows() -> None:
    broker = EventBroker(subscriber_queue_size=1)
    stream = broker.subscribe("run-backpressure")

    waiting = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    await broker.publish("run-backpressure", "created")
    first = await asyncio.wait_for(waiting, timeout=1)
    assert (first.sequence, first.type) == (1, "created")

    await broker.publish("run-backpressure", "token", {"token_index": 0})
    await broker.publish("run-backpressure", "metrics", {"generated": 1})
    marker = await asyncio.wait_for(anext(stream), timeout=1)
    assert marker.sequence == 3
    assert marker.type == "resync_required"
    # Sequence 2 was removed from the queue, so reconnecting after 2 would
    # permanently skip it. The marker must point to the last delivered event.
    assert marker.payload == {"resume_after": 1}

    await broker.publish("run-backpressure", "completed")
    terminal = await asyncio.wait_for(anext(stream), timeout=1)
    assert (terminal.sequence, terminal.type) == (4, "completed")
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_event_published_during_replay_is_not_lost_before_live_registration() -> None:
    persisted: list[dict[str, Any]] = []
    replay_snapshotted = asyncio.Event()
    release_replay = asyncio.Event()

    async def persist(event: dict[str, Any]) -> None:
        persisted.append(dict(event))

    async def replay(_run_id: str, after: int) -> list[dict[str, Any]]:
        snapshot = [event for event in persisted if event["sequence"] > after]
        replay_snapshotted.set()
        await release_replay.wait()
        return snapshot

    broker = EventBroker(persist=persist, replay=replay)
    await broker.publish("run-replay-race", "created")
    stream = broker.subscribe("run-replay-race")
    first_wait = asyncio.create_task(anext(stream))
    await asyncio.wait_for(replay_snapshotted.wait(), timeout=1)

    # This lands after the replay query took its snapshot but before subscribe
    # can register for live delivery.
    await broker.publish("run-replay-race", "token", {"token_index": 0})
    release_replay.set()
    first = await asyncio.wait_for(first_wait, timeout=1)
    assert (first.sequence, first.type) == (1, "created")

    second_wait = asyncio.create_task(anext(stream))
    try:
        second = await asyncio.wait_for(second_wait, timeout=1)
    finally:
        if not second_wait.done():
            second_wait.cancel()
    assert (second.sequence, second.type) == (2, "token")
    await stream.aclose()


@pytest.mark.asyncio
async def test_concurrent_publishers_deliver_in_sequence_order() -> None:
    first_persist_started = asyncio.Event()
    release_first_persist = asyncio.Event()

    async def persist(event: dict[str, Any]) -> None:
        if event["sequence"] == 1:
            first_persist_started.set()
            await release_first_persist.wait()

    broker = EventBroker(persist=persist)
    stream = broker.subscribe("run-concurrent")
    first_wait = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    first_publish = asyncio.create_task(broker.publish("run-concurrent", "stage"))
    await asyncio.wait_for(first_persist_started.wait(), timeout=1)
    second_publish = asyncio.create_task(broker.publish("run-concurrent", "token"))
    await asyncio.sleep(0)
    release_first_persist.set()
    await asyncio.gather(first_publish, second_publish)

    first = await asyncio.wait_for(first_wait, timeout=1)
    second = await asyncio.wait_for(anext(stream), timeout=1)
    assert [(first.sequence, first.type), (second.sequence, second.type)] == [
        (1, "stage"),
        (2, "token"),
    ]
    await stream.aclose()
