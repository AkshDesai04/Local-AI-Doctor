from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.services.events import EventBroker, RunEvent


class _TerminalGateBroker(EventBroker):
    """Hold a terminal publisher until its subscriber has inspected persistence."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.terminal_observed = asyncio.Event()

    async def publish(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        event = await super().publish(run_id, event_type, payload)
        if event_type in {"completed", "cancelled", "error"}:
            await asyncio.wait_for(self.terminal_observed.wait(), timeout=2)
        return event


class _Reservation:
    def __init__(self) -> None:
        self.acquired = False
        self.released = False

    async def acquire(self) -> None:
        self.acquired = True

    def release(self) -> None:
        self.released = True


def test_terminal_subscriber_observes_final_run_message_and_token_state(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None

    async def fake_load_reserved(
        _reservation: object,
        _model_id: str,
        **_kwargs: object,
    ) -> dict[str, str]:
        return {"status": "loaded"}

    async def fake_generate(**_kwargs: object) -> AsyncIterator[dict[str, Any]]:
        yield {
            "event_type": "stage",
            "payload": {
                "stage": "prefill",
                "reasoning_primed": True,
                "rendered_prompt": "User: test\nAssistant: <think>\n",
                "prompt_tokens": 5,
            },
        }
        yield {
            "event_type": "token",
            "payload": {
                "token_index": 0,
                "token_id": 42,
                "piece": "done",
                "escaped_bytes": "done",
                "display_text": "done",
                "alternatives": {},
            },
        }
        yield {
            "event_type": "completed",
            "payload": {
                "finish_reason": "length",
                "total_generation_ms": 1.0,
            },
        }

    monkeypatch.setattr(services.registry, "load_reserved", fake_load_reserved)
    monkeypatch.setattr(services.worker, "generate", fake_generate)

    async def scenario() -> tuple[RunEvent, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        repository = services.repository

        async def persist_event(event: Mapping[str, Any]) -> None:
            if event["type"] in {"completed", "cancelled", "error"}:
                await services.telemetry.flush()
            await repository.append_raw_event(event)

        broker = _TerminalGateBroker(
            persist=persist_event,
            replay=repository.get_raw_events,
        )
        run_id = f"terminal-order-{uuid.uuid4()}"
        chat = await repository.create_chat("Terminal ordering")
        message = await repository.create_message(
            chat_id=chat["id"],
            role="assistant",
            content="",
            status="pending",
        )
        await repository.create_run(
            {
                "id": run_id,
                "message_id": message["id"],
                "kind": "generation",
                "status": "queued",
                "effective_seed": 7,
            }
        )
        reservation = _Reservation()

        async def observe_terminal() -> tuple[
            RunEvent,
            dict[str, Any],
            dict[str, Any],
            list[dict[str, Any]],
        ]:
            async for event in broker.subscribe(run_id):
                if event.type == "completed":
                    try:
                        persisted_run = await repository.get_run(run_id)
                        persisted_message = await repository.get_message(message["id"])
                        assert persisted_run is not None
                        assert persisted_message is not None
                        tokens = await repository.list_run_tokens(run_id)
                        return event, persisted_run, persisted_message, tokens
                    finally:
                        broker.terminal_observed.set()
            raise AssertionError("generation stream ended without a completed event")

        observer = asyncio.create_task(observe_terminal())
        await asyncio.sleep(0)
        original_events = services.runs.events
        services.runs.events = broker
        try:
            await services.runs._execute_generation(
                run_id=run_id,
                message_id=message["id"],
                model_id="test-model",
                messages=[],
                sampling={},
                effective_seed=7,
                instrumentation="token",
                deterministic_reference_mode=False,
                device=None,
                dtype=None,
                reservation=reservation,
            )
            observed = await asyncio.wait_for(observer, timeout=2)
        finally:
            services.runs.events = original_events
            if not observer.done():
                observer.cancel()
        assert reservation.acquired is True
        assert reservation.released is True
        return observed

    event, run, message, tokens = portal.call(scenario)
    assert event.type == "completed"
    assert run["status"] == "complete"
    assert run["finish_reason"] == "length"
    assert run["generated_token_count"] == 1
    assert run["completed_at"] is not None
    assert message["status"] == "complete"
    assert message["content"] == "done"
    assert message["metadata"] == {"reasoning_primed": True}
    assert [(token["token_index"], token["token_id"]) for token in tokens] == [(0, 42)]


def test_generation_persists_and_publishes_error_when_telemetry_has_failed(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    private_failure = RuntimeError(
        r"telemetry failed for C:\Users\alice\private-model\weights.safetensors"
    )

    async def fake_load_reserved(
        _reservation: object,
        _model_id: str,
        **_kwargs: object,
    ) -> dict[str, str]:
        return {"status": "loaded"}

    async def fake_generate(**_kwargs: object) -> AsyncIterator[dict[str, Any]]:
        yield {
            "event_type": "token",
            "payload": {
                "token_index": 0,
                "token_id": 42,
                "piece": "partial",
                "escaped_bytes": "partial",
                "display_text": "partial",
                "alternatives": {},
            },
        }

    async def failed_submit(_sql: str, _rows: object) -> None:
        raise private_failure

    async def failed_flush() -> None:
        raise private_failure

    monkeypatch.setattr(services.registry, "load_reserved", fake_load_reserved)
    monkeypatch.setattr(services.worker, "generate", fake_generate)
    monkeypatch.setattr(services.telemetry, "submit", failed_submit)
    monkeypatch.setattr(services.telemetry, "flush", failed_flush)

    async def scenario() -> tuple[RunEvent, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        repository = services.repository
        run_id = f"failed-telemetry-{uuid.uuid4()}"
        chat = await repository.create_chat("Failed telemetry")
        message = await repository.create_message(
            chat_id=chat["id"],
            role="assistant",
            content="",
            status="pending",
        )
        await repository.create_run(
            {
                "id": run_id,
                "message_id": message["id"],
                "kind": "generation",
                "status": "queued",
                "effective_seed": 7,
            }
        )
        reservation = _Reservation()

        async def observe_terminal() -> RunEvent:
            async for event in services.events.subscribe(run_id):
                if event.type in {"completed", "cancelled", "error"}:
                    return event
            raise AssertionError("generation stream ended without a terminal event")

        observer = asyncio.create_task(observe_terminal())
        await asyncio.sleep(0)
        await services.runs._execute_generation(
            run_id=run_id,
            message_id=message["id"],
            model_id="test-model",
            messages=[],
            sampling={},
            effective_seed=7,
            instrumentation="token",
            deterministic_reference_mode=False,
            device=None,
            dtype=None,
            reservation=reservation,
        )
        terminal = await asyncio.wait_for(observer, timeout=2)
        persisted_run = await repository.get_run(run_id)
        persisted_message = await repository.get_message(message["id"])
        raw_events = await repository.get_raw_events(run_id, 0)
        assert persisted_run is not None
        assert persisted_message is not None
        assert reservation.released is True
        await repository.delete_chat(chat["id"])
        return terminal, persisted_run, persisted_message, raw_events

    with caplog.at_level(logging.ERROR, logger="local_ai_doctor.services.runs"):
        terminal, run, message, raw_events = portal.call(scenario)

    assert terminal.type == "error"
    assert terminal.payload["code"] == "internal_error"
    assert run["status"] == "failed"
    assert run["error_message"] == "generation failed at an application boundary"
    assert message["status"] == "failed"
    assert [event["type"] for event in raw_events if event["type"] in {"error", "completed"}] == [
        "error"
    ]
    assert "alice" not in caplog.text.lower()
    assert "weights.safetensors" not in caplog.text
