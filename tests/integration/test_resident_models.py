"""Several resident models: generation placement, resident status, unload, and 507 errors."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

API_TOKEN = "integration-test-token"
API_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def _model_id(client: TestClient) -> str:
    return str(client.get("/api/v1/models", headers=API_HEADERS).json()["models"][0]["id"])


def _wait_for_run(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        run = client.get(f"/api/v1/runs/{run_id}", headers=API_HEADERS).json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            return dict(run)
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_generation_passes_strict_vram_through_and_records_the_real_placement(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    loads: list[dict[str, Any]] = []
    streams: list[dict[str, Any]] = []

    async def fake_load_reserved(_reservation: object, model_id: str, **kwargs: Any) -> Any:
        loads.append({"model_id": model_id, **kwargs})
        return {
            "model_key": "resident-key",
            "placement": "offload",
            "strict_vram": kwargs["strict_vram"],
            "quantization": "none",
            "evicted_model_keys": ["older-key"],
        }

    async def fake_generate(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        streams.append(kwargs)
        yield {
            "event_type": "completed",
            "payload": {
                "finish_reason": "length",
                "scheduling": {"concurrent_sessions_max": 1, "interleaved": False},
            },
        }

    monkeypatch.setattr(services.registry, "load_reserved", fake_load_reserved)
    monkeypatch.setattr(services.worker, "generate", fake_generate)
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={}).json()

    response = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": _model_id(api_client),
            "content": "hello",
            "settings": {"strictVram": False, "quantization": "none", "temperature": 0},
        },
    )

    assert response.status_code == 202, response.text
    run = _wait_for_run(api_client, response.json()["runId"])
    assert run["status"] == "complete"
    assert loads[0]["strict_vram"] is False
    assert loads[0]["pinned"] == frozenset()
    assert streams[0]["model_key"] == "resident-key"
    reproducibility = run["reproducibility"]
    assert reproducibility["strict_vram"] is False
    assert reproducibility["quantization"] == "none"
    assert reproducibility["placement"] == "offload"
    assert reproducibility["model_key"] == "resident-key"
    assert reproducibility["scheduling"] == {"concurrent_sessions_max": 1, "interleaved": False}
    assert run["settings"]["strict_vram"] is False
    events = api_client.get(f"/api/v1/runs/{run['id']}/events", headers=API_HEADERS).json()[
        "events"
    ]
    loaded = next(event for event in events if event["type"] == "model_loaded")
    assert loaded["payload"]["evicted_model_keys"] == ["older-key"]

    # Replay reuses the recorded Strict VRAM choice rather than the configured default.
    replay = api_client.post(f"/api/v1/runs/{run['id']}/replay", headers=API_HEADERS)
    assert replay.status_code == 202, replay.text
    _wait_for_run(api_client, replay.json()["runId"])
    assert loads[1]["strict_vram"] is False


def test_generation_defaults_to_the_configured_strict_vram(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    loads: list[dict[str, Any]] = []

    async def fake_load_reserved(_reservation: object, _model_id: str, **kwargs: Any) -> Any:
        loads.append(kwargs)
        return {"status": "loaded"}

    async def fake_generate(**_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"event_type": "completed", "payload": {"finish_reason": "length"}}

    monkeypatch.setattr(services.registry, "load_reserved", fake_load_reserved)
    monkeypatch.setattr(services.worker, "generate", fake_generate)
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={}).json()

    response = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={"chat_id": chat["id"], "model_id": _model_id(api_client), "prompt": "hi"},
    )

    run = _wait_for_run(api_client, response.json()["runId"])
    assert loads[0]["strict_vram"] is True
    assert run["reproducibility"]["strict_vram"] is True
    assert "placement" not in run["reproducibility"]
    unsupported = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={
            "chat_id": chat["id"],
            "model_id": _model_id(api_client),
            "prompt": "hi",
            "quantization": "bitsandbytes-4bit",
        },
    )
    assert unsupported.status_code == 409
    assert unsupported.json()["error"]["code"] == "capability_unavailable"
