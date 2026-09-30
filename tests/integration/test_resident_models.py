"""Several resident models: generation placement, resident status, unload, and 507 errors."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.workers import WorkerFailure

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


RESIDENT = {
    "model_key": "k1",
    "model_id": "m1",
    "display_name": "M1",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "quantization": "none",
    "strict_vram": True,
    "placement": "gpu",
    "gpu_bytes": 2048,
    "cpu_bytes": 64,
    "kv_reserve_bytes": 128,
    "load_seconds": 1.5,
    "last_used_at": "2026-01-01T00:00:00+00:00",
    "memory": {"process_rss_bytes": 1},
}


def test_health_and_resident_status_list_every_resident(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    second = {**RESIDENT, "model_key": "k2", "model_id": "m2", "placement": "offload"}
    monkeypatch.setattr(services.worker, "_resident", OrderedDict(k1=RESIDENT, k2=second))

    health = api_client.get("/api/v1/health", headers=API_HEADERS).json()
    assert (health["status"], health["database"], health["worker"]) == ("ok", "ready", "ready")
    assert health["loaded_model"]["model_key"] == "k2"
    assert health["loaded_models"] == [
        {
            "model_key": key,
            "model_id": model_id,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "quantization": "none",
            "placement": placement,
            "strict_vram": True,
        }
        for key, model_id, placement in (("k1", "m1", "gpu"), ("k2", "m2", "offload"))
    ]

    status = api_client.get("/api/v1/models/resident", headers=API_HEADERS)
    assert status.status_code == 200
    body = status.json()
    assert set(body) == {"models", "memory", "max_loaded_models"}
    assert body["max_loaded_models"] == 4
    assert body["models"][0] == {
        "model_key": "k1",
        "model_id": "m1",
        "display_name": "M1",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "quantization": "none",
        "strict_vram": True,
        "placement": "gpu",
        "gpu_bytes": 2048,
        "cpu_bytes": 64,
        "kv_reserve_bytes": 128,
        "load_seconds": 1.5,
        "last_used_at": "2026-01-01T00:00:00+00:00",
        "in_use": False,
    }
    memory = body["memory"]
    assert set(memory) == {
        "device",
        "total_bytes",
        "free_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "cap_bytes",
        "process_rss_bytes",
        "system_available_bytes",
        "safety_margin_bytes",
        "ledger_age_seconds",
        "stale",
    }
    # The idle worker answered memory_status, so the ledger is fresh.
    assert memory["stale"] is False
    assert memory["process_rss_bytes"] > 0
    assert memory["safety_margin_bytes"] == 512 * 1024**2


def test_unloading_by_model_or_key_requires_a_resident(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    model_id = _model_id(api_client)

    missing = api_client.post(f"/api/v1/models/{model_id}/unload", headers=API_HEADERS)
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "model_not_resident"
    unknown = api_client.post("/api/v1/models/resident/nope/unload", headers=API_HEADERS)
    assert unknown.status_code == 409
    assert unknown.json()["error"]["code"] == "model_not_resident"

    unloaded: list[str | None] = []

    async def fake_unload(*, model_key: str | None = None, timeout_seconds: float) -> Any:
        unloaded.append(model_key)
        services.worker._resident.pop(model_key, None)
        return {"unloaded_model_keys": [model_key], "freed_bytes": 10, "leaked_bytes": 0}

    monkeypatch.setattr(
        services.worker,
        "_resident",
        OrderedDict(
            a={**RESIDENT, "model_key": "a", "model_id": model_id},
            b={**RESIDENT, "model_key": "b", "model_id": model_id},
            c={**RESIDENT, "model_key": "c"},
        ),
    )
    monkeypatch.setattr(services.worker, "unload", fake_unload)

    one = api_client.post("/api/v1/models/resident/c/unload", headers=API_HEADERS)
    assert one.status_code == 200
    assert one.json()["unloaded_model_keys"] == ["c"]
    both = api_client.post(f"/api/v1/models/{model_id}/unload", headers=API_HEADERS)
    assert both.status_code == 200
    assert both.json()["lifecycle"] == "unloaded"
    assert both.json()["unload"]["unloaded_model_keys"] == ["a", "b"]
    assert both.json()["unload"]["freed_bytes"] == 20
    assert unloaded == ["c", "a", "b"]


def test_a_load_that_cannot_fit_returns_a_structured_507(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    requested: list[dict[str, Any]] = []

    async def insufficient(_descriptor: object, runtime: dict[str, Any], **_kwargs: Any) -> Any:
        requested.append(runtime)
        raise WorkerFailure(
            {
                "code": "insufficient_memory",
                "message": "the model does not fit the available system memory",
                "memory_kind": "ram",
                "required_bytes": 900,
                "available_bytes": 100,
                "estimate": {"weights": 800, "kv_reserve": 100, "margin": 50},
            }
        )

    monkeypatch.setattr(services.worker, "load_model", insufficient)

    response = api_client.post(
        f"/api/v1/models/{_model_id(api_client)}/load",
        headers=API_HEADERS,
        json={"strictVram": True},
    )

    assert response.status_code == 507
    error = response.json()["error"]
    assert error["code"] == "out_of_memory"
    assert error["retryable"] is True
    assert error["hint"]
    assert error["details"] == {
        "memory_kind": "ram",
        "required_bytes": 900,
        "available_bytes": 100,
        "estimate": {"weights": 800, "kv_reserve": 100, "margin": 50},
        "strict_vram": True,
        "placement": "cpu",
        "resident_model_keys": [],
        "pinned_model_keys": [],
        "evicted_model_keys": [],
    }
    assert requested[0]["placement"] == "cpu"
    assert requested[0]["safety_margin_bytes"] == 512 * 1024**2
    assert requested[0]["max_concurrent_runs"] == 2
