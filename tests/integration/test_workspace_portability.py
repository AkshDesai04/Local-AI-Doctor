from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from local_ai_doctor.persistence.database import utc_now

API_HEADERS = {"Authorization": "Bearer integration-test-token"}


def test_chat_workspace_round_trip_remaps_ids_and_preserves_token_telemetry(
    api_client: TestClient,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={"title": "portable"}).json()
    user = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="Explain 2 + 2",
        )
    )
    assistant = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="4",
            parent_id=user["id"],
            status="complete",
        )
    )
    run = portal.call(
        services.repository.create_run,
        {
            "message_id": assistant["id"],
            "status": "complete",
            "effective_seed": 42,
            "settings": {"sampling": {"temperature": 0.0}},
            "reproducibility": {"backend": "test"},
            "generated_token_count": 1,
        },
    )
    now = utc_now()
    portal.call(
        services.database.execute,
        """
        INSERT INTO token_events(
            run_id, token_index, token_id, piece, escaped_bytes, display_text,
            raw_logprob, raw_probability, segment, selected_experts_json, created_at
        ) VALUES (?, 0, 7, '4', '\\x34', '4', -0.25, 0.75, 'answer', '[1,2]', ?)
        """,
        (run["id"], now),
    )
    portal.call(
        services.database.execute,
        """
        INSERT INTO token_alternatives(
            run_id, token_index, distribution, rank, token_id, piece,
            probability, survived_filter
        ) VALUES (?, 0, 'raw', 1, 7, '4', 0.75, 1)
        """,
        (run["id"],),
    )

    exported_response = api_client.get(f"/api/v1/chats/{chat['id']}/export", headers=API_HEADERS)
    assert exported_response.status_code == 200
    assert exported_response.headers["content-disposition"] == (
        'attachment; filename="chat-workspace.json"'
    )
    exported = exported_response.json()
    assert exported["schema"] == "local-ai-doctor/chat-workspace"
    assert exported["schema_version"] == 1
    assert exported["runs"][0]["tokens"][0]["selected_experts"] == [1, 2]
    assert exported["runs"][0]["tokens"][0]["alternatives"][0]["probability"] == 0.75
    assert "storage_name" not in exported_response.text
    assert "canonical_path" not in exported_response.text

    imported_response = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=exported)
    assert imported_response.status_code == 201
    imported = imported_response.json()
    imported_chat = imported["chat"]
    assert imported["imported"] == {"messages": 2, "runs": 1, "tokens": 1}
    assert imported_chat["id"] != chat["id"]
    assert {message["id"] for message in imported_chat["messages"]}.isdisjoint(
        {user["id"], assistant["id"]}
    )
    imported_runs = portal.call(services.repository.list_chat_runs, imported_chat["id"])
    assert len(imported_runs) == 1
    assert imported_runs[0]["id"] != run["id"]
    assert imported_runs[0]["message_id"] in {
        message["id"] for message in imported_chat["messages"]
    }
    imported_tokens = portal.call(services.repository.list_run_tokens, imported_runs[0]["id"])
    assert imported_tokens[0]["piece"] == "4"
    assert imported_tokens[0]["alternatives"][0]["probability"] == 0.75

    assert api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS).status_code == 204
    assert (
        api_client.delete(f"/api/v1/chats/{imported_chat['id']}", headers=API_HEADERS).status_code
        == 204
    )


def test_chat_workspace_import_rejects_wrong_versions_paths_and_cycles(
    api_client: TestClient,
) -> None:
    base: dict[str, Any] = {
        "schema": "local-ai-doctor/chat-workspace",
        "schema_version": 1,
        "chat": {"title": "invalid", "pinned": False, "archived": False},
        "messages": [],
        "runs": [],
    }
    wrong_version = deepcopy(base)
    wrong_version["schema_version"] = 2
    response = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=wrong_version)
    assert response.status_code == 422

    arbitrary_path = deepcopy(base)
    arbitrary_path["source_path"] = "C:/private/workspace.json"
    response = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=arbitrary_path)
    assert response.status_code == 422

    cycle = deepcopy(base)
    cycle["messages"] = [
        {
            "id": "a",
            "parent_id": "b",
            "role": "user",
            "content": "a",
            "status": "complete",
            "branch_index": 0,
            "metadata": {},
        },
        {
            "id": "b",
            "parent_id": "a",
            "role": "assistant",
            "content": "b",
            "status": "complete",
            "branch_index": 0,
            "metadata": {},
        },
    ]
    response = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=cycle)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_replay_uses_recorded_configuration_and_creates_an_assistant_branch(
    api_client: TestClient, monkeypatch: Any
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    chat = portal.call(services.repository.create_chat, "replay")
    user = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="A seeded question",
        )
    )
    source_message = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="Original answer",
            parent_id=user["id"],
            status="complete",
            branch_index=0,
        )
    )
    source_run = portal.call(
        services.repository.create_run,
        {
            "message_id": source_message["id"],
            "model_id": model.id,
            "status": "complete",
            "effective_seed": 123456,
            "requested_seed": 123456,
            "settings": {
                "sampling": {
                    "max_output_tokens": 17,
                    "temperature": 0.6,
                    "top_k": 5,
                    "top_p": 0.9,
                },
                "instrumentation": "full",
                "deterministic_reference_mode": True,
            },
            "effective_config": {"source": "recorded"},
            "reproducibility": {
                "device": {"requested": "cpu"},
                "dtype": "float32",
            },
            "model_fingerprint": model.fingerprint.value,
        },
    )
    captured: dict[str, Any] = {}

    async def fake_execute_generation(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(services.runs, "_execute_generation", fake_execute_generation)
    response = api_client.post(f"/api/v1/runs/{source_run['id']}/replay", headers=API_HEADERS)
    assert response.status_code == 202
    replayed = response.json()
    assert replayed["parentRunId"] == source_run["id"]
    assert replayed["run"]["parent_run_id"] == source_run["id"]
    assert replayed["run"]["effective_seed"] == "123456"
    assert replayed["run"]["settings"]["sampling"]["max_output_tokens"] == 17
    assert replayed["run"]["effective_config"] == services.settings.inference_snapshot()
    assert replayed["assistant_message"]["parent_id"] == user["id"]
    assert replayed["assistant_message"]["branch_index"] == 1
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [{"role": "user", "content": "A seeded question"}]
    assert captured["effective_seed"] == 123456
    assert captured["sampling"]["top_k"] == 5
    assert captured["deterministic_reference_mode"] is True

    portal.call(
        partial(
            services.repository.update_run,
            replayed["run"]["id"],
            status="complete",
            completed_at=utc_now(),
        )
    )
    portal.call(services.repository.delete_chat, chat["id"])


def test_generation_uses_only_the_selected_message_lineage(
    api_client: TestClient, monkeypatch: Any
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    chat = portal.call(services.repository.create_chat, "lineage")
    first_user = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="First question",
        )
    )
    first_assistant = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="First answer",
            parent_id=first_user["id"],
            status="complete",
        )
    )
    portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="Unrelated root branch",
        )
    )
    captured: dict[str, Any] = {}

    async def fake_execute_generation(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(services.runs, "_execute_generation", fake_execute_generation)
    response = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chat_id": chat["id"],
            "model_id": model.id,
            "prompt": "Follow-up question",
            "parent_message_id": first_assistant["id"],
            "sampling": {"max_output_tokens": 1},
        },
    )
    assert response.status_code == 202
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up question"},
    ]

    portal.call(
        partial(
            services.repository.update_run,
            response.json()["run"]["id"],
            status="complete",
            completed_at=utc_now(),
        )
    )
    portal.call(services.repository.delete_chat, chat["id"])


def test_generation_rejects_parent_lineage_from_a_different_chat(
    api_client: TestClient,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    source_chat = portal.call(services.repository.create_chat, "source")
    foreign_parent = portal.call(
        partial(
            services.repository.create_message,
            chat_id=source_chat["id"],
            role="assistant",
            content="Foreign branch",
        )
    )
    target_chat = portal.call(services.repository.create_chat, "target")

    direct = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chat_id": target_chat["id"],
            "model_id": model.id,
            "prompt": "Do not attach this",
            "parent_message_id": foreign_parent["id"],
        },
    )
    assert direct.status_code == 422
    assert direct.json()["error"]["code"] == "invalid_request"

    cross_chat_bridge = api_client.post(
        f"/api/v1/chats/{target_chat['id']}/messages",
        headers=API_HEADERS,
        json={
            "role": "assistant",
            "content": "Do not create a cross-chat bridge",
            "parent_id": foreign_parent["id"],
        },
    )
    assert cross_chat_bridge.status_code == 422
    assert cross_chat_bridge.json()["error"]["code"] == "invalid_request"

    target_after = portal.call(services.repository.get_chat, target_chat["id"])
    assert target_after is not None
    assert target_after["messages"] == []
    assert portal.call(services.repository.list_chat_runs, target_chat["id"]) == []

    portal.call(services.repository.delete_chat, target_chat["id"])
    portal.call(services.repository.delete_chat, source_chat["id"])


def test_generation_byte_limit_covers_messages_created_through_generic_api(
    api_client: TestClient, monkeypatch: Any
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    monkeypatch.setattr(services.runs.settings.limits, "prompt_bytes", 1024)

    chat = api_client.post(
        "/api/v1/chats", headers=API_HEADERS, json={"title": "history limit"}
    ).json()
    historical = api_client.post(
        f"/api/v1/chats/{chat['id']}/messages",
        headers=API_HEADERS,
        json={"role": "system", "content": "x" * 1000},
    )
    assert historical.status_code == 201

    response = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chat_id": chat["id"],
            "model_id": model.id,
            "prompt": "small new prompt",
            "parent_message_id": historical.json()["id"],
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"] == {
        "maximum_bytes": 1024,
        "rendered_history_bytes": 1042,
        "message_count": 2,
    }

    persisted = portal.call(services.repository.get_chat, chat["id"])
    assert persisted is not None
    assert [message["id"] for message in persisted["messages"]] == [historical.json()["id"]]
    assert portal.call(services.repository.list_chat_runs, chat["id"]) == []
    portal.call(services.repository.delete_chat, chat["id"])


def test_retention_requires_confirmation_and_preserves_non_run_data(
    api_client: TestClient, tmp_path: Path
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    chat = portal.call(services.repository.create_chat, "retention")
    old_message = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="old but retained",
        )
    )
    old_run = portal.call(
        services.repository.create_run,
        {
            "message_id": old_message["id"],
            "status": "complete",
            "effective_seed": 0,
        },
    )
    portal.call(
        partial(
            services.repository.update_run,
            old_run["id"],
            completed_at="2020-01-01T00:00:00+00:00",
        )
    )
    current_run = portal.call(
        services.repository.create_run,
        {"status": "complete", "effective_seed": 1},
    )
    portal.call(
        partial(
            services.repository.update_run,
            current_run["id"],
            completed_at="2030-01-01T00:00:00+00:00",
        )
    )
    now = utc_now()
    portal.call(
        services.database.execute,
        "INSERT INTO token_events(run_id, token_index, token_id, piece, escaped_bytes, display_text, created_at) VALUES (?, 0, 1, 'x', '\\x78', 'x', ?)",
        (old_run["id"], now),
    )
    portal.call(
        services.repository.append_raw_event,
        {
            "run_id": old_run["id"],
            "sequence": 1,
            "type": "completed",
            "monotonic_ns": 1,
            "payload": {},
        },
    )
    attachment = portal.call(
        services.repository.create_attachment,
        {
            "id": "retention-attachment",
            "sha256": "b" * 64,
            "storage_name": f"{'b' * 64}.txt",
            "original_name": "retained.txt",
            "media_type": "text/plain",
            "size_bytes": 8,
        },
    )
    retained_file = tmp_path / "retained.txt"
    retained_file.write_text("retained", encoding="utf-8")
    models_before = portal.call(services.repository.list_models)

    rejected = api_client.delete(
        "/api/v1/storage/telemetry",
        headers=API_HEADERS,
        params={"before": "2025-01-01T00:00:00Z"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "invalid_request"
    deleted = api_client.delete(
        "/api/v1/storage/telemetry",
        headers=API_HEADERS,
        params={"before": "2025-01-01T00:00:00Z", "confirm": "true"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"]["runs"] == 1
    assert deleted.json()["deleted"]["token_events"] == 1
    assert deleted.json()["deleted"]["raw_events"] == 1
    assert portal.call(services.repository.get_run, old_run["id"]) is None
    assert portal.call(services.repository.get_run, current_run["id"]) is not None
    assert portal.call(services.repository.get_message, old_message["id"]) is not None
    assert portal.call(services.repository.get_attachment, attachment["id"]) is not None
    assert portal.call(services.repository.list_models) == models_before
    assert retained_file.read_text(encoding="utf-8") == "retained"

    portal.call(services.repository.delete_chat, chat["id"])
