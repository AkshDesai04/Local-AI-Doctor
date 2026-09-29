from __future__ import annotations

import asyncio
import sqlite3
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.persistence import Database, WorkspaceRepository
from local_ai_doctor.persistence.database import utc_now

API_HEADERS = {"Authorization": "Bearer integration-test-token"}


def _create_chat(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/api/v1/chats", headers=API_HEADERS, json={"title": "sp", **body})
    assert response.status_code == 201, response.text
    chat: dict[str, Any] = response.json()
    return chat


def _patch_chat(client: TestClient, chat_id: str, **body: Any) -> Any:
    return client.patch(f"/api/v1/chats/{chat_id}", headers=API_HEADERS, json=body)


def test_chat_system_prompt_is_created_and_patched_with_absent_null_and_blank_semantics(
    api_client: TestClient,
) -> None:
    kept = _create_chat(api_client, systemPrompt="  Be terse.  ")
    snake = _create_chat(api_client, system_prompt="Answer in French.")
    plain = _create_chat(api_client)
    blank = _create_chat(api_client, systemPrompt=" \t ")
    # The author's text is stored verbatim; only whitespace-only counts as empty.
    assert kept["system_prompt"] == "  Be terse.  "
    assert snake["system_prompt"] == "Answer in French."
    assert plain["system_prompt"] is None
    assert blank["system_prompt"] is None

    chat_id = kept["id"]
    renamed = _patch_chat(api_client, chat_id, title="renamed").json()
    assert renamed["title"] == "renamed"
    assert renamed["system_prompt"] == "  Be terse.  "  # absent leaves it unchanged

    assert _patch_chat(api_client, chat_id, systemPrompt="New rules").json()["system_prompt"] == (
        "New rules"
    )
    for clearing in (None, "", "   \n"):
        _patch_chat(api_client, chat_id, systemPrompt="Something")
        cleared = _patch_chat(api_client, chat_id, systemPrompt=clearing)
        assert cleared.status_code == 200
        assert cleared.json()["system_prompt"] is None
    assert _patch_chat(api_client, chat_id, system_prompt="Snake").json()["system_prompt"] == (
        "Snake"
    )

    fetched = api_client.get(f"/api/v1/chats/{chat_id}", headers=API_HEADERS).json()
    listed = {
        chat["id"]: chat for chat in api_client.get("/api/v1/chats", headers=API_HEADERS).json()
    }
    assert fetched["system_prompt"] == "Snake"
    assert listed[chat_id]["system_prompt"] == "Snake"
    assert listed[plain["id"]]["system_prompt"] is None
    assert _patch_chat(api_client, "missing-chat", systemPrompt="x").status_code == 404
    assert _patch_chat(api_client, chat_id, systemPrompt=7).status_code == 422

    for chat in (kept, snake, plain, blank):
        api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


def test_chat_system_prompt_is_bounded_by_utf8_prompt_bytes(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services
    monkeypatch.setattr(services.settings.limits, "prompt_bytes", 1024)
    chat = _create_chat(api_client)

    fits = "é" * 500  # 500 characters, 1000 UTF-8 bytes
    too_long = "é" * 600  # 600 characters, 1200 UTF-8 bytes
    assert _patch_chat(api_client, chat["id"], systemPrompt=fits).status_code == 200
    rejected = _patch_chat(api_client, chat["id"], systemPrompt=too_long)
    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "limit_exceeded"
    assert rejected.json()["error"]["details"] == {"maximum_bytes": 1024}
    created = api_client.post(
        "/api/v1/chats", headers=API_HEADERS, json={"title": "big", "systemPrompt": too_long}
    )
    assert created.status_code == 413
    stored = api_client.get(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS).json()
    assert stored["system_prompt"] == fits

    api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


def _capture_execution(services: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_execute_generation(**kwargs: Any) -> None:
        captured.clear()
        captured.update(kwargs)

    monkeypatch.setattr(services.runs, "_execute_generation", fake_execute_generation)
    return captured


def test_generation_sends_the_chat_system_prompt_and_snapshots_it_on_the_run(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    captured = _capture_execution(services, monkeypatch)

    chat = _create_chat(api_client, systemPrompt="Be terse.")
    response = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={"chatId": chat["id"], "modelId": model.id, "content": "hello"},
    )
    assert response.status_code == 202, response.text
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hello"},
    ]
    detail = api_client.get(f"/api/v1/runs/{response.json()['runId']}", headers=API_HEADERS).json()
    assert detail["settings"]["system_prompt"] == "Be terse."
    exported = api_client.get(
        f"/api/v1/runs/{response.json()['runId']}/export", headers=API_HEADERS
    ).json()
    assert exported["run"]["settings"]["system_prompt"] == "Be terse."
    # The stored user message never contains the system prompt.
    stored = api_client.get(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS).json()
    assert [message["role"] for message in stored["messages"]] == ["user", "assistant"]

    # A follow-up turn keeps a single system message at the start of the lineage.
    portal.call(
        partial(
            services.repository.update_message,
            response.json()["messageId"],
            content="ok",
            status="complete",
        )
    )
    follow_up = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": model.id,
            "content": "again",
            "parentMessageId": response.json()["messageId"],
        },
    )
    assert follow_up.status_code == 202, follow_up.text
    portal.call(asyncio.sleep, 0)
    assert [message["role"] for message in captured["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    # Clearing the prompt means later runs carry none and record null.
    assert _patch_chat(api_client, chat["id"], systemPrompt=None).status_code == 200
    plain = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={"chatId": chat["id"], "modelId": model.id, "content": "no prompt"},
    )
    assert plain.status_code == 202, plain.text
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [{"role": "user", "content": "no prompt"}]
    assert plain.json()["run"]["settings"]["system_prompt"] is None

    api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


def test_an_existing_leading_system_message_wins_over_the_chat_system_prompt(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    captured = _capture_execution(services, monkeypatch)
    chat = _create_chat(api_client, systemPrompt="Chat level")
    system_message = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="system",
            content="Message level",
        )
    )

    response = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": model.id,
            "content": "hello",
            "parentMessageId": system_message["id"],
        },
    )

    assert response.status_code == 202, response.text
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [
        {"role": "system", "content": "Message level"},
        {"role": "user", "content": "hello"},
    ]
    assert response.json()["run"]["settings"]["system_prompt"] is None
    api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


def test_system_prompt_counts_toward_the_rendered_history_byte_limit(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    _capture_execution(services, monkeypatch)
    monkeypatch.setattr(services.settings.limits, "prompt_bytes", 1024)
    chat = _create_chat(api_client, systemPrompt="s" * 900)

    response = api_client.post(
        "/api/v1/runs",
        headers=API_HEADERS,
        json={"chatId": chat["id"], "modelId": model.id, "content": "u" * 200},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


def test_replay_and_token_branch_use_the_source_runs_snapshot_not_the_current_chat_prompt(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    captured = _capture_execution(services, monkeypatch)
    chat = _create_chat(api_client, systemPrompt="Snapshot prompt")
    user = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="Pick a color",
        )
    )
    assistant = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="red car",
            parent_id=user["id"],
            status="complete",
        )
    )
    source_run = portal.call(
        services.repository.create_run,
        {
            "message_id": assistant["id"],
            "model_id": model.id,
            "status": "complete",
            "effective_seed": 5,
            "requested_seed": 5,
            "settings": {
                "sampling": {"max_output_tokens": 8, "temperature": 0.7},
                "instrumentation": "token",
                "system_prompt": "Snapshot prompt",
            },
            "reproducibility": {"device": {"requested": "cpu"}, "dtype": "float32"},
            "model_fingerprint": model.fingerprint.value,
            "generated_token_count": 2,
        },
    )
    now = utc_now()
    portal.call(
        services.database.executemany,
        """
        INSERT INTO token_events(
            run_id, token_index, token_id, piece, escaped_bytes, display_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (source_run["id"], 0, 11, "red", "red", "red", now),
            (source_run["id"], 1, 12, " car", " car", " car", now),
        ],
    )
    portal.call(
        services.database.execute,
        """
        INSERT INTO token_alternatives(
            run_id, token_index, distribution, rank, token_id, piece, probability,
            survived_filter
        ) VALUES (?, 1, 'sampling', 2, 42, ' bike', 0.2, 1)
        """,
        (source_run["id"],),
    )
    # The chat prompt is edited after the run; replays must not pick the edit up.
    assert _patch_chat(api_client, chat["id"], systemPrompt="Edited later").status_code == 200

    replay = api_client.post(f"/api/v1/runs/{source_run['id']}/replay", headers=API_HEADERS)
    assert replay.status_code == 202, replay.text
    portal.call(asyncio.sleep, 0)
    expected = [
        {"role": "system", "content": "Snapshot prompt"},
        {"role": "user", "content": "Pick a color"},
    ]
    assert captured["messages"] == expected
    assert replay.json()["run"]["settings"]["system_prompt"] == "Snapshot prompt"

    branch = api_client.post(
        f"/api/v1/runs/{source_run['id']}/branch",
        headers=API_HEADERS,
        json={"tokenIndex": 1, "distribution": "sampling", "rank": 2, "tokenId": 42},
    )
    assert branch.status_code == 202, branch.text
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == expected
    assert branch.json()["run"]["settings"]["system_prompt"] == "Snapshot prompt"
    # The branch chat inherits the snapshot so its follow-up turns stay consistent.
    assert branch.json()["chat"]["system_prompt"] == "Snapshot prompt"

    # A run recorded before system prompts existed replays without one.
    legacy_assistant = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="legacy",
            parent_id=user["id"],
            status="complete",
            branch_index=5,
        )
    )
    legacy_run = portal.call(
        services.repository.create_run,
        {
            "message_id": legacy_assistant["id"],
            "model_id": model.id,
            "status": "complete",
            "effective_seed": 6,
            "settings": {"sampling": {"max_output_tokens": 8}},
            "reproducibility": {"device": {"requested": "cpu"}, "dtype": "float32"},
            "model_fingerprint": model.fingerprint.value,
        },
    )
    legacy = api_client.post(f"/api/v1/runs/{legacy_run['id']}/replay", headers=API_HEADERS)
    assert legacy.status_code == 202, legacy.text
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [{"role": "user", "content": "Pick a color"}]
    assert legacy.json()["run"]["settings"]["system_prompt"] is None

    api_client.delete(f"/api/v1/chats/{branch.json()['chatId']}", headers=API_HEADERS)
    api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


def test_workspace_export_and_import_carry_the_system_prompt(api_client: TestClient) -> None:
    chat = _create_chat(api_client, systemPrompt="Portable rules")
    plain = _create_chat(api_client)

    exported = api_client.get(f"/api/v1/chats/{chat['id']}/export", headers=API_HEADERS).json()
    assert exported["schema_version"] == 1
    assert exported["chat"]["system_prompt"] == "Portable rules"
    assert (
        api_client.get(f"/api/v1/chats/{plain['id']}/export", headers=API_HEADERS).json()["chat"][
            "system_prompt"
        ]
        is None
    )

    imported = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=exported)
    assert imported.status_code == 201, imported.text
    assert imported.json()["chat"]["system_prompt"] == "Portable rules"

    # Documents written before the field existed still import, and blanks normalize away.
    legacy = {**exported, "chat": {"title": "legacy"}}
    legacy_import = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=legacy)
    assert legacy_import.status_code == 201, legacy_import.text
    assert legacy_import.json()["chat"]["system_prompt"] is None
    blank = {**exported, "chat": {"title": "blank", "system_prompt": "  "}}
    blank_import = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=blank)
    assert blank_import.json()["chat"]["system_prompt"] is None

    for chat_id in (chat["id"], plain["id"], imported.json()["chat"]["id"]):
        api_client.delete(f"/api/v1/chats/{chat_id}", headers=API_HEADERS)
    for imported_chat in (legacy_import, blank_import):
        api_client.delete(
            f"/api/v1/chats/{imported_chat.json()['chat']['id']}", headers=API_HEADERS
        )


def test_workspace_import_enforces_the_system_prompt_byte_limit(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services
    chat = _create_chat(api_client)
    exported = api_client.get(f"/api/v1/chats/{chat['id']}/export", headers=API_HEADERS).json()
    monkeypatch.setattr(services.settings.limits, "prompt_bytes", 1024)
    exported["chat"]["system_prompt"] = "x" * 1025

    response = api_client.post("/api/v1/chats/import", headers=API_HEADERS, json=exported)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "limit_exceeded"
    api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)


@pytest.mark.asyncio
async def test_database_at_migration_0003_gains_a_nullable_system_prompt_column(
    tmp_path: Path,
) -> None:
    migrations = sorted(
        (Path(__file__).parents[2] / "backend/local_ai_doctor/persistence/migrations").glob("*.sql")
    )
    assert [path.stem for path in migrations][:4] == [
        "0001_initial",
        "0002_token_reasoning_slices",
        "0003_token_attention_attribution",
        "0004_chat_system_prompt",
    ]
    database_path = tmp_path / "state" / "workbench.sqlite3"
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as legacy:
        legacy.execute("CREATE TABLE schema_migrations(version TEXT PRIMARY KEY, applied_at TEXT)")
        for migration in migrations[:3]:
            legacy.executescript(migration.read_text(encoding="utf-8"))
            legacy.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (migration.stem, utc_now()),
            )
        legacy.execute(
            "INSERT INTO chats(id, title, created_at, updated_at) VALUES ('old', 'Old', 'a', 'a')"
        )
        columns = {row[1] for row in legacy.execute("PRAGMA table_info(chats)")}
        assert "system_prompt" not in columns

    database = Database(database_path, tmp_path / "backups")
    await database.initialize()
    repository = WorkspaceRepository(database)
    try:
        old = await repository.get_chat("old")
        assert old is not None
        assert old["system_prompt"] is None
        updated = await repository.update_chat("old", system_prompt="Upgraded")
        assert updated is not None
        assert updated["system_prompt"] == "Upgraded"
        created = await repository.create_chat("New", system_prompt="Fresh")
        assert created["system_prompt"] == "Fresh"
        assert len(list((tmp_path / "backups").glob("*.sqlite3"))) == 1
    finally:
        await database.close()
