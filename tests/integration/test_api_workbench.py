from __future__ import annotations

import base64
import hashlib
import json
from functools import partial
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from local_ai_doctor.config import AppSettings, ProfileName, SettingsLoader

API_TOKEN = "integration-test-token"
API_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
API_AUTH_PROTOCOL = "lad.auth." + base64.urlsafe_b64encode(API_TOKEN.encode("utf-8")).decode(
    "ascii"
).rstrip("=")


def test_health_configuration_redaction_and_model_scan(
    api_client: TestClient, api_settings: AppSettings
) -> None:
    unauthorized = api_client.get("/api/v1/health")
    assert unauthorized.status_code == 401

    health = api_client.get("/api/v1/health", headers=API_HEADERS)
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "database": "ready",
        "worker": "ready",
        "loaded_model": None,
        "protocol_version": 1,
    }
    assert health.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in health.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in health.headers["content-security-policy"]
    assert "font-src 'self' data:" in health.headers["content-security-policy"]
    assert (
        "connect-src 'self' ws://127.0.0.1 wss://127.0.0.1"
        in health.headers["content-security-policy"]
    )
    assert health.headers["x-frame-options"] == "DENY"
    assert health.headers["cache-control"] == "no-store"

    configuration = api_client.get("/api/v1/configuration", headers=API_HEADERS)
    assert configuration.status_code == 200
    effective = configuration.json()["effective"]
    assert effective["server"]["authentication_token"] == "<redacted-secret>"
    assert API_TOKEN not in configuration.text
    assert effective["paths"] == {
        "model_roots": ["<redacted-path:model_roots:0>"],
        "database": "<redacted-path:database>",
        "uploads": "<redacted-path:uploads>",
        "cache": "<redacted-path:cache>",
        "exports": "<redacted-path:exports>",
        "backups": "<redacted-path:backups>",
    }
    assert str(api_settings.paths.database) not in configuration.json().values()

    models = api_client.get("/api/v1/models", headers=API_HEADERS)
    assert models.status_code == 200
    report = models.json()
    assert report["roots"] == [{"root": "<model-root:0>", "discovered_count": 1, "diagnostics": []}]
    assert len(report["models"]) == 1
    descriptor = report["models"][0]
    assert descriptor["display_name"] == "Tiny-Generation-Model"
    assert descriptor["task"] == "text_generation"
    assert descriptor["path"] == "<model-root>/Tiny-Generation-Model"
    assert descriptor["loadable"] is True

    refreshed = api_client.post("/api/v1/models/refresh", headers=API_HEADERS)
    assert refreshed.status_code == 200
    assert refreshed.json()["models"][0]["id"] == descriptor["id"]


def test_model_root_settings_are_persisted_and_rescanned(
    api_client: TestClient,
    api_settings: AppSettings,
    api_user_config_path: Path,
    tmp_path: Path,
) -> None:
    current = api_client.get("/api/v1/configuration/model-roots", headers=API_HEADERS)
    assert current.status_code == 200
    assert current.json() == {
        "model_roots": [str(api_settings.paths.model_roots[0])],
        "writable": True,
        "source": "user-local configuration",
        "reason": None,
        "containerized": False,
    }

    original_root = api_settings.paths.model_roots[0]
    alternate_root = tmp_path / "alternate-models"
    alternate_root.mkdir()
    try:
        updated = api_client.put(
            "/api/v1/configuration/model-roots",
            headers=API_HEADERS,
            json={"model_roots": [str(original_root), str(alternate_root)]},
        )
        assert updated.status_code == 200
        assert [root["discovered_count"] for root in updated.json()["models"]["roots"]] == [1, 0]
        reloaded = SettingsLoader().load(
            user_path=api_user_config_path,
            profile=ProfileName.TEST,
            environ={},
            base_dir=api_user_config_path.parent,
        )
        assert reloaded.paths.model_roots == (original_root.resolve(), alternate_root.resolve())
    finally:
        restored = api_client.put(
            "/api/v1/configuration/model-roots",
            headers=API_HEADERS,
            json={"model_roots": [str(original_root)]},
        )
        assert restored.status_code == 200

    assert "test:" in api_user_config_path.read_text(encoding="utf-8")
    assert str(original_root) in api_user_config_path.read_text(encoding="utf-8")

    relative = api_client.put(
        "/api/v1/configuration/model-roots",
        headers=API_HEADERS,
        json={"model_roots": ["relative/models"]},
    )
    assert relative.status_code == 422
    assert relative.json()["error"]["code"] == "configuration_invalid"


def test_model_root_settings_require_an_unloaded_worker(
    api_client: TestClient,
    api_settings: AppSettings,
    api_user_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = api_client.app.state.services
    before = api_user_config_path.read_bytes()
    monkeypatch.setattr(services.worker, "_loaded", {"model_id": "still-in-vram"})

    response = api_client.put(
        "/api/v1/configuration/model-roots",
        headers=API_HEADERS,
        json={"model_roots": [str(api_settings.paths.model_roots[0])]},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "capability_unavailable"
    assert "Unload" in response.json()["error"]["hint"]
    assert api_user_config_path.read_bytes() == before


def test_live_api_rejects_oversized_body_before_route_parsing(
    api_client: TestClient, api_settings: AppSettings
) -> None:
    maximum = api_settings.limits.prompt_bytes + 1024 * 1024
    response = api_client.post(
        "/api/v1/chats",
        headers={**API_HEADERS, "Content-Type": "application/json"},
        content=b"x" * (maximum + 1),
    )

    assert response.status_code == 413
    assert response.json()["error"] == {
        "code": "limit_exceeded",
        "message": "request body exceeds the configured size limit",
        "retryable": False,
        "details": {"maximum_bytes": maximum},
    }


def test_message_attachments_are_associated_and_served_from_confined_storage(
    api_client: TestClient,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    content = b"\x89PNG\r\n\x1a\nvalidated-test-payload"
    digest = hashlib.sha256(content).hexdigest()
    storage_name = f"{digest}.png"
    services.settings.paths.uploads.mkdir(parents=True, exist_ok=True)
    (services.settings.paths.uploads / storage_name).write_bytes(content)
    attachment = portal.call(
        services.repository.create_attachment,
        {
            "id": "persisted-preview",
            "sha256": digest,
            "storage_name": storage_name,
            "original_name": "preview.png",
            "media_type": "image/png",
            "size_bytes": len(content),
            "preprocessing": {"path": "native_model_processor", "modality": "image"},
        },
    )
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={"title": "media"}).json()

    created = api_client.post(
        f"/api/v1/chats/{chat['id']}/messages",
        headers=API_HEADERS,
        json={
            "role": "user",
            "content": "Attached image",
            "attachment_ids": [attachment["id"]],
        },
    )
    assert created.status_code == 201

    messages = api_client.get(f"/api/v1/chats/{chat['id']}/messages", headers=API_HEADERS).json()
    assert messages[0]["attachments"][0]["original_name"] == "preview.png"
    assert "storage_name" not in messages[0]["attachments"][0]

    served = api_client.get(f"/api/v1/attachments/{attachment['id']}/content", headers=API_HEADERS)
    assert served.status_code == 200
    assert served.content == content
    assert served.headers["content-type"] == "image/png"

    missing = api_client.get("/api/v1/attachments/not-found/content", headers=API_HEADERS)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "attachment_not_found"

    assert api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS).status_code == 204


def test_health_degrades_after_worker_exit_and_next_operation_restarts_it(
    api_client: TestClient,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    original_pid = services.worker._process.pid
    assert original_pid is not None

    services.worker._process.terminate()
    services.worker._process.join(5.0)
    assert services.worker.alive is False

    degraded = api_client.get("/api/v1/health", headers=API_HEADERS)
    assert degraded.status_code == 503
    assert degraded.json() == {
        "status": "degraded",
        "database": "ready",
        "worker": "unavailable",
        "loaded_model": None,
        "protocol_version": 1,
    }

    restarted = portal.call(partial(services.worker.unload, timeout_seconds=10.0))
    assert restarted["unloaded_model_id"] is None
    assert services.worker.alive is True
    assert services.worker._process.pid != original_pid
    assert api_client.get("/api/v1/health", headers=API_HEADERS).status_code == 200


def test_http_origin_and_websocket_authentication_are_enforced(api_client: TestClient) -> None:
    # The static shell must load before authentication so the user can enter a token.
    assert api_client.get("/").status_code != 401

    # A browser's CORS preflight never carries the bearer credential. The
    # configured origin must reach CORSMiddleware before authentication.
    preflight = api_client.options(
        "/api/v1/chats",
        headers={
            "Origin": "http://vite.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://vite.test"

    malformed_preflight = api_client.options(
        "/api/v1/chats",
        headers={
            "Host": "attacker.invalid@127.0.0.1",
            "Origin": "http://vite.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert malformed_preflight.status_code == 400
    assert malformed_preflight.json()["error"]["code"] == "host_rejected"

    rejected_preflight = api_client.options(
        "/api/v1/chats",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert rejected_preflight.status_code == 400
    assert "access-control-allow-origin" not in rejected_preflight.headers

    wrong_token = api_client.get(
        "/api/v1/health",
        headers={
            "Authorization": "Bearer definitely-wrong",
            "Origin": "http://vite.test",
        },
    )
    assert wrong_token.status_code == 401
    assert wrong_token.headers["www-authenticate"] == "Bearer"
    assert wrong_token.headers["access-control-allow-origin"] == "http://vite.test"

    rejected_origin = api_client.post(
        "/api/v1/chats",
        headers={**API_HEADERS, "Origin": "https://untrusted.example"},
        json={"title": "blocked"},
    )
    assert rejected_origin.status_code == 403
    assert rejected_origin.json()["error"]["code"] == "origin_rejected"

    same_origin = api_client.post(
        "/api/v1/chats",
        headers={**API_HEADERS, "Origin": "http://127.0.0.1"},
        json={"title": "same-origin production UI"},
    )
    assert same_origin.status_code == 201
    assert (
        api_client.delete(
            f"/api/v1/chats/{same_origin.json()['id']}",
            headers={**API_HEADERS, "Origin": "http://127.0.0.1"},
        ).status_code
        == 204
    )

    with (
        pytest.raises(WebSocketDisconnect) as missing_auth,
        api_client.websocket_connect(
            "/ws/v1/runs/does-not-matter",
            subprotocols=["lad.events.v1"],
            headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1"},
        ),
    ):
        pass
    assert missing_auth.value.code == 4401

    with (
        pytest.raises(WebSocketDisconnect) as query_credential,
        api_client.websocket_connect(
            f"/ws/v1/runs/does-not-matter?access_token={API_TOKEN}",
            subprotocols=["lad.events.v1"],
            headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1"},
        ),
    ):
        pass
    assert query_credential.value.code == 4401

    with (
        pytest.raises(WebSocketDisconnect) as wrong_origin,
        api_client.websocket_connect(
            "/ws/v1/runs/does-not-matter",
            subprotocols=["lad.events.v1", API_AUTH_PROTOCOL],
            headers={"Host": "127.0.0.1", "Origin": "https://untrusted.example"},
        ),
    ):
        pass
    assert wrong_origin.value.code == 4403

    with api_client.websocket_connect(
        "/ws/v1/runs/does-not-matter",
        subprotocols=["lad.events.v1", API_AUTH_PROTOCOL],
        headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1"},
    ):
        pass

    # Reviewed cross-origin browser clients use the subprotocol credential;
    # non-browser clients may use the ordinary Authorization header.
    with api_client.websocket_connect(
        "/ws/v1/runs/does-not-matter",
        subprotocols=["lad.events.v1", API_AUTH_PROTOCOL],
        headers={"Host": "127.0.0.1", "Origin": "http://vite.test"},
    ):
        pass
    with api_client.websocket_connect(
        "/ws/v1/runs/does-not-matter",
        subprotocols=["lad.events.v1"],
        headers={"Host": "127.0.0.1", **API_HEADERS},
    ):
        pass

    with (
        pytest.raises(WebSocketDisconnect) as rebound_websocket,
        api_client.websocket_connect(
            "/ws/v1/runs/does-not-matter",
            subprotocols=["lad.events.v1", API_AUTH_PROTOCOL],
            headers={"Host": "attacker.invalid@127.0.0.1", "Origin": "http://vite.test"},
        ),
    ):
        pass
    assert rebound_websocket.value.code == 4403

    rebound = api_client.get(
        "/api/v1/health",
        headers={**API_HEADERS, "Host": "attacker.invalid", "Origin": "http://attacker.invalid"},
    )
    assert rebound.status_code == 400
    assert rebound.json()["error"]["code"] == "host_rejected"

    for malformed_host in ("attacker.invalid@127.0.0.1", "127.0.0.1:8000:bad"):
        malformed = api_client.get("/", headers={"Host": malformed_host})
        assert malformed.status_code == 400
        assert malformed.json()["error"]["code"] == "host_rejected"


def test_chat_crud_treats_sql_and_xss_payloads_as_opaque_data(api_client: TestClient) -> None:
    sql_title = "Robert'); DROP TABLE chats;--"
    xss_content = '<script>globalThis.pwned=true</script><img src=x onerror="alert(1)">'
    created = api_client.post("/api/v1/chats", headers=API_HEADERS, json={"title": sql_title})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    assert created.json()["title"] == sql_title

    message = api_client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=API_HEADERS,
        json={"role": "user", "content": xss_content},
    )
    assert message.status_code == 201
    assert message.json()["content"] == xss_content

    fetched = api_client.get(f"/api/v1/chats/{chat_id}", headers=API_HEADERS)
    assert fetched.status_code == 200
    assert fetched.json()["messages"][0]["content"] == xss_content
    assert fetched.headers["content-type"].startswith("application/json")

    renamed_title = '<img src=x onerror="alert(document.domain)">'
    updated = api_client.patch(
        f"/api/v1/chats/{chat_id}",
        headers=API_HEADERS,
        json={"title": renamed_title, "pinned": True},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == renamed_title
    assert updated.json()["pinned"] == 1

    injected_search = api_client.get(
        "/api/v1/chats", headers=API_HEADERS, params={"search": "' OR 1=1 --"}
    )
    assert injected_search.status_code == 200
    assert injected_search.json() == []
    literal_wildcard = api_client.get("/api/v1/chats", headers=API_HEADERS, params={"search": "%"})
    assert literal_wildcard.status_code == 200
    assert literal_wildcard.json() == []

    still_usable = api_client.post(
        "/api/v1/chats", headers=API_HEADERS, json={"title": "table still exists"}
    )
    assert still_usable.status_code == 201
    second_id = still_usable.json()["id"]

    assert api_client.delete(f"/api/v1/chats/{chat_id}", headers=API_HEADERS).status_code == 204
    assert api_client.get(f"/api/v1/chats/{chat_id}", headers=API_HEADERS).status_code == 404
    assert api_client.delete(f"/api/v1/chats/{second_id}", headers=API_HEADERS).status_code == 204


def test_active_generation_returns_structured_conflict_for_destructive_chat_actions(
    api_client: TestClient,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    chat = api_client.post(
        "/api/v1/chats", headers=API_HEADERS, json={"title": "active generation"}
    ).json()
    message = api_client.post(
        f"/api/v1/chats/{chat['id']}/messages",
        headers=API_HEADERS,
        json={"role": "assistant", "content": "pending"},
    ).json()
    run = portal.call(
        services.repository.create_run,
        {
            "id": "api-active-generation",
            "message_id": message["id"],
            "kind": "generation",
            "status": "running",
            "effective_seed": 0,
        },
    )

    deleted = api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS)
    assert deleted.status_code == 409
    assert deleted.json() == {
        "error": {
            "code": "active_run_conflict",
            "message": "chat cannot be deleted while it owns an active generation",
            "retryable": True,
            "hint": "Cancel the generation or wait for it to finish, then retry.",
            "details": {"chat_id": chat["id"], "run_ids": [run["id"]]},
        }
    }

    cleared = api_client.delete("/api/v1/chats", headers=API_HEADERS, params={"confirm": "true"})
    assert cleared.status_code == 409
    assert cleared.json()["error"]["code"] == "active_run_conflict"
    assert api_client.get(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS).status_code == 200

    portal.call(
        partial(
            services.repository.update_run,
            run["id"],
            status="cancelled",
            completed_at="2026-01-01T00:00:00+00:00",
        )
    )
    assert api_client.delete(f"/api/v1/chats/{chat['id']}", headers=API_HEADERS).status_code == 204


def test_terminal_event_cannot_overtake_batched_tokens_during_replay(
    api_client: TestClient,
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    run_id = "event-persistence-order"
    portal.call(
        services.repository.create_run,
        {
            "id": run_id,
            "status": "running",
            "effective_seed": 0,
        },
    )
    portal.call(services.events.publish, run_id, "token", {"token_index": 0})
    portal.call(services.events.publish, run_id, "completed", {"finish_reason": "length"})

    persisted = api_client.get(f"/api/v1/runs/{run_id}", headers=API_HEADERS)
    assert persisted.status_code == 200
    assert "events" not in persisted.json()
    events = api_client.get(f"/api/v1/runs/{run_id}/events", headers=API_HEADERS)
    assert events.status_code == 200
    assert [(event["sequence"], event["type"]) for event in events.json()["events"]] == [
        (1, "token"),
        (2, "completed"),
    ]
    assert all("payload_json" not in event for event in events.json()["events"])

    async def reconnect() -> list[tuple[int, str]]:
        return [
            (event.sequence, event.type)
            async for event in services.events.subscribe(run_id, after_sequence=0)
        ]

    assert portal.call(reconnect) == [(1, "token"), (2, "completed")]


def test_model_refresh_replaces_fingerprint_derived_id_atomically(
    api_client: TestClient, api_settings: AppSettings
) -> None:
    before = api_client.get("/api/v1/models", headers=API_HEADERS).json()["models"][0]
    model_directory = api_settings.paths.model_roots[0] / "Tiny-Generation-Model"
    config_path = model_directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_position_embeddings"] = int(config["max_position_embeddings"]) + 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    refreshed = api_client.post("/api/v1/models/refresh", headers=API_HEADERS)
    assert refreshed.status_code == 200
    after = refreshed.json()["models"][0]
    assert after["id"] != before["id"]

    portal = api_client.portal
    assert portal is not None
    persisted = portal.call(api_client.app.state.services.repository.list_models)
    assert len(persisted) == 1
    assert persisted[0]["id"] == after["id"]
