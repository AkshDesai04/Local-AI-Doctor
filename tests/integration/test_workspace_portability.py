from __future__ import annotations

import asyncio
import json
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
    attention_attribution = {
        "method": "mean_causal_self_attention",
        "semantics": "attention_weights_not_causal_contributions",
        "total_source_count": 1,
        "retained_source_count": 1,
        "retained_weight": 1.0,
        "omitted_weight": 0.0,
        "source_tokens": [
            {
                "context_index": 0,
                "token_id": 5,
                "piece": "Explain",
                "display_text": "Explain",
                "source_kind": "prompt",
                "weight": 1.0,
            }
        ],
    }
    portal.call(
        services.database.execute,
        """
        INSERT INTO token_events(
            run_id, token_index, token_id, piece, escaped_bytes, display_text,
            raw_logprob, raw_probability, segment, reasoning_slices_json,
            selected_experts_json, attention_attribution_json, created_at
        ) VALUES (?, 0, 7, '4', '\\x34', '4', -0.25, 0.75, 'answer', ?, '[1,2]', ?, ?)
        """,
        (
            run["id"],
            '[{"start":0,"end":1,"classification":"answer","delimiter":false}]',
            json.dumps(attention_attribution),
            now,
        ),
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
    assert exported["runs"][0]["tokens"][0]["reasoning_slices"] == [
        {"start": 0, "end": 1, "classification": "answer", "delimiter": False}
    ]
    assert exported["runs"][0]["tokens"][0]["attention_attribution"] == (attention_attribution)
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
    assert imported_tokens[0]["reasoning_slices"][0]["classification"] == "answer"
    assert imported_tokens[0]["attention_attribution"] == attention_attribution
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
                "reasoning": False,
            },
            "effective_config": {"source": "recorded"},
            "reproducibility": {
                "device": {"requested": "cpu"},
                "dtype": "float32",
                "reasoning": True,
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
    assert replayed["run"]["settings"]["reasoning"] is False
    assert replayed["run"]["reproducibility"]["reasoning"] is False
    assert replayed["run"]["effective_config"] == services.settings.inference_snapshot()
    assert replayed["assistant_message"]["parent_id"] == user["id"]
    assert replayed["assistant_message"]["branch_index"] == 1
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [{"role": "user", "content": "A seeded question"}]
    assert captured["effective_seed"] == 123456
    assert captured["sampling"]["top_k"] == 5
    assert captured["deterministic_reference_mode"] is True
    assert captured["reasoning"] is False

    portal.call(
        partial(
            services.repository.update_run,
            replayed["run"]["id"],
            status="complete",
            completed_at=utc_now(),
        )
    )
    portal.call(services.repository.delete_chat, chat["id"])


def test_token_branch_validates_telemetry_and_clones_lineage_into_a_new_chat(
    api_client: TestClient, monkeypatch: Any
) -> None:
    services = api_client.app.state.services
    portal = api_client.portal
    assert portal is not None
    assert services.registry.report is not None
    model = services.registry.report.models[0]
    chat = portal.call(services.repository.create_chat, "token branch")
    earlier_user = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="Earlier question",
        )
    )
    earlier_assistant = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="assistant",
            content="Earlier answer",
            parent_id=earlier_user["id"],
            status="complete",
        )
    )
    earlier_run = portal.call(
        services.repository.create_run,
        {
            "message_id": earlier_assistant["id"],
            "model_id": model.id,
            "status": "complete",
            "effective_seed": 123,
            "settings": {"sampling": {"max_output_tokens": 8, "temperature": 0.7}},
            "model_fingerprint": model.fingerprint.value,
        },
    )
    earlier_now = utc_now()
    cloned_attention_attribution = {
        "method": "mean_causal_self_attention",
        "semantics": "attention_weights_not_causal_contributions",
        "total_source_count": 1,
        "retained_source_count": 1,
        "retained_weight": 1.0,
        "omitted_weight": 0.0,
        "source_tokens": [],
    }
    portal.call(
        partial(
            services.repository.update_run,
            earlier_run["id"],
            rendered_prompt="Earlier question ->",
            prompt_token_count=2,
            generated_token_count=1,
            finish_reason="stop",
            completed_at=earlier_now,
        )
    )
    portal.call(
        partial(
            services.repository.save_environment_snapshot,
            earlier_run["id"],
            hardware={"device": "test-gpu"},
            software={"torch": "test"},
            backend={"name": "test-backend"},
        )
    )
    portal.call(
        partial(
            services.repository.upsert_phase_metric,
            earlier_run["id"],
            "generation",
            duration_ms=12.5,
            details={"decode_tokens_per_second": 80.0},
        )
    )
    portal.call(
        services.database.execute,
        """
        INSERT INTO token_events(
            run_id, token_index, token_id, piece, escaped_bytes, display_text,
            selected_experts_json, created_at, reasoning_slices_json,
            attention_attribution_json
        ) VALUES (?, 0, 7, 'Earlier', 'Earlier', 'Earlier', '[1]', ?,
                  '[{"kind":"answer","start":0,"end":7}]', ?)
        """,
        (
            earlier_run["id"],
            earlier_now,
            json.dumps(cloned_attention_attribution),
        ),
    )
    portal.call(
        services.database.execute,
        """
        INSERT INTO token_alternatives(
            run_id, token_index, distribution, rank, token_id, piece, probability,
            survived_filter
        ) VALUES (?, 0, 'sampling', 1, 8, 'Alternative', 0.25, 1)
        """,
        (earlier_run["id"],),
    )
    portal.call(
        services.database.execute,
        """
        INSERT INTO router_events(
            run_id, token_index, layer_index, selected_json, executed_json,
            router_entropy, dropped_assignments
        ) VALUES (?, 0, 0, '[{"expert_id":1}]', '[{"expert_id":1}]', 0.4, 0)
        """,
        (earlier_run["id"],),
    )
    portal.call(
        services.database.execute,
        """
        INSERT INTO router_aggregates(
            run_id, layer_index, expert_id, activation_count, average_weight, load_fraction
        ) VALUES (?, 0, 1, 1, 0.75, 1.0)
        """,
        (earlier_run["id"],),
    )
    portal.call(
        services.repository.append_raw_event,
        {
            "run_id": earlier_run["id"],
            "sequence": 1,
            "type": "warning",
            "monotonic_ns": 1,
            "payload": {"message": "historical warning"},
        },
    )
    portal.call(
        services.repository.append_raw_event,
        {
            "run_id": earlier_run["id"],
            "sequence": 2,
            "type": "completed",
            "monotonic_ns": 2,
            "payload": {"finish_reason": "stop"},
        },
    )
    user = portal.call(
        partial(
            services.repository.create_message,
            chat_id=chat["id"],
            role="user",
            content="Pick a color",
            parent_id=earlier_assistant["id"],
        )
    )
    source_message = portal.call(
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
            "message_id": source_message["id"],
            "model_id": model.id,
            "status": "complete",
            "effective_seed": 987,
            "requested_seed": 987,
            "settings": {
                "sampling": {"max_output_tokens": 8, "temperature": 0.7},
                "instrumentation": "full",
            },
            "reproducibility": {
                "device": {"requested": "cpu"},
                "dtype": "float32",
            },
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
    persisted_source = api_client.get(
        f"/api/v1/runs/{source_run['id']}", headers=API_HEADERS
    ).json()
    assert persisted_source["branchable_through_token_index"] == 1
    assert "events" not in persisted_source
    captured: dict[str, Any] = {}

    async def fake_execute_generation(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(services.runs, "_execute_generation", fake_execute_generation)
    oversized = api_client.post(
        f"/api/v1/runs/{source_run['id']}/branch",
        headers=API_HEADERS,
        json={
            "tokenIndex": 100_001,
            "distribution": "sampling",
            "rank": 2,
            "tokenId": 42,
        },
    )
    assert oversized.status_code == 422
    stale = api_client.post(
        f"/api/v1/runs/{source_run['id']}/branch",
        headers=API_HEADERS,
        json={"tokenIndex": 1, "distribution": "sampling", "rank": 2, "tokenId": 41},
    )
    assert stale.status_code == 422
    assert stale.json()["error"]["code"] == "invalid_request"

    response = api_client.post(
        f"/api/v1/runs/{source_run['id']}/branch",
        headers=API_HEADERS,
        json={"tokenIndex": 1, "distribution": "sampling", "rank": 2, "tokenId": 42},
    )
    assert response.status_code == 202
    branched = response.json()
    assert branched["chatId"] != chat["id"]
    assert branched["sourceRunId"] == source_run["id"]
    assert "parentRunId" not in branched
    assert branched["run"]["parent_run_id"] is None
    assert branched["run"]["settings"]["token_branch"] == {
        "source_run_id": source_run["id"],
        "source_token_index": 1,
        "source_token_id": 12,
        "selected_token_id": 42,
        "selected_piece": " bike",
        "distribution": "sampling",
        "rank": 2,
        "forced_prefix_token_count": 2,
    }
    assert branched["run"]["reproducibility"]["generator_device"] == "cpu"
    assert branched["run"]["reproducibility"]["quantization"] == (
        services.settings.runtime.quantization.value
    )
    assert branched["run"]["reproducibility"]["attention_implementation"] == (
        services.settings.runtime.attention_backend.value
    )
    new_chat = api_client.get(f"/api/v1/chats/{branched['chatId']}", headers=API_HEADERS).json()
    assert new_chat["title"] == "token branch · Branch"
    assert [message["role"] for message in new_chat["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert new_chat["messages"][0]["content"] == "Earlier question"
    assert new_chat["messages"][0]["id"] != earlier_user["id"]
    cloned_history_run_id = new_chat["messages"][1]["run_id"]
    assert cloned_history_run_id != earlier_run["id"]
    assert new_chat["messages"][1]["metadata"]["cloned_from_run_id"] == earlier_run["id"]
    assert new_chat["messages"][2]["content"] == "Pick a color"
    assert new_chat["messages"][2]["id"] != user["id"]
    assert new_chat["messages"][3]["parent_id"] == new_chat["messages"][2]["id"]
    assert [message["created_at"] for message in new_chat["messages"]] == sorted(
        message["created_at"] for message in new_chat["messages"]
    )
    portal.call(asyncio.sleep, 0)
    assert captured["messages"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Pick a color"},
    ]
    assert captured["forced_prefix_token_ids"] == [11, 42]
    assert captured["effective_seed"] == 987

    assert portal.call(services.repository.delete_chat, chat["id"])
    assert (
        api_client.get(f"/api/v1/runs/{earlier_run['id']}", headers=API_HEADERS).status_code == 404
    )
    cloned_details_response = api_client.get(
        f"/api/v1/runs/{cloned_history_run_id}", headers=API_HEADERS
    )
    assert cloned_details_response.status_code == 200
    cloned_details = cloned_details_response.json()
    assert cloned_details["message_id"] == new_chat["messages"][1]["id"]
    assert cloned_details["parent_run_id"] is None
    assert cloned_details["effective_seed"] == "123"
    assert cloned_details["rendered_prompt"] == "Earlier question ->"
    assert cloned_details["finish_reason"] == "stop"
    assert cloned_details["environment"]["hardware"] == {"device": "test-gpu"}
    assert cloned_details["phases"][0]["details"] == {"decode_tokens_per_second": 80.0}
    assert cloned_details["tokens"][0]["reasoning_slices"] == [
        {"kind": "answer", "start": 0, "end": 7}
    ]
    assert cloned_details["tokens"][0]["attention_attribution"] == (cloned_attention_attribution)
    assert cloned_details["tokens"][0]["alternatives"][0]["token_id"] == 8
    assert cloned_details["summary"] == {"finish_reason": "stop"}
    assert cloned_details["warnings"] == ["historical warning"]
    cloned_events = api_client.get(
        f"/api/v1/runs/{cloned_history_run_id}/events", headers=API_HEADERS
    ).json()["events"]
    assert [event["type"] for event in cloned_events] == ["warning", "completed"]
    for table in ("router_events", "router_aggregates"):
        assert portal.call(
            services.database.fetch_one,
            f"SELECT COUNT(*) AS count FROM {table} WHERE run_id = ?",
            (cloned_history_run_id,),
        ) == {"count": 1}

    nested_branch = api_client.post(
        f"/api/v1/runs/{cloned_history_run_id}/branch",
        headers=API_HEADERS,
        json={"tokenIndex": 0, "distribution": "sampling", "rank": 1, "tokenId": 8},
    )
    assert nested_branch.status_code == 202
    assert nested_branch.json()["sourceRunId"] == cloned_history_run_id
    portal.call(
        partial(
            services.repository.update_run,
            nested_branch.json()["runId"],
            status="complete",
            completed_at=utc_now(),
        )
    )
    portal.call(services.repository.delete_chat, nested_branch.json()["chatId"])

    replay = api_client.post(f"/api/v1/runs/{cloned_history_run_id}/replay", headers=API_HEADERS)
    assert replay.status_code == 202
    assert replay.json()["parentRunId"] == cloned_history_run_id
    assert replay.json()["assistant_message"]["chat_id"] == branched["chatId"]
    portal.call(
        partial(
            services.repository.update_message,
            replay.json()["messageId"],
            content="replayed",
            status="complete",
        )
    )
    portal.call(
        partial(
            services.repository.update_run,
            replay.json()["runId"],
            status="complete",
            completed_at=utc_now(),
        )
    )

    portal.call(
        partial(
            services.repository.update_run,
            branched["runId"],
            status="complete",
            completed_at=utc_now(),
        )
    )
    branch_export = api_client.get(
        f"/api/v1/chats/{branched['chatId']}/export", headers=API_HEADERS
    )
    assert branch_export.status_code == 200
    assert branch_export.json()["runs"][0]["parent_run_id"] is None
    branch_import = api_client.post(
        "/api/v1/chats/import", headers=API_HEADERS, json=branch_export.json()
    )
    assert branch_import.status_code == 201
    portal.call(services.repository.delete_chat, branch_import.json()["chat"]["id"])
    portal.call(services.repository.delete_chat, branched["chatId"])


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
