"""Chat attachments reach the worker only for generators with a usable media capability."""

from __future__ import annotations

import io
import json
import struct
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from local_ai_doctor.config import AppSettings

API_TOKEN = "integration-test-token"
API_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
TEXT_MODEL = "Tiny-Generation-Model"
VISION_MODEL = "Tiny-Vision-Generator"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_tiny_safetensors(path: Path) -> None:
    header = json.dumps(
        {"lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))


@pytest.fixture(scope="module")
def api_settings(tmp_path_factory: pytest.TempPathFactory) -> AppSettings:
    root = tmp_path_factory.mktemp("media-workspace")
    model_root = root / "models"
    text = model_root / TEXT_MODEL
    vision = model_root / VISION_MODEL
    for directory in (text, vision):
        directory.mkdir(parents=True)
        _write_json(directory / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"x": 0}}})
        _write_json(directory / "tokenizer_config.json", {"model_max_length": 1024})
        _write_tiny_safetensors(directory / "model.safetensors")
    _write_json(
        text / "config.json",
        {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
    )
    # Mirrors Qwen3-VL-2B-Thinking's packaging: a generator with image and video processors.
    _write_json(
        vision / "config.json",
        {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": {"max_position_embeddings": 1024},
            "vision_config": {"hidden_size": 8},
            "image_token_id": 7,
            "video_token_id": 8,
        },
    )
    _write_json(vision / "preprocessor_config.json", {"processor_class": "Qwen3VLProcessor"})
    _write_json(vision / "video_preprocessor_config.json", {"fps": 2})
    return AppSettings.model_validate(
        {
            "active_profile": "test",
            "paths": {
                "model_roots": [model_root],
                "database": root / "state" / "workbench.sqlite3",
                "uploads": root / "state" / "uploads",
                "cache": root / "state" / "cache",
                "exports": root / "state" / "exports",
                "backups": root / "state" / "backups",
            },
            "server": {"host": "127.0.0.1", "authentication_token": API_TOKEN},
            "runtime": {"device": "cpu", "cpu_threads": 1, "queue_limit": 2},
            "workers": {"shutdown_grace_seconds": 1.0},
        }
    )


def _model_id(client: TestClient, name: str) -> str:
    models = client.get("/api/v1/models", headers=API_HEADERS).json()["models"]
    return str(next(model["id"] for model in models if model["display_name"] == name))


def _upload_png(client: TestClient, model_id: str) -> dict[str, Any]:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    response = client.post(
        "/api/v1/uploads",
        headers=API_HEADERS,
        data={"model_id": model_id},
        files={"file": ("red.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


def _wait_for_run(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        run = client.get(f"/api/v1/runs/{run_id}", headers=API_HEADERS).json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            return dict(run)
        time.sleep(0.02)
    raise AssertionError("run did not finish")


@pytest.fixture
def captured_generations(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    captured: list[dict[str, Any]] = []

    async def fake_load_reserved(*_args: object, **_kwargs: object) -> dict[str, str]:
        return {"status": "loaded"}

    async def fake_generate(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        captured.append(kwargs)
        yield {"event_type": "completed", "payload": {"finish_reason": "length"}}

    monkeypatch.setattr(services.registry, "load_reserved", fake_load_reserved)
    monkeypatch.setattr(services.worker, "generate", fake_generate)
    return captured


def test_capability_matrix_marks_generator_media_partial(api_client: TestClient) -> None:
    models = api_client.get("/api/v1/models", headers=API_HEADERS).json()["models"]
    vision = next(model for model in models if model["display_name"] == VISION_MODEL)
    entries = vision["capabilities"]["entries"]
    assert entries["vision"]["state"] == "partial"
    assert entries["vision"]["reason"].startswith("validated on Qwen3-VL")
    assert entries["video"]["state"] == "partial"
    assert entries["audio"]["state"] == "unsupported"


def test_generation_sends_current_and_history_attachments_to_the_worker(
    api_client: TestClient, captured_generations: list[dict[str, Any]]
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    model_id = _model_id(api_client, VISION_MODEL)
    attachment = _upload_png(api_client, model_id)
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={}).json()

    first = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": model_id,
            "content": "What colour is this?",
            "attachmentIds": [attachment["id"]],
        },
    )
    assert first.status_code == 202, first.text
    assert _wait_for_run(api_client, first.json()["runId"])["status"] == "complete"

    media = captured_generations[0]["messages"][-1]["attachments"]
    assert [item["kind"] for item in media] == ["image"]
    stored = Path(media[0]["path"])
    assert stored.parent == services.uploads.root
    messages = api_client.get(f"/api/v1/chats/{chat['id']}/messages", headers=API_HEADERS).json()
    assert [item["id"] for item in messages[0]["attachments"]] == [attachment["id"]]

    # A follow-up without new media still re-renders the earlier image.
    second = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": model_id,
            "content": "And its shape?",
            "parentMessageId": first.json()["messageId"],
        },
    )
    assert second.status_code == 202, second.text
    _wait_for_run(api_client, second.json()["runId"])
    history = captured_generations[1]["messages"]
    assert history[0]["attachments"] == media
    assert "attachments" not in history[-1]

    replay = api_client.post(f"/api/v1/runs/{first.json()['runId']}/replay", headers=API_HEADERS)
    assert replay.status_code in {200, 202}, replay.text
    _wait_for_run(api_client, replay.json()["run"]["id"])
    assert captured_generations[2]["messages"][-1]["attachments"] == media


def test_generation_rejects_media_for_models_without_the_capability(
    api_client: TestClient, captured_generations: list[dict[str, Any]]
) -> None:
    attachment = _upload_png(api_client, _model_id(api_client, VISION_MODEL))
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={}).json()

    rejected = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": _model_id(api_client, TEXT_MODEL),
            "content": "Describe the image.",
            "attachmentIds": [attachment["id"]],
        },
    )

    assert rejected.status_code == 409
    error = rejected.json()["error"]
    assert error["code"] == "capability_unavailable"
    assert error["details"]["media_kind"] == "image"
    assert "processor" in error["details"]["reason"]
    assert captured_generations == []
    # Rejection happens before any chat or run state is written.
    assert api_client.get(f"/api/v1/chats/{chat['id']}/messages", headers=API_HEADERS).json() == []


def test_generation_rejects_unknown_attachment_ids(
    api_client: TestClient, captured_generations: list[dict[str, Any]]
) -> None:
    chat = api_client.post("/api/v1/chats", headers=API_HEADERS, json={}).json()
    missing = api_client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": _model_id(api_client, VISION_MODEL),
            "content": "Describe the image.",
            "attachmentIds": ["not-an-attachment"],
        },
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "attachment_not_found"
    assert captured_generations == []
