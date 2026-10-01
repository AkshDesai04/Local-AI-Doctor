"""The token influence endpoint: validation, recorded-prefix handoff, caching, and refusals."""

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
from local_ai_doctor.workers import WorkerFailure

API_TOKEN = "integration-test-token"
API_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
DECODER = "Tiny-Generation-Model"
SEQ2SEQ = "Tiny-Seq2Seq-Model"
VISION = "Tiny-Vision-Generator"
PROMPT_TOKENS = 20
GENERATED = [(101, "Hel", "Hel"), (102, "lo", "lo"), (103, "Ġthere", " there")]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="module")
def api_settings(tmp_path_factory: pytest.TempPathFactory) -> AppSettings:
    root = tmp_path_factory.mktemp("influence-workspace")
    model_root = root / "models"
    for name, config in (
        (DECODER, {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        (
            SEQ2SEQ,
            {
                "architectures": ["T5ForConditionalGeneration"],
                "model_type": "t5",
                "is_encoder_decoder": True,
            },
        ),
        (
            VISION,
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "vision_config": {"hidden_size": 8},
                "image_token_id": 7,
                "video_token_id": 8,
            },
        ),
    ):
        directory = model_root / name
        directory.mkdir(parents=True)
        _write_json(directory / "config.json", {**config, "max_position_embeddings": 1024})
        _write_json(directory / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"x": 0}}})
        _write_json(directory / "tokenizer_config.json", {"model_max_length": 1024})
        header = json.dumps(
            {"lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            separators=(",", ":"),
        ).encode("utf-8")
        (directory / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + bytes(4)
        )
        if name == VISION:
            _write_json(
                directory / "preprocessor_config.json", {"processor_class": "Qwen3VLProcessor"}
            )
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


def _model(client: TestClient, name: str) -> dict[str, Any]:
    models = client.get("/api/v1/models", headers=API_HEADERS).json()["models"]
    return dict(next(model for model in models if model["display_name"] == name))


def _wait_for_run(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        run = client.get(f"/api/v1/runs/{run_id}", headers=API_HEADERS).json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            return dict(run)
        time.sleep(0.02)
    raise AssertionError("run did not finish")


class Fakes:
    def __init__(self) -> None:
        self.loads: list[dict[str, Any]] = []
        self.analyses: list[dict[str, Any]] = []
        self.failure: dict[str, Any] | None = None


@pytest.fixture
def fakes(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Fakes:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    recorded = Fakes()

    async def fake_load_reserved(*_args: object, **kwargs: Any) -> dict[str, Any]:
        recorded.loads.append(kwargs)
        return {"model_key": "resident-key", "placement": "cpu"}

    async def fake_generate(**_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "event_type": "stage",
            "payload": {
                "stage": "prefill",
                "rendered_prompt": "<|user|>hi<|assistant|>",
                "prompt_tokens": PROMPT_TOKENS,
                "prompt_renderer": "chat_template",
                "template_ms": 1.0,
                "tokenization_ms": 1.0,
            },
        }
        text = ""
        for index, (token_id, piece, display) in enumerate(GENERATED):
            yield {
                "event_type": "token",
                "payload": {
                    "token_index": index,
                    "token_id": token_id,
                    "piece": piece,
                    "escaped_bytes": "",
                    "display_text": display,
                    "replace_from": len(text),
                    "alternatives": {"raw": [], "sampling": []},
                },
            }
            text += display
        yield {"event_type": "completed", "payload": {"finish_reason": "length"}}

    async def fake_analyze(command: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        recorded.analyses.append(command)
        if recorded.failure is not None:
            raise WorkerFailure(recorded.failure)
        generated = len(command["generated_token_ids"])
        sources = [
            {
                "source_kind": "prompt",
                "context_index": 0,
                "span": None,
                "token_count": 1,
                "token_id": 7,
                "piece": "<|user|>",
                "display_text": "<|user|>",
                "generated_token_index": None,
                "media_index": None,
                "is_special": True,
                "weight": 0.25,
            },
            {
                "source_kind": "generated",
                "context_index": PROMPT_TOKENS,
                "span": None,
                "token_count": 1,
                "token_id": GENERATED[0][0],
                "piece": "single-token-piece",
                "display_text": "single-token-decode",
                "generated_token_index": 0,
                "media_index": None,
                "is_special": False,
                "weight": 0.75,
            },
        ]
        return {
            "method": command["method"],
            "context_token_count": PROMPT_TOKENS + generated,
            "prompt_token_count": PROMPT_TOKENS,
            "source_limit": command["source_limit"],
            "sources": sources,
            "retained_weight": 1.0,
            "omitted_weight": 0.0,
            "layers": (
                [{"layer": 0, "sources": sources, "retained_weight": 1.0, "omitted_weight": 0.0}]
                if command.get("layers") == "all"
                else None
            ),
            "captured_layers": [0] if command["method"] == "attention" else [],
            "heads_per_layer": 4 if command["method"] == "attention" else None,
            "objective": None if command["method"] == "attention" else "log_probability",
            "objective_value": None,
            "alternative_piece": None,
            "semantics": "allocation, not causal attribution",
            "normalization": "sum_to_one",
            "duration_ms": 12.5,
        }

    monkeypatch.setattr(services.registry, "load_reserved", fake_load_reserved)
    monkeypatch.setattr(services.worker, "generate", fake_generate)
    monkeypatch.setattr(services.worker, "analyze_influence", fake_analyze)
    return recorded


def _completed_run(
    client: TestClient, model: str = DECODER, attachment_ids: list[str] | None = None
) -> str:
    chat = client.post("/api/v1/chats", headers=API_HEADERS, json={}).json()
    created = client.post(
        "/api/v1/runs/generation",
        headers=API_HEADERS,
        json={
            "chatId": chat["id"],
            "modelId": _model(client, model)["id"],
            "content": "hi",
            "attachmentIds": attachment_ids or [],
        },
    )
    assert created.status_code == 202, created.text
    run = _wait_for_run(client, created.json()["runId"])
    assert run["status"] == "complete"
    return str(run["id"])


def _influence(client: TestClient, run_id: str, index: int, body: dict[str, Any]) -> Any:
    return client.post(
        f"/api/v1/runs/{run_id}/tokens/{index}/influence", headers=API_HEADERS, json=body
    )


def test_influence_re_runs_the_recorded_prefix_and_caches_the_result(
    api_client: TestClient, fakes: Fakes
) -> None:
    run_id = _completed_run(api_client)
    generation_loads = len(fakes.loads)

    first = _influence(api_client, run_id, 2, {"method": "attention", "layers": "all"})

    assert first.status_code == 200, first.text
    body = first.json()
    assert body["cached"] is False
    assert body["run_id"] == run_id and body["token_index"] == 2
    assert body["target"] == {
        "token_id": 103,
        "piece": "Ġthere",
        "display_text": " there",
        "alternative_token_id": None,
        "alternative_piece": None,
    }
    assert body["normalization"] == "sum_to_one"
    assert body["layers"][0]["layer"] == 0
    # Generated sources carry the exact text the run streamed, not a one-token decode.
    generated = next(item for item in body["sources"] if item["source_kind"] == "generated")
    assert (generated["piece"], generated["display_text"]) == ("Hel", "Hel")
    assert body["model_fingerprint"] == _model(api_client, DECODER)["fingerprint"]["value"]
    [command] = fakes.analyses
    assert command["model_key"] == "resident-key"
    assert command["rendered_prompt"] == "<|user|>hi<|assistant|>"
    assert command["prompt_renderer"] == "chat_template"
    assert command["expected_prompt_token_count"] == PROMPT_TOKENS
    assert command["generated_token_ids"] == [101, 102]
    assert command["target_token_id"] == 103
    assert command["media"] == []
    assert command["layers"] == "all" and command["source_limit"] == 128
    [load] = fakes.loads[generation_loads:]
    assert {key: str(value) for key, value in load.items() if key != "pinned"} == {
        "device": "cpu",
        "dtype": "float32",
        "quantization": "none",
        "strict_vram": "True",
    }

    again = _influence(api_client, run_id, 2, {"method": "attention", "layers": "all"})
    assert again.status_code == 200
    assert again.json()["cached"] is True
    assert {**again.json(), "cached": False} == body
    assert len(fakes.analyses) == 1

    other = _influence(api_client, run_id, 2, {"method": "attention", "sourceLimit": 32})
    assert other.json()["cached"] is False
    gradient = _influence(
        api_client, run_id, 1, {"method": "gradient_x_input", "alternativeTokenId": 55}
    )
    assert gradient.status_code == 200, gradient.text
    assert gradient.json()["target"]["alternative_token_id"] == 55
    assert fakes.analyses[-1]["alternative_token_id"] == 55
    assert fakes.analyses[-1]["generated_token_ids"] == [101]
    assert len(fakes.analyses) == 3


@pytest.mark.parametrize(
    ("index", "body"),
    [
        (3, {"method": "attention"}),
        (-1, {"method": "attention"}),
        (0, {"method": "gradient_x_input", "layers": "mean"}),
        (0, {"method": "attention", "alternativeTokenId": 4}),
        (0, {"method": "attention", "layers": [-1]}),
        (0, {"method": "attention", "sourceLimit": 4}),
        (0, {"method": "attention", "sourceLimit": 513}),
        (0, {"method": "integrated_gradients"}),
        (0, {"method": "gradient_x_input", "alternativeTokenId": 101}),
        (0, {"method": "attention", "unexpected": True}),
    ],
)
def test_invalid_requests_are_rejected_before_the_worker_runs(
    api_client: TestClient, fakes: Fakes, index: int, body: dict[str, Any]
) -> None:
    run_id = _completed_run(api_client)

    response = _influence(api_client, run_id, index, body)

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_request"
    assert fakes.analyses == []


def test_a_changed_model_fingerprint_is_a_conflict(api_client: TestClient, fakes: Fakes) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    portal = api_client.portal
    assert portal is not None
    run_id = _completed_run(api_client)
    portal.call(
        services.database.execute,
        "UPDATE inference_runs SET model_fingerprint = ? WHERE id = ?",
        ("an-older-checkpoint", run_id),
    )

    response = _influence(api_client, run_id, 0, {"method": "attention"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_fingerprint_changed"
    assert fakes.analyses == []


def test_runs_that_are_not_finished_or_not_decoder_only_are_conflicts(
    api_client: TestClient, fakes: Fakes
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    portal = api_client.portal
    assert portal is not None
    decoder = _model(api_client, DECODER)
    seq2seq = _model(api_client, SEQ2SEQ)
    for run_id, model, status in (
        ("influence-running", decoder, "running"),
        ("influence-seq2seq", seq2seq, "complete"),
    ):
        portal.call(
            services.repository.create_run,
            {
                "id": run_id,
                "model_id": model["id"],
                "kind": "generation",
                "status": status,
                "effective_seed": 0,
                "model_fingerprint": model["fingerprint"]["value"],
            },
        )

    running = _influence(api_client, "influence-running", 0, {"method": "attention"})
    seq2seq_response = _influence(api_client, "influence-seq2seq", 0, {"method": "attention"})
    missing = _influence(api_client, "no-such-run", 0, {"method": "attention"})

    assert running.status_code == 409
    assert running.json()["error"]["code"] == "influence_unavailable"
    assert running.json()["error"]["details"]["status"] == "running"
    assert seq2seq_response.status_code == 409
    assert "decoder-only" in seq2seq_response.json()["error"]["message"]
    assert missing.status_code == 404
    assert fakes.analyses == []


def test_long_prefixes_are_refused_for_gradients_before_any_model_load(
    api_client: TestClient, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    run_id = _completed_run(api_client)
    generation_loads = len(fakes.loads)
    monkeypatch.setattr(services.settings.inference, "influence_max_gradient_tokens", 16)

    refused = _influence(api_client, run_id, 2, {"method": "gradient_x_input"})
    attention = _influence(api_client, run_id, 2, {"method": "attention"})

    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "influence_sequence_too_long"
    assert error["details"] == {"sequence_tokens": PROMPT_TOKENS + 2, "max_sequence_tokens": 16}
    assert attention.status_code == 200
    assert len(fakes.loads) == generation_loads + 1


def test_worker_refusals_keep_their_code_and_details(api_client: TestClient, fakes: Fakes) -> None:
    run_id = _completed_run(api_client)
    fakes.failure = {
        "code": "influence_prompt_mismatch",
        "message": "re-tokenizing the persisted prompt did not reproduce the recorded token count",
        "expected_prompt_tokens": PROMPT_TOKENS,
        "actual_prompt_tokens": PROMPT_TOKENS + 1,
    }

    response = _influence(api_client, run_id, 0, {"method": "attention"})

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "influence_prompt_mismatch"
    assert error["details"] == {
        "expected_prompt_tokens": PROMPT_TOKENS,
        "actual_prompt_tokens": PROMPT_TOKENS + 1,
    }
    # A failed analysis is not cached.
    fakes.failure = None
    assert _influence(api_client, run_id, 0, {"method": "attention"}).json()["cached"] is False


def test_retention_removes_cached_influence_with_its_run(
    api_client: TestClient, fakes: Fakes
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    portal = api_client.portal
    assert portal is not None
    run_id = _completed_run(api_client)
    assert _influence(api_client, run_id, 0, {"method": "attention"}).status_code == 200

    deleted = api_client.delete(
        "/api/v1/storage/telemetry",
        headers=API_HEADERS,
        params={"before": "2999-01-01T00:00:00+00:00", "confirm": "true"},
    )

    assert deleted.status_code == 200
    assert deleted.json()["deleted"]["token_influence"] >= 1
    remaining = portal.call(
        services.database.fetch_all,
        "SELECT run_id FROM token_influence WHERE run_id = ?",
        (run_id,),
    )
    assert remaining == []


def test_chat_images_are_resolved_in_conversation_order_for_the_worker(
    api_client: TestClient, fakes: Fakes
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 255)).save(buffer, format="PNG")
    uploaded = api_client.post(
        "/api/v1/uploads",
        headers=API_HEADERS,
        data={"model_id": _model(api_client, VISION)["id"]},
        files={"file": ("blue.png", buffer.getvalue(), "image/png")},
    )
    assert uploaded.status_code == 201, uploaded.text
    run_id = _completed_run(api_client, VISION, [uploaded.json()["id"]])

    response = _influence(api_client, run_id, 0, {"method": "attention"})

    assert response.status_code == 200, response.text
    [media] = fakes.analyses[-1]["media"]
    assert media["kind"] == "image"
    assert Path(media["path"]).parent == services.uploads.root
