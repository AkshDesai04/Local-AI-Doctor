"""Load-time quantization and Flush to storage through the HTTP API (worker faked)."""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.workers import WorkerFailure

API_HEADERS = {"Authorization": "Bearer integration-test-token"}
SOURCE = "Tiny-Generation-Model"


def _source(client: TestClient) -> dict[str, Any]:
    models = client.get("/api/v1/models", headers=API_HEADERS).json()["models"]
    return next(model for model in models if model["display_name"] == SOURCE)


def _resident(model_id: str, **values: Any) -> dict[str, Any]:
    return {
        "model_key": "nf4-key",
        "model_id": model_id,
        "display_name": SOURCE,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "quantization": "bitsandbytes-4bit",
        "strict_vram": True,
        "placement": "gpu",
        "gpu_bytes": 1024,
        **values,
    }


def _flush(client: TestClient, body: dict[str, Any], key: str = "nf4-key") -> Any:
    return client.post(f"/api/v1/models/resident/{key}/flush", headers=API_HEADERS, json=body)


@pytest.fixture
def model_root(api_client: TestClient) -> Path:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    return Path(services.settings.paths.model_roots[0]).resolve()


@pytest.fixture
def resident(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    entry = _resident(_source(api_client)["id"])
    monkeypatch.setattr(services.worker, "_resident", OrderedDict({entry["model_key"]: entry}))
    return entry


@pytest.mark.parametrize("name", ["../escape", "CON", "nul.txt", "trailing.", ".hidden", "a b"])
def test_unsafe_folder_names_are_rejected_before_anything_is_written(
    api_client: TestClient, resident: dict[str, Any], model_root: Path, name: str
) -> None:
    before = sorted(path.name for path in model_root.iterdir())
    response = _flush(api_client, {"targetRootIndex": 0, "folderName": name})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert sorted(path.name for path in model_root.iterdir()) == before


def test_flush_preconditions_are_conflicts(
    api_client: TestClient,
    resident: dict[str, Any],
    model_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    unknown = _flush(api_client, {"targetRootIndex": 0, "folderName": "x"}, key="missing")
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (409, "model_not_resident")

    exists = _flush(api_client, {"targetRootIndex": 0, "folderName": SOURCE})
    assert (exists.status_code, exists.json()["error"]["code"]) == (409, "target_exists")
    assert exists.json()["error"]["details"]["folder"] == f"<model-root:0>/{SOURCE}"

    outside = _flush(api_client, {"targetRootIndex": 5, "folderName": "x"})
    assert outside.status_code == 422

    def read_only(self: Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(13, "read-only file system")

    monkeypatch.setattr(Path, "mkdir", read_only)
    blocked = _flush(api_client, {"targetRootIndex": 0, "folderName": "x"})
    monkeypatch.undo()
    error = blocked.json()["error"]
    assert (blocked.status_code, error["code"]) == (409, "model_root_read_only")
    assert "Docker mounts /models read-only" in error["hint"]

    for values in ({"quantization": "none"}, {"placement": "offload"}):
        entry = _resident(resident["model_id"], **values)
        monkeypatch.setattr(services.worker, "_resident", OrderedDict({"nf4-key": entry}))
        refused = _flush(api_client, {"targetRootIndex": 0, "folderName": "x"})
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "flush_requires_quantized_resident"
    assert not (model_root / "x").exists()


def test_a_worker_failure_removes_only_the_staging_folder(
    api_client: TestClient,
    resident: dict[str, Any],
    model_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    before = sorted(path.name for path in model_root.iterdir())
    staged: list[Path] = []

    async def failing_flush(*, staging_dir: str, **_: Any) -> dict[str, Any]:
        staged.append(Path(staging_dir))
        (Path(staging_dir) / "model.safetensors").write_bytes(bytes(8))
        raise WorkerFailure({"code": "model_worker_error", "message": "disk full"})

    monkeypatch.setattr(services.worker, "flush", failing_flush)
    response = _flush(api_client, {"targetRootIndex": 0, "folderName": "Partial"})

    assert response.status_code == 502
    assert staged and staged[0].parent == model_root and staged[0].name.startswith(".lad-staging-")
    assert not staged[0].exists()
    assert sorted(path.name for path in model_root.iterdir()) == before


def test_flush_renames_the_staging_folder_and_registers_the_derived_model(
    api_client: TestClient,
    resident: dict[str, Any],
    model_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = api_client.app.state.services  # type: ignore[attr-defined]
    source_dir = model_root / SOURCE
    calls: list[dict[str, Any]] = []

    async def fake_flush(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        staging = Path(kwargs["staging_dir"])
        for path in source_dir.iterdir():
            shutil.copyfile(path, staging / path.name)
        config = json.loads((source_dir / "config.json").read_text("utf-8"))
        config["quantization_config"] = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        }
        (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (staging / "local_ai_doctor_derivation.json").write_text(
            json.dumps({"schema_version": 1, **kwargs["derivation"]}), encoding="utf-8"
        )
        return {"bytes_written": 123, "files": []}

    monkeypatch.setattr(services.worker, "flush", fake_flush)
    response = _flush(api_client, {"targetRootIndex": 0, "folderName": "Tiny-bnb-nf4"})

    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {"model", "folder", "bytes_written", "derivation"}
    assert body["folder"] == "<model-root:0>/Tiny-bnb-nf4"
    assert body["bytes_written"] == 123
    assert calls[0]["model_key"] == "nf4-key"
    assert calls[0]["source_dir"] == str(source_dir)
    source = _source(api_client)
    assert calls[0]["derivation"] == {
        "source_model_id": source["id"],
        "source_fingerprint": source["fingerprint"]["value"],
        "source_display_name": SOURCE,
    }
    assert body["derivation"]["source_display_name"] == SOURCE
    model = body["model"]
    assert model["path"] == "<model-root>/Tiny-bnb-nf4"
    assert model["root_index"] == 0
    assert model["metadata"]["weight_quantization"]["method"] == "bitsandbytes"
    assert model["derivation"] == body["derivation"]
    listed = api_client.get("/api/v1/models", headers=API_HEADERS).json()["models"]
    assert model["id"] in {item["id"] for item in listed}
    assert [path.name for path in model_root.iterdir() if path.name.startswith(".")] == []
    assert (model_root / "Tiny-bnb-nf4" / "config.json").is_file()

    again = _flush(api_client, {"targetRootIndex": 0, "folderName": "Tiny-bnb-nf4"})
    assert again.json()["error"]["code"] == "target_exists"


def test_load_and_generation_reject_quantization_this_backend_cannot_run(
    api_client: TestClient,
) -> None:
    model_id = _source(api_client)["id"]
    load = api_client.post(
        f"/api/v1/models/{model_id}/load",
        headers=API_HEADERS,
        json={"quantization": "bitsandbytes-8bit"},
    )
    error = load.json()["error"]
    assert (load.status_code, error["code"]) == (409, "capability_unavailable")
    assert "CUDA only" in error["message"]
    assert error["details"]["quantization"] == "bitsandbytes-8bit"

    legacy = api_client.post(
        f"/api/v1/models/{model_id}/load", headers=API_HEADERS, json={"quantization": "int4"}
    )
    assert legacy.status_code == 409
    assert "legacy" in legacy.json()["error"]["message"]
