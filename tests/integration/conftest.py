from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.config import AppSettings
from local_ai_doctor.main import create_app

API_TOKEN = "integration-test-token"
API_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


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
    root = tmp_path_factory.mktemp("api-workspace")
    model_root = root / "models"
    model = model_root / "Tiny-Generation-Model"
    model.mkdir(parents=True)
    _write_json(
        model / "config.json",
        {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "max_position_embeddings": 1024,
        },
    )
    _write_json(model / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"x": 0}}})
    _write_json(model / "tokenizer_config.json", {"model_max_length": 1024})
    _write_tiny_safetensors(model / "model.safetensors")

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
            "server": {
                "host": "127.0.0.1",
                "allowed_origins": ["http://vite.test"],
                "authentication_token": API_TOKEN,
            },
            "runtime": {"device": "cpu", "cpu_threads": 1, "queue_limit": 2},
            "workers": {"shutdown_grace_seconds": 1.0},
            # Keep batches pending long enough for the reconnect ordering test
            # to prove that terminal events explicitly establish a barrier.
            "telemetry": {"write_batch_size": 128, "write_flush_interval_ms": 60_000},
        }
    )


@pytest.fixture(scope="module")
def api_client(api_settings: AppSettings) -> Iterator[TestClient]:
    with TestClient(create_app(api_settings), base_url="http://127.0.0.1") as client:
        yield client
