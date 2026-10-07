from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.config import AppSettings
from local_ai_doctor.main import create_app

INDEX_HTML = "<!doctype html><title>shell</title>"


@pytest.fixture
def spa_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An app whose working directory holds a built frontend, as in a packaged run."""

    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(tmp_path)
    settings = AppSettings.model_validate(
        {
            "active_profile": "test",
            "paths": {
                "model_roots": [tmp_path / "models"],
                "database": tmp_path / "state" / "workbench.sqlite3",
                "uploads": tmp_path / "state" / "uploads",
                "cache": tmp_path / "state" / "cache",
                "exports": tmp_path / "state" / "exports",
                "backups": tmp_path / "state" / "backups",
            },
            "runtime": {"device": "cpu", "cpu_threads": 1},
            "workers": {"shutdown_grace_seconds": 1.0},
        }
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        yield client


def test_framework_http_errors_use_the_error_envelope(spa_client: TestClient) -> None:
    wrong_method = spa_client.post("/api/v1/health")
    assert wrong_method.status_code == 405
    assert wrong_method.json() == {
        "error": {
            "code": "method_not_allowed",
            "message": "method not allowed",
            "retryable": False,
        }
    }
    assert "GET" in wrong_method.headers["allow"]

    unknown_post = spa_client.post("/api/v1/does-not-exist", json={})
    assert unknown_post.status_code == 405
    assert unknown_post.json()["error"]["code"] == "method_not_allowed"

    unknown_get = spa_client.get("/api/v1/does-not-exist")
    assert unknown_get.status_code == 404
    assert unknown_get.json()["error"]["code"] == "not_found"

    missing_asset = spa_client.get("/assets/missing.js")
    assert missing_asset.status_code == 404
    assert missing_asset.json()["error"]["code"] == "not_found"


def test_spa_fallback_still_serves_the_shell_for_client_side_routes(spa_client: TestClient) -> None:
    for path in ("/", "/chats/some-chat", "/settings/models"):
        response = spa_client.get(path)
        assert response.status_code == 200
        assert response.text == INDEX_HTML
