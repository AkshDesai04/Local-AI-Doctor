from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from local_ai_doctor.api.body_limit import RequestBodyLimitMiddleware


def _scope(*, headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/runs",
        "raw_path": b"/api/v1/runs",
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    }


@pytest.mark.asyncio
async def test_body_limit_rejects_declared_oversize_without_reading() -> None:
    app_called = False

    async def app(_scope: Any, _receive: Any, _send: Any) -> None:
        nonlocal app_called
        app_called = True

    receive_called = False

    async def receive() -> dict[str, Any]:
        nonlocal receive_called
        receive_called = True
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(app, json_bytes=8, upload_bytes=8)
    await middleware(
        _scope(headers=[(b"content-length", b"9"), (b"content-type", b"application/json")]),
        receive,
        send,
    )

    assert app_called is False
    assert receive_called is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "limit_exceeded"


@pytest.mark.asyncio
async def test_body_limit_counts_streamed_chunks_without_content_length() -> None:
    messages = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]
    )

    async def receive() -> dict[str, Any]:
        return next(messages)

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def consuming_app(
        _scope: Any,
        receive_message: Callable[[], Awaitable[dict[str, Any]]],
        _send: Any,
    ) -> None:
        while (await receive_message()).get("more_body"):
            pass

    middleware = RequestBodyLimitMiddleware(consuming_app, json_bytes=8, upload_bytes=8)
    await middleware(_scope(), receive, send)

    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["details"]["maximum_bytes"] == 8
