"""ASGI request-body limits enforced while bytes arrive from the client."""

from __future__ import annotations

import json
from collections.abc import Mapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject oversized JSON and multipart bodies before route parsing completes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        json_bytes: int,
        upload_bytes: int,
        multipart_overhead_bytes: int = 1024 * 1024,
    ) -> None:
        self.app = app
        self.json_bytes = json_bytes
        self.upload_bytes = upload_bytes
        self.multipart_overhead_bytes = multipart_overhead_bytes

    @staticmethod
    def _headers(scope: Scope) -> Mapping[bytes, bytes]:
        return {key.lower(): value for key, value in scope.get("headers", [])}

    def _limit(self, scope: Scope) -> int:
        path = str(scope.get("path", ""))
        content_type = self._headers(scope).get(b"content-type", b"").lower()
        if path in {"/api/v1/attachments", "/api/v1/uploads"} or content_type.startswith(
            b"multipart/form-data"
        ):
            return self.upload_bytes + self.multipart_overhead_bytes
        return self.json_bytes

    @staticmethod
    async def _response(send: Send, status: int, code: str, message: str, limit: int) -> None:
        content = json.dumps(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "retryable": False,
                    "details": {"maximum_bytes": limit},
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(content)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": content})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or str(scope.get("method", "GET")).upper() not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        limit = self._limit(scope)
        raw_length = self._headers(scope).get(b"content-length")
        if raw_length is not None:
            try:
                declared_length = int(raw_length)
            except ValueError:
                await self._response(
                    send, 400, "invalid_request", "Content-Length must be an integer", limit
                )
                return
            if declared_length < 0:
                await self._response(
                    send, 400, "invalid_request", "Content-Length must not be negative", limit
                )
                return
            if declared_length > limit:
                await self._response(
                    send,
                    413,
                    "limit_exceeded",
                    "request body exceeds the configured size limit",
                    limit,
                )
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._response(
                send,
                413,
                "limit_exceeded",
                "request body exceeds the configured size limit",
                limit,
            )


__all__ = ["RequestBodyLimitMiddleware"]
