from __future__ import annotations

import base64

import pytest

from local_ai_doctor.config import AppSettings, ServerSettings
from local_ai_doctor.main import (
    _auth_valid,
    _hostname,
    _is_same_origin,
    _websocket_protocol_token,
)


@pytest.mark.parametrize(
    ("origin", "scheme", "netloc"),
    [
        ("http://localhost", "http", "LOCALHOST:80"),
        ("https://example.test", "https", "example.test:443"),
        ("http://[::1]:8000", "ws", "[::1]:8000"),
        ("https://example.test:8443", "wss", "example.test:8443"),
    ],
)
def test_same_origin_normalizes_ws_aliases_case_and_default_ports(
    origin: str, scheme: str, netloc: str
) -> None:
    assert _is_same_origin(origin, scheme=scheme, netloc=netloc)


@pytest.mark.parametrize(
    ("origin", "scheme", "netloc"),
    [
        ("https://example.test", "http", "example.test"),
        ("http://example.test:8001", "http", "example.test:8000"),
        ("http://user@example.test", "http", "example.test"),
        ("http://example.test/path", "http", "example.test"),
    ],
)
def test_same_origin_rejects_scheme_port_credentials_and_path_mismatches(
    origin: str, scheme: str, netloc: str
) -> None:
    assert not _is_same_origin(origin, scheme=scheme, netloc=netloc)


@pytest.mark.parametrize(
    "netloc",
    [
        "attacker.invalid@127.0.0.1",
        "127.0.0.1:8000:bad",
        "127.0.0.1:",
        "127.0.0.1/path",
        "[::1]garbage",
    ],
)
def test_hostname_rejects_ambiguous_or_malformed_authorities(netloc: str) -> None:
    assert _hostname(netloc) is None


def test_websocket_subprotocol_auth_decodes_utf8_without_accepting_malformed_values() -> None:
    token = "correct-horse-battery-staple"
    encoded = base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")
    settings = AppSettings(server=ServerSettings(authentication_token=token))

    assert _websocket_protocol_token(f"lad.events.v1, lad.auth.{encoded}") == token
    assert _auth_valid(settings, None, token)
    assert _websocket_protocol_token("lad.events.v1, lad.auth.!!!") is None
    assert not _auth_valid(settings, None, "wrong-token")
