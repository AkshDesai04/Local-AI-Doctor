from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from local_ai_doctor.config import (
    AppSettings,
    PathSettings,
    ProfileName,
    ServerSettings,
    SettingsLoader,
    migrate_config,
)
from local_ai_doctor.errors import ConfigurationError


def test_configuration_precedence_and_relative_path_resolution(tmp_path: Path) -> None:
    portable = tmp_path / "config.json"
    portable.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_profile": "test",
                "defaults": {"server": {"port": 7000}, "runtime": {"cpu_threads": 2}},
                "profiles": {"test": {"server": {"port": 7001}}},
            }
        ),
        encoding="utf-8",
    )
    user = tmp_path / "local.json"
    user.write_text(
        json.dumps({"schema_version": 1, "defaults": {"server": {"port": 7002}}}),
        encoding="utf-8",
    )

    settings = SettingsLoader().load(
        default_path=portable,
        user_path=user,
        environ={"LAD_SERVER__PORT": "7003", "LAD_RUNTIME__CPU_THREADS": "3"},
        cli_overrides={"server": {"port": 7004}},
    )

    assert settings.active_profile is ProfileName.TEST
    assert settings.server.port == 7004
    assert settings.runtime.cpu_threads == 3
    assert settings.paths.database.is_absolute()
    assert settings.paths.database == (tmp_path / "data/test.sqlite3").resolve()


def test_cli_hook_parses_nested_typed_values() -> None:
    options = SettingsLoader.parse_cli(
        ["--profile", "production", "--set", "runtime.device=cpu", "--set", "runtime.cpu_threads=4"]
    )
    assert options.profile is ProfileName.PRODUCTION
    assert options.overrides == {"runtime": {"device": "cpu", "cpu_threads": 4}}


def test_environment_unknown_key_has_actionable_validation_error() -> None:
    with pytest.raises(ConfigurationError) as caught:
        SettingsLoader().load(environ={"LAD_RUNTIME__NOT_A_FIELD": "true"})
    payload = caught.value.to_dict()
    assert payload["code"] == "configuration_invalid"
    assert payload["details"]["issues"][0]["field"] == "runtime.not_a_field"


def test_non_loopback_binding_requires_opt_in_and_authentication() -> None:
    with pytest.raises(ConfigurationError) as caught:
        SettingsLoader().load(environ={"LAD_SERVER__HOST": "0.0.0.0"})
    assert "external" in caught.value.to_dict()["details"]["issues"][0]["message"]


@pytest.mark.parametrize("token", ["", " ", "contains whitespace"])
def test_authentication_token_must_be_usable_as_a_bearer_credential(token: str) -> None:
    with pytest.raises(ValueError, match="non-empty and contain no whitespace"):
        ServerSettings(authentication_token=SecretStr(token))


@pytest.mark.parametrize(
    "origin",
    ["*", "ftp://example.test", "https://user@example.test", "https://example.test/path"],
)
def test_allowed_origins_are_exact_http_origins(origin: str) -> None:
    with pytest.raises(ValueError, match="origin"):
        ServerSettings(allowed_origins=(origin,))


def test_redacted_config_and_inference_snapshot_hide_sensitive_values(tmp_path: Path) -> None:
    settings = AppSettings(
        paths=PathSettings(model_roots=(tmp_path / "private-models",)),
        server=ServerSettings(
            host="0.0.0.0",
            allow_external_access=True,
            authentication_token=SecretStr("do-not-leak"),
            allowed_origins=("https://example.test",),
        ),
    )
    redacted = settings.redacted_effective_config()
    assert redacted["server"]["authentication_token"] == "<redacted-secret>"
    assert "private-models" not in json.dumps(redacted)
    snapshot = settings.inference_snapshot()
    assert len(snapshot["configuration_digest"]) == 64
    assert "server" not in snapshot and "paths" not in snapshot


def test_schema_zero_migration_is_explicit_and_non_mutating() -> None:
    source = {"models_path": "somewhere", "database_path": "state.db"}
    result = migrate_config(source)
    assert source == {"models_path": "somewhere", "database_path": "state.db"}
    assert result["schema_version"] == 1
    assert result["paths"] == {"model_roots": ["somewhere"], "database": "state.db"}
