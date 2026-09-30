from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

import local_ai_doctor.config as config_module
from local_ai_doctor.config import (
    AppSettings,
    PathSettings,
    ProfileName,
    ServerSettings,
    SettingsLoader,
    migrate_config,
    persist_user_model_roots,
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


def test_model_root_update_persists_in_selected_user_profile(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    user = tmp_path / "local.yaml"
    user.write_text(
        "schema_version: 1\nprofiles:\n  native-windows:\n    runtime:\n      cpu_threads: 3\n",
        encoding="utf-8",
    )

    persist_user_model_roots(user, ProfileName.NATIVE_WINDOWS, (model_root,))

    saved = SettingsLoader().load(
        user_path=user,
        profile=ProfileName.NATIVE_WINDOWS,
        environ={},
    )
    assert saved.paths.model_roots == (model_root.resolve(),)
    assert saved.runtime.cpu_threads == 3


@pytest.mark.parametrize("replacement_errno", [errno.EACCES, errno.EPERM, errno.EBUSY])
def test_model_root_update_falls_back_when_bind_mount_replacement_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_errno: int,
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    user = tmp_path / "local.yaml"
    user.write_text("schema_version: 1\n", encoding="utf-8")

    def deny_replace(_source: Path, target: Path) -> None:
        raise PermissionError(replacement_errno, "bind mount denied replacement", str(target))

    monkeypatch.setattr(config_module.os, "replace", deny_replace)

    persist_user_model_roots(user, ProfileName.NATIVE_WINDOWS, (model_root,))

    saved = SettingsLoader().load(
        user_path=user,
        profile=ProfileName.NATIVE_WINDOWS,
        environ={},
    )
    assert saved.paths.model_roots == (model_root.resolve(),)


@pytest.mark.parametrize("chmod_errno", [errno.EACCES, errno.EPERM, errno.EROFS])
def test_model_root_update_tolerates_unsupported_bind_mount_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chmod_errno: int,
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    user = tmp_path / "local.yaml"
    user.write_text("schema_version: 1\n", encoding="utf-8")

    def deny_chmod(path: Path, _mode: int) -> None:
        raise PermissionError(chmod_errno, "bind mount does not support chmod", str(path))

    monkeypatch.setattr(config_module.os, "chmod", deny_chmod)

    persist_user_model_roots(user, ProfileName.NATIVE_WINDOWS, (model_root,))

    saved = SettingsLoader().load(
        user_path=user,
        profile=ProfileName.NATIVE_WINDOWS,
        environ={},
    )
    assert saved.paths.model_roots == (model_root.resolve(),)


def test_model_root_update_tolerates_combined_bind_mount_denials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    user = tmp_path / "local.yaml"
    user.write_text("schema_version: 1\n", encoding="utf-8")

    def deny_chmod(path: Path, _mode: int) -> None:
        raise PermissionError(errno.EPERM, "bind mount does not support chmod", str(path))

    def deny_replace(_source: Path, target: Path) -> None:
        raise PermissionError(errno.EACCES, "bind mount denied replacement", str(target))

    monkeypatch.setattr(config_module.os, "chmod", deny_chmod)
    monkeypatch.setattr(config_module.os, "replace", deny_replace)

    persist_user_model_roots(user, ProfileName.NATIVE_WINDOWS, (model_root,))

    saved = SettingsLoader().load(
        user_path=user,
        profile=ProfileName.NATIVE_WINDOWS,
        environ={},
    )
    assert saved.paths.model_roots == (model_root.resolve(),)


def test_unload_timeout_is_separate_from_the_shutdown_grace() -> None:
    assert AppSettings().workers.unload_timeout_seconds == 60.0
    assert AppSettings().workers.shutdown_grace_seconds == 15.0

    portable = Path(__file__).parents[2] / "config" / "default.yaml"
    loaded = SettingsLoader().load(default_path=portable, environ={})
    assert loaded.workers.unload_timeout_seconds == 60.0

    override = AppSettings.model_validate({"workers": {"unload_timeout_seconds": 5}})
    assert override.workers.unload_timeout_seconds == 5.0
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"workers": {"unload_timeout_seconds": 0.5}})


def test_runtime_defaults_allow_several_residents_under_strict_vram() -> None:
    runtime = AppSettings().runtime

    assert runtime.strict_vram is True
    assert runtime.vram_safety_margin_bytes == 512 * 1024**2
    assert runtime.kv_reserve_tokens == 4096
    assert runtime.load_one_model_at_a_time is False
    assert runtime.max_loaded_models == 4
    assert runtime.max_concurrent_runs == 2
    snapshot = AppSettings().inference_snapshot()["runtime"]
    assert snapshot["strict_vram"] is True
    assert snapshot["vram_safety_margin_bytes"] == 512 * 1024**2


def test_runtime_rejects_superseded_cpu_offload_and_incoherent_limits() -> None:
    with pytest.raises(ValidationError, match="superseded by runtime.strict_vram=false"):
        AppSettings.model_validate({"runtime": {"cpu_offload": True}})
    with pytest.raises(ValidationError, match="max_loaded_models must be 1"):
        AppSettings.model_validate(
            {"runtime": {"load_one_model_at_a_time": True, "max_loaded_models": 2}}
        )
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"runtime": {"vram_safety_margin_bytes": 17 * 1024**3}})
    assert AppSettings.model_validate({"runtime": {"cpu_offload": False}}).runtime.strict_vram
