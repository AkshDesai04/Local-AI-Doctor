"""Central, typed application configuration and precedence-aware loading.

Precedence, from lowest to highest, is:

1. model defaults;
2. the portable configuration's ``defaults`` section;
3. the selected built-in profile;
4. the selected profile in the portable configuration;
5. the user-local configuration (defaults, then selected profile);
6. ``LAD_`` environment variables using ``__`` for nesting; and
7. command-line ``--set dotted.path=value`` overrides.

No other core module reads process environment variables directly.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import hashlib
import ipaddress
import json
import os
import stat
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import ConfigurationError, ConfigurationIOError, ConfigurationVersionError

CURRENT_SCHEMA_VERSION: Final[int] = 1
ENV_PREFIX: Final[str] = "LAD_"


class ProfileName(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    NATIVE_WINDOWS = "native-windows"
    NATIVE_WSL = "native-wsl"
    CONTAINER_CPU = "container-cpu"
    CONTAINER_NVIDIA = "container-nvidia"
    PRODUCTION = "production"


class DeviceMode(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DType(StrEnum):
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class Quantization(StrEnum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    BITSANDBYTES_8BIT = "bitsandbytes-8bit"
    BITSANDBYTES_4BIT = "bitsandbytes-4bit"


class AttentionBackend(StrEnum):
    AUTO = "auto"
    EAGER = "eager"
    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash-attention-2"


class InferenceBackend(StrEnum):
    AUTO = "auto"
    REFERENCE = "reference"
    TRANSFORMERS = "transformers"


class InstrumentationLevel(StrEnum):
    OFF = "off"
    BASIC = "basic"
    TOKEN = "token"
    FULL = "full"
    EXPERT = "expert"


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_roots: tuple[Path, ...] = (Path("models"),)
    database: Path = Path("data/workbench.sqlite3")
    uploads: Path = Path("data/uploads")
    cache: Path = Path("data/cache")
    exports: Path = Path("data/exports")
    backups: Path = Path("data/backups")

    @field_validator("model_roots")
    @classmethod
    def require_model_root(cls, roots: tuple[Path, ...]) -> tuple[Path, ...]:
        if not roots:
            raise ValueError("configure at least one model root")
        if len({str(path) for path in roots}) != len(roots):
            raise ValueError("model roots must not contain duplicates")
        return roots


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    allowed_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    allow_external_access: bool = False
    authentication_token: SecretStr | None = None
    csrf_protection: bool = True

    @field_validator("authentication_token")
    @classmethod
    def require_usable_authentication_token(cls, token: SecretStr | None) -> SecretStr | None:
        if token is None:
            return None
        value = token.get_secret_value()
        if not value or any(character.isspace() or ord(character) < 0x20 for character in value):
            raise ValueError("authentication token must be non-empty and contain no whitespace")
        return token

    @field_validator("allowed_origins")
    @classmethod
    def require_exact_http_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(origins)) != len(origins):
            raise ValueError("allowed origins must not contain duplicates")
        for origin in origins:
            if origin == "*":
                raise ValueError("wildcard CORS origins are not allowed")
            parsed = urlsplit(origin)
            try:
                _ = parsed.port
            except ValueError as exc:
                raise ValueError(f"invalid allowed origin: {origin!r}") from exc
            if (
                parsed.scheme.casefold() not in {"http", "https"}
                or parsed.hostname is None
                or parsed.netloc.endswith(":")
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "allowed origins must be exact HTTP(S) origins without paths or credentials"
                )
        return origins

    @model_validator(mode="after")
    def external_binding_requires_explicit_security(self) -> ServerSettings:
        host = self.host.strip().lower()
        loopback = host == "localhost"
        with contextlib.suppress(ValueError):
            loopback = loopback or ipaddress.ip_address(host).is_loopback
        if not loopback and not self.allow_external_access:
            raise ValueError(
                "non-loopback host requires server.allow_external_access=true; "
                "use 127.0.0.1 for local-only access"
            )
        if not loopback and self.authentication_token is None:
            raise ValueError("non-loopback host requires server.authentication_token")
        return self


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: InferenceBackend = InferenceBackend.AUTO
    device: DeviceMode = DeviceMode.AUTO
    allow_cpu_fallback: bool = True
    cpu_threads: int = Field(default=max(1, os.cpu_count() or 1), ge=1, le=1024)
    ram_budget_bytes: int | None = Field(default=None, ge=256 * 1024**2)
    vram_budget_bytes: int | None = Field(default=None, ge=256 * 1024**2)
    low_memory_loading: bool = True
    # Superseded by strict_vram=false; kept so existing files that say `false` still load.
    cpu_offload: bool = False
    device_placement: str = "sequential"
    dtype: DType = DType.AUTO
    quantization: Quantization = Quantization.NONE
    attention_backend: AttentionBackend = AttentionBackend.AUTO
    # Strict VRAM never lets weights spill into system RAM: a load that does not fit
    # fails instead of offloading layers. Each load may override it.
    strict_vram: bool = True
    vram_safety_margin_bytes: int = Field(default=512 * 1024**2, ge=0, le=16 * 1024**3)
    kv_reserve_tokens: int = Field(default=4096, ge=0, le=10_000_000)
    load_one_model_at_a_time: bool = False
    max_loaded_models: int = Field(default=4, ge=1, le=32)
    max_batch_size: int = Field(default=1, ge=1, le=1024)
    max_concurrent_runs: int = Field(default=2, ge=1, le=256)
    queue_limit: int = Field(default=32, ge=0, le=100_000)

    @model_validator(mode="after")
    def sequential_loading_is_coherent(self) -> RuntimeSettings:
        if self.load_one_model_at_a_time and self.max_loaded_models != 1:
            raise ValueError(
                "runtime.max_loaded_models must be 1 when "
                "runtime.load_one_model_at_a_time is enabled"
            )
        if self.cpu_offload:
            raise ValueError("runtime.cpu_offload is superseded by runtime.strict_vram=false")
        return self


class SamplingDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int = Field(default=512, ge=1, le=100_000)
    temperature: float = Field(default=0.7, ge=0.0, le=10.0)
    top_k: int = Field(default=50, ge=0, le=1_000_000)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=10.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    alternatives: int = Field(default=10, ge=0, le=1000)


class InferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conservative_context_limit: int = Field(default=4096, ge=128, le=10_000_000)
    reserved_output_tokens: int = Field(default=512, ge=1, le=1_000_000)
    max_prompt_tokens: int = Field(default=32_768, ge=1, le=10_000_000)
    # Gradient x input keeps the whole prefix's activations for one backward pass.
    influence_max_gradient_tokens: int = Field(default=2048, ge=16, le=1_000_000)
    defaults: SamplingDefaults = Field(default_factory=SamplingDefaults)
    deterministic_reference_mode: bool = False
    instrumentation: InstrumentationLevel = InstrumentationLevel.TOKEN
    local_files_only: bool = True
    trust_remote_code: bool = False

    @field_validator("trust_remote_code")
    @classmethod
    def prohibit_global_remote_code(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "global trust_remote_code is prohibited; reviewed bundled code must be "
                "enabled narrowly by a model-specific isolated adapter"
            )
        return value


class LimitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_bytes: int = Field(default=100 * 1024**2, ge=1024, le=100 * 1024**3)
    prompt_bytes: int = Field(default=4 * 1024**2, ge=1024, le=1024**3)
    trace_bytes_per_run: int = Field(default=128 * 1024**2, ge=1024, le=100 * 1024**3)
    attachment_count: int = Field(default=16, ge=0, le=10_000)
    telemetry_events_per_run: int = Field(default=100_000, ge=1, le=100_000_000)
    image_pixels: int = Field(default=40_000_000, ge=1, le=1_000_000_000)
    video_frames: int = Field(default=256, ge=1, le=100_000)
    video_frame_pixels: int = Field(default=8_500_000, ge=1, le=1_000_000_000)
    decoded_media_pixels: int = Field(default=500_000_000, ge=1, le=100_000_000_000)
    media_duration_seconds: float = Field(default=600.0, gt=0.0, le=86_400.0)


class WorkerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=1, ge=1, le=128)
    startup_timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    load_timeout_seconds: float = Field(default=600.0, gt=0.0, le=86_400.0)
    unload_timeout_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)
    inference_timeout_seconds: float = Field(default=3600.0, gt=0.0, le=7 * 86_400.0)
    shutdown_grace_seconds: float = Field(default=15.0, ge=0.0, le=600.0)


class TelemetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retention_days: int = Field(default=30, ge=0, le=3650)
    persist_token_events: bool = True
    persist_router_traces: bool = False
    router_trace_token_limit: int = Field(default=256, ge=0, le=1_000_000)
    hardware_sample_interval_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    write_batch_size: int = Field(default=128, ge=1, le=100_000)
    write_flush_interval_ms: int = Field(default=100, ge=1, le=60_000)


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    level: LogLevel = LogLevel.INFO
    json_output: bool = Field(default=True, alias="json", serialization_alias="json")
    redact_paths: bool = True
    log_prompts: bool = False
    log_model_output: bool = False

    @model_validator(mode="after")
    def private_content_is_never_logged(self) -> LoggingSettings:
        if self.log_prompts or self.log_model_output:
            raise ValueError("prompts and model output must not be written to normal logs")
        return self


class FeatureSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attention_probe: bool = False
    hidden_state_probe: bool = False
    activation_probe: bool = False
    logit_lens: bool = False
    full_router_traces: bool = False
    multi_gpu: bool = False


class PlatformSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wsl_distribution: str | None = None
    container_model_root: Path = Path("/models")
    container_data_root: Path = Path("/data")
    container_gpu_profile: bool = False


class AppSettings(BaseModel):
    """Single validated source of runtime configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = CURRENT_SCHEMA_VERSION
    active_profile: ProfileName = ProfileName.DEVELOPMENT
    paths: PathSettings = Field(default_factory=PathSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    platform: PlatformSettings = Field(default_factory=PlatformSettings)

    @field_validator("schema_version")
    @classmethod
    def current_version_only(cls, value: int) -> int:
        if value != CURRENT_SCHEMA_VERSION:
            raise ValueError(f"expected migrated schema version {CURRENT_SCHEMA_VERSION}")
        return value

    def redacted_effective_config(self) -> dict[str, Any]:
        """Return a stable view suitable for the UI and support bundles."""

        data = self.model_dump(mode="json", by_alias=True)
        paths = data.get("paths", {})
        for key, value in tuple(paths.items()):
            if isinstance(value, list):
                paths[key] = [f"<redacted-path:{key}:{index}>" for index, _ in enumerate(value)]
            else:
                paths[key] = f"<redacted-path:{key}>"
        platform = data.get("platform", {})
        if platform.get("wsl_distribution"):
            platform["wsl_distribution"] = "<redacted-local-identifier>"
        for key in ("container_model_root", "container_data_root"):
            if key in platform:
                platform[key] = f"<redacted-path:{key}>"
        server = data.get("server", {})
        if server.get("authentication_token") is not None:
            server["authentication_token"] = "<redacted-secret>"
        return data

    def inference_snapshot(self) -> dict[str, Any]:
        """Configuration subset that can materially affect an inference run."""

        snapshot = {
            "schema_version": self.schema_version,
            "active_profile": self.active_profile.value,
            "runtime": self.runtime.model_dump(mode="json"),
            "inference": self.inference.model_dump(mode="json"),
            "limits": {
                "prompt_bytes": self.limits.prompt_bytes,
                "trace_bytes_per_run": self.limits.trace_bytes_per_run,
                "telemetry_events_per_run": self.limits.telemetry_events_per_run,
            },
            "features": self.features.model_dump(mode="json"),
        }
        canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        snapshot["configuration_digest"] = hashlib.sha256(canonical).hexdigest()
        return snapshot


BUILTIN_PROFILES: Final[dict[ProfileName, dict[str, Any]]] = {
    ProfileName.DEVELOPMENT: {
        "logging": {"level": "debug", "json": False},
        "telemetry": {"retention_days": 7},
    },
    ProfileName.TEST: {
        "paths": {
            "model_roots": ["tests/fixtures/models"],
            "database": "data/test.sqlite3",
            "uploads": "data/test-uploads",
            "cache": "data/test-cache",
            "exports": "data/test-exports",
            "backups": "data/test-backups",
        },
        "runtime": {"device": "cpu", "cpu_threads": 1},
        "inference": {"instrumentation": "full", "conservative_context_limit": 1024},
        "logging": {"level": "warning", "json": False},
        "telemetry": {"retention_days": 0},
    },
    ProfileName.NATIVE_WINDOWS: {"runtime": {"device": "auto"}},
    ProfileName.NATIVE_WSL: {"runtime": {"device": "auto"}},
    ProfileName.CONTAINER_CPU: {
        "paths": {
            "model_roots": ["/models"],
            "database": "/data/workbench.sqlite3",
            "uploads": "/data/uploads",
            "cache": "/data/cache",
            "exports": "/data/exports",
            "backups": "/data/backups",
        },
        "runtime": {"device": "cpu", "allow_cpu_fallback": True},
        "logging": {"json": True},
    },
    ProfileName.CONTAINER_NVIDIA: {
        "paths": {
            "model_roots": ["/models"],
            "database": "/data/workbench.sqlite3",
            "uploads": "/data/uploads",
            "cache": "/data/cache",
            "exports": "/data/exports",
            "backups": "/data/backups",
        },
        "runtime": {"device": "cuda", "allow_cpu_fallback": False},
        "platform": {"container_gpu_profile": True},
        "logging": {"json": True},
    },
    ProfileName.PRODUCTION: {
        "logging": {"level": "info", "json": True},
        "telemetry": {"retention_days": 30},
    },
}


class CLIOptions(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config: Path | None = None
    user_config: Path | None = None
    profile: ProfileName | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        current = base.get(key)
        if isinstance(current, MutableMapping) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            base[key] = copy.deepcopy(value)


def _parse_scalar(value: str) -> Any:
    """Parse a safe YAML/JSON-like scalar while preserving ordinary strings."""

    try:
        import yaml

        parsed = yaml.safe_load(value)
    except ImportError:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
    return parsed


def _set_nested(target: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    if not path or any(not segment for segment in path):
        raise ConfigurationError("override key contains an empty path segment")
    cursor = target
    for segment in path[:-1]:
        existing = cursor.get(segment)
        if existing is None:
            child: dict[str, Any] = {}
            cursor[segment] = child
            cursor = child
        elif isinstance(existing, MutableMapping):
            cursor = existing
        else:
            raise ConfigurationError(
                f"override cannot descend through non-object key {segment!r}",
                hint="Check the nested override name and use double underscores for environment variables.",
            )
    cursor[path[-1]] = value


def environment_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reserved = {f"{ENV_PREFIX}PROFILE", f"{ENV_PREFIX}CONFIG", f"{ENV_PREFIX}USER_CONFIG"}
    for key, raw_value in environ.items():
        if not key.startswith(ENV_PREFIX) or key in reserved:
            continue
        nested = tuple(part.lower() for part in key[len(ENV_PREFIX) :].split("__"))
        _set_nested(result, nested, _parse_scalar(raw_value))
    return result


def parse_key_value_overrides(values: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        key, separator, raw_value = item.partition("=")
        if not separator or not key.strip():
            raise ConfigurationError(
                f"invalid --set override {item!r}",
                hint="Use --set section.field=value, for example --set runtime.device=cpu.",
            )
        _set_nested(
            result, tuple(part.strip() for part in key.split(".")), _parse_scalar(raw_value)
        )
    return result


def migrate_config(document: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a configuration document without mutating the caller's data."""

    migrated = copy.deepcopy(dict(document))
    version = migrated.get("schema_version", 0)
    if not isinstance(version, int) or version < 0:
        raise ConfigurationVersionError("schema_version must be a non-negative integer")
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigurationVersionError(
            f"configuration schema {version} is newer than supported schema {CURRENT_SCHEMA_VERSION}",
            hint="Upgrade Local AI Doctor before using this configuration.",
        )
    if version == 0:
        # The pre-release flat keys are the only historical shape.  Retaining a
        # real migration makes schema evolution testable instead of ceremonial.
        paths = migrated.setdefault("paths", {})
        if "models_path" in migrated:
            paths.setdefault("model_roots", [migrated.pop("models_path")])
        if "database_path" in migrated:
            paths.setdefault("database", migrated.pop("database_path"))
        migrated["schema_version"] = 1
    return migrated


def _read_document(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationIOError(
            f"could not read configuration file {path.name!r}",
            hint="Verify that the file exists and is readable.",
            details={"filename": path.name, "reason": type(exc).__name__},
        ) from exc
    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            parsed = json.loads(raw)
        elif suffix == ".toml":
            parsed = tomllib.loads(raw.decode("utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationIOError(
                    "YAML configuration requires PyYAML",
                    hint="Install the backend dependencies or use JSON/TOML configuration.",
                ) from exc
            parsed = yaml.safe_load(raw) or {}
        else:
            raise ConfigurationIOError(
                f"unsupported configuration extension {suffix or '<none>'}",
                hint="Use .yaml, .yml, .json, or .toml.",
            )
    except (UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationIOError(
            f"configuration file {path.name!r} is malformed",
            details={"filename": path.name, "reason": type(exc).__name__},
        ) from exc
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        # PyYAML uses library-specific parse exceptions.  Keep them behind the
        # same structured boundary without returning their path-rich messages.
        raise ConfigurationIOError(
            f"configuration file {path.name!r} is malformed",
            details={"filename": path.name, "reason": type(exc).__name__},
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ConfigurationError("configuration root must be an object/mapping")
    return migrate_config(parsed)


def _serialize_document(path: Path, document: Mapping[str, Any]) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationIOError(
                "YAML configuration requires PyYAML",
                hint="Install the backend dependencies or use a JSON user configuration.",
            ) from exc
        return yaml.safe_dump(
            dict(document),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).encode("utf-8")
    raise ConfigurationError(
        "this user configuration format cannot be edited from the workbench",
        hint="Use a .yaml, .yml, or .json user-local configuration file.",
    )


def persist_user_model_roots(
    path: Path,
    profile: ProfileName,
    roots: Sequence[Path],
) -> None:
    """Persist model roots in the selected profile of the user-local wrapper.

    Writes use a same-directory temporary file and replacement so a crash cannot
    leave a partially serialized configuration. Some bind-mounted filesystems
    reject replacement with ``EBUSY``, ``EACCES``, or ``EPERM`` even when the
    existing file itself is writable; for those cases we overwrite and fsync the
    mounted file after the complete payload has been prepared.
    """

    target = path.expanduser().resolve(strict=False)
    document: dict[str, Any] = _read_document(target) if target.is_file() else {"schema_version": 1}
    profiles = document.setdefault("profiles", {})
    if not isinstance(profiles, MutableMapping):
        raise ConfigurationError("configuration profiles must be an object/mapping")
    selected = profiles.setdefault(profile.value, {})
    if not isinstance(selected, MutableMapping):
        raise ConfigurationError(f"profile {profile.value!r} must be an object/mapping")
    paths = selected.setdefault("paths", {})
    if not isinstance(paths, MutableMapping):
        raise ConfigurationError(f"profile {profile.value!r} paths must be an object/mapping")
    paths["model_roots"] = [str(root) for root in roots]
    document["schema_version"] = CURRENT_SCHEMA_VERSION
    payload = _serialize_document(target, document)

    def overwrite_mounted_file() -> None:
        with target.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EROFS} or not target.is_file():
                raise
            # A read-only container root can still contain one explicitly
            # writable file bind mount, while rejecting sibling temp files.
            overwrite_mounted_file()
            return
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, existing_mode)
        except OSError as exc:
            chmod_unsupported = exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}
            if not chmod_unsupported:
                raise
            # Windows-backed Docker bind mounts can permit file creation and
            # replacement while rejecting chmod. Their host-side permission
            # model remains authoritative, so continue with the writable file.
        try:
            os.replace(temporary, target)
            temporary = None
        except OSError as exc:
            replacement_denied = exc.errno in {errno.EBUSY, errno.EACCES, errno.EPERM}
            if not replacement_denied or not target.is_file():
                raise
            # Docker Desktop bind mounts backed by Windows/WSL can reject
            # rename-over-existing even when both the file and directory are
            # writable. Keep the non-atomic fallback limited to that condition.
            overwrite_mounted_file()
    except ConfigurationError:
        raise
    except OSError as exc:
        raise ConfigurationIOError(
            "could not update the user-local configuration file",
            hint="Verify that the configured user-local file and its directory are writable.",
            details={"filename": target.name, "reason": type(exc).__name__},
        ) from exc
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


def _split_document(
    document: Mapping[str, Any], profile: ProfileName
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "defaults" in document:
        defaults = document.get("defaults", {})
        if not isinstance(defaults, Mapping):
            raise ConfigurationError("configuration defaults must be an object/mapping")
    else:
        defaults = {
            key: value
            for key, value in document.items()
            if key not in {"profiles", "active_profile"}
        }
    profiles = document.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ConfigurationError("configuration profiles must be an object/mapping")
    selected = profiles.get(profile.value, {})
    if not isinstance(selected, Mapping):
        raise ConfigurationError(f"profile {profile.value!r} must be an object/mapping")
    return dict(defaults), dict(selected)


def _resolve_paths(data: MutableMapping[str, Any], base_dir: Path) -> None:
    path_settings = data.get("paths")
    if isinstance(path_settings, MutableMapping):
        roots = path_settings.get("model_roots")
        if isinstance(roots, (list, tuple)):
            path_settings["model_roots"] = [
                str((base_dir / Path(item)).resolve())
                if not Path(item).is_absolute()
                else str(Path(item))
                for item in roots
            ]
        for key in ("database", "uploads", "cache", "exports", "backups"):
            item = path_settings.get(key)
            if item is not None and not Path(item).is_absolute():
                path_settings[key] = str((base_dir / Path(item)).resolve())


def _validation_error(exc: ValidationError) -> ConfigurationError:
    issues: list[dict[str, Any]] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        issues.append({"field": location, "message": error.get("msg", "invalid value")})
    return ConfigurationError(
        "effective configuration is invalid",
        hint="Correct the listed fields in the highest-precedence configuration source.",
        details={"issues": issues},
    )


class SettingsLoader:
    """Pure loader with explicit environment and argv inputs for reliable tests."""

    @staticmethod
    def parse_cli(argv: Sequence[str]) -> CLIOptions:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--config", type=Path)
        parser.add_argument("--user-config", type=Path)
        parser.add_argument("--profile", choices=[profile.value for profile in ProfileName])
        parser.add_argument("--set", dest="sets", action="append", default=[])
        try:
            namespace, _unknown = parser.parse_known_args(list(argv))
        except SystemExit as exc:
            raise ConfigurationError(
                "invalid configuration command-line arguments",
                hint="Check --profile and --set values.",
            ) from exc
        return CLIOptions(
            config=namespace.config,
            user_config=namespace.user_config,
            profile=ProfileName(namespace.profile) if namespace.profile else None,
            overrides=parse_key_value_overrides(namespace.sets),
        )

    def load(
        self,
        *,
        default_path: Path | None = None,
        user_path: Path | None = None,
        profile: ProfileName | str | None = None,
        environ: Mapping[str, str] | None = None,
        cli_overrides: Mapping[str, Any] | None = None,
        base_dir: Path | None = None,
    ) -> AppSettings:
        environment = dict(os.environ if environ is None else environ)
        default_document = _read_document(default_path) if default_path else {}
        profile_value = (
            profile
            or environment.get(f"{ENV_PREFIX}PROFILE")
            or default_document.get("active_profile")
            or ProfileName.DEVELOPMENT
        )
        try:
            selected_profile = ProfileName(profile_value)
        except ValueError as exc:
            raise ConfigurationError(
                f"unknown profile {profile_value!r}",
                details={"available_profiles": [item.value for item in ProfileName]},
            ) from exc

        # Materialize model defaults first so relative default paths are
        # resolved by the same central policy as file/env/CLI supplied paths.
        effective: dict[str, Any] = AppSettings().model_dump(mode="python", by_alias=True)
        defaults, selected = _split_document(default_document, selected_profile)
        _deep_merge(effective, defaults)
        _deep_merge(effective, BUILTIN_PROFILES[selected_profile])
        _deep_merge(effective, selected)

        if user_path:
            user_document = _read_document(user_path)
            user_defaults, user_selected = _split_document(user_document, selected_profile)
            _deep_merge(effective, user_defaults)
            _deep_merge(effective, user_selected)

        _deep_merge(effective, environment_overrides(environment))
        if cli_overrides:
            _deep_merge(effective, cli_overrides)
        effective["schema_version"] = CURRENT_SCHEMA_VERSION
        effective["active_profile"] = selected_profile.value
        anchor = (base_dir or (default_path.parent if default_path else Path.cwd())).resolve()
        _resolve_paths(effective, anchor)
        try:
            return AppSettings.model_validate(effective)
        except ValidationError as exc:
            raise _validation_error(exc) from exc

    def load_from_cli(
        self,
        argv: Sequence[str],
        *,
        environ: Mapping[str, str] | None = None,
        fallback_config: Path | None = None,
        fallback_user_config: Path | None = None,
    ) -> AppSettings:
        options = self.parse_cli(argv)
        environment = dict(os.environ if environ is None else environ)
        config = options.config or (
            Path(environment[f"{ENV_PREFIX}CONFIG"])
            if f"{ENV_PREFIX}CONFIG" in environment
            else fallback_config
        )
        user_config = options.user_config or (
            Path(environment[f"{ENV_PREFIX}USER_CONFIG"])
            if f"{ENV_PREFIX}USER_CONFIG" in environment
            else fallback_user_config
        )
        return self.load(
            default_path=config,
            user_path=user_config,
            profile=options.profile,
            environ=environment,
            cli_overrides=options.overrides,
        )


__all__ = [
    "BUILTIN_PROFILES",
    "CURRENT_SCHEMA_VERSION",
    "AppSettings",
    "AttentionBackend",
    "CLIOptions",
    "DType",
    "DeviceMode",
    "InferenceBackend",
    "InstrumentationLevel",
    "ProfileName",
    "Quantization",
    "SettingsLoader",
    "environment_overrides",
    "migrate_config",
    "parse_key_value_overrides",
    "persist_user_model_roots",
]
