"""FastAPI application factory for the local workbench."""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import io
import ipaddress
import json
import os
import secrets
import sys
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import urlsplit

from fastapi import (
    APIRouter,
    Body,
    FastAPI,
    File,
    Form,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.body_limit import RequestBodyLimitMiddleware
from .api.schemas import (
    ChatCreate,
    ChatUpdate,
    ChatWorkspaceDocument,
    EmbeddingRunCreate,
    GenerationRunCreate,
    MessageCreate,
    ModelLoadRequest,
    ModelRootsUpdate,
    PromptScoreRequest,
    TokenBranchCreate,
)
from .config import AppSettings, ProfileName, SettingsLoader, persist_user_model_roots
from .errors import (
    CapabilityUnavailableError,
    ChatNotFoundError,
    ConfigurationError,
    ConfirmationRequiredError,
    InvalidRequestError,
    ModelNotLoadedError,
    NotFoundError,
    PayloadTooLargeError,
    RunNotCancellableError,
    RunNotFoundError,
    WorkbenchError,
    worker_failure_response,
)
from .persistence import Database, TelemetryWriter, WorkspaceRepository
from .services.events import EventBroker
from .services.models import ModelRegistry
from .services.runs import RunManager
from .services.uploads import UploadStore
from .services.workspace import WorkspaceService
from .workers import ModelWorkerSupervisor, WorkerFailure


@dataclass(slots=True)
class ApplicationServices:
    settings: AppSettings
    user_config_path: Path | None
    model_roots_override_source: str | None
    configuration_lock: asyncio.Lock
    database: Database
    repository: WorkspaceRepository
    telemetry: TelemetryWriter
    events: EventBroker
    worker: ModelWorkerSupervisor
    registry: ModelRegistry
    runs: RunManager
    uploads: UploadStore
    workspace: WorkspaceService


def _load_settings() -> AppSettings:
    project_root = Path.cwd()
    default_path = project_root / "config" / "default.yaml"
    local_path = project_root / "config" / "local.yaml"
    return SettingsLoader().load_from_cli(
        sys.argv[1:],
        fallback_config=default_path if default_path.is_file() else None,
        fallback_user_config=local_path if local_path.is_file() else None,
    )


def _user_configuration_path() -> Path:
    options = SettingsLoader.parse_cli(sys.argv[1:])
    configured = options.user_config or (
        Path(os.environ["LAD_USER_CONFIG"])
        if "LAD_USER_CONFIG" in os.environ
        else Path.cwd() / "config" / "local.yaml"
    )
    return configured.expanduser().resolve(strict=False)


def _model_roots_override_source() -> str | None:
    if "LAD_PATHS__MODEL_ROOTS" in os.environ:
        return "LAD_PATHS__MODEL_ROOTS environment override"
    overrides = SettingsLoader.parse_cli(sys.argv[1:]).overrides
    paths = overrides.get("paths")
    if isinstance(paths, Mapping) and "model_roots" in paths:
        return "--set paths.model_roots command-line override"
    return None


def _json_error(code: str, message: str, *, hint: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message, "retryable": False}
    if hint:
        payload["hint"] = hint
    return {"error": payload}


def _is_same_origin(origin: str, *, scheme: str, netloc: str) -> bool:
    """Compare an HTTP Origin with the request target, including WS scheme aliases."""

    expected_scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
    parsed = urlsplit(origin)
    target = urlsplit(f"{expected_scheme}://{netloc}")
    if (
        parsed.scheme.casefold() != expected_scheme.casefold()
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or target.username is not None
        or target.password is not None
    ):
        return False
    try:
        parsed_port = parsed.port or {"http": 80, "https": 443}.get(parsed.scheme.casefold())
        target_port = target.port or {"http": 80, "https": 443}.get(target.scheme.casefold())
    except ValueError:
        return False
    return (
        parsed.hostname is not None
        and target.hostname is not None
        and parsed.hostname.casefold() == target.hostname.casefold()
        and parsed_port == target_port
    )


def _hostname(netloc: str) -> str | None:
    """Extract a normalized hostname while rejecting malformed Host values."""

    if not netloc or netloc.endswith(":") or any(character in netloc for character in "\r\n\t /\\"):
        return None
    try:
        parsed = urlsplit(f"//{netloc}")
        # Accessing ``port`` validates bracket use, numeric syntax, and range.
        _ = parsed.port
    except ValueError:
        return None
    if (
        parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.hostname


def _websocket_csp_sources(netloc: str) -> str:
    """Build same-authority WS sources from an already validated request Host."""

    parsed = urlsplit(f"//{netloc}")
    hostname = parsed.hostname or ""
    if ":" in hostname:
        hostname = f"[{hostname}]"
    authority = f"{hostname}:{parsed.port}" if parsed.port is not None else hostname
    return f"ws://{authority} wss://{authority}"


def _trusted_request_host(settings: AppSettings, netloc: str) -> bool:
    """Reject DNS-rebinding Host values before serving even read-only data."""

    hostname = _hostname(netloc)
    if hostname is None:
        return False
    trusted = {settings.server.host.strip().casefold()}
    for origin in settings.server.allowed_origins:
        parsed_host = _hostname(urlsplit(origin).netloc)
        if parsed_host:
            trusted.add(parsed_host.casefold())
    configured_host = settings.server.host.strip().casefold()
    with suppress(ValueError):
        if ipaddress.ip_address(configured_host).is_loopback:
            trusted.update({"127.0.0.1", "::1", "localhost"})
    if configured_host == "localhost":
        trusted.update({"127.0.0.1", "::1", "localhost"})
    return hostname.casefold() in trusted


def _websocket_protocol_token(header: str | None) -> str | None:
    """Decode a bearer token carried in a WebSocket subprotocol, never the URL."""

    prefix = "lad.auth."
    for protocol in (header or "").split(","):
        candidate = protocol.strip()
        if not candidate.startswith(prefix):
            continue
        encoded = candidate[len(prefix) :]
        if not encoded or len(encoded) > 16_384:
            return None
        padding = "=" * (-len(encoded) % 4)
        try:
            decoded = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
            return decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None
    return None


def _is_frontend_shell_request(request: Request) -> bool:
    """Allow the static login shell to load before bearer authentication."""

    return request.method in {"GET", "HEAD"} and not request.url.path.startswith(("/api/", "/ws/"))


def _services(request: Request) -> ApplicationServices:
    return cast(ApplicationServices, request.app.state.services)


def _public_hardware(services: ApplicationServices) -> dict[str, Any]:
    inventory = services.registry.hardware
    if inventory is None:
        return {}
    data = inventory.model_dump(mode="json")
    for device in data.get("accelerators", []):
        device["identifier"] = f"{device['backend']}:{device['index']}"
    return data


def _auth_valid(
    settings: AppSettings,
    authorization: str | None,
    access_token: str | None = None,
) -> bool:
    expected = settings.server.authentication_token
    if expected is None:
        return True
    scheme, _, supplied = (authorization or "").partition(" ")
    candidate = supplied if scheme.lower() == "bearer" else (access_token or "")
    return bool(candidate) and secrets.compare_digest(candidate, expected.get_secret_value())


def _model_roots_configuration(services: ApplicationServices) -> dict[str, Any]:
    path = services.user_config_path
    reason: str | None = None
    if services.model_roots_override_source:
        reason = (
            f"Model roots are controlled by the {services.model_roots_override_source}; "
            "remove that higher-precedence override before editing them here."
        )
    elif path is None:
        reason = "No user-local configuration file is configured for this backend."
    elif path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        reason = "Workbench edits require a YAML or JSON user-local configuration file."
    elif path.exists() and (not path.is_file() or not os.access(path, os.W_OK)):
        reason = "The user-local configuration file is not writable by the backend."
    return {
        "model_roots": [str(root) for root in services.settings.paths.model_roots],
        "writable": reason is None,
        "source": "user-local configuration",
        "reason": reason,
        "containerized": services.settings.active_profile
        in {ProfileName.CONTAINER_CPU, ProfileName.CONTAINER_NVIDIA},
    }


def _validated_model_roots(values: list[str]) -> tuple[Path, ...]:
    resolved_roots: list[Path] = []
    for index, value in enumerate(values):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise ConfigurationError(
                "model root paths must be absolute in the backend environment",
                hint="For Docker, use the mounted container path (normally /models).",
                details={"index": index},
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError(
                "a configured model root does not exist or cannot be resolved",
                hint="Choose an existing directory visible to the backend process.",
                details={"index": index, "reason": type(exc).__name__},
            ) from exc
        if resolved == Path(resolved.anchor) or not resolved.is_dir():
            raise ConfigurationError(
                "a configured model root is not an eligible directory",
                hint="Choose a dedicated directory containing local model folders.",
                details={"index": index},
            )
        if not os.access(resolved, os.R_OK | os.X_OK):
            raise ConfigurationError(
                "a configured model root is not readable by the backend",
                hint="Grant the backend read access or choose another directory.",
                details={"index": index},
            )
        if resolved in resolved_roots:
            raise ConfigurationError(
                "model roots resolve to the same directory",
                details={"index": index},
            )
        resolved_roots.append(resolved)
    return tuple(resolved_roots)


def create_app(
    settings: AppSettings | None = None,
    *,
    user_config_path: Path | None = None,
) -> FastAPI:
    effective_settings = settings or _load_settings()
    configured_user_path = (
        user_config_path.expanduser().resolve(strict=False)
        if user_config_path is not None
        else _user_configuration_path()
        if settings is None
        else None
    )
    roots_override_source = _model_roots_override_source() if settings is None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(
            effective_settings.paths.database,
            effective_settings.paths.backups,
        )
        await database.initialize()
        repository = WorkspaceRepository(database)
        telemetry = TelemetryWriter(
            database,
            batch_size=effective_settings.telemetry.write_batch_size,
            flush_interval_seconds=effective_settings.telemetry.write_flush_interval_ms / 1000,
        )
        telemetry.start()
        raw_token_counts: dict[str, int] = {}
        raw_trace_bytes: dict[str, int] = {}

        async def persist_event(event: Mapping[str, Any]) -> None:
            # Token events are high volume and go through the bounded batch writer;
            # lifecycle events are committed immediately for reliable reconnection.
            if event["type"] == "token":
                if not effective_settings.telemetry.persist_token_events:
                    return
                run_id = str(event["run_id"])
                encoded_payload = json.dumps(
                    event["payload"], ensure_ascii=False, separators=(",", ":")
                )
                encoded_bytes = len(encoded_payload.encode("utf-8"))
                if (
                    raw_token_counts.get(run_id, 0)
                    >= effective_settings.limits.telemetry_events_per_run
                    or raw_trace_bytes.get(run_id, 0) + encoded_bytes
                    > effective_settings.limits.trace_bytes_per_run
                ):
                    return
                raw_token_counts[run_id] = raw_token_counts.get(run_id, 0) + 1
                raw_trace_bytes[run_id] = raw_trace_bytes.get(run_id, 0) + encoded_bytes
                await telemetry.submit(
                    """
                    INSERT OR REPLACE INTO raw_events(
                        run_id, sequence, event_type, protocol_version, monotonic_ns,
                        payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event["run_id"],
                            event["sequence"],
                            event["type"],
                            event["version"],
                            event["monotonic_ns"],
                            encoded_payload,
                            event["server_time"],
                        )
                    ],
                )
            else:
                if event["type"] in {"completed", "cancelled", "error"}:
                    # A failed token writer must not prevent durable lifecycle
                    # state or the terminal raw event from being committed.
                    with suppress(Exception):
                        await telemetry.flush()
                await repository.append_raw_event(event)
                if event["type"] in {"completed", "cancelled", "error"}:
                    raw_token_counts.pop(str(event["run_id"]), None)
                    raw_trace_bytes.pop(str(event["run_id"]), None)

        events = EventBroker(
            persist=persist_event,
            replay=repository.get_raw_events,
            subscriber_queue_size=256,
        )
        worker = ModelWorkerSupervisor(
            queue_limit=effective_settings.runtime.queue_limit or 1,
            startup_timeout_seconds=effective_settings.workers.startup_timeout_seconds,
        )
        await worker.start()
        registry = ModelRegistry(effective_settings, repository, worker)
        await registry.refresh()
        await repository.recover_incomplete_runs()
        runs = RunManager(effective_settings, repository, registry, worker, events, telemetry)
        uploads = UploadStore(
            effective_settings.paths.uploads,
            repository,
            effective_settings.limits.upload_bytes,
            maximum_image_pixels=effective_settings.limits.image_pixels,
            maximum_video_frames=effective_settings.limits.video_frames,
            maximum_video_frame_pixels=effective_settings.limits.video_frame_pixels,
            maximum_decoded_media_pixels=effective_settings.limits.decoded_media_pixels,
            maximum_media_duration_seconds=effective_settings.limits.media_duration_seconds,
        )
        workspace = WorkspaceService(repository)
        app.state.services = ApplicationServices(
            settings=effective_settings,
            user_config_path=configured_user_path,
            model_roots_override_source=roots_override_source,
            configuration_lock=asyncio.Lock(),
            database=database,
            repository=repository,
            telemetry=telemetry,
            events=events,
            worker=worker,
            registry=registry,
            runs=runs,
            uploads=uploads,
            workspace=workspace,
        )
        try:
            yield
        finally:
            await runs.close()
            await worker.close(grace_seconds=effective_settings.workers.shutdown_grace_seconds)
            await telemetry.stop()
            await database.close()

    app = FastAPI(
        title="Local AI Doctor",
        version=__version__,
        docs_url="/api/docs"
        if effective_settings.active_profile is ProfileName.DEVELOPMENT
        else None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def secure_transport(request: Request, call_next: Any) -> Response:
        request_host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if not _is_frontend_shell_request(request) and not _auth_valid(
            effective_settings, request.headers.get("authorization")
        ):
            return JSONResponse(
                _json_error("authentication_required", "valid bearer authentication is required"),
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if (
            effective_settings.server.csrf_protection
            and request.method in {"POST", "PATCH", "PUT", "DELETE"}
            and origin
            and origin not in effective_settings.server.allowed_origins
            and not _is_same_origin(
                origin,
                scheme=request.url.scheme,
                netloc=request_host,
            )
        ):
            return JSONResponse(
                _json_error("origin_rejected", "request origin is not allowed"),
                status_code=status.HTTP_403_FORBIDDEN,
            )
        response = cast(Response, await call_next(request))
        websocket_sources = _websocket_csp_sources(request_host)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
            "form-action 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "font-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
            f"connect-src 'self' {websocket_sources}"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # Count bytes as they arrive, before Starlette parses JSON or spools a
    # multipart upload. The small envelope allowance covers JSON/form framing;
    # domain services still enforce exact prompt and file payload limits.
    app.add_middleware(
        RequestBodyLimitMiddleware,
        json_bytes=effective_settings.limits.prompt_bytes + 1024 * 1024,
        upload_bytes=effective_settings.limits.upload_bytes,
    )

    # This must wrap the authentication middleware: browsers do not attach the
    # bearer header to a CORS preflight, and an allowed frontend must be able to
    # inspect a 401 response before it has collected the tab-scoped token.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(effective_settings.server.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )

    # Host validation remains outside CORS so even a middleware-handled
    # preflight cannot use an untrusted or syntactically malformed authority.
    @app.middleware("http")
    async def trusted_host_transport(request: Request, call_next: Any) -> Response:
        if not _trusted_request_host(effective_settings, request.headers.get("host", "")):
            return JSONResponse(
                _json_error("host_rejected", "request host is not allowed"),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return cast(Response, await call_next(request))

    @app.exception_handler(WorkbenchError)
    async def workbench_error(_request: Request, exc: WorkbenchError) -> JSONResponse:
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.http_status)

    @app.exception_handler(WorkerFailure)
    async def worker_failure(_request: Request, exc: WorkerFailure) -> JSONResponse:
        http_status, payload = worker_failure_response(exc.error)
        return JSONResponse({"error": payload}, status_code=http_status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        issues = [
            {"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
            for item in exc.errors()
        ]
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "request validation failed",
                    "retryable": False,
                    "details": {"issues": issues},
                }
            },
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            _json_error(
                "internal_error",
                "the request failed at an internal boundary",
                hint="Review the local structured log for private diagnostic details.",
            ),
            status_code=500,
        )

    api = APIRouter(prefix="/api/v1")

    @api.get("/health")
    async def health(request: Request) -> JSONResponse:
        services = _services(request)
        worker_ready = services.worker.available
        payload = {
            "status": "ok" if worker_ready else "degraded",
            "database": "ready",
            "worker": "ready" if worker_ready else "unavailable",
            "loaded_model": services.worker.loaded,
            "protocol_version": 1,
        }
        if not worker_ready:
            return JSONResponse(payload, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse(payload)

    @api.get("/configuration")
    async def configuration(request: Request) -> dict[str, Any]:
        services = _services(request)
        return {
            "effective": services.settings.redacted_effective_config(),
            "precedence": [
                "model defaults",
                "portable defaults",
                "built-in profile",
                "portable profile",
                "user-local configuration",
                "LAD_ environment overrides",
                "--set command-line overrides",
            ],
        }

    @api.get("/configuration/model-roots")
    async def model_root_configuration(request: Request) -> dict[str, Any]:
        return _model_roots_configuration(_services(request))

    @api.put("/configuration/model-roots")
    async def update_model_root_configuration(
        request: Request,
        body: ModelRootsUpdate,
    ) -> dict[str, Any]:
        services = _services(request)
        availability = _model_roots_configuration(services)
        if not availability["writable"] or services.user_config_path is None:
            raise ConfigurationError(
                "model roots cannot be edited from this backend",
                hint=str(availability["reason"]),
            )
        roots = _validated_model_roots(body.model_roots)
        updated_paths = type(services.settings.paths).model_validate(
            {
                **services.settings.paths.model_dump(mode="python"),
                "model_roots": roots,
            }
        )
        async with (
            services.configuration_lock,
            services.registry.admission.lifecycle("model_roots_update"),
        ):
            if services.worker.loaded is not None:
                raise CapabilityUnavailableError(
                    "model roots cannot be changed while a model is loaded",
                    hint="Unload the current model first so it cannot remain hidden while occupying memory.",
                )
            await asyncio.to_thread(
                persist_user_model_roots,
                services.user_config_path,
                services.settings.active_profile,
                roots,
            )
            services.settings.paths = updated_paths
            await services.registry.refresh()
        return {
            **_model_roots_configuration(services),
            "models": services.registry.public_report(),
        }

    @api.get("/hardware")
    async def hardware(request: Request) -> dict[str, Any]:
        services = _services(request)
        selection = services.registry.choose_hardware()
        return {
            "inventory": _public_hardware(services),
            "selection": selection.model_dump(mode="json"),
        }

    @api.get("/models")
    async def models(request: Request) -> dict[str, Any]:
        return _services(request).registry.public_report()

    @api.post("/models/refresh")
    async def refresh_models(request: Request) -> dict[str, Any]:
        services = _services(request)
        await services.registry.refresh()
        return services.registry.public_report()

    @api.post("/models/{model_id}/load")
    async def load_model(
        request: Request,
        model_id: str,
        body: Annotated[ModelLoadRequest | None, Body()] = None,
    ) -> dict[str, Any]:
        body = body or ModelLoadRequest()
        return await _services(request).registry.load(
            model_id, device=body.device, dtype=body.dtype
        )

    @api.post("/models/unload")
    async def unload_model(request: Request) -> dict[str, Any]:
        return await _services(request).registry.unload()

    @api.post("/models/{model_id}/unload")
    async def unload_selected_model(request: Request, model_id: str) -> dict[str, Any]:
        services = _services(request)
        descriptor = services.registry.get(model_id)
        loaded = services.worker.loaded
        if loaded is not None and loaded.get("model_id") != model_id:
            raise ModelNotLoadedError(
                "a different model is loaded",
                hint="Only the resident model can be unloaded by ID; use POST /models/unload instead.",
                details={"model_id": model_id},
            )
        unload = await services.registry.unload()
        return {
            **descriptor.public_dict(reveal_path=False),
            "lifecycle": "unloaded",
            "loaded_device": None,
            "unload": unload,
        }

    @api.get("/models/{model_id}/inspect")
    async def inspect_model(request: Request, model_id: str) -> dict[str, Any]:
        descriptor = _services(request).registry.get(model_id)
        result: dict[str, Any] = {"descriptor": descriptor.public_dict(reveal_path=False)}
        for filename, key in (
            ("tokenizer_config.json", "tokenizer"),
            ("generation_config.json", "generation"),
            ("special_tokens_map.json", "special_tokens"),
        ):
            path = descriptor.path / filename
            if path.is_file() and path.stat().st_size <= 4 * 1024**2:
                try:
                    result[key] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    result[key] = {"diagnostic": f"{filename} is unreadable"}
        template_path = descriptor.path / "chat_template.jinja"
        if template_path.is_file() and template_path.stat().st_size <= 1024**2:
            result["chat_template"] = template_path.read_text(encoding="utf-8")
        elif isinstance(result.get("tokenizer"), dict):
            result["chat_template"] = result["tokenizer"].get("chat_template")
        result["sampling_pipeline"] = [
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "renormalize",
            "sample",
        ]
        return result

    @api.post("/chats", status_code=201)
    async def create_chat(request: Request, body: ChatCreate) -> dict[str, Any]:
        return await _services(request).repository.create_chat(body.title)

    @api.post("/chats/import", status_code=201)
    async def import_chat(request: Request, body: ChatWorkspaceDocument) -> dict[str, Any]:
        return await _services(request).workspace.import_chat(body)

    @api.get("/chats")
    async def list_chats(
        request: Request,
        search: str | None = Query(default=None, max_length=200),
        archived: bool = False,
    ) -> list[dict[str, Any]]:
        return await _services(request).repository.list_chats(search=search, archived=archived)

    @api.get("/chats/{chat_id}")
    async def get_chat(request: Request, chat_id: str) -> dict[str, Any]:
        chat = await _services(request).repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError("chat not found", details={"chat_id": chat_id})
        return chat

    @api.get("/chats/{chat_id}/export")
    async def export_chat(request: Request, chat_id: str) -> Response:
        document = await _services(request).workspace.export_chat(chat_id)
        if document is None:
            raise ChatNotFoundError("chat not found", details={"chat_id": chat_id})
        return JSONResponse(
            document,
            headers={"Content-Disposition": 'attachment; filename="chat-workspace.json"'},
        )

    @api.get("/chats/{chat_id}/messages")
    async def get_chat_messages(request: Request, chat_id: str) -> list[dict[str, Any]]:
        chat = await _services(request).repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError("chat not found", details={"chat_id": chat_id})
        return cast(list[dict[str, Any]], chat["messages"])

    @api.patch("/chats/{chat_id}")
    async def update_chat(request: Request, chat_id: str, body: ChatUpdate) -> dict[str, Any]:
        chat = await _services(request).repository.update_chat(
            chat_id, title=body.title, pinned=body.pinned, archived=body.archived
        )
        if chat is None:
            raise ChatNotFoundError("chat not found", details={"chat_id": chat_id})
        return chat

    @api.delete("/chats/{chat_id}", status_code=204)
    async def delete_chat(request: Request, chat_id: str) -> Response:
        deleted = await _services(request).repository.delete_chat(chat_id)
        if not deleted:
            raise ChatNotFoundError("chat not found", details={"chat_id": chat_id})
        return Response(status_code=204)

    @api.delete("/chats")
    async def clear_chats(
        request: Request, confirm: bool, include_archived: bool = False
    ) -> dict[str, Any]:
        if not confirm:
            raise ConfirmationRequiredError(
                "explicit confirmation is required",
                hint="Repeat the request with confirm=true after reviewing what will be deleted.",
            )
        count = await _services(request).repository.clear_chats(include_archived=include_archived)
        return {"deleted": count}

    @api.post("/chats/{chat_id}/messages", status_code=201)
    async def create_message(request: Request, chat_id: str, body: MessageCreate) -> dict[str, Any]:
        services = _services(request)
        if len(body.content.encode("utf-8")) > services.settings.limits.prompt_bytes:
            raise PayloadTooLargeError(
                "message exceeds the configured byte limit",
                details={"maximum_bytes": services.settings.limits.prompt_bytes},
            )
        if len(body.attachment_ids) > services.settings.limits.attachment_count:
            raise InvalidRequestError(
                "attachment count exceeds the configured per-message limit",
                details={"maximum_count": services.settings.limits.attachment_count},
            )
        return await services.repository.create_message(
            chat_id=chat_id,
            role=body.role,
            content=body.content,
            parent_id=body.parent_id,
            branch_index=body.branch_index,
            attachment_ids=body.attachment_ids,
        )

    @api.post("/runs", status_code=202)
    @api.post("/runs/generation", status_code=202)
    async def generation(request: Request, body: GenerationRunCreate) -> dict[str, Any]:
        created = await _services(request).runs.create_generation(body)
        return {
            "runId": created["run"]["id"],
            "messageId": created["assistant_message"]["id"],
            "websocketUrl": f"/ws/v1/runs/{created['run']['id']}",
            **created,
        }

    @api.get("/runs/{run_id}")
    async def get_run(request: Request, run_id: str) -> dict[str, Any]:
        services = _services(request)
        run = await services.repository.get_run(run_id)
        if run is None:
            raise RunNotFoundError("run not found", details={"run_id": run_id})
        tokens = await services.repository.list_run_tokens(run_id)
        run["tokens"] = tokens
        branchable_through = -1
        for token in tokens:
            if int(token["token_index"]) != branchable_through + 1:
                break
            branchable_through += 1
        run["branchable_through_token_index"] = (
            branchable_through if branchable_through >= 0 else None
        )
        run["phases"] = await services.repository.list_phase_metrics(run_id)
        run["environment"] = await services.repository.get_environment_snapshot(run_id)
        terminal_event = await services.repository.get_latest_terminal_event(run_id)
        run["summary"] = terminal_event["payload"] if terminal_event else None
        run["warnings"] = await services.repository.get_run_warnings(run_id)
        return run

    @api.get("/runs/{run_id}/events")
    async def get_run_events(request: Request, run_id: str) -> dict[str, Any]:
        services = _services(request)
        if await services.repository.get_run(run_id) is None:
            raise RunNotFoundError("run not found", details={"run_id": run_id})
        return {"events": await services.repository.get_raw_events(run_id)}

    @api.post("/runs/{run_id}/cancel")
    async def cancel_run(request: Request, run_id: str) -> dict[str, Any]:
        accepted = await _services(request).runs.cancel(run_id)
        if not accepted:
            raise RunNotCancellableError(
                "run is not cancellable",
                hint="Only queued, loading, or running runs can be cancelled.",
                details={"run_id": run_id},
            )
        return {"accepted": True}

    @api.post("/runs/{run_id}/replay", status_code=202)
    async def replay_run(request: Request, run_id: str) -> dict[str, Any]:
        created = await _services(request).runs.replay_generation(run_id)
        return {
            "runId": created["run"]["id"],
            "messageId": created["assistant_message"]["id"],
            "parentRunId": run_id,
            "websocketUrl": f"/ws/v1/runs/{created['run']['id']}",
            **created,
        }

    @api.post("/runs/{run_id}/branch", status_code=202)
    async def branch_run(request: Request, run_id: str, body: TokenBranchCreate) -> dict[str, Any]:
        created = await _services(request).runs.branch_generation(run_id, body)
        return {
            "chatId": created["chat"]["id"],
            "runId": created["run"]["id"],
            "messageId": created["assistant_message"]["id"],
            "sourceRunId": run_id,
            "websocketUrl": f"/ws/v1/runs/{created['run']['id']}",
            **created,
        }

    @api.get("/runs/{run_id}/export")
    async def export_run(
        request: Request,
        run_id: str,
        format: Annotated[str, Query(pattern="^(json|jsonl|csv)$")] = "json",
    ) -> Response:
        services = _services(request)
        run = await services.repository.get_run(run_id)
        if run is None:
            raise RunNotFoundError("run not found", details={"run_id": run_id})
        tokens = await services.repository.list_run_tokens(run_id)
        if format == "json":
            return JSONResponse({"schema_version": 1, "run": run, "tokens": tokens})
        if format == "jsonl":
            events = await services.repository.get_raw_events(run_id)
            content = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
            return PlainTextResponse(
                content,
                media_type="application/x-ndjson",
                headers={"Content-Disposition": f'attachment; filename="run-{run_id}.jsonl"'},
            )
        output = io.StringIO(newline="")
        fields = [
            "token_index",
            "token_id",
            "piece",
            "escaped_bytes",
            "display_text",
            "raw_logprob",
            "raw_probability",
            "raw_rank",
            "sample_logprob",
            "sample_probability",
            "entropy",
            "surprise",
            "running_perplexity",
            "decode_ms",
            "sample_ms",
            "inter_token_ms",
            "segment",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tokens)
        return PlainTextResponse(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="tokens-{run_id}.csv"'},
        )

    @api.get("/runs/compare/summary")
    async def compare_runs(
        request: Request,
        ids: Annotated[list[str], Query(min_length=2, max_length=8)],
    ) -> dict[str, Any]:
        runs = []
        for run_id in ids:
            run = await _services(request).repository.get_run(run_id)
            if run is not None:
                runs.append(run)
        return {
            "runs": runs,
            "missing_ids": [run_id for run_id in ids if run_id not in {r["id"] for r in runs}],
        }

    @api.post("/attachments", status_code=201)
    @api.post("/uploads", status_code=201)
    async def upload_attachment(
        request: Request,
        model_id: Annotated[str, Form()],
        file: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        services = _services(request)
        descriptor = services.registry.get(model_id)
        attachment = await services.uploads.save(
            file,
            supported_modalities=descriptor.modalities,
            allow_extracted_text=False,
        )
        attachment.pop("storage_name", None)
        media_type = str(attachment["media_type"])
        family = media_type.split("/", maxsplit=1)[0]
        return {
            **attachment,
            "name": attachment["original_name"],
            "mimeType": media_type,
            "sizeBytes": attachment["size_bytes"],
            "kind": family if family in {"image", "audio", "video"} else "document",
            "status": "ready",
            "nativeProcessing": family in {"image", "audio", "video"},
        }

    @api.get("/attachments/{attachment_id}/content")
    async def attachment_content(request: Request, attachment_id: str) -> Response:
        services = _services(request)
        attachment, path = await services.uploads.resolve(attachment_id)
        return FileResponse(
            path,
            media_type=str(attachment["media_type"]),
            headers={"Content-Disposition": "inline"},
        )

    @api.post("/embeddings")
    @api.post("/runs/embeddings")
    async def embeddings(request: Request, body: EmbeddingRunCreate) -> dict[str, Any]:
        services = _services(request)
        attachment_count = sum(1 for item in body.inputs if item.attachment_id is not None)
        if attachment_count > services.settings.limits.attachment_count:
            raise PayloadTooLargeError(
                "attachment count exceeds the configured per-request limit",
                details={"maximum_count": services.settings.limits.attachment_count},
            )
        resolved: list[dict[str, Any]] = []
        for item in body.inputs:
            value: dict[str, Any] = {
                "input_id": item.id or str(uuid.uuid4()),
                "modality": item.modality,
                "text": item.text,
            }
            if item.attachment_id:
                attachment, path = await services.uploads.resolve(item.attachment_id)
                value.update(
                    {
                        "attachment_id": item.attachment_id,
                        "content_path": str(path),
                        "media_kind": attachment["media_type"].split("/", maxsplit=1)[0],
                    }
                )
            resolved.append(value)
        result = await services.runs.create_embedding(body, resolved)
        vectors = [
            {
                "inputId": item["input_id"],
                "values": item["vector"],
                "dimensions": item["output_dimension"],
                "dtype": item["output_dtype"],
                "l2Norm": item["l2_norm"],
                "minimum": item["statistics"]["minimum"],
                "maximum": item["statistics"]["maximum"],
                "mean": item["statistics"]["mean"],
                "standardDeviation": item["statistics"]["standard_deviation"],
                "finite": item["statistics"]["finite"],
            }
            for item in result["results"]
        ]
        similarity: list[list[float]] = []
        for left in vectors:
            row: list[float] = []
            for right in vectors:
                row.append(sum(a * b for a, b in zip(left["values"], right["values"], strict=True)))
            similarity.append(row)
        return {
            **result,
            "id": result["run_id"],
            "modelId": body.model_id,
            "status": "complete",
            "inputs": [item.model_dump(mode="json") for item in body.inputs],
            "vectors": vectors,
            "requestedDimensions": body.dimensions,
            "outputDimensions": vectors[0]["dimensions"],
            "normalized": body.normalize,
            "jointSpace": result.get("joint_embedding_space"),
            "preprocessingMs": result.get("preprocessing_ms"),
            "forwardMs": result.get("forward_ms"),
            "totalMs": result.get("total_ms"),
            "itemsPerSecond": result.get("items_per_second"),
            "similarityMatrix": similarity,
        }

    @api.post("/runs/prompt-score")
    async def prompt_score(request: Request, body: PromptScoreRequest) -> dict[str, Any]:
        return await _services(request).runs.score_prompt(body)

    @api.get("/storage")
    async def storage_stats(request: Request) -> dict[str, Any]:
        return await _services(request).repository.database_stats()

    @api.delete("/storage/telemetry")
    async def delete_old_telemetry(
        request: Request,
        before: datetime,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if not confirm:
            raise InvalidRequestError(
                "explicit confirmation is required",
                hint="Repeat the request with confirm=true after reviewing the cutoff.",
            )
        if before.tzinfo is None or before.utcoffset() is None:
            raise InvalidRequestError("the telemetry cutoff must include a timezone")
        services = _services(request)
        await services.telemetry.flush()
        cutoff = before.astimezone(UTC).isoformat()
        deleted = await services.repository.delete_run_telemetry_before(cutoff)
        return {"before": cutoff, "deleted": deleted}

    app.include_router(api)

    @app.websocket("/ws/v1/runs/{run_id}")
    async def run_events(
        websocket: WebSocket,
        run_id: str,
        after: int = 0,
    ) -> None:
        request_host = websocket.headers.get("host", "")
        if not _trusted_request_host(effective_settings, request_host):
            await websocket.close(code=4403, reason="host rejected")
            return
        if not _auth_valid(
            effective_settings,
            websocket.headers.get("authorization"),
            _websocket_protocol_token(websocket.headers.get("sec-websocket-protocol")),
        ):
            await websocket.close(code=4401, reason="authentication required")
            return
        origin = websocket.headers.get("origin")
        if (
            origin
            and origin not in effective_settings.server.allowed_origins
            and not _is_same_origin(
                origin,
                scheme=websocket.url.scheme,
                netloc=request_host,
            )
        ):
            await websocket.close(code=4403, reason="origin rejected")
            return
        await websocket.accept(subprotocol="lad.events.v1")
        try:
            services: ApplicationServices = websocket.app.state.services
            async for event in services.events.subscribe(run_id, after_sequence=max(0, after)):
                await websocket.send_json(event.model_dump(mode="json"))
        except WebSocketDisconnect:
            # Generation intentionally continues; partial output is already durable.
            return

    frontend = Path.cwd() / "frontend" / "dist"
    assets = frontend / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend_route(path: str) -> Response:
        index = frontend / "index.html"
        if index.is_file() and not path.startswith(("api/", "ws/")):
            return FileResponse(index)
        raise NotFoundError("not found")

    return app
