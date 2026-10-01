"""Structured, safe-to-serialize application errors.

The core never exposes raw exception strings at transport boundaries.  Those
strings frequently contain private paths, prompts, or implementation details.
Instead, callers receive a stable error code, a concise message, and optional
structured details that have been explicitly selected for disclosure.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar


class ErrorCode(StrEnum):
    CONFIGURATION_INVALID = "configuration_invalid"
    CONFIGURATION_IO = "configuration_io"
    CONFIGURATION_VERSION = "configuration_version"
    PATH_OUTSIDE_ROOT = "path_outside_root"
    MODEL_NOT_FOUND = "model_not_found"
    CHAT_NOT_FOUND = "chat_not_found"
    RUN_NOT_FOUND = "run_not_found"
    NOT_FOUND = "not_found"
    ATTACHMENT_NOT_FOUND = "attachment_not_found"
    MODEL_INVALID = "model_invalid"
    MODEL_CORRUPT = "model_corrupt"
    MISSING_COMPONENT = "missing_component"
    UNSUPPORTED_ARCHITECTURE = "unsupported_architecture"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    ADAPTER_NOT_FOUND = "adapter_not_found"
    ADAPTER_CONFLICT = "adapter_conflict"
    MODEL_NOT_LOADED = "model_not_loaded"
    MODEL_ALREADY_LOADED = "model_already_loaded"
    MODEL_NOT_RESIDENT = "model_not_resident"
    TARGET_EXISTS = "target_exists"
    FLUSH_REQUIRES_QUANTIZED_RESIDENT = "flush_requires_quantized_resident"
    MODEL_ROOT_READ_ONLY = "model_root_read_only"
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_SAMPLING_SETTINGS = "invalid_sampling_settings"
    INVALID_LOGITS = "invalid_logits"
    INVALID_REQUEST = "invalid_request"
    LIMIT_EXCEEDED = "limit_exceeded"
    INVALID_UPLOAD = "invalid_upload"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    WORKER_BUSY = "worker_busy"
    ACTIVE_RUN_CONFLICT = "active_run_conflict"
    RUN_NOT_CANCELLABLE = "run_not_cancellable"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


def _safe_detail(value: Any) -> Any:
    """Convert deliberately supplied detail values to JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        # Paths are redacted by default.  A caller may disclose a safe basename
        # explicitly if that improves an error shown to the local user.
        return f"<path:{value.name or 'root'}>"
    if isinstance(value, Mapping):
        return {str(key): _safe_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_detail(item) for item in value]
    return type(value).__name__


class WorkbenchError(Exception):
    """Base class for errors that can safely cross an API boundary."""

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL
    http_status: ClassVar[int] = 500
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = dict(details or {})
        self.retryable = self.default_retryable if retryable is None else retryable

    def to_dict(self, *, include_details: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.hint:
            payload["hint"] = self.hint
        if include_details and self.details:
            payload["details"] = _safe_detail(self.details)
        return payload


class ConfigurationError(WorkbenchError):
    code = ErrorCode.CONFIGURATION_INVALID
    http_status = 422


class ConfigurationIOError(WorkbenchError):
    code = ErrorCode.CONFIGURATION_IO
    http_status = 500


class ConfigurationVersionError(ConfigurationError):
    code = ErrorCode.CONFIGURATION_VERSION


class PathSecurityError(WorkbenchError):
    code = ErrorCode.PATH_OUTSIDE_ROOT
    http_status = 403


class ModelNotFoundError(WorkbenchError):
    code = ErrorCode.MODEL_NOT_FOUND
    http_status = 404


class ChatNotFoundError(WorkbenchError):
    code = ErrorCode.CHAT_NOT_FOUND
    http_status = 404


class RunNotFoundError(WorkbenchError):
    code = ErrorCode.RUN_NOT_FOUND
    http_status = 404


class NotFoundError(WorkbenchError):
    code = ErrorCode.NOT_FOUND
    http_status = 404


class AttachmentNotFoundError(WorkbenchError):
    code = ErrorCode.ATTACHMENT_NOT_FOUND
    http_status = 404


class ModelInvalidError(WorkbenchError):
    code = ErrorCode.MODEL_INVALID
    http_status = 422


class ModelCorruptError(ModelInvalidError):
    code = ErrorCode.MODEL_CORRUPT


class MissingComponentError(ModelInvalidError):
    code = ErrorCode.MISSING_COMPONENT


class UnsupportedArchitectureError(WorkbenchError):
    code = ErrorCode.UNSUPPORTED_ARCHITECTURE
    http_status = 422


class CapabilityUnavailableError(WorkbenchError):
    code = ErrorCode.CAPABILITY_UNAVAILABLE
    http_status = 409


class BackendUnavailableError(WorkbenchError):
    code = ErrorCode.BACKEND_UNAVAILABLE
    http_status = 503


class AdapterNotFoundError(WorkbenchError):
    code = ErrorCode.ADAPTER_NOT_FOUND
    http_status = 422


class AdapterConflictError(WorkbenchError):
    code = ErrorCode.ADAPTER_CONFLICT
    http_status = 409


class InvalidSamplingSettingsError(WorkbenchError):
    code = ErrorCode.INVALID_SAMPLING_SETTINGS
    http_status = 422


class InvalidLogitsError(WorkbenchError):
    code = ErrorCode.INVALID_LOGITS
    http_status = 422


class InvalidRequestError(WorkbenchError):
    code = ErrorCode.INVALID_REQUEST
    http_status = 422


class InvalidUploadError(WorkbenchError):
    """The uploaded file itself is malformed, mislabeled, or undecodable."""

    code = ErrorCode.INVALID_UPLOAD
    http_status = 422


class UnsupportedMediaTypeError(WorkbenchError):
    code = ErrorCode.UNSUPPORTED_MEDIA_TYPE
    http_status = 415


class OutOfMemoryError(WorkbenchError):
    code = ErrorCode.OUT_OF_MEMORY
    http_status = 507
    default_retryable = True


class LimitExceededError(WorkbenchError):
    code = ErrorCode.LIMIT_EXCEEDED
    http_status = 429
    default_retryable = True


class PayloadTooLargeError(WorkbenchError):
    code = ErrorCode.LIMIT_EXCEEDED
    http_status = 413


class WorkerBusyError(WorkbenchError):
    code = ErrorCode.WORKER_BUSY
    http_status = 409
    default_retryable = True


class ActiveRunConflictError(WorkbenchError):
    """Raised when destructive workspace changes would invalidate an active run."""

    code = ErrorCode.ACTIVE_RUN_CONFLICT
    http_status = 409
    default_retryable = True


class ModelNotResidentError(WorkbenchError):
    code = ErrorCode.MODEL_NOT_RESIDENT
    http_status = 409


class TargetExistsError(WorkbenchError):
    code = ErrorCode.TARGET_EXISTS
    http_status = 409


class FlushRequiresQuantizedResidentError(WorkbenchError):
    code = ErrorCode.FLUSH_REQUIRES_QUANTIZED_RESIDENT
    http_status = 409


class ModelRootReadOnlyError(WorkbenchError):
    code = ErrorCode.MODEL_ROOT_READ_ONLY
    http_status = 409


class RunNotCancellableError(WorkbenchError):
    code = ErrorCode.RUN_NOT_CANCELLABLE
    http_status = 409


class ConfirmationRequiredError(WorkbenchError):
    code = ErrorCode.CONFIRMATION_REQUIRED
    http_status = 409


class CancelledError(WorkbenchError):
    code = ErrorCode.CANCELLED
    http_status = 499


_WORKER_CODE_STATUS = {
    "model_out_of_memory": 507,
    "out_of_memory": 507,
    "insufficient_memory": 507,
    "model_worker_timeout": 504,
    "inference_timeout": 504,
    "model_worker_state_mismatch": 409,
    "model_not_resident": 409,
    "worker_busy": 409,
    "flush_requires_quantized_resident": 409,
}
# Codes-and-numbers fields the worker reports next to its code (never paths or text).
_WORKER_DETAIL_FIELDS = (
    "memory_kind",
    "required_bytes",
    "available_bytes",
    "estimate",
    "max_concurrent_runs",
)


def worker_failure_response(error: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Map a model-worker error dict to an HTTP status and the public error envelope.

    The dict is already redacted at the process boundary (a code, a canned
    message, an optional hint, and supervisor-selected details), so nothing
    here reads exception text. Out-of-memory codes are normalized to the API's
    ``out_of_memory``; every other worker code is passed through unchanged. Numeric
    fields a worker reports beside its code (``insufficient_memory`` byte counts)
    become details.
    """

    worker_code = str(error.get("code") or "model_worker_error")[:64]
    status = _WORKER_CODE_STATUS.get(worker_code, 502)
    code = ErrorCode.OUT_OF_MEMORY.value if status == 507 else worker_code
    details = dict(error["details"]) if isinstance(error.get("details"), Mapping) else {}
    details.update({field: error[field] for field in _WORKER_DETAIL_FIELDS if field in error})
    if code != worker_code:
        details["worker_code"] = worker_code
    payload: dict[str, Any] = {
        "code": code,
        "message": str(error.get("message") or "model worker operation failed"),
        "retryable": status in {504, 507},
    }
    if error.get("hint"):
        payload["hint"] = str(error["hint"])
    if details:
        payload["details"] = _safe_detail(details)
    return status, payload
