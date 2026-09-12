"""Domain models describing discovered local model directories."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .capabilities import CapabilityMatrix


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Diagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: DiagnosticSeverity
    message: str
    hint: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ModelTask(StrEnum):
    TEXT_GENERATION = "text_generation"
    ENCODER_DECODER_GENERATION = "encoder_decoder_generation"
    EMBEDDING = "embedding"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
    UNKNOWN = "unknown"


class TrustDecision(StrEnum):
    BUILTIN_ONLY = "builtin_only"
    REVIEWED_BUNDLED_CODE = "reviewed_bundled_code"
    REJECTED_CUSTOM_CODE = "rejected_custom_code"


class ModelFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str = "sha256"
    value: str
    strength: str
    manifest_version: int = 1
    file_count: int = Field(ge=0)
    total_weight_bytes: int = Field(ge=0)
    metadata_digest: str
    weights_digest: str

    @field_validator("value", "metadata_digest", "weights_digest")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        lowered = value.lower()
        if len(lowered) != 64 or any(char not in "0123456789abcdef" for char in lowered):
            raise ValueError("expected a 64-character hexadecimal SHA-256 digest")
        return lowered


class ModelComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: bool = False
    safetensors: bool = False
    tokenizer: bool = False
    chat_template: bool = False
    generation_config: bool = False
    processor: bool = False
    pooling: bool = False
    custom_code: bool = False


class ContextValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    value: int = Field(gt=0)
    note: str | None = None


class ModelDescriptor(BaseModel):
    """Immutable discovery result safe to persist as a point-in-time snapshot."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str
    display_name: str
    path: Path
    fingerprint: ModelFingerprint
    task: ModelTask
    architectures: tuple[str, ...] = ()
    model_type: str | None = None
    dtype: str | None = None
    parameter_count: int | None = Field(default=None, ge=0)
    weight_dtypes: dict[str, int] = Field(default_factory=dict)
    modalities: frozenset[str] = frozenset({"text"})
    components: ModelComponents
    capabilities: CapabilityMatrix
    context_values: tuple[ContextValue, ...] = ()
    effective_context_limit: int | None = Field(default=None, gt=0)
    reasoning_delimiters: tuple[str, str] | None = None
    is_moe: bool = False
    trust_decision: TrustDecision = TrustDecision.BUILTIN_ONLY
    diagnostics: tuple[Diagnostic, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def loadable(self) -> bool:
        return not any(item.severity is DiagnosticSeverity.ERROR for item in self.diagnostics)

    def public_dict(self, *, reveal_path: bool = False) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["path"] = str(self.path) if reveal_path else f"<model-root>/{self.path.name}"
        data["loadable"] = self.loadable
        return data
