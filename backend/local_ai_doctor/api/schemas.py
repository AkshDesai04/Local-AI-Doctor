"""Versioned API request and response boundary models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import DeviceMode, DType, InstrumentationLevel, Quantization, SamplingDefaults
from ..domain.quantization import folder_name_error

_SAMPLING_DEFAULTS = SamplingDefaults()


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ModelRootsUpdate(StrictRequest):
    model_roots: list[str] = Field(min_length=1, max_length=16)

    @field_validator("model_roots")
    @classmethod
    def model_roots_are_bounded_and_unique(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            candidate = value.strip()
            if not candidate:
                raise ValueError("model root paths must not be empty")
            if len(candidate.encode("utf-8")) > 4096:
                raise ValueError("model root paths must be at most 4096 UTF-8 bytes")
            if any(ord(character) < 0x20 for character in candidate):
                raise ValueError("model root paths must not contain control characters")
            normalized.append(candidate)
        if len(set(normalized)) != len(normalized):
            raise ValueError("model root paths must not contain duplicates")
        return normalized


class _SystemPromptRequest(StrictRequest):
    @field_validator("system_prompt", check_fields=False)
    @classmethod
    def blank_system_prompt_clears(cls, value: str | None) -> str | None:
        """Whitespace-only clears the prompt; otherwise the author's text is kept verbatim."""

        return value if value is not None and value.strip() else None


class ChatCreate(_SystemPromptRequest):
    title: str = Field(default="New chat", max_length=200)
    system_prompt: str | None = Field(default=None, alias="systemPrompt")


class ChatUpdate(_SystemPromptRequest):
    title: str | None = Field(default=None, max_length=200)
    pinned: bool | None = None
    archived: bool | None = None
    # Absent leaves the prompt unchanged (see model_fields_set); null or blank clears it.
    system_prompt: str | None = Field(default=None, alias="systemPrompt")


class MessageCreate(StrictRequest):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = Field(min_length=1)
    parent_id: str | None = Field(default=None, max_length=200)
    branch_index: int = Field(default=0, ge=0)
    attachment_ids: list[str] = Field(default_factory=list, max_length=1024)

    @field_validator("attachment_ids")
    @classmethod
    def attachment_ids_are_unique_and_bounded(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("attachment IDs must be unique")
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("attachment IDs must be non-empty and at most 200 characters")
        return value


class SamplingRequest(StrictRequest):
    max_output_tokens: int = Field(default=_SAMPLING_DEFAULTS.max_output_tokens, ge=1, le=100_000)
    temperature: float = Field(default=_SAMPLING_DEFAULTS.temperature, ge=0.0, le=10.0)
    top_k: int = Field(default=_SAMPLING_DEFAULTS.top_k, ge=0, le=1_000_000)
    top_p: float = Field(default=_SAMPLING_DEFAULTS.top_p, gt=0.0, le=1.0)
    min_p: float = Field(default=_SAMPLING_DEFAULTS.min_p, ge=0.0, le=1.0)
    repetition_penalty: float = Field(
        default=_SAMPLING_DEFAULTS.repetition_penalty, gt=0.0, le=10.0
    )
    frequency_penalty: float = Field(default=_SAMPLING_DEFAULTS.frequency_penalty, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=_SAMPLING_DEFAULTS.presence_penalty, ge=-2.0, le=2.0)
    stop_sequences: list[str] = Field(default_factory=list, max_length=32)
    alternatives: int = Field(default=10, ge=0, le=1000)

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_are_bounded(cls, value: list[str]) -> list[str]:
        if any(not item or len(item.encode("utf-8")) > 1024 for item in value):
            raise ValueError("stop sequences must be non-empty and at most 1024 UTF-8 bytes")
        return value


# Keys of the browser-shaped `settings` object (the frontend's GenerationSettings).
_BROWSER_SETTINGS_KEYS = frozenset(
    {
        "device",
        "dtype",
        "instrumentation",
        "reasoning",
        "seed",
        "deterministic",
        "quantization",
        "strictVram",
    }
)
_BROWSER_SAMPLING_KEYS = {
    "temperature": "temperature",
    "alternatives": "alternatives",
    "maxOutputTokens": "max_output_tokens",
    "topK": "top_k",
    "topP": "top_p",
    "minP": "min_p",
    "repetitionPenalty": "repetition_penalty",
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "stopSequences": "stop_sequences",
}


class GenerationRunCreate(StrictRequest):
    chat_id: str
    model_id: str
    prompt: str = Field(min_length=1)
    parent_message_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    device: DeviceMode | None = None
    dtype: DType | None = None
    instrumentation: InstrumentationLevel | None = None
    reasoning: bool | None = None
    deterministic_reference_mode: bool = False
    # None means the configured runtime default applies.
    quantization: Quantization | None = None
    strict_vram: bool | None = None
    sampling: SamplingRequest = Field(default_factory=SamplingRequest)
    attachment_ids: list[str] = Field(default_factory=list, max_length=1024)

    @model_validator(mode="before")
    @classmethod
    def accept_browser_shape(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        aliases = {
            "chatId": "chat_id",
            "modelId": "model_id",
            "content": "prompt",
            "parentMessageId": "parent_message_id",
            "attachmentIds": "attachment_ids",
        }
        for source, target in aliases.items():
            if source in data and target not in data:
                data[target] = data.pop(source)
        browser_settings = data.pop("settings", None)
        if browser_settings is not None:
            if not isinstance(browser_settings, Mapping):
                raise ValueError("settings must be an object")
            settings = dict(browser_settings)
            unknown = sorted(
                str(key)
                for key in settings
                if key not in _BROWSER_SETTINGS_KEYS and key not in _BROWSER_SAMPLING_KEYS
            )
            if unknown:
                raise ValueError(f"unknown settings fields: {', '.join(unknown[:10])}")
            # Top-level canonical fields always win over the nested browser shape.
            for name in ("device", "dtype", "instrumentation", "reasoning", "quantization"):
                if name in settings and name not in data:
                    data[name] = settings[name]
            if "strictVram" in settings and "strict_vram" not in data:
                data["strict_vram"] = settings["strictVram"]
            raw_seed = settings.get("seed")
            if raw_seed not in (None, "") and "seed" not in data:
                try:
                    data["seed"] = int(raw_seed)
                except (TypeError, ValueError):
                    raise ValueError("settings.seed must be an integer") from None
            if "deterministic" in settings and "deterministic_reference_mode" not in data:
                data["deterministic_reference_mode"] = settings["deterministic"]
            explicit_sampling = data.get("sampling")
            if explicit_sampling is None or isinstance(explicit_sampling, Mapping):
                data["sampling"] = {
                    **{
                        _BROWSER_SAMPLING_KEYS[key]: item
                        for key, item in settings.items()
                        if key in _BROWSER_SAMPLING_KEYS
                    },
                    **(explicit_sampling or {}),
                }
        return data


class TokenBranchCreate(StrictRequest):
    """Select one persisted alternative and continue generation from that token."""

    token_index: int = Field(alias="tokenIndex", ge=0, le=100_000)
    distribution: Literal["raw", "sampling"]
    rank: int = Field(ge=1)
    token_id: int = Field(alias="tokenId", ge=0)


class EmbeddingItemRequest(StrictRequest):
    id: str | None = Field(default=None, max_length=200)
    modality: Literal["text", "image", "video", "audio", "mixed"]
    text: str | None = None
    attachment_id: str | None = Field(default=None, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def accept_browser_shape(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "kind" in data and "modality" not in data:
            data["modality"] = data.pop("kind")
        attachments = data.pop("attachmentIds", None)
        if isinstance(attachments, list) and attachments and "attachment_id" not in data:
            data["attachment_id"] = attachments[0]
        data.pop("label", None)
        return data

    @model_validator(mode="after")
    def require_fields_for_declared_modality(self) -> EmbeddingItemRequest:
        has_text = self.text is not None and bool(self.text)
        has_attachment = self.attachment_id is not None and bool(self.attachment_id)
        if self.modality == "text":
            if not has_text or has_attachment:
                raise ValueError("text inputs require text and must not include an attachment")
        elif self.modality == "mixed":
            if not has_text or not has_attachment:
                raise ValueError("mixed inputs require both text and one attachment")
        elif not has_attachment or has_text:
            raise ValueError(
                f"{self.modality} inputs require one attachment and must not include text"
            )
        return self


class EmbeddingRunCreate(StrictRequest):
    model_id: str
    inputs: list[EmbeddingItemRequest] = Field(min_length=1, max_length=128)
    dimensions: int | None = Field(default=None, ge=1, le=65536)
    normalize: bool = True
    persist_vectors: bool = False

    @model_validator(mode="before")
    @classmethod
    def accept_browser_shape(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "modelId" in data and "model_id" not in data:
            data["model_id"] = data.pop("modelId")
        if "persistVectors" in data and "persist_vectors" not in data:
            data["persist_vectors"] = data.pop("persistVectors")
        return data


class ModelLoadRequest(StrictRequest):
    device: DeviceMode | None = None
    dtype: DType | None = None
    quantization: Quantization | None = None
    strict_vram: bool | None = Field(default=None, alias="strictVram")


class ModelFlushRequest(StrictRequest):
    target_root_index: int = Field(ge=0, alias="targetRootIndex")
    folder_name: str = Field(alias="folderName")

    @field_validator("folder_name")
    @classmethod
    def folder_name_is_safe(cls, value: str) -> str:
        error = folder_name_error(value)
        if error:
            raise ValueError(error)
        return value


class PromptScoreRequest(StrictRequest):
    model_id: str
    text: str = Field(min_length=1)
    device: DeviceMode | None = None
    dtype: DType | None = None


class ChatExportDescriptor(_SystemPromptRequest):
    title: str = Field(max_length=200)
    pinned: bool = False
    archived: bool = False
    system_prompt: str | None = None


class ChatExportMessage(StrictRequest):
    id: str = Field(min_length=1, max_length=200)
    parent_id: str | None = Field(default=None, max_length=200)
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=4_000_000)
    status: Literal["pending", "streaming", "complete", "cancelled", "failed"]
    branch_index: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatExportAlternative(StrictRequest):
    distribution: Literal["raw", "sampling"]
    rank: int = Field(ge=1)
    token_id: int
    piece: str = Field(max_length=65_536)
    logit: float | None = None
    log_probability: float | None = None
    probability: float | None = None
    survived_filter: bool = False


class ChatExportToken(StrictRequest):
    token_index: int = Field(ge=0)
    token_id: int
    piece: str = Field(max_length=65_536)
    escaped_bytes: str = Field(max_length=262_144)
    display_text: str = Field(max_length=262_144)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    raw_logit: float | None = None
    raw_logprob: float | None = None
    raw_probability: float | None = None
    raw_rank: int | None = Field(default=None, ge=1)
    processed_logit: float | None = None
    sample_logprob: float | None = None
    sample_probability: float | None = None
    entropy: float | None = None
    surprise: float | None = None
    cumulative_logprob: float | None = None
    running_perplexity: float | None = None
    decode_ms: float | None = Field(default=None, ge=0)
    sample_ms: float | None = Field(default=None, ge=0)
    emit_ms: float | None = Field(default=None, ge=0)
    inter_token_ms: float | None = Field(default=None, ge=0)
    cumulative_ms: float | None = Field(default=None, ge=0)
    instantaneous_tps: float | None = Field(default=None, ge=0)
    rolling_tps: float | None = Field(default=None, ge=0)
    segment: Literal["reasoning", "answer", "unknown"] = "unknown"
    reasoning_slices: list[dict[str, Any]] = Field(default_factory=list, max_length=1_000)
    selected_experts: Any = None
    attention_attribution: dict[str, Any] | None = None
    alternatives: list[ChatExportAlternative] = Field(default_factory=list, max_length=2_000)


class ChatExportRun(StrictRequest):
    id: str = Field(min_length=1, max_length=200)
    message_id: str | None = Field(default=None, max_length=200)
    parent_run_id: str | None = Field(default=None, max_length=200)
    model_id: str | None = Field(default=None, max_length=500)
    kind: Literal["generation", "embedding", "prompt_score", "benchmark"]
    status: Literal[
        "queued", "loading", "running", "complete", "cancelled", "failed", "disconnected"
    ]
    requested_seed: str | None = Field(default=None, max_length=100)
    effective_seed: str = Field(max_length=100)
    rng_algorithm: str = Field(max_length=200)
    generator_device: str = Field(max_length=500)
    settings: dict[str, Any] = Field(default_factory=dict)
    effective_config: dict[str, Any] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    model_fingerprint: str | None = Field(default=None, max_length=500)
    tokenizer_fingerprint: str | None = Field(default=None, max_length=500)
    rendered_prompt: str | None = Field(default=None, max_length=4_000_000)
    prompt_token_count: int | None = Field(default=None, ge=0)
    generated_token_count: int = Field(default=0, ge=0)
    finish_reason: str | None = Field(default=None, max_length=500)
    error_code: str | None = Field(default=None, max_length=500)
    error_message: str | None = Field(default=None, max_length=4_000)
    tokens: list[ChatExportToken] = Field(default_factory=list, max_length=100_000)


class ChatWorkspaceDocument(StrictRequest):
    schema_identifier: Literal["local-ai-doctor/chat-workspace"] = Field(alias="schema")
    schema_version: Literal[1]
    chat: ChatExportDescriptor
    messages: list[ChatExportMessage] = Field(default_factory=list, max_length=10_000)
    runs: list[ChatExportRun] = Field(default_factory=list, max_length=10_000)


class ErrorEnvelope(BaseModel):
    error: dict[str, Any]
