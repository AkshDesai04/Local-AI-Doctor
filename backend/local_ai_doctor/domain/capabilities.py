"""Machine-readable model/backend capability declarations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CapabilityState(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE_ON_BACKEND = "unavailable_on_backend"


class Capability(StrEnum):
    TEXT_GENERATION = "text_generation"
    ENCODER_DECODER_GENERATION = "encoder_decoder_generation"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    NATIVE_FILE_INPUT = "native_file_input"
    EXTRACTED_TEXT_FILE_INPUT = "extracted_text_file_input"
    REASONING_CHANNEL = "reasoning_channel"
    MOE_ROUTING = "moe_routing"
    RAW_LOGITS = "raw_logits"
    PROCESSED_LOGITS = "processed_logits"
    TOP_K_ALTERNATIVES = "top_k_alternatives"
    PROMPT_SCORING = "prompt_scoring"
    ATTENTION_CAPTURE = "attention_capture"
    HIDDEN_STATE_CAPTURE = "hidden_state_capture"
    STREAMING = "streaming"
    BATCHING = "batching"
    DETERMINISTIC_SEEDING = "deterministic_seeding"
    CPU_OFFLOAD = "cpu_offload"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"


class CapabilitySupport(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: CapabilityState
    reason: str | None = None
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def non_full_requires_reason(self) -> CapabilitySupport:
        if self.state is not CapabilityState.FULL and not self.reason:
            raise ValueError("a reason is required whenever capability support is not full")
        return self


class CapabilityMatrix(BaseModel):
    """Complete matrix: absence is never confused with lack of support."""

    model_config = ConfigDict(frozen=True)

    entries: dict[Capability, CapabilitySupport] = Field(default_factory=dict)

    @model_validator(mode="after")
    def contains_every_capability(self) -> CapabilityMatrix:
        missing = set(Capability).difference(self.entries)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"capability matrix is incomplete; missing: {names}")
        return self

    def support(self, capability: Capability) -> CapabilitySupport:
        return self.entries[capability]

    def supports(self, capability: Capability) -> bool:
        return self.support(capability).state is CapabilityState.FULL

    @classmethod
    def all_unsupported(cls, reason: str) -> CapabilityMatrix:
        return cls(
            entries={
                capability: CapabilitySupport(
                    state=CapabilityState.UNSUPPORTED,
                    reason=reason,
                )
                for capability in Capability
            }
        )
