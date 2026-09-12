"""Evidence-based capability inference; unknowns are never promoted to support."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.capabilities import (
    Capability,
    CapabilityMatrix,
    CapabilityState,
    CapabilitySupport,
)
from ..domain.models import ModelTask
from ..hardware.models import BackendKind


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    task: ModelTask
    modalities: frozenset[str]
    has_tokenizer: bool
    has_processor: bool
    has_reasoning_delimiters: bool
    is_moe: bool
    architecture_known: bool
    available_backends: frozenset[BackendKind] = frozenset({BackendKind.CPU})


def _full() -> CapabilitySupport:
    return CapabilitySupport(state=CapabilityState.FULL)


def _unsupported(reason: str) -> CapabilitySupport:
    return CapabilitySupport(state=CapabilityState.UNSUPPORTED, reason=reason)


def _partial(reason: str, *limitations: str) -> CapabilitySupport:
    return CapabilitySupport(
        state=CapabilityState.PARTIAL,
        reason=reason,
        limitations=limitations,
    )


def _backend(evidence: ModelEvidence, backend: BackendKind) -> CapabilitySupport:
    if backend in evidence.available_backends:
        return _full()
    return CapabilitySupport(
        state=CapabilityState.UNAVAILABLE_ON_BACKEND,
        reason=f"no usable {backend.value} runtime was discovered on this host",
    )


def build_capability_matrix(evidence: ModelEvidence) -> CapabilityMatrix:
    generation = evidence.task in {
        ModelTask.TEXT_GENERATION,
        ModelTask.ENCODER_DECODER_GENERATION,
    }
    embedding = evidence.task in {ModelTask.EMBEDDING, ModelTask.MULTIMODAL_EMBEDDING}
    entries: dict[Capability, CapabilitySupport] = {}

    entries[Capability.TEXT_GENERATION] = (
        _full()
        if evidence.task is ModelTask.TEXT_GENERATION and evidence.has_tokenizer
        else _unsupported(
            "the checkpoint was not identified as a tokenizer-backed causal generator"
        )
    )
    entries[Capability.ENCODER_DECODER_GENERATION] = (
        _full()
        if evidence.task is ModelTask.ENCODER_DECODER_GENERATION and evidence.has_tokenizer
        else _unsupported("the checkpoint was not identified as an encoder-decoder generator")
    )
    entries[Capability.EMBEDDINGS] = (
        _full()
        if embedding
        else _unsupported("the checkpoint has no discovered embedding pipeline")
    )

    for capability, modality in (
        (Capability.VISION, "image"),
        (Capability.AUDIO, "audio"),
        (Capability.VIDEO, "video"),
    ):
        entries[capability] = (
            _full()
            if modality in evidence.modalities and evidence.has_processor
            else _unsupported(f"no native {modality} processor was discovered")
        )

    native_modalities = evidence.modalities.intersection({"image", "audio", "video"})
    entries[Capability.NATIVE_FILE_INPUT] = (
        _partial(
            "only processor-declared native media types are accepted",
            *sorted(native_modalities),
        )
        if native_modalities
        else _unsupported("the model exposes no native file modality")
    )
    entries[Capability.EXTRACTED_TEXT_FILE_INPUT] = (
        _partial(
            "document text extraction is an application preprocessing path, not a native model input"
        )
        if evidence.has_tokenizer
        else _unsupported("no tokenizer-backed text input was discovered")
    )
    entries[Capability.REASONING_CHANNEL] = (
        _full()
        if generation and evidence.has_reasoning_delimiters
        else _unsupported(
            "no explicit emitted reasoning channel or configured delimiter pair was discovered"
        )
    )
    entries[Capability.MOE_ROUTING] = (
        _partial(
            "router tensors require an architecture-specific instrumentation adapter",
            "optimized or fused backends may hide routing",
        )
        if evidence.is_moe
        else _unsupported("not applicable: the checkpoint is dense, not mixture-of-experts")
    )

    generation_only = {
        Capability.RAW_LOGITS: "raw logits exist only for generation/scoring adapters",
        Capability.PROCESSED_LOGITS: "sampler logits exist only for generation adapters",
        Capability.TOP_K_ALTERNATIVES: "token alternatives do not apply to embedding runs",
        Capability.PROMPT_SCORING: "prompt scoring requires a tokenizer-backed generation model",
        Capability.STREAMING: "token streaming does not apply to embedding runs",
        Capability.DETERMINISTIC_SEEDING: "sampling seeds do not select embedding outputs",
    }
    for capability, reason in generation_only.items():
        entries[capability] = _full() if generation else _unsupported(reason)
    entries[Capability.ATTENTION_CAPTURE] = (
        _partial("attention capture is opt-in and can disable optimized attention paths")
        if evidence.architecture_known
        else _unsupported("no compatible architecture adapter was identified")
    )
    entries[Capability.HIDDEN_STATE_CAPTURE] = (
        _partial("hidden-state capture is opt-in and has a bounded trace size")
        if evidence.architecture_known
        else _unsupported("no compatible architecture adapter was identified")
    )
    entries[Capability.BATCHING] = (
        _partial("batching depends on compatible input shapes and the selected adapter")
        if generation or embedding
        else _unsupported("the model task is unknown")
    )

    entries[Capability.CPU] = _backend(evidence, BackendKind.CPU)
    entries[Capability.CUDA] = _backend(evidence, BackendKind.CUDA)
    entries[Capability.ROCM] = _backend(evidence, BackendKind.ROCM)
    entries[Capability.MPS] = _backend(evidence, BackendKind.MPS)
    return CapabilityMatrix(entries=entries)
