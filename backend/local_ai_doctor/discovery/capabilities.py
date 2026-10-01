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
    # `quantization_config.quant_method` of a checkpoint that stores quantized weights.
    weight_quantization: str | None = None
    # Whether the bitsandbytes package is installed on this host.
    quantization_backend: bool = False
    # Codes of error-severity discovery diagnostics. Any one of them means the
    # checkpoint cannot be loaded, so no capability may be advertised.
    blocking_diagnostics: tuple[str, ...] = ()


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
    if backend not in {BackendKind.CPU, BackendKind.CUDA}:
        return CapabilitySupport(
            state=CapabilityState.UNAVAILABLE_ON_BACKEND,
            reason=f"the {backend.value} execution adapter is not implemented",
        )
    if backend in evidence.available_backends:
        return _full()
    return CapabilitySupport(
        state=CapabilityState.UNAVAILABLE_ON_BACKEND,
        reason=f"no usable {backend.value} runtime was discovered on this host",
    )


def _weight_quantization(evidence: ModelEvidence) -> CapabilitySupport:
    cuda = BackendKind.CUDA in evidence.available_backends
    if evidence.weight_quantization == "bitsandbytes":
        return (
            _partial("pre-quantized bitsandbytes checkpoint; loads as-is; cannot be re-quantized")
            if cuda
            else CapabilitySupport(
                state=CapabilityState.UNAVAILABLE_ON_BACKEND,
                reason="pre-quantized bitsandbytes checkpoints need a usable CUDA runtime",
            )
        )
    if evidence.weight_quantization is not None:
        return _unsupported(
            f"the checkpoint is pre-quantized with {evidence.weight_quantization}; "
            "only bitsandbytes is recognized"
        )
    if evidence.task is not ModelTask.TEXT_GENERATION:
        return _unsupported(
            "load-time weight quantization is implemented only for decoder-only text generation"
        )
    if not cuda or not evidence.quantization_backend:
        return CapabilitySupport(
            state=CapabilityState.UNAVAILABLE_ON_BACKEND,
            reason=(
                "bitsandbytes quantization needs a usable CUDA runtime"
                if not cuda
                else "the bitsandbytes package is not installed (install the CUDA extra)"
            ),
        )
    return _partial(
        "bitsandbytes NF4 4-bit / LLM.int8 8-bit at load time; CUDA only; outputs differ "
        "from the checkpoint dtype",
        "the quantized copy lives only in GPU memory until Flush to storage writes it",
    )


def build_capability_matrix(evidence: ModelEvidence) -> CapabilityMatrix:
    if evidence.blocking_diagnostics:
        return CapabilityMatrix.all_unsupported(
            "the model is not loadable; blocking discovery diagnostic: "
            + ", ".join(evidence.blocking_diagnostics)
        )
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
        if modality not in evidence.modalities or not evidence.has_processor:
            entries[capability] = _unsupported(f"no native {modality} processor was discovered")
        elif embedding:
            entries[capability] = _full()
        elif evidence.task is not ModelTask.TEXT_GENERATION or modality == "audio":
            entries[capability] = _unsupported(
                f"{modality} chat input is not implemented for this generation path"
            )
        else:
            # Chat media goes through the checkpoint's own processor and chat template.
            entries[capability] = _partial(
                "validated on Qwen3-VL; other processor families best-effort",
                "media is decoded only from uploaded attachments",
                "video frames are sampled by the processor's own defaults",
            )

    native_modalities = {
        modality
        for capability, modality in (
            (Capability.VISION, "image"),
            (Capability.AUDIO, "audio"),
            (Capability.VIDEO, "video"),
        )
        if entries[capability].state is not CapabilityState.UNSUPPORTED
    }
    entries[Capability.NATIVE_FILE_INPUT] = (
        _partial(
            "only processor-declared native media types are accepted",
            *sorted(native_modalities),
        )
        if native_modalities
        else _unsupported("the model exposes no native file modality")
    )
    entries[Capability.EXTRACTED_TEXT_FILE_INPUT] = _unsupported(
        "no extracted-text inference adapter is currently registered"
    )
    entries[Capability.REASONING_CHANNEL] = (
        _full()
        if generation and evidence.has_reasoning_delimiters
        else _unsupported(
            "no explicit emitted reasoning channel or configured delimiter pair was discovered"
        )
    )
    entries[Capability.MOE_ROUTING] = (
        _unsupported(
            "the checkpoint appears to use experts, but no production router instrumentation adapter is registered"
        )
        if evidence.is_moe
        else _unsupported("not applicable: the checkpoint is dense, not mixture-of-experts")
    )

    generation_only = {
        Capability.RAW_LOGITS: "raw logits exist only for generation/scoring adapters",
        Capability.PROCESSED_LOGITS: "sampler logits exist only for generation adapters",
        Capability.TOP_K_ALTERNATIVES: "token alternatives do not apply to embedding runs",
        Capability.STREAMING: "token streaming does not apply to embedding runs",
        Capability.DETERMINISTIC_SEEDING: "sampling seeds do not select embedding outputs",
    }
    for capability, reason in generation_only.items():
        entries[capability] = _full() if generation else _unsupported(reason)
    entries[Capability.PROMPT_SCORING] = (
        _full()
        if evidence.task is ModelTask.TEXT_GENERATION
        else _unsupported(
            "encoder-decoder prompt scoring requires separate source and target text"
            if evidence.task is ModelTask.ENCODER_DECODER_GENERATION
            else "prompt scoring requires a tokenizer-backed generation model"
        )
    )
    entries[Capability.ATTENTION_CAPTURE] = (
        _partial(
            "full/expert instrumentation captures causal self-attention for supported decoder models",
            "attention weights are model-internal allocations, not causal contribution scores",
            "the retained top source positions are bounded and report omitted attention mass",
        )
        if evidence.task is ModelTask.TEXT_GENERATION
        else _unsupported(
            "causal self-attention capture is available only for decoder-only text generation"
        )
    )
    entries[Capability.TOKEN_INFLUENCE] = (
        _partial(
            "attention allocation and gradient x input saliency; not causal attribution",
            "recomputed on demand from the persisted prefix of a completed or cancelled run",
            "gradient x input needs the model on GPU or CPU (not layer offload) and a prefix "
            "within inference.influence_max_gradient_tokens",
        )
        if evidence.task is ModelTask.TEXT_GENERATION
        else _unsupported(
            "token influence re-runs a decoder-only prefix; encoder-decoder cross-attention "
            "and embedding models are not analyzed"
            if generation
            else "token influence applies only to decoder-only text generation"
        )
    )
    entries[Capability.HIDDEN_STATE_CAPTURE] = _unsupported(
        "the reference adapter does not currently capture hidden-state traces"
    )
    entries[Capability.BATCHING] = (
        _partial("batching depends on compatible input shapes and the selected adapter")
        if generation or embedding
        else _unsupported("the model task is unknown")
    )

    entries[Capability.CPU_OFFLOAD] = (
        _partial(
            "accelerate layer offload when Strict VRAM is off; much slower",
            "layers placed in system RAM are copied to the GPU on every forward pass",
        )
        if generation and BackendKind.CUDA in evidence.available_backends
        else _unsupported(
            "layer offload needs a usable CUDA runtime"
            if generation
            else "layer offload applies only to generation models"
        )
    )
    entries[Capability.WEIGHT_QUANTIZATION] = _weight_quantization(evidence)
    entries[Capability.CPU] = _backend(evidence, BackendKind.CPU)
    entries[Capability.CUDA] = _backend(evidence, BackendKind.CUDA)
    if evidence.weight_quantization == "bitsandbytes":
        entries[Capability.CPU] = _unsupported("bitsandbytes-quantized weights run on CUDA only")
        entries[Capability.CPU_OFFLOAD] = _unsupported(
            "a pre-quantized bitsandbytes checkpoint cannot be offloaded to system RAM"
        )
    entries[Capability.ROCM] = _backend(evidence, BackendKind.ROCM)
    entries[Capability.MPS] = _backend(evidence, BackendKind.MPS)
    return CapabilityMatrix(entries=entries)
