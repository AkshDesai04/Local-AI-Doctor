"""Stable extension contracts and deterministic adapter selection."""

from .base import (
    AdapterProbe,
    EmbeddingAdapter,
    EmbeddingInput,
    EmbeddingResult,
    GenerationAdapter,
    GenerationEvent,
    GenerationRequest,
    HardwareBackend,
    LoadedModel,
    ModelAdapter,
    ProcessorAdapter,
    ReasoningSegmentAdapter,
    RouterInstrumentationAdapter,
    TelemetrySink,
)
from .registry import AdapterRegistry

__all__ = [
    "AdapterProbe",
    "AdapterRegistry",
    "EmbeddingAdapter",
    "EmbeddingInput",
    "EmbeddingResult",
    "GenerationAdapter",
    "GenerationEvent",
    "GenerationRequest",
    "HardwareBackend",
    "LoadedModel",
    "ModelAdapter",
    "ProcessorAdapter",
    "ReasoningSegmentAdapter",
    "RouterInstrumentationAdapter",
    "TelemetrySink",
]
