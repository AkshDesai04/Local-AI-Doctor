"""Interfaces between architecture-neutral orchestration and model runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import InstrumentationLevel
from ..domain.capabilities import CapabilityMatrix
from ..domain.models import ModelDescriptor
from ..hardware.models import HardwareInventory, HardwareSelection
from ..sampling.metrics import SamplingSettings


class AdapterProbe(BaseModel):
    """A transparent, comparable claim that an adapter can load a model."""

    model_config = ConfigDict(frozen=True)

    supported: bool
    confidence: int = Field(ge=0, le=100)
    reason: str
    limitations: tuple[str, ...] = ()


class LoadedModel(BaseModel):
    """Opaque model-session identity; heavyweight objects stay inside workers."""

    model_config = ConfigDict(frozen=True)

    handle_id: str
    model_id: str
    adapter_name: str
    backend: str
    device: str
    dtype: str
    loaded_at: datetime
    capabilities: CapabilityMatrix


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int = Field(ge=1, le=1_000_000)
    requested_seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    effective_seed: int = Field(ge=0, le=2**64 - 1)
    sampling_settings: SamplingSettings
    stop_token_ids: tuple[int, ...] = ()
    stop_sequences: tuple[str, ...] = ()
    instrumentation: InstrumentationLevel = InstrumentationLevel.TOKEN


class GenerationEventType(StrEnum):
    STAGE = "stage"
    TOKEN = "token"
    WARNING = "warning"
    COMPLETE = "complete"


class GenerationEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    type: GenerationEventType
    monotonic_seconds: float = Field(ge=0.0)
    payload: dict[str, Any] = Field(default_factory=dict)


class EmbeddingInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_id: str
    modality: str
    text: str | None = None
    content_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_id: str
    vector: tuple[float, ...]
    output_dimension: int = Field(ge=1)
    output_dtype: str
    normalized: bool
    l2_norm: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedInput(BaseModel):
    """Serializable processor result suitable for an isolated worker boundary."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, Any]
    modality_positions: dict[str, int] = Field(default_factory=dict)
    truncation: dict[str, Any] = Field(default_factory=dict)


class RouterCapture(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_index: int = Field(ge=0)
    token_index: int = Field(ge=0)
    selected_expert_ids: tuple[int, ...]
    selected_weights: tuple[float, ...]
    executed_expert_ids: tuple[int, ...] = ()
    router_entropy: float | None = Field(default=None, ge=0.0)
    dropped_assignments: int | None = Field(default=None, ge=0)


class ModelAdapter(ABC):
    """Architecture-specific model lifecycle contract."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def probe(self, model: ModelDescriptor, hardware: HardwareSelection) -> AdapterProbe:
        """Inspect metadata only; this method must not load model weights."""

    @abstractmethod
    async def load(
        self,
        model: ModelDescriptor,
        hardware: HardwareSelection,
        effective_config: Mapping[str, Any],
    ) -> LoadedModel: ...

    @abstractmethod
    async def unload(self, model: LoadedModel) -> None: ...

    @abstractmethod
    def generation(self, model: LoadedModel) -> GenerationAdapter | None: ...

    @abstractmethod
    def embeddings(self, model: LoadedModel) -> EmbeddingAdapter | None: ...

    @abstractmethod
    def processor(self, model: LoadedModel) -> ProcessorAdapter | None: ...


class GenerationAdapter(ABC):
    """Own prompt prefill, KV-cache decoding, sampling, and cancellation."""

    @abstractmethod
    def generate(
        self,
        model: LoadedModel,
        request: GenerationRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[GenerationEvent]: ...

    @abstractmethod
    async def score_prompt(
        self,
        model: LoadedModel,
        token_ids: Sequence[int],
        attention_mask: Sequence[bool] | None = None,
    ) -> Mapping[str, Any]: ...


class EmbeddingAdapter(ABC):
    @abstractmethod
    async def embed(
        self,
        model: LoadedModel,
        inputs: Sequence[EmbeddingInput],
        *,
        output_dimension: int | None = None,
        normalize: bool = True,
    ) -> Sequence[EmbeddingResult]: ...


class ProcessorAdapter(ABC):
    @property
    @abstractmethod
    def supported_modalities(self) -> frozenset[str]: ...

    @abstractmethod
    async def process(
        self,
        inputs: Sequence[EmbeddingInput],
        *,
        maximum_length: int,
    ) -> ProcessedInput: ...


class ReasoningSegmentAdapter(ABC):
    @abstractmethod
    def feed(self, token_index: int, text: str) -> Sequence[Any]: ...

    @abstractmethod
    def finalize(self) -> Sequence[Any]: ...


class RouterInstrumentationAdapter(ABC):
    @property
    @abstractmethod
    def changes_execution_path(self) -> bool: ...

    @abstractmethod
    async def start_capture(
        self,
        model: LoadedModel,
        *,
        include_prompt: bool,
        token_limit: int,
    ) -> None: ...

    @abstractmethod
    async def stop_capture(self) -> Sequence[RouterCapture]: ...


class HardwareBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def discover(self) -> HardwareInventory: ...

    @abstractmethod
    def select(self, inventory: HardwareInventory) -> HardwareSelection: ...

    @abstractmethod
    async def release_memory(self, selection: HardwareSelection) -> None: ...


class TelemetrySink(ABC):
    """Non-blocking hot-path boundary; implementations should batch writes."""

    @abstractmethod
    def emit_nowait(self, event: Mapping[str, Any]) -> bool:
        """Queue an event and return False when bounded backpressure rejects it."""

    @abstractmethod
    async def flush(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class CancellationToken(ABC):
    @property
    @abstractmethod
    def cancelled(self) -> bool: ...

    @abstractmethod
    def raise_if_cancelled(self) -> None: ...
