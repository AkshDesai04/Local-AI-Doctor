from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from local_ai_doctor.adapters import (
    AdapterProbe,
    AdapterRegistry,
    EmbeddingAdapter,
    GenerationAdapter,
    LoadedModel,
    ModelAdapter,
    ProcessorAdapter,
)
from local_ai_doctor.config import DeviceMode
from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.domain import ModelDescriptor
from local_ai_doctor.errors import AdapterConflictError, AdapterNotFoundError
from local_ai_doctor.hardware import (
    BackendKind,
    CPUInfo,
    HardwareInventory,
    MemoryInfo,
    select_hardware,
)
from local_ai_doctor.hardware.models import HardwareSelection


class FixtureAdapter(ModelAdapter):
    def __init__(self, name: str, confidence: int, *, supported: bool = True) -> None:
        self._name = name
        self._confidence = confidence
        self._supported = supported

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1"

    def probe(self, model: ModelDescriptor, hardware: HardwareSelection) -> AdapterProbe:
        return AdapterProbe(
            supported=self._supported,
            confidence=self._confidence,
            reason="fixture supports it" if self._supported else "fixture rejects it",
        )

    async def load(
        self,
        model: ModelDescriptor,
        hardware: HardwareSelection,
        effective_config: Mapping[str, Any],
    ) -> LoadedModel:
        raise NotImplementedError

    async def unload(self, model: LoadedModel) -> None:
        raise NotImplementedError

    def generation(self, model: LoadedModel) -> GenerationAdapter | None:
        return None

    def embeddings(self, model: LoadedModel) -> EmbeddingAdapter | None:
        return None

    def processor(self, model: LoadedModel) -> ProcessorAdapter | None:
        return None


def cpu_selection() -> HardwareSelection:
    inventory = HardwareInventory(
        operating_system="test",
        os_release="test",
        python_version="3.12",
        cpu=CPUInfo(logical_cores=1, architecture="fixture"),
        memory=MemoryInfo(source="fixture"),
    )
    selection = select_hardware(inventory, DeviceMode.CPU)
    assert selection.selected_backend is BackendKind.CPU
    return selection


def test_registry_selects_unique_highest_confidence_adapter(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    selected = AdapterRegistry(
        [FixtureAdapter("generic", 50), FixtureAdapter("specific", 90)]
    ).resolve(model, cpu_selection())
    assert selected.name == "specific"


def test_registry_reports_all_probe_reasons_when_none_support(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    registry = AdapterRegistry([FixtureAdapter("rejecting", 0, supported=False)])
    with pytest.raises(AdapterNotFoundError) as caught:
        registry.resolve(model, cpu_selection())
    assert (
        caught.value.to_dict()["details"]["probes"]["rejecting"]["reason"] == "fixture rejects it"
    )


def test_registry_rejects_ambiguous_ties(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    registry = AdapterRegistry([FixtureAdapter("a", 80), FixtureAdapter("b", 80)])
    with pytest.raises(AdapterConflictError):
        registry.resolve(model, cpu_selection())
