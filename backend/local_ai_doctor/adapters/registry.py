"""Explicit adapter registry with explainable, deterministic resolution."""

from __future__ import annotations

from collections.abc import Iterable

from ..domain.models import ModelDescriptor
from ..errors import AdapterConflictError, AdapterNotFoundError
from ..hardware.models import HardwareSelection
from .base import AdapterProbe, ModelAdapter


class AdapterRegistry:
    def __init__(self, adapters: Iterable[ModelAdapter] = ()) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ModelAdapter) -> None:
        name = adapter.name.strip()
        if not name:
            raise ValueError("adapter name must not be empty")
        if name in self._adapters:
            raise AdapterConflictError(
                f"adapter {name!r} is already registered",
                details={"adapter": name},
            )
        self._adapters[name] = adapter

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def probes(
        self, model: ModelDescriptor, hardware: HardwareSelection
    ) -> tuple[tuple[ModelAdapter, AdapterProbe], ...]:
        return tuple(
            (adapter, adapter.probe(model, hardware))
            for adapter in sorted(self._adapters.values(), key=lambda item: item.name)
        )

    def resolve(self, model: ModelDescriptor, hardware: HardwareSelection) -> ModelAdapter:
        probes = self.probes(model, hardware)
        supported = [(adapter, probe) for adapter, probe in probes if probe.supported]
        if not supported:
            raise AdapterNotFoundError(
                f"no registered adapter can load model {model.display_name!r}",
                hint="Inspect the capability report or add a model-specific adapter.",
                details={
                    "model_id": model.id,
                    "probes": {
                        adapter.name: {
                            "reason": probe.reason,
                            "limitations": probe.limitations,
                        }
                        for adapter, probe in probes
                    },
                },
            )
        highest = max(probe.confidence for _, probe in supported)
        winners = [(adapter, probe) for adapter, probe in supported if probe.confidence == highest]
        if len(winners) > 1:
            raise AdapterConflictError(
                "multiple adapters have equal highest confidence",
                hint="Tighten adapter probes so the most specific adapter wins unambiguously.",
                details={
                    "model_id": model.id,
                    "confidence": highest,
                    "adapters": [adapter.name for adapter, _ in winners],
                },
            )
        return winners[0][0]
