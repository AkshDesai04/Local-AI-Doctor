"""Model registry, discovery persistence, and explainable lifecycle selection."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ..config import AppSettings, DeviceMode, DType
from ..discovery.scanner import ModelScanner, ModelScanReport
from ..domain.models import ModelDescriptor
from ..errors import (
    CapabilityUnavailableError,
    ModelInvalidError,
    ModelNotFoundError,
    OutOfMemoryError,
)
from ..hardware.models import BackendKind, HardwareInventory, HardwareSelection
from ..hardware.probe import SystemHardwareProbe
from ..hardware.selection import available_backends, select_hardware
from ..persistence import WorkspaceRepository
from ..workers import InferenceReservation, ModelWorkerSupervisor, SingleWorkerAdmission


class ModelRegistry:
    def __init__(
        self,
        settings: AppSettings,
        repository: WorkspaceRepository,
        worker: ModelWorkerSupervisor,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.worker = worker
        self.hardware: HardwareInventory | None = None
        self.report: ModelScanReport | None = None
        self._models: dict[str, ModelDescriptor] = {}
        self.admission = SingleWorkerAdmission(settings.runtime.queue_limit)

    def _discover_hardware(self) -> HardwareInventory:
        return SystemHardwareProbe().discover(
            probe_runtime=self.settings.runtime.device is not DeviceMode.CPU
        )

    async def refresh(self) -> ModelScanReport:
        self.hardware = self._discover_hardware()
        scanner = ModelScanner(
            self.settings.paths.model_roots,
            conservative_context_limit=self.settings.inference.conservative_context_limit,
            available_backends=available_backends(self.hardware),
        )
        self.report = await asyncio.to_thread(scanner.scan)
        self._models = self.report.by_id
        for descriptor in self.report.models:
            await self.repository.upsert_model(
                {
                    "id": descriptor.id,
                    "canonical_path": str(descriptor.path),
                    "display_name": descriptor.display_name,
                    "architecture": descriptor.architectures[0]
                    if descriptor.architectures
                    else None,
                    "task": descriptor.task.value,
                    "fingerprint": descriptor.fingerprint.value,
                    "fingerprint_algorithm": descriptor.fingerprint.algorithm,
                    "parameter_count": descriptor.parameter_count,
                    "weight_bytes": descriptor.fingerprint.total_weight_bytes,
                    "metadata": descriptor.metadata,
                    "diagnostics": [
                        item.model_dump(mode="json") for item in descriptor.diagnostics
                    ],
                }
            )
            await self.repository.replace_capabilities(
                descriptor.id,
                {
                    capability.value: support.model_dump(mode="json")
                    for capability, support in descriptor.capabilities.entries.items()
                },
            )
        return self.report

    def get(self, model_id: str) -> ModelDescriptor:
        descriptor = self._models.get(model_id)
        if descriptor is None:
            raise ModelNotFoundError(
                "the selected model is not in the current registry",
                hint="Refresh the model registry and select a discovered model.",
                details={"model_id": model_id},
            )
        return descriptor

    def public_models(self) -> list[dict[str, Any]]:
        return [descriptor.public_dict(reveal_path=False) for descriptor in self._models.values()]

    def public_report(self) -> dict[str, Any]:
        report = self.report
        if report is None:
            return {"models": [], "roots": []}
        return {"models": self.public_models(), "roots": report.public_roots()}

    def choose_hardware(
        self,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
    ) -> HardwareSelection:
        if self.hardware is None:
            self.hardware = self._discover_hardware()
        return select_hardware(
            self.hardware,
            device or self.settings.runtime.device,
            requested_dtype=dtype or self.settings.runtime.dtype,
            allow_cpu_fallback=self.settings.runtime.allow_cpu_fallback,
        )

    def reserve_inference(self, inference_id: str, kind: str) -> InferenceReservation:
        return self.admission.reserve(inference_id, kind)

    @asynccontextmanager
    async def inference_session(
        self,
        reservation: InferenceReservation,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Hold exclusive worker ownership from model load through inference."""

        async with reservation:
            yield await self._load_for_worker(model_id, device=device, dtype=dtype)

    async def load_reserved(
        self,
        reservation: InferenceReservation,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
    ) -> dict[str, Any]:
        if not reservation.active:
            raise RuntimeError("worker model load requires an active inference reservation")
        return await self._load_for_worker(model_id, device=device, dtype=dtype)

    async def load(
        self,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
    ) -> dict[str, Any]:
        async with self.admission.lifecycle("load"):
            return await self._load_for_worker(model_id, device=device, dtype=dtype)

    async def _load_for_worker(
        self,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
    ) -> dict[str, Any]:
        descriptor = self.get(model_id)
        if not descriptor.loadable:
            raise ModelInvalidError(
                "the selected model has blocking discovery diagnostics",
                hint="Open the model registry diagnostics before loading it.",
                details={
                    "model_id": model_id,
                    "errors": [
                        item.code
                        for item in descriptor.diagnostics
                        if item.severity.value == "error"
                    ],
                },
            )
        selection = self.choose_hardware(device=device, dtype=dtype)
        if self.settings.runtime.quantization.value != "none":
            raise CapabilityUnavailableError(
                "the requested model-weight quantization has no installed compatible adapter",
                hint="Use runtime.quantization=none or install and register a reviewed quantization adapter.",
                details={"quantization": self.settings.runtime.quantization.value},
            )
        if self.settings.runtime.cpu_offload:
            raise CapabilityUnavailableError(
                "CPU offload is configured but is not exposed by the selected reference adapter",
                hint="Disable runtime.cpu_offload or add a capability-gated placement adapter.",
            )
        weight_bytes = descriptor.fingerprint.total_weight_bytes
        budget = (
            self.settings.runtime.vram_budget_bytes
            if selection.selected_backend is BackendKind.CUDA
            else self.settings.runtime.ram_budget_bytes
        )
        if budget is not None and weight_bytes > budget:
            raise OutOfMemoryError(
                "the checkpoint weights exceed the configured memory budget",
                hint="Raise the local budget, choose CPU, or select a smaller compatible model.",
                details={
                    "selected_backend": selection.selected_backend.value,
                    "weight_bytes": weight_bytes,
                    "configured_budget_bytes": budget,
                },
            )
        runtime = {
            "device": selection.device_identifier,
            "dtype": selection.effective_dtype.value,
            "cpu_threads": self.settings.runtime.cpu_threads,
            "attention_backend": self.settings.runtime.attention_backend.value,
            "low_memory_loading": self.settings.runtime.low_memory_loading,
            "cpu_offload": self.settings.runtime.cpu_offload,
        }
        loaded = await self.worker.load_model(
            descriptor,
            runtime,
            timeout_seconds=self.settings.workers.load_timeout_seconds,
        )
        return {
            **descriptor.public_dict(reveal_path=False),
            "lifecycle": "loaded",
            "loaded_device": loaded.get("device"),
            "selection": selection.model_dump(mode="json"),
            "load": loaded,
        }

    async def unload(self) -> dict[str, Any]:
        async with self.admission.lifecycle("unload"):
            return await self.worker.unload(
                timeout_seconds=self.settings.workers.unload_timeout_seconds
            )
