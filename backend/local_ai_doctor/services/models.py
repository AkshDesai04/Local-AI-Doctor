"""Model registry, discovery persistence, and explainable lifecycle selection."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ..config import AppSettings, DeviceMode, DType, Quantization
from ..discovery.scanner import ModelScanner, ModelScanReport
from ..domain.capabilities import Capability, CapabilityState
from ..domain.models import ModelDescriptor
from ..domain.quantization import (
    bitsandbytes_available,
    quantization_rejection,
)
from ..errors import (
    CapabilityUnavailableError,
    ModelInvalidError,
    ModelNotFoundError,
    ModelNotResidentError,
    OutOfMemoryError,
    WorkerBusyError,
)
from ..hardware.models import BackendKind, HardwareInventory, HardwareSelection
from ..hardware.probe import SystemHardwareProbe
from ..hardware.selection import available_backends, select_hardware
from ..persistence import WorkspaceRepository
from ..workers import (
    InferenceReservation,
    ModelWorkerSupervisor,
    SingleWorkerAdmission,
    WorkerFailure,
)


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
        # Keys each admitted job has loaded, so its residents are never evicted under it.
        self._job_keys: dict[str, tuple[InferenceReservation, set[str]]] = {}

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
            yield await self.load_reserved(reservation, model_id, device=device, dtype=dtype)

    async def load_reserved(
        self,
        reservation: InferenceReservation,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
        quantization: Quantization | None = None,
        strict_vram: bool | None = None,
        pinned: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Make a model resident for the admitted job that owns ``reservation``.

        ``pinned`` names residents the same job still needs (for example the other
        side of a comparison); they are never evicted to make room.
        """

        if not reservation.active:
            raise RuntimeError("worker model load requires an active inference reservation")
        self._in_use_keys()  # forget finished jobs
        job_keys = self._job_keys.setdefault(reservation.inference_id, (reservation, set()))[1]
        loaded = await self._ensure_resident(
            model_id,
            device=device,
            dtype=dtype,
            quantization=quantization,
            strict_vram=strict_vram,
            pinned=frozenset(pinned) | frozenset(job_keys),
        )
        job_keys.add(str(loaded["model_key"]))
        return loaded

    async def load(
        self,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
        quantization: Quantization | None = None,
        strict_vram: bool | None = None,
    ) -> dict[str, Any]:
        async with self.admission.lifecycle("load"):
            return await self._ensure_resident(
                model_id,
                device=device,
                dtype=dtype,
                quantization=quantization,
                strict_vram=strict_vram,
            )

    def _runtime_payload(
        self,
        selection: HardwareSelection,
        *,
        placement: str,
        strict_vram: bool,
        quantization: str,
    ) -> dict[str, Any]:
        runtime = self.settings.runtime
        return {
            "device": selection.device_identifier,
            "dtype": selection.effective_dtype.value,
            "cpu_threads": runtime.cpu_threads,
            "attention_backend": runtime.attention_backend.value,
            "low_memory_loading": runtime.low_memory_loading,
            "placement": placement,
            "strict_vram": strict_vram,
            "quantization": quantization,
            "kv_reserve_tokens": runtime.kv_reserve_tokens,
            "safety_margin_bytes": runtime.vram_safety_margin_bytes,
            "vram_budget_bytes": runtime.vram_budget_bytes,
            "ram_budget_bytes": runtime.ram_budget_bytes,
            "max_concurrent_runs": runtime.max_concurrent_runs,
        }

    def _lru_victims(self, pinned: frozenset[str]) -> list[dict[str, Any]]:
        """Residents that may be evicted, least recently used first."""

        return [entry for entry in self.worker.resident if entry.get("model_key") not in pinned]

    async def _evict(self, model_key: str) -> None:
        await self.worker.unload(
            model_key=model_key, timeout_seconds=self.settings.workers.unload_timeout_seconds
        )

    async def _ensure_resident(
        self,
        model_id: str,
        *,
        device: DeviceMode | None = None,
        dtype: DType | None = None,
        quantization: Quantization | None = None,
        strict_vram: bool | None = None,
        pinned: frozenset[str] = frozenset(),
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
        quantization_value = self.check_quantization(descriptor, selection, quantization)
        strict = self.settings.runtime.strict_vram if strict_vram is None else strict_vram
        cuda = selection.selected_backend is BackendKind.CUDA
        key = model_key(descriptor, selection, quantization_value)
        existing = next(
            (entry for entry in self.worker.resident if entry.get("model_key") == key), None
        )
        if existing is not None and (not strict or not cuda or existing.get("placement") == "gpu"):
            self.worker.touch(key)
            return self._loaded_view(descriptor, selection, existing, evicted=[])
        evicted: list[str] = []
        if existing is not None:
            # Strict VRAM asks for this model entirely on the GPU; re-place the offloaded one.
            await self._evict(key)

        while len(self.worker.resident) >= self.settings.runtime.max_loaded_models:
            victims = self._lru_victims(pinned)
            if not victims:
                raise WorkerBusyError(
                    "resident model limit reached",
                    hint="Unload a model or raise runtime.max_loaded_models.",
                    details={
                        "max_loaded_models": self.settings.runtime.max_loaded_models,
                        "resident_model_keys": [
                            entry.get("model_key") for entry in self.worker.resident
                        ],
                        "pinned_model_keys": sorted(pinned),
                    },
                )
            victim = str(victims[0]["model_key"])
            await self._evict(victim)
            evicted.append(victim)

        placement = "gpu_only" if cuda else "cpu"
        while True:
            try:
                loaded = await self.worker.load_model(
                    descriptor,
                    self._runtime_payload(
                        selection,
                        placement=placement,
                        strict_vram=strict,
                        quantization=quantization_value,
                    ),
                    model_key=key,
                    timeout_seconds=self.settings.workers.load_timeout_seconds,
                )
                break
            except WorkerFailure as exc:
                if exc.error.get("code") != "insufficient_memory":
                    raise
                failure = exc.error
            required = int(failure.get("required_bytes") or 0)
            available = int(failure.get("available_bytes") or 0)
            measure = "gpu_bytes" if failure.get("memory_kind") == "vram" else "cpu_bytes"
            # Evict least-recently-used idle residents only when together they free enough.
            plan: list[str] = []
            reclaimable = available
            for entry in self._lru_victims(pinned):
                if reclaimable >= required:
                    break
                plan.append(str(entry["model_key"]))
                reclaimable += int(entry.get(measure) or 0)
            if plan and reclaimable >= required and placement != "offload":
                for victim in plan:
                    await self._evict(victim)
                    evicted.append(victim)
                continue
            offload_capable = descriptor.capabilities.support(Capability.CPU_OFFLOAD).state in {
                CapabilityState.FULL,
                CapabilityState.PARTIAL,
            }
            if not strict and cuda and placement == "gpu_only" and offload_capable:
                placement = "offload"
                continue
            raise OutOfMemoryError(
                "the model does not fit the available memory",
                hint=(
                    "Quantize the model, unload another model, or turn off Strict VRAM to allow "
                    "system-RAM offload."
                    if failure.get("memory_kind") == "vram"
                    else "Unload another model or choose a smaller model."
                ),
                details={
                    "memory_kind": failure.get("memory_kind"),
                    "required_bytes": required,
                    "available_bytes": available,
                    "estimate": failure.get("estimate"),
                    "strict_vram": strict,
                    "placement": placement,
                    "resident_model_keys": [
                        entry.get("model_key") for entry in self.worker.resident
                    ],
                    "pinned_model_keys": sorted(pinned),
                    "evicted_model_keys": evicted,
                },
            )
        return self._loaded_view(descriptor, selection, loaded, evicted=evicted)

    def check_quantization(
        self,
        descriptor: ModelDescriptor,
        selection: HardwareSelection,
        quantization: Quantization | None,
    ) -> str:
        """The effective quantization value, or a 409 naming why it cannot load."""

        value = (quantization or self.settings.runtime.quantization).value
        stored = descriptor.metadata.get("weight_quantization") or {}
        reason = quantization_rejection(
            value,
            task=descriptor.task.value,
            cuda=selection.selected_backend is BackendKind.CUDA,
            prequantized=stored.get("method") == "bitsandbytes",
            backend_available=bitsandbytes_available(),
        )
        if reason:
            raise CapabilityUnavailableError(
                reason,
                hint=(
                    "Choose quantization None, or a CUDA device and a decoder-only text "
                    "generation model."
                ),
                details={"quantization": value, "model_id": descriptor.id},
            )
        return value

    @staticmethod
    def _loaded_view(
        descriptor: ModelDescriptor,
        selection: HardwareSelection,
        loaded: dict[str, Any],
        *,
        evicted: list[str],
    ) -> dict[str, Any]:
        return {
            **descriptor.public_dict(reveal_path=False),
            "lifecycle": "loaded",
            "loaded_device": loaded.get("device"),
            "selection": selection.model_dump(mode="json"),
            "load": loaded,
            "model_key": loaded.get("model_key"),
            "placement": loaded.get("placement"),
            "quantization": loaded.get("quantization"),
            "strict_vram": loaded.get("strict_vram"),
            "evicted_model_keys": evicted,
        }

    async def unload(self, model_key: str | None = None) -> dict[str, Any]:
        async with self.admission.lifecycle("unload"):
            timeout = self.settings.workers.unload_timeout_seconds
            if model_key is None:
                return await self.worker.unload(timeout_seconds=timeout)
            resident = [
                entry for entry in self.worker.resident if entry.get("model_key") == model_key
            ]
            self._require_resident(
                [str(entry["model_key"]) for entry in resident], model_key=model_key
            )
            return await self.worker.unload(model_key=model_key, timeout_seconds=timeout)

    async def unload_model(self, model_id: str) -> dict[str, Any]:
        """Unload every resident of one model (each device/dtype selection is its own key)."""

        async with self.admission.lifecycle("unload"):
            keys = [
                str(entry["model_key"])
                for entry in self.worker.resident
                if entry.get("model_id") == model_id
            ]
            self._require_resident(keys, model_id=model_id)
            results = [
                await self.worker.unload(
                    model_key=key, timeout_seconds=self.settings.workers.unload_timeout_seconds
                )
                for key in keys
            ]
        return {
            "unloaded_model_id": model_id,
            "unloaded_model_keys": keys,
            "freed_bytes": sum(int(item.get("freed_bytes") or 0) for item in results),
            "leaked_bytes": sum(int(item.get("leaked_bytes") or 0) for item in results),
            "ledger": results[-1].get("ledger"),
        }

    @staticmethod
    def _require_resident(keys: list[str], **details: str) -> None:
        if not keys:
            raise ModelNotResidentError(
                "the model is not resident",
                hint="Refresh the resident model list; it may already have been unloaded.",
                details=details,
            )

    def _in_use_keys(self) -> set[str]:
        """Resident keys an admitted, still-active job has loaded."""

        for job_id, (reservation, _keys) in tuple(self._job_keys.items()):
            if not reservation.active:
                del self._job_keys[job_id]
        return {key for _reservation, keys in self._job_keys.values() for key in keys}

    async def resident_status(self) -> dict[str, Any]:
        """Resident models plus the worker memory ledger.

        The ledger is refreshed only while the worker is idle: a request that times
        out recycles the worker, so an admitted job is never interrupted for it.
        """

        stale = True
        if self.worker.available:
            try:
                async with self.admission.lifecycle("memory_status"):
                    await self.worker.memory_status(
                        timeout_seconds=self.settings.workers.unload_timeout_seconds
                    )
                stale = False
            except (WorkerBusyError, WorkerFailure):
                pass
        in_use = self._in_use_keys()
        ledger = self.worker.ledger or {}
        memory = {
            field: ledger.get(field)
            for field in (
                "device",
                "total_bytes",
                "free_bytes",
                "torch_allocated_bytes",
                "torch_reserved_bytes",
                "cap_bytes",
                "process_rss_bytes",
                "system_available_bytes",
            )
        }
        age = self.worker.ledger_age_seconds
        memory.update(
            safety_margin_bytes=self.settings.runtime.vram_safety_margin_bytes,
            ledger_age_seconds=age,
            stale=stale or age is None,
        )
        return {
            "models": [
                {
                    "model_key": entry.get("model_key"),
                    "model_id": entry.get("model_id"),
                    "display_name": entry.get("display_name"),
                    "device": entry.get("device"),
                    "dtype": entry.get("dtype"),
                    "quantization": entry.get("quantization"),
                    "strict_vram": entry.get("strict_vram"),
                    "placement": entry.get("placement"),
                    "gpu_bytes": entry.get("gpu_bytes"),
                    "cpu_bytes": entry.get("cpu_bytes"),
                    "kv_reserve_bytes": entry.get("kv_reserve_bytes"),
                    "load_seconds": entry.get("load_seconds"),
                    "last_used_at": entry.get("last_used_at"),
                    "in_use": entry.get("model_key") in in_use,
                }
                for entry in self.worker.resident
            ],
            "memory": memory,
            "max_loaded_models": self.settings.runtime.max_loaded_models,
        }


def model_key(descriptor: ModelDescriptor, selection: HardwareSelection, quantization: str) -> str:
    """Stable resident identity: checkpoint, device, effective dtype, and quantization."""

    identity = [
        descriptor.id,
        descriptor.fingerprint.value,
        selection.device_identifier,
        selection.effective_dtype.value,
        quantization,
    ]
    return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:20]
