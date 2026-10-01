"""Resident-model lifecycle policy: keys, LRU eviction, strict VRAM, and offload retry."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_ai_doctor.config import AppSettings, DType, Quantization
from local_ai_doctor.domain.capabilities import Capability, CapabilityState
from local_ai_doctor.domain.models import ModelTask
from local_ai_doctor.errors import (
    CapabilityUnavailableError,
    ModelNotResidentError,
    OutOfMemoryError,
    WorkerBusyError,
)
from local_ai_doctor.hardware.models import (
    AcceleratorDevice,
    BackendKind,
    CPUInfo,
    HardwareInventory,
    MemoryInfo,
)
from local_ai_doctor.services.models import ModelRegistry
from local_ai_doctor.workers import WorkerFailure

GiB = 1024**3


class FakeSupervisor:
    """Simulates VRAM: a GPU-only load that does not fit reports insufficient_memory."""

    def __init__(self, free: int, sizes: dict[str, int]) -> None:
        self.free = free
        self.sizes = sizes
        self.entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.loads: list[tuple[str, str]] = []
        self.unloads: list[str] = []
        self.available = True
        self.ledger: dict[str, Any] | None = None
        self.ledger_age_seconds: float | None = None

    @property
    def resident(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries.values()]

    def touch(self, key: str) -> None:
        self.entries.move_to_end(key)

    async def load_model(
        self, descriptor: Any, runtime: dict[str, Any], *, model_key: str, timeout_seconds: float
    ) -> dict[str, Any]:
        placement = runtime["placement"]
        self.loads.append((descriptor.id, placement))
        size = self.sizes[descriptor.id]
        if placement == "gpu_only" and size > self.free:
            raise WorkerFailure(
                {
                    "code": "insufficient_memory",
                    "message": "the model does not fit the available GPU memory",
                    "memory_kind": "vram",
                    "required_bytes": size,
                    "available_bytes": self.free,
                    "estimate": {"weights": size, "kv_reserve": 0, "margin": 0},
                }
            )
        gpu = size if placement == "gpu_only" else min(size, self.free)
        self.free -= gpu
        entry = {
            "model_key": model_key,
            "model_id": descriptor.id,
            "device": runtime["device"],
            "dtype": runtime["dtype"],
            "quantization": runtime["quantization"],
            "strict_vram": runtime["strict_vram"],
            "placement": "gpu" if gpu == size else "offload",
            "gpu_bytes": gpu,
            "cpu_bytes": size - gpu,
        }
        self.entries[model_key] = entry
        return dict(entry)

    async def unload(
        self, *, model_key: str | None = None, timeout_seconds: float
    ) -> dict[str, Any]:
        keys = [model_key] if model_key is not None else list(self.entries)
        freed = 0
        for key in keys:
            entry = self.entries.pop(key)
            freed += entry["gpu_bytes"]
            self.unloads.append(key)
        self.free += freed
        return {"unloaded_model_keys": keys, "freed_bytes": freed, "leaked_bytes": 0, "ledger": {}}

    async def memory_status(self, *, timeout_seconds: float) -> dict[str, Any]:
        self.ledger = {"device": "cuda:0", "total_bytes": 8 * GiB, "free_bytes": self.free}
        self.ledger_age_seconds = 0.0
        return self.ledger


def _descriptor(model_id: str, *, offload: bool = True) -> Any:
    state = CapabilityState.PARTIAL if offload else CapabilityState.UNSUPPORTED
    support = SimpleNamespace(state=state)
    return SimpleNamespace(
        id=model_id,
        display_name=model_id.title(),
        loadable=True,
        diagnostics=[],
        task=ModelTask.TEXT_GENERATION,
        metadata={},
        fingerprint=SimpleNamespace(value=model_id * 8),
        capabilities=SimpleNamespace(
            support=lambda capability: (
                support
                if capability is Capability.CPU_OFFLOAD
                else SimpleNamespace(state=CapabilityState.FULL)
            )
        ),
        public_dict=lambda reveal_path=False: {"id": model_id},
    )


def _registry(
    free: int, sizes: dict[str, int], **runtime: Any
) -> tuple[ModelRegistry, FakeSupervisor]:
    worker = FakeSupervisor(free, sizes)
    registry = ModelRegistry(
        AppSettings.model_validate({"runtime": runtime}),
        repository=cast(Any, None),
        worker=cast(Any, worker),
    )
    registry.hardware = HardwareInventory(
        operating_system="test",
        os_release="1",
        python_version="3.12",
        cpu=CPUInfo(logical_cores=4, architecture="x86_64"),
        memory=MemoryInfo(source="test"),
        accelerators=(
            AcceleratorDevice(
                backend=BackendKind.CUDA,
                index=0,
                name="GPU",
                identifier="cuda:0",
                compute_capability=(8, 9),
                runtime_available=True,
            ),
        ),
    )
    registry._models = {
        model_id: _descriptor(model_id, offload=model_id != "embedder") for model_id in sizes
    }
    return registry, worker


def _keys(worker: FakeSupervisor) -> list[str]:
    return [str(entry["model_id"]) for entry in worker.resident]


def test_models_share_the_gpu_and_the_same_selection_reuses_its_key() -> None:
    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 4 * GiB, "b": 4 * GiB})
        first = await registry.load("a")
        await registry.load("b")
        again = await registry.load("a")

        assert again["model_key"] == first["model_key"]
        assert again["evicted_model_keys"] == []
        assert worker.loads == [("a", "gpu_only"), ("b", "gpu_only")]
        assert _keys(worker) == ["b", "a"]  # reuse touches: most recently used last
        other_dtype = await registry.load("a", dtype=DType.FLOAT32)
        assert other_dtype["model_key"] != first["model_key"]

    asyncio.run(scenario())


def test_lru_eviction_frees_room_but_never_evicts_pinned_residents() -> None:
    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 4 * GiB, "b": 4 * GiB, "c": 4 * GiB})
        a = await registry.load("a")
        b = await registry.load("b")

        reservation = SimpleNamespace(active=True, inference_id="job")
        c = await registry.load_reserved(
            reservation,
            "c",
            pinned=frozenset({a["model_key"]}),  # type: ignore[arg-type]
        )

        assert worker.unloads == [b["model_key"]]
        assert c["evicted_model_keys"] == [b["model_key"]]
        assert _keys(worker) == ["a", "c"]
        status = await registry.resident_status()
        assert [(item["model_id"], item["in_use"]) for item in status["models"]] == [
            ("a", False),
            ("c", True),
        ]
        assert status["memory"]["stale"] is False
        assert status["max_loaded_models"] == 4

    asyncio.run(scenario())


def test_strict_load_that_cannot_fit_is_a_structured_507() -> None:
    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 8 * GiB, "b": 4 * GiB, "c": 1 * GiB})
        a = await registry.load("a")
        c = await registry.load("c")
        reservation = SimpleNamespace(active=True, inference_id="job")

        with pytest.raises(OutOfMemoryError) as caught:
            await registry.load_reserved(
                reservation,
                "b",
                pinned=frozenset({a["model_key"]}),  # type: ignore[arg-type]
            )

        error = caught.value
        assert error.http_status == 507
        assert "turn off Strict VRAM" in (error.hint or "")
        assert error.details["required_bytes"] == 4 * GiB
        assert error.details["available_bytes"] == 1 * GiB
        assert error.details["pinned_model_keys"] == [a["model_key"]]
        assert error.details["resident_model_keys"] == [a["model_key"], c["model_key"]]
        assert error.details["evicted_model_keys"] == []
        # Evicting the idle 1 GiB resident could not have made room, so it stays.
        assert worker.unloads == []

    asyncio.run(scenario())


def test_non_strict_load_retries_once_with_offload() -> None:
    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 8 * GiB, "b": 4 * GiB})
        a = await registry.load("a")
        reservation = SimpleNamespace(active=True, inference_id="job")

        b = await registry.load_reserved(
            reservation,  # type: ignore[arg-type]
            "b",
            strict_vram=False,
            pinned=frozenset({a["model_key"]}),
        )

        assert worker.loads == [("a", "gpu_only"), ("b", "gpu_only"), ("b", "offload")]
        assert b["placement"] == "offload"
        assert b["strict_vram"] is False

    asyncio.run(scenario())


def test_non_strict_models_without_offload_support_still_fail_clearly() -> None:
    async def scenario() -> None:
        registry, _worker = _registry(1 * GiB, {"embedder": 4 * GiB})

        with pytest.raises(OutOfMemoryError):
            await registry.load("embedder", strict_vram=False)

    asyncio.run(scenario())


def test_strict_request_replaces_an_offloaded_resident_with_a_gpu_one() -> None:
    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 8 * GiB, "b": 4 * GiB})
        a = await registry.load("a")
        reservation = SimpleNamespace(active=True, inference_id="job")
        offloaded = await registry.load_reserved(
            reservation,  # type: ignore[arg-type]
            "b",
            strict_vram=False,
            pinned=frozenset({a["model_key"]}),
        )
        assert offloaded["placement"] == "offload"
        await registry.unload(a["model_key"])

        # Non-strict reuse keeps the offloaded resident; strict re-places it on the GPU.
        assert (await registry.load("b", strict_vram=False))["placement"] == "offload"
        placed = await registry.load("b")

        assert placed["model_key"] == offloaded["model_key"]
        assert placed["placement"] == "gpu"
        assert worker.unloads == [a["model_key"], offloaded["model_key"]]

    asyncio.run(scenario())


def test_max_loaded_models_evicts_the_least_recently_used_idle_resident() -> None:
    async def scenario() -> None:
        registry, worker = _registry(100 * GiB, {"a": GiB, "b": GiB, "c": GiB}, max_loaded_models=2)
        a = await registry.load("a")
        b = await registry.load("b")

        c = await registry.load("c")
        assert c["evicted_model_keys"] == [a["model_key"]]
        assert _keys(worker) == ["b", "c"]

        reservation = SimpleNamespace(active=True, inference_id="job")
        with pytest.raises(WorkerBusyError, match="resident model limit reached"):
            await registry.load_reserved(
                reservation,  # type: ignore[arg-type]
                "a",
                pinned=frozenset({b["model_key"], c["model_key"]}),
            )

    asyncio.run(scenario())


def test_unloading_a_model_that_is_not_resident_is_a_409() -> None:
    async def scenario() -> None:
        registry, _worker = _registry(10 * GiB, {"a": GiB, "b": GiB})
        await registry.load("a", dtype=DType.BFLOAT16)
        await registry.load("a", dtype=DType.FLOAT32)

        with pytest.raises(ModelNotResidentError) as caught:
            await registry.unload_model("b")
        assert caught.value.http_status == 409
        with pytest.raises(ModelNotResidentError):
            await registry.unload("no-such-key")
        result = await registry.unload_model("a")
        assert len(result["unloaded_model_keys"]) == 2

    asyncio.run(scenario())


def test_quantized_loads_get_their_own_key_and_reach_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_ai_doctor.services.models.bitsandbytes_available", lambda: True)

    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 4 * GiB})
        plain = await registry.load("a")
        nf4 = await registry.load("a", quantization=Quantization.BITSANDBYTES_4BIT)

        assert nf4["model_key"] != plain["model_key"]
        assert worker.entries[nf4["model_key"]]["quantization"] == "bitsandbytes-4bit"

    asyncio.run(scenario())


def test_quantization_the_backend_cannot_run_is_refused_before_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_ai_doctor.services.models.bitsandbytes_available", lambda: False)

    async def scenario() -> None:
        registry, worker = _registry(10 * GiB, {"a": 4 * GiB})
        with pytest.raises(CapabilityUnavailableError, match="not installed"):
            await registry.load("a", quantization=Quantization.BITSANDBYTES_8BIT)
        registry._models["a"].metadata = {"weight_quantization": {"method": "bitsandbytes"}}
        with pytest.raises(CapabilityUnavailableError, match="re-quantized"):
            await registry.load("a", quantization=Quantization.BITSANDBYTES_4BIT)
        assert worker.loads == []

    asyncio.run(scenario())
