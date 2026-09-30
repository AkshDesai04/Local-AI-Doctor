"""Opt-in GPU evidence for several resident models in one worker.

Uses real checkpoints from ``LAD_REAL_MODEL_ROOT`` (read-only). Worker-level tests run
the runtime in this process so device memory can be measured directly; the eviction
test drives the real supervisor, spawned worker, and registry together.
"""

from __future__ import annotations

import asyncio
import gc
import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest

from local_ai_doctor.config import AppSettings
from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.discovery.scanner import ModelScanReport
from local_ai_doctor.domain.models import ModelDescriptor
from local_ai_doctor.hardware.models import (
    AcceleratorDevice,
    BackendKind,
    CPUInfo,
    HardwareInventory,
    MemoryInfo,
)
from local_ai_doctor.services.models import ModelRegistry
from local_ai_doctor.workers import ModelWorkerSupervisor
from local_ai_doctor.workers.runtime import WorkerReportedError, WorkerRuntime
from local_ai_doctor.workers.supervisor import worker_model_payload

pytestmark = [pytest.mark.real_model, pytest.mark.gpu]

GEMMA = "gemma-3-1b-it"
LLAMA = "Llama-3.2-1B"
QWEN = "Qwen3-1.7B"
PHI = "Phi-3-mini-4k-instruct"
MiB = 1024**2
_HEADROOM_BYTES = 64 * MiB


class _Queue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


class _NeverCancelled:
    @staticmethod
    def is_set() -> bool:
        return False


@lru_cache(maxsize=1)
def _report() -> ModelScanReport | None:
    configured = os.environ.get("LAD_REAL_MODEL_ROOT")
    if not configured or not Path(configured).is_dir():
        return None
    backends = frozenset({BackendKind.CPU, BackendKind.CUDA})
    return ModelScanner([Path(configured)], available_backends=backends).scan()


def _descriptor(name: str) -> ModelDescriptor:
    report = _report()
    if report is None:
        pytest.skip("set LAD_REAL_MODEL_ROOT to opt in to real-model execution")
    match = next((model for model in report.models if model.display_name == name), None)
    if match is None or not match.loadable:
        pytest.skip(f"{name} is not available under LAD_REAL_MODEL_ROOT")
    return match


@pytest.fixture
def torch() -> Iterator[Any]:
    module = pytest.importorskip("torch")
    if not module.cuda.is_available():
        pytest.skip("these tests need a usable CUDA device")
    gc.collect()
    module.cuda.empty_cache()
    yield module
    module.cuda.set_per_process_memory_fraction(1.0, 0)
    gc.collect()
    module.cuda.empty_cache()


def _require_free_vram(torch: Any, gib: float) -> None:
    """Skip, like the real-model matrix, when another process holds the shared GPU."""

    free, _total = torch.cuda.mem_get_info(0)
    if free < gib * 1024**3:
        pytest.skip(f"needs {gib:g} GiB of free VRAM; {free / 1024**3:.1f} GiB is free")


def _runtime() -> tuple[WorkerRuntime, _Queue]:
    output = _Queue()
    return WorkerRuntime(cast(Any, _Queue()), cast(Any, output), _NeverCancelled()), output


def _load(
    runtime: WorkerRuntime, name: str, key: str, *, placement: str = "gpu_only", strict: bool = True
) -> dict[str, Any]:
    return runtime._load(
        {
            "model_key": key,
            "model": worker_model_payload(_descriptor(name)),
            "runtime": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention_backend": "auto",
                "placement": placement,
                "strict_vram": strict,
                "kv_reserve_tokens": 512,
                "safety_margin_bytes": 256 * MiB,
                "max_concurrent_runs": 2,
            },
        }
    )


def _command(run: str, key: str, content: str, tokens: int = 16) -> dict[str, Any]:
    return {
        "op": "generate",
        "run_id": run,
        "request_id": f"request-{run}",
        "model_key": key,
        "messages": [{"role": "user", "content": content}],
        "effective_seed": 0,
        "instrumentation": "token",
        "reasoning": False,
        "max_prompt_tokens": 4096,
        "reserved_output_tokens": 64,
        "sampling": {
            "max_output_tokens": tokens,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "alternatives": 0,
            "stop_sequences": [],
        },
    }


def _tokens(output: _Queue, run: str) -> list[int]:
    events = [item for item in output.items if item.get("run_id") == run]
    assert not any(item["event_type"] == "error" for item in events), events[-1]
    return [item["payload"]["token_id"] for item in events if item["event_type"] == "token"]


def test_two_small_models_are_resident_together_and_both_generate(torch: Any) -> None:
    _require_free_vram(torch, 5.5)
    baseline = torch.cuda.memory_allocated()
    runtime, output = _runtime()
    try:
        gemma = _load(runtime, GEMMA, "gemma")
        llama = _load(runtime, LLAMA, "llama")
        assert (gemma["placement"], llama["placement"]) == ("gpu", "gpu")
        assert gemma["gpu_bytes"] > 1024 * MiB and llama["gpu_bytes"] > 1024 * MiB
        assert [item["model_key"] for item in llama["ledger"]["residents"]] == ["gemma", "llama"]
        assert llama["ledger"]["cap_bytes"] is not None

        runtime._generate(_command("gemma-run", "gemma", "Name three primary colors."))
        runtime._generate(_command("llama-run", "llama", "The capital of France is"))
        assert _tokens(output, "gemma-run")
        assert _tokens(output, "llama-run")

        released = runtime._unload("gemma")
        assert released["freed_bytes"] >= gemma["gpu_bytes"] - _HEADROOM_BYTES
        assert released["leaked_bytes"] < _HEADROOM_BYTES
        assert list(runtime.residents) == ["llama"]
    finally:
        runtime._unload()
    assert torch.cuda.memory_allocated() - baseline < _HEADROOM_BYTES
    assert runtime._vram_cap_fraction == 1.0


def test_interleaved_sessions_match_their_solo_greedy_tokens(torch: Any) -> None:
    _require_free_vram(torch, 5.5)
    runtime, output = _runtime()
    runs = [
        ("gemma-a", "gemma", "Name three primary colors."),
        ("llama-b", "llama", "The capital of France is"),
        ("gemma-c", "gemma", "Write one sentence about the sea."),
    ]
    try:
        _load(runtime, GEMMA, "gemma")
        _load(runtime, LLAMA, "llama")
        solo = {}
        for run, key, content in runs:
            runtime._generate(_command(f"solo-{run}", key, content))
            solo[run] = _tokens(output, f"solo-{run}")

        # Two different residents interleaved, then two sessions on one resident.
        for pair in ((runs[0], runs[1]), (runs[0], runs[2])):
            for run, key, content in pair:
                runtime._handle(_command(f"pair-{run}", key, content))
            assert len(runtime.sessions) == 2
            while runtime.sessions:
                runtime._step_round()
            for run, _key, _content in pair:
                assert _tokens(output, f"pair-{run}") == solo[run], run
                output.items = [
                    item for item in output.items if item.get("run_id") != f"pair-{run}"
                ]
        completed = [
            item["payload"]
            for item in output.items
            if item.get("event_type") == "completed" and str(item["run_id"]).startswith("solo-")
        ]
        assert all(not item["scheduling"]["interleaved"] for item in completed)
    finally:
        runtime._unload()


def test_strict_cap_turns_a_spill_into_out_of_memory(torch: Any) -> None:
    _require_free_vram(torch, 4.5)
    runtime, _output = _runtime()
    try:
        _load(runtime, GEMMA, "gemma")
        # A wide margin puts the cap well below the physically free memory, so an
        # out-of-memory error here comes from the cap and not from the device.
        runtime._safety_margin_bytes = 1536 * MiB
        runtime._apply_vram_cap()
        fraction = runtime._vram_cap_fraction
        assert fraction < 1.0
        _free, total = torch.cuda.mem_get_info(0)
        beyond = int(total * fraction) - torch.cuda.memory_reserved(0) + 256 * MiB
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(beyond, dtype=torch.uint8, device="cuda:0")
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        uncapped = torch.empty(beyond, dtype=torch.uint8, device="cuda:0")
        del uncapped
        torch.cuda.empty_cache()
        runtime._vram_cap_fraction = 1.0
    finally:
        runtime._unload()
    assert runtime._vram_cap_fraction == 1.0


def test_phi3_is_refused_under_strict_vram_and_offloads_when_not_strict(torch: Any) -> None:
    _require_free_vram(torch, 0.5)
    baseline = torch.cuda.memory_allocated()
    runtime, output = _runtime()
    try:
        with pytest.raises(WorkerReportedError) as caught:
            _load(runtime, PHI, "phi-strict")
        error = caught.value.error
        assert error["code"] == "insufficient_memory"
        assert error["memory_kind"] == "vram"
        assert error["required_bytes"] > error["available_bytes"]
        assert runtime.residents == {}
        assert torch.cuda.memory_allocated() - baseline < _HEADROOM_BYTES

        loaded = _load(runtime, PHI, "phi", placement="offload", strict=False)
        assert loaded["placement"] == "offload"
        assert loaded["device_map_summary"].get("cpu", 0) > 0
        runtime._generate(_command("phi-run", "phi", "The capital of France is", tokens=8))
        assert len(_tokens(output, "phi-run")) == 8
    finally:
        runtime._unload()
    assert torch.cuda.memory_allocated() - baseline < _HEADROOM_BYTES


def test_registry_evicts_the_least_recently_used_resident_to_fit_a_vram_budget(
    torch: Any,
) -> None:
    _require_free_vram(torch, 6.5)
    descriptors = {name: _descriptor(name) for name in (LLAMA, GEMMA, QWEN)}
    major, minor = torch.cuda.get_device_capability(0)
    settings = AppSettings.model_validate(
        {
            "runtime": {
                "device": "cuda",
                "allow_cpu_fallback": False,
                "kv_reserve_tokens": 512,
                "vram_safety_margin_bytes": 256 * MiB,
            },
            "workers": {"load_timeout_seconds": 600},
        }
    )

    async def scenario() -> None:
        supervisor = ModelWorkerSupervisor(startup_timeout_seconds=120)
        registry = ModelRegistry(settings, repository=cast(Any, None), worker=supervisor)
        registry.hardware = HardwareInventory(
            operating_system="test",
            os_release="1",
            python_version="3.12",
            cpu=CPUInfo(logical_cores=os.cpu_count() or 1, architecture="x86_64"),
            memory=MemoryInfo(source="test"),
            accelerators=(
                AcceleratorDevice(
                    backend=BackendKind.CUDA,
                    index=0,
                    name="gpu",
                    identifier="cuda:0",
                    compute_capability=(major, minor),
                    runtime_available=True,
                ),
            ),
        )
        registry._models = {descriptor.id: descriptor for descriptor in descriptors.values()}
        await supervisor.start()
        try:
            llama = await registry.load(descriptors[LLAMA].id)
            gemma = await registry.load(descriptors[GEMMA].id)
            status = await registry.resident_status()
            allocated = int(status["memory"]["torch_allocated_bytes"])
            qwen_bytes = int(descriptors[QWEN].fingerprint.total_weight_bytes)
            # Room for Qwen only once the least recently used resident (Llama) is gone.
            settings.runtime.vram_budget_bytes = (
                allocated + qwen_bytes - int(llama["load"]["gpu_bytes"] * 0.5)
            )

            qwen = await registry.load(descriptors[QWEN].id)

            assert qwen["evicted_model_keys"] == [llama["model_key"]]
            assert qwen["placement"] == "gpu"
            resident = await registry.resident_status()
            assert [item["model_key"] for item in resident["models"]] == [
                gemma["model_key"],
                qwen["model_key"],
            ]
            assert resident["memory"]["stale"] is False
        finally:
            await supervisor.close(grace_seconds=30)

    asyncio.run(scenario())
