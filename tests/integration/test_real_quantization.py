"""Opt-in GPU evidence for load-time bitsandbytes quantization and Flush to storage.

Source checkpoints come from ``LAD_REAL_MODEL_ROOT`` and are only read. The flush
test writes into a temporary model root created for the test, never into the
real one.
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

from local_ai_doctor.config import AppSettings, Quantization
from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.discovery.scanner import ModelScanReport
from local_ai_doctor.domain.models import ModelDescriptor
from local_ai_doctor.hardware.models import BackendKind
from local_ai_doctor.services.models import ModelRegistry
from local_ai_doctor.workers import ModelWorkerSupervisor
from local_ai_doctor.workers.runtime import WorkerRuntime
from local_ai_doctor.workers.supervisor import worker_model_payload

pytestmark = [pytest.mark.real_model, pytest.mark.gpu]

QWEN = "Qwen3-1.7B"
QWEN_VL = "Qwen3-VL-2B-Thinking"
PHI = "Phi-3-mini-4k-instruct"
MiB = 1024**2
GiB = 1024**3
QUESTION = "What is the capital of France? Answer in one word."
STORY = "Write a long paragraph about the history of the sea."


class _Queue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


class _NeverCancelled:
    @staticmethod
    def is_set() -> bool:
        return False


class _NullRepository:
    async def upsert_model(self, model: Any) -> None:
        return None

    async def replace_capabilities(self, model_id: str, capabilities: Any) -> None:
        return None


def _root() -> Path:
    configured = os.environ.get("LAD_REAL_MODEL_ROOT")
    if not configured or not Path(configured).is_dir():
        pytest.skip("set LAD_REAL_MODEL_ROOT to opt in to real-model execution")
    return Path(configured)


@lru_cache(maxsize=1)
def _report() -> ModelScanReport:
    return ModelScanner(
        [_root()], available_backends=frozenset({BackendKind.CPU, BackendKind.CUDA})
    ).scan()


def _descriptor(name: str) -> ModelDescriptor:
    match = next((model for model in _report().models if model.display_name == name), None)
    if match is None or not match.loadable:
        pytest.skip(f"{name} is not available under LAD_REAL_MODEL_ROOT")
    return match


@pytest.fixture
def torch() -> Iterator[Any]:
    module = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    if not module.cuda.is_available():
        pytest.skip("these tests need a usable CUDA device")
    gc.collect()
    module.cuda.empty_cache()
    yield module
    module.cuda.set_per_process_memory_fraction(1.0, 0)
    gc.collect()
    module.cuda.empty_cache()


def _require_free_vram(torch: Any, gib: float) -> None:
    free, _total = torch.cuda.mem_get_info(0)
    if free < gib * GiB:
        pytest.skip(f"needs {gib:g} GiB of free VRAM; {free / GiB:.1f} GiB is free")


def _runtime() -> tuple[WorkerRuntime, _Queue]:
    output = _Queue()
    return WorkerRuntime(cast(Any, _Queue()), cast(Any, output), _NeverCancelled()), output


def _load(runtime: WorkerRuntime, name: str, key: str, quantization: str) -> dict[str, Any]:
    return runtime._load(
        {
            "model_key": key,
            "model": worker_model_payload(_descriptor(name)),
            "runtime": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention_backend": "auto",
                "placement": "gpu_only",
                "strict_vram": True,
                "quantization": quantization,
                "kv_reserve_tokens": 512,
                "safety_margin_bytes": 256 * MiB,
            },
        }
    )


def _sampling(tokens: int) -> dict[str, Any]:
    return {
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
    }


def _generate(
    runtime: WorkerRuntime, output: _Queue, key: str, tokens: int = 8, prompt: str = STORY
) -> list[int]:
    run = f"run-{key}-{len(output.items)}"
    runtime._generate(
        {
            "op": "generate",
            "run_id": run,
            "request_id": f"request-{run}",
            "model_key": key,
            "messages": [{"role": "user", "content": prompt}],
            "effective_seed": 0,
            "instrumentation": "token",
            "reasoning": False,
            "max_prompt_tokens": 4096,
            "reserved_output_tokens": 64,
            "sampling": _sampling(tokens),
        }
    )
    events = [item for item in output.items if item.get("run_id") == run]
    assert not any(item["event_type"] == "error" for item in events), events[-1]
    return [item["payload"]["token_id"] for item in events if item["event_type"] == "token"]


def test_qwen3_4bit_stays_small_in_vram_and_generates(torch: Any) -> None:
    _require_free_vram(torch, 2.5)
    runtime, output = _runtime()
    try:
        loaded = _load(runtime, QWEN, "qwen-nf4", "bitsandbytes-4bit")
        assert loaded["quantization"] == "bitsandbytes-4bit"
        assert loaded["placement"] == "gpu"
        assert 0 < loaded["gpu_bytes"] < 1.6 * GiB
        assert loaded["estimate"]["weights"] < 1.6 * GiB
        assert len(_generate(runtime, output, "qwen-nf4")) == 8
        answer = _generate(runtime, output, "qwen-nf4", tokens=16, prompt=QUESTION)
        assert "Paris" in runtime.tokenizer.decode(answer, skip_special_tokens=True)
    finally:
        runtime._unload()


def test_qwen3_8bit_loads_and_generates(torch: Any) -> None:
    _require_free_vram(torch, 3.0)
    runtime, output = _runtime()
    try:
        loaded = _load(runtime, QWEN, "qwen-int8", "bitsandbytes-8bit")
        assert loaded["quantization"] == "bitsandbytes-8bit"
        assert loaded["placement"] == "gpu"
        assert loaded["gpu_bytes"] < 2.6 * GiB
        assert len(_generate(runtime, output, "qwen-int8")) == 8
    finally:
        runtime._unload()


def test_phi3_mini_4bit_fits_under_strict_vram_and_generates(torch: Any) -> None:
    _require_free_vram(torch, 3.5)
    runtime, output = _runtime()
    try:
        loaded = _load(runtime, PHI, "phi-nf4", "bitsandbytes-4bit")
        assert loaded["placement"] == "gpu"
        assert loaded["strict_vram"] is True
        assert loaded["gpu_bytes"] < 3 * GiB
        assert len(_generate(runtime, output, "phi-nf4")) == 8
    finally:
        runtime._unload()


def test_qwen3_vl_quantizes_the_language_model_and_keeps_the_vision_tower(torch: Any) -> None:
    _require_free_vram(torch, 3.0)
    runtime, output = _runtime()
    try:
        _load(runtime, QWEN_VL, "vl-nf4", "bitsandbytes-4bit")
        modules = dict(runtime.model.named_modules())
        quantized = {
            name for name, module in modules.items() if "Linear4bit" in type(module).__name__
        }
        assert any(".language_model." in name for name in quantized)
        assert not any(".visual." in name or name.startswith("visual") for name in quantized)
        assert "lm_head" not in quantized
        answer = _generate(runtime, output, "vl-nf4", tokens=16, prompt=QUESTION)
        assert "Paris" in runtime.tokenizer.decode(answer, skip_special_tokens=True)
    finally:
        runtime._unload()


def test_flush_writes_a_prequantized_folder_that_reloads_with_identical_tokens(
    torch: Any, tmp_path: Path
) -> None:
    _require_free_vram(torch, 3.0)
    source = _descriptor(QWEN)
    target_root = tmp_path / "flush-root"
    target_root.mkdir()
    settings = AppSettings.model_validate(
        {
            "paths": {"model_roots": [_root(), target_root]},
            "runtime": {
                "device": "cuda",
                "allow_cpu_fallback": False,
                "kv_reserve_tokens": 512,
                "vram_safety_margin_bytes": 256 * MiB,
            },
            "workers": {"load_timeout_seconds": 600},
        }
    )

    async def greedy(supervisor: ModelWorkerSupervisor, key: str, run: str) -> list[int]:
        tokens: list[int] = []
        async for event in supervisor.generate(
            run_id=run,
            messages=[{"role": "user", "content": STORY}],
            sampling=_sampling(16),
            effective_seed=0,
            instrumentation="token",
            deterministic_reference_mode=False,
            max_prompt_tokens=4096,
            reserved_output_tokens=64,
            timeout_seconds=600,
            reasoning=False,
            model_key=key,
        ):
            assert event["event_type"] != "error", event
            if event["event_type"] == "token":
                tokens.append(int(event["payload"]["token_id"]))
        return tokens

    async def scenario() -> None:
        supervisor = ModelWorkerSupervisor(startup_timeout_seconds=120)
        registry = ModelRegistry(
            settings, repository=cast(Any, _NullRepository()), worker=supervisor
        )
        await supervisor.start()
        try:
            await registry.refresh()
            loaded = await registry.load(source.id, quantization=Quantization.BITSANDBYTES_4BIT)
            before = await greedy(supervisor, loaded["model_key"], "in-memory")

            flushed = await registry.flush(loaded["model_key"], 1, "Qwen3-1.7B-bnb-nf4")

            assert flushed["folder"] == "<model-root:1>/Qwen3-1.7B-bnb-nf4"
            assert flushed["bytes_written"] > 1 * GiB
            model = flushed["model"]
            assert model["path"] == "<model-root>/Qwen3-1.7B-bnb-nf4"
            assert model["root_index"] == 1
            assert model["parameter_count"] is None
            assert model["metadata"]["weight_quantization"] == {
                "method": "bitsandbytes",
                "bits": 4,
                "quant_type": "nf4",
            }
            assert model["capabilities"]["entries"]["weight_quantization"]["state"] == "partial"
            assert flushed["derivation"]["source_display_name"] == QWEN
            assert flushed["derivation"]["source_fingerprint"] == source.fingerprint.value
            written = target_root / "Qwen3-1.7B-bnb-nf4"
            derivation_text = (written / "local_ai_doctor_derivation.json").read_text("utf-8")
            for host_path in (str(_root()), str(tmp_path), str(Path.home())):
                assert host_path not in derivation_text
                assert host_path.replace("\\", "\\\\") not in derivation_text
            assert (written / "LICENSE").is_file()
            assert [path.name for path in target_root.iterdir()] == ["Qwen3-1.7B-bnb-nf4"]

            await registry.unload(loaded["model_key"])
            reloaded = await registry.load(model["id"])
            assert reloaded["quantization"] == "none"
            after = await greedy(supervisor, reloaded["model_key"], "reloaded")
            assert after == before
        finally:
            await supervisor.close(grace_seconds=30)

    asyncio.run(scenario())
