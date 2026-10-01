"""Opt-in GPU evidence for token influence on real checkpoints.

Uses ``LAD_REAL_MODEL_ROOT`` (read-only). Worker-level tests run the runtime in this
process on a short greedy Qwen3-1.7B run; the cache test drives the real app, the
spawned worker, and the endpoint end to end.
"""

from __future__ import annotations

import gc
import math
import os
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from local_ai_doctor.config import AppSettings
from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.discovery.scanner import ModelScanReport
from local_ai_doctor.domain.models import ModelDescriptor
from local_ai_doctor.hardware.models import BackendKind
from local_ai_doctor.main import create_app
from local_ai_doctor.workers.runtime import WorkerRuntime
from local_ai_doctor.workers.supervisor import worker_model_payload

pytestmark = [pytest.mark.real_model, pytest.mark.gpu]

QWEN = "Qwen3-1.7B"
QWEN_VL = "Qwen3-VL-2B-Thinking"
MiB = 1024**2
TOKEN_INDEX = 3


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
    free, _total = torch.cuda.mem_get_info(0)
    if free < gib * 1024**3:
        pytest.skip(f"needs {gib:g} GiB of free VRAM; {free / 1024**3:.1f} GiB is free")


def _loaded(name: str) -> tuple[WorkerRuntime, _Queue]:
    output = _Queue()
    runtime = WorkerRuntime(cast(Any, _Queue()), cast(Any, output), _NeverCancelled())
    runtime._load(
        {
            "model_key": "influence",
            "model": worker_model_payload(_descriptor(name)),
            "runtime": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention_backend": "auto",
                "placement": "gpu_only",
                "strict_vram": True,
                "kv_reserve_tokens": 512,
                "safety_margin_bytes": 256 * MiB,
                "max_concurrent_runs": 1,
            },
        }
    )
    return runtime, output


def _generate(runtime: WorkerRuntime, output: _Queue, message: dict[str, Any]) -> dict[str, Any]:
    runtime._generate(
        {
            "run_id": "influence-run",
            "request_id": "influence-request",
            "model_key": "influence",
            "messages": [message],
            "effective_seed": 0,
            "instrumentation": "token",
            "reasoning": False,
            "max_prompt_tokens": 4096,
            "reserved_output_tokens": 64,
            "sampling": {
                "max_output_tokens": 8,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "alternatives": 5,
                "stop_sequences": [],
            },
        }
    )
    events = [item for item in output.items if item.get("kind") == "run_event"]
    assert not any(item["event_type"] == "error" for item in events), events[-1]
    prefill = next(
        item["payload"]
        for item in events
        if item["event_type"] == "stage" and item["payload"].get("stage") == "prefill"
    )
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    assert len(tokens) > TOKEN_INDEX
    return {"prefill": prefill, "tokens": tokens}


def _analysis(run: dict[str, Any], method: str, **overrides: Any) -> dict[str, Any]:
    prefill = run["prefill"]
    tokens = run["tokens"]
    return {
        "model_key": "influence",
        "method": method,
        "layers": "all" if method == "attention" else None,
        "alternative_token_id": None,
        "source_limit": 512,
        "rendered_prompt": prefill["rendered_prompt"],
        "prompt_renderer": prefill["prompt_renderer"],
        "media": [],
        "expected_prompt_token_count": prefill["prompt_tokens"],
        "generated_token_ids": [token["token_id"] for token in tokens[:TOKEN_INDEX]],
        "target_token_id": tokens[TOKEN_INDEX]["token_id"],
        "max_gradient_tokens": 2048,
        **overrides,
    }


def _weights(sources: list[dict[str, Any]]) -> list[float]:
    return [item["weight"] for item in sorted(sources, key=lambda item: item["context_index"])]


@pytest.fixture
def qwen_run(torch: Any) -> Iterator[tuple[WorkerRuntime, dict[str, Any]]]:
    _require_free_vram(torch, 4.5)
    runtime, output = _loaded(QWEN)
    try:
        yield (
            runtime,
            _generate(runtime, output, {"role": "user", "content": "The capital of France is"}),
        )
    finally:
        runtime._unload()


def test_qwen3_attention_rows_per_layer_each_sum_to_one(
    torch: Any, qwen_run: tuple[WorkerRuntime, dict[str, Any]]
) -> None:
    runtime, run = qwen_run
    result = runtime._analyze_influence(_analysis(run, "attention"))

    context = run["prefill"]["prompt_tokens"] + TOKEN_INDEX
    assert result["context_token_count"] == context
    assert len(result["captured_layers"]) == 28
    rows = torch.tensor([_weights(layer["sources"]) for layer in result["layers"]])
    assert rows.shape == (28, context)
    assert torch.allclose(rows.sum(dim=1), torch.ones(28, dtype=rows.dtype), atol=1e-4)
    assert math.fsum(_weights(result["sources"])) == pytest.approx(1.0, abs=1e-4)
    assert any(item["is_special"] for item in result["sources"])
    assert runtime.model.config._attn_implementation == runtime.loaded_attention_implementation


def test_qwen3_gradient_x_input_is_finite_and_an_alternative_changes_it(
    torch: Any, qwen_run: tuple[WorkerRuntime, dict[str, Any]]
) -> None:
    runtime, run = qwen_run
    baseline = torch.cuda.memory_allocated()
    chosen = runtime._analyze_influence(_analysis(run, "gradient_x_input"))
    target = run["tokens"][TOKEN_INDEX]
    alternative = next(
        item["token_id"]
        for item in target["alternatives"]["raw"]
        if item["token_id"] != target["token_id"]
    )
    contrast = runtime._analyze_influence(
        _analysis(run, "gradient_x_input", alternative_token_id=alternative)
    )

    for result in (chosen, contrast):
        weights = _weights(result["sources"])
        assert all(math.isfinite(value) and value >= 0.0 for value in weights)
        assert math.fsum(weights) == pytest.approx(1.0, abs=1e-6)
    assert chosen["dtype"] == "bfloat16" and chosen["objective"] == "log_probability"
    # The recomputed objective is the run's own raw log probability for that token.
    assert chosen["objective_value"] == pytest.approx(target["raw_logprob"], abs=0.15)
    assert contrast["objective"] == "logit_difference"
    assert contrast["objective_value"] > 0  # greedy: the chosen logit beats every alternative
    difference = max(
        abs(left - right)
        for left, right in zip(
            _weights(chosen["sources"]), _weights(contrast["sources"]), strict=True
        )
    )
    assert difference > 1e-3
    # The backward graph is freed; only allocator noise remains.
    assert torch.cuda.memory_allocated() - baseline < 64 * MiB


def test_qwen3_vl_image_placeholders_group_into_one_source(torch: Any, tmp_path: Path) -> None:
    from PIL import Image

    _require_free_vram(torch, 5.5)
    image = tmp_path / "gradient.png"
    Image.new("RGB", (128, 128), (200, 40, 40)).save(image)
    runtime, output = _loaded(QWEN_VL)
    try:
        media = [{"kind": "image", "path": str(image)}]
        run = _generate(
            runtime,
            output,
            {"role": "user", "content": "What colour is this image?", "attachments": media},
        )
        pads = run["prefill"]["media"]["placeholder_tokens"]
        assert pads > 1
        for method in ("attention", "gradient_x_input"):
            result = runtime._analyze_influence(_analysis(run, method, media=media))
            [image_source] = [item for item in result["sources"] if item["source_kind"] == "image"]
            assert image_source["token_count"] == pads
            assert image_source["span"][1] - image_source["span"][0] == pads
            assert image_source["media_index"] == 0
            assert image_source["token_id"] is None and image_source["is_special"] is False
            weights = _weights(result["sources"])
            assert math.fsum(weights) == pytest.approx(1.0, abs=1e-4)
            # One image source plus one source per remaining position.
            assert len(result["sources"]) == result["context_token_count"] - pads + 1
    finally:
        runtime._unload()


def test_the_endpoint_caches_a_real_analysis(torch: Any, tmp_path: Path) -> None:
    _require_free_vram(torch, 4.5)
    descriptor = _descriptor(QWEN)
    root = Path(os.environ["LAD_REAL_MODEL_ROOT"])
    settings = AppSettings.model_validate(
        {
            "active_profile": "test",
            "paths": {
                "model_roots": [root],
                "database": tmp_path / "state" / "workbench.sqlite3",
                "uploads": tmp_path / "state" / "uploads",
                "cache": tmp_path / "state" / "cache",
                "exports": tmp_path / "state" / "exports",
                "backups": tmp_path / "state" / "backups",
            },
            "server": {"host": "127.0.0.1"},
            "runtime": {
                "device": "cuda",
                "allow_cpu_fallback": False,
                "kv_reserve_tokens": 512,
                "vram_safety_margin_bytes": 256 * MiB,
            },
            "inference": {"instrumentation": "token"},
            "workers": {"shutdown_grace_seconds": 30.0},
        }
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        chat = client.post("/api/v1/chats", json={}).json()
        created = client.post(
            "/api/v1/runs/generation",
            json={
                "chatId": chat["id"],
                "modelId": descriptor.id,
                "content": "The capital of France is",
                "settings": {"temperature": 0, "maxOutputTokens": 6, "reasoning": False},
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["runId"]
        for _ in range(900):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"complete", "failed", "cancelled"}:
                break
            time.sleep(0.2)
        assert run["status"] == "complete", run.get("error_message")
        url = f"/api/v1/runs/{run_id}/tokens/2/influence"

        first = client.post(url, json={"method": "gradient_x_input"})
        second = client.post(url, json={"method": "gradient_x_input"})
        attention = client.post(url, json={"method": "attention", "layers": "all"})

        assert first.status_code == 200, first.text
        assert first.json()["cached"] is False
        assert second.json() == {**first.json(), "cached": True}
        assert first.json()["placement"] == "gpu"
        assert first.json()["target"]["token_id"] == run["tokens"][2]["token_id"]
        assert attention.status_code == 200, attention.text
        assert len(attention.json()["layers"]) == 28
