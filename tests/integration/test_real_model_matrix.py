"""Opt-in real-checkpoint matrix for every model in ``LAD_REAL_MODEL_ROOT``.

Each generator is loaded once through the worker runtime (in this process, so the
loaded module can also run the Transformers reference ``generate()``), exercised
with short generations, and unloaded. Nothing in the root is modified, and a
model whose weights cannot fit the free accelerator memory is skipped with the
reason rather than attempted.
"""

from __future__ import annotations

import gc
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.discovery.scanner import ModelScanReport
from local_ai_doctor.domain import Capability, CapabilityState, ModelTask
from local_ai_doctor.domain.models import ModelDescriptor, TrustDecision
from local_ai_doctor.hardware.models import BackendKind
from local_ai_doctor.workers.runtime import WorkerRuntime

pytestmark = pytest.mark.real_model

_PROMPT = [{"role": "user", "content": "Name three primary colors."}]
# Instruction-tuned models answer _PROMPT almost deterministically even at T=1.0,
# so seed sensitivity is checked on a deliberately open-ended request.
_OPEN_PROMPT = [{"role": "user", "content": "Invent a name for a new fruit and describe it."}]
_CUDA_HEADROOM_BYTES = 64 * 1024**2


def _root() -> Path | None:
    configured = os.environ.get("LAD_REAL_MODEL_ROOT")
    return Path(configured) if configured and Path(configured).is_dir() else None


@lru_cache(maxsize=1)
def _report() -> ModelScanReport | None:
    root = _root()
    if root is None:
        return None
    return ModelScanner([root], available_backends=frozenset({BackendKind.CPU})).scan()


def _names(*tasks: ModelTask) -> list[str]:
    report = _report()
    if report is None:
        return ["<unset>"]
    return [model.display_name for model in report.models if model.loadable and model.task in tasks]


def _descriptor(name: str) -> ModelDescriptor:
    report = _report()
    if report is None:
        pytest.skip("set LAD_REAL_MODEL_ROOT to opt in to real-model execution")
    return next(model for model in report.models if model.display_name == name)


class _Queue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


class _NeverCancelled:
    @staticmethod
    def is_set() -> bool:
        return False


def _placement() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        return "cuda:0", "bfloat16" if major >= 8 else "float16"
    return "cpu", "float32"


def _load(descriptor: ModelDescriptor) -> tuple[WorkerRuntime, _Queue]:
    import torch

    device, dtype = _placement()
    if device.startswith("cuda"):
        free, _total = torch.cuda.mem_get_info(0)
        weights = descriptor.fingerprint.total_weight_bytes
        if weights + 512 * 1024**2 > free:
            pytest.skip(
                f"{descriptor.display_name}: {weights / 1024**3:.1f} GiB of weights do not fit "
                f"the {free / 1024**3:.1f} GiB of free VRAM with room for activations; this "
                "needs quantization or CPU offload, which the runtime does not provide yet"
            )
    output = _Queue()
    runtime = WorkerRuntime(_Queue(), output, _NeverCancelled())  # type: ignore[arg-type]
    metadata = descriptor.metadata
    model = {
        "id": descriptor.id,
        "display_name": descriptor.display_name,
        "path": str(descriptor.path),
        "task": descriptor.task.value,
        "model_type": descriptor.model_type,
        "effective_context_limit": descriptor.effective_context_limit,
        "reasoning_delimiters": descriptor.reasoning_delimiters,
        "fingerprint": descriptor.fingerprint.value,
        "embedding_pooling": metadata.get("pooling"),
        "joint_embedding_space": metadata.get("joint_embedding_space", False),
        "trust_remote_code": descriptor.trust_decision is TrustDecision.REVIEWED_BUNDLED_CODE,
    }
    try:
        runtime._load(
            {
                "model": model,
                "runtime": {
                    "device": device,
                    "dtype": dtype,
                    "cpu_threads": os.cpu_count() or 1,
                    "attention_backend": "auto",
                    "low_memory_loading": True,
                },
            }
        )
    except torch.cuda.OutOfMemoryError:
        runtime._unload()
        pytest.skip(f"{descriptor.display_name}: out of accelerator memory while loading")
    return runtime, output


def _generate(
    runtime: WorkerRuntime,
    output: _Queue,
    *,
    temperature: float = 0.0,
    seed: int = 0,
    tokens: int = 20,
    instrumentation: str = "token",
    reasoning: bool | None = None,
    messages: list[dict[str, str]] = _PROMPT,
) -> dict[str, Any]:
    output.items.clear()
    runtime._generate(
        {
            "run_id": "real-model-run",
            "request_id": "real-model-request",
            "messages": messages,
            "effective_seed": seed,
            "instrumentation": instrumentation,
            "reasoning": reasoning,
            "max_prompt_tokens": 4096,
            "reserved_output_tokens": 64,
            "sampling": {
                "max_output_tokens": tokens,
                "temperature": temperature,
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
    )
    events = [item for item in output.items if item["kind"] == "run_event"]
    assert not any(item["event_type"] == "error" for item in events)
    token_events = [item["payload"] for item in events if item["event_type"] == "token"]
    return {
        "stage": next(item["payload"] for item in events if item["event_type"] == "stage"),
        "tokens": [item["token_id"] for item in token_events],
        "segments": [item["segment"] for item in token_events],
        "attention": [item["attention_attribution"] for item in token_events],
        "completed": next(item["payload"] for item in events if item["event_type"] == "completed"),
    }


@pytest.mark.parametrize("name", _names(ModelTask.TEXT_GENERATION))
def test_real_generator_matrix(name: str) -> None:
    torch = pytest.importorskip("torch")
    descriptor = _descriptor(name)
    cuda = torch.cuda.is_available()
    if cuda:
        gc.collect()
        torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() if cuda else 0
    runtime, output = _load(descriptor)
    # Reasoning models get one extra answer window when reasoning fills the budget;
    # disabling reasoning keeps these parity runs at exactly the requested length.
    quiet = False if descriptor.reasoning_delimiters else None
    try:
        greedy = _generate(runtime, output, reasoning=quiet)
        assert greedy["tokens"], "greedy generation produced no tokens"
        assert _generate(runtime, output, reasoning=quiet)["tokens"] == greedy["tokens"]

        # The full tier captures the exact prompt ids in the attention catalogue and
        # runs the split eager prefill; both must describe the same forward pass.
        full = _generate(runtime, output, instrumentation="full", tokens=4, reasoning=quiet)
        assert full["attention"][0] is not None, "attention capture was unavailable"
        prompt_ids = [item["token_id"] for item in full["attention"][0]["context_tokens"]]
        assert len(prompt_ids) == full["stage"]["prompt_tokens"]
        if full["stage"]["prompt_renderer"] == "plain_text_fallback":
            bos = runtime.tokenizer.bos_token_id
            assert bos is None or prompt_ids[0] == bos

        # Transformers' own greedy decode is the reference for positions and cache use.
        eos = runtime.model.generation_config.eos_token_id
        reference = runtime.model.generate(
            input_ids=torch.tensor([prompt_ids], device=runtime.device),
            attention_mask=torch.ones(
                (1, len(prompt_ids)), dtype=torch.long, device=runtime.device
            ),
            max_new_tokens=len(greedy["tokens"]),
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos[0] if isinstance(eos, list) else eos,
        )[0, len(prompt_ids) :].tolist()
        shared = min(len(reference), len(greedy["tokens"]))
        assert greedy["tokens"][:shared] == reference[:shared]

        sampled = {
            tuple(
                _generate(
                    runtime,
                    output,
                    temperature=1.0,
                    seed=seed,
                    tokens=10,
                    reasoning=quiet,
                    messages=_OPEN_PROMPT,
                )["tokens"]
            )
            for seed in (1, 2, 3)
        }
        assert len(sampled) > 1, "three seeds produced identical sampled continuations"

        template = str(getattr(runtime.tokenizer, "chat_template", "") or "")
        if "enable_thinking" in template and descriptor.reasoning_delimiters:
            thinking = _generate(runtime, output, reasoning=True, tokens=12)
            plain = _generate(runtime, output, reasoning=False, tokens=12)
            assert thinking["stage"]["reasoning_primed"] or "reasoning" in thinking["segments"]
            assert "reasoning" not in plain["segments"]
    finally:
        runtime._unload()
    if cuda:
        assert torch.cuda.memory_allocated() - baseline < _CUDA_HEADROOM_BYTES


@pytest.mark.parametrize("name", _names(ModelTask.EMBEDDING, ModelTask.MULTIMODAL_EMBEDDING))
def test_real_embedding_matrix(name: str) -> None:
    torch = pytest.importorskip("torch")
    descriptor = _descriptor(name)
    cuda = torch.cuda.is_available()
    baseline = torch.cuda.memory_allocated() if cuda else 0
    runtime, _output = _load(descriptor)
    try:
        result = runtime._embed(
            {
                "inputs": [
                    {"input_id": "a", "modality": "text", "text": "The sky is blue."},
                    {"input_id": "b", "modality": "text", "text": "Grass is green."},
                ],
                "normalize": True,
                "batch_size": 2,
            }
        )
    finally:
        runtime._unload()
    widths = {item["output_dimension"] for item in result["results"]}
    assert widths == {descriptor.metadata["embedding_dimension"]}
    for item in result["results"]:
        assert item["statistics"]["finite"]
        # Normalization happens in the loaded dtype; BF16 keeps about 3 digits.
        assert math.isclose(item["l2_norm"], 1.0, rel_tol=1e-2)
    if cuda:
        assert torch.cuda.memory_allocated() - baseline < _CUDA_HEADROOM_BYTES


def test_real_scan_reports_supplied_checkpoint_facts() -> None:
    report = _report()
    root = _root()
    if report is None or root is None:
        pytest.skip("set LAD_REAL_MODEL_ROOT to opt in to the read-only real-model scan")
    models = {model.display_name: model for model in report.models}

    def codes(name: str) -> set[str]:
        return {item.code for item in models[name].diagnostics}

    if "Krutrim-1-instruct" in models:
        assert "bundled_code_incompatible" in codes("Krutrim-1-instruct")
        assert not models["Krutrim-1-instruct"].loadable
    if "Krutrim-2-instruct" in models:
        assert "pickle_weights_only" in codes("Krutrim-2-instruct")
        support = models["Krutrim-2-instruct"].capabilities.support(Capability.TEXT_GENERATION)
        assert support.state is CapabilityState.UNSUPPORTED
    if (root / "Ministral-3-3B-Reasoning-2512-GGUF").is_dir():
        root_codes = {item.code for item in report.roots[0].diagnostics}
        assert root_codes & {"gguf_only_directory", "empty_model_directory"}
    if "Llama-3.2-1B" in models:
        llama = models["Llama-3.2-1B"]
        assert (
            llama.fingerprint.total_weight_bytes
            == (llama.path / "model.safetensors").stat().st_size
        )
    if "Qwen3-VL-Embedding-2B" in models:
        embedding = models["Qwen3-VL-Embedding-2B"]
        assert embedding.reasoning_delimiters is None
        assert (
            embedding.capabilities.support(Capability.REASONING_CHANNEL).state
            is CapabilityState.UNSUPPORTED
        )
    for name in ("DeepSeek-R1-Distill-Qwen-1.5B", "Qwen3-1.7B", "Qwen3-VL-2B-Thinking"):
        if name in models:
            assert models[name].reasoning_delimiters == ("<think>", "</think>")
