"""Load-time quantization rules, pre-quantized discovery, and the worker flush op."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_ai_doctor.api.schemas import ModelFlushRequest
from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.discovery.capabilities import ModelEvidence, build_capability_matrix
from local_ai_doctor.domain import Capability, CapabilityState, ModelTask
from local_ai_doctor.domain.quantization import (
    bitsandbytes_config,
    folder_name_error,
    quantization_rejection,
    weight_quantization,
)
from local_ai_doctor.hardware.models import BackendKind
from local_ai_doctor.workers.runtime import ResidentModel, WorkerReportedError, WorkerRuntime

CUDA = frozenset({BackendKind.CPU, BackendKind.CUDA})
NF4_CONFIG = {
    "quant_method": "bitsandbytes",
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


def test_bitsandbytes_config_arguments_per_mode() -> None:
    assert bitsandbytes_config("none", compute_dtype="bfloat16", placement="gpu_only") == {}
    assert bitsandbytes_config(
        "bitsandbytes-4bit", compute_dtype="bfloat16", placement="gpu_only"
    ) == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "uint8",
        "llm_int8_enable_fp32_cpu_offload": False,
    }
    assert bitsandbytes_config(
        "bitsandbytes-8bit", compute_dtype="float16", placement="offload"
    ) == {
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "llm_int8_enable_fp32_cpu_offload": True,
    }
    media = bitsandbytes_config(
        "bitsandbytes-4bit", compute_dtype="bfloat16", placement="gpu_only", media=True
    )
    assert media["llm_int8_skip_modules"][:2] == ["lm_head", "visual"]


def test_bitsandbytes_config_arguments_build_a_transformers_config() -> None:
    pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    args = bitsandbytes_config("bitsandbytes-4bit", compute_dtype="bfloat16", placement="gpu_only")
    config = transformers.BitsAndBytesConfig(**args)
    assert config.bnb_4bit_compute_dtype is torch.bfloat16
    assert config.bnb_4bit_quant_storage is torch.uint8
    assert config.llm_int8_skip_modules is None  # the default LM-head skip list applies


@pytest.mark.parametrize(
    ("quantization", "options", "fragment"),
    [
        ("int4", {}, "legacy"),
        ("int8", {}, "legacy"),
        ("bitsandbytes-4bit", {"task": "embedding"}, "decoder-only text generation"),
        (
            "bitsandbytes-8bit",
            {"task": "encoder_decoder_generation"},
            "decoder-only text generation",
        ),
        ("bitsandbytes-4bit", {"cuda": False}, "CUDA only"),
        ("bitsandbytes-4bit", {"backend_available": False}, "not installed"),
        ("bitsandbytes-8bit", {"prequantized": True}, "cannot be re-quantized"),
        ("none", {"prequantized": True, "cuda": False}, "CUDA only"),
    ],
)
def test_quantization_rejections_name_their_reason(
    quantization: str, options: dict[str, Any], fragment: str
) -> None:
    arguments = {
        "task": "text_generation",
        "cuda": True,
        "prequantized": False,
        "backend_available": True,
        **options,
    }
    reason = quantization_rejection(quantization, **arguments)
    assert reason is not None and fragment in reason


def test_quantization_is_allowed_for_cuda_text_generators_and_none_everywhere() -> None:
    base = {"task": "text_generation", "cuda": True, "prequantized": False}
    assert quantization_rejection("bitsandbytes-4bit", backend_available=True, **base) is None
    assert quantization_rejection("bitsandbytes-8bit", backend_available=True, **base) is None
    assert (
        quantization_rejection(
            "none", task="embedding", cuda=False, prequantized=False, backend_available=False
        )
        is None
    )
    assert (
        quantization_rejection(
            "none", task="text_generation", cuda=True, prequantized=True, backend_available=True
        )
        is None
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".hidden",
        "-dash",
        "has space",
        "a/b",
        "a\\b",
        "x" * 101,
        "CON",
        "con.txt",
        "Lpt9.bin",
        "nul",
        "COM1",
        "trailing.",
    ],
)
def test_folder_names_that_could_escape_or_collide_are_rejected(name: str) -> None:
    assert folder_name_error(name) is not None
    with pytest.raises(ValueError, match="folder"):
        ModelFlushRequest.model_validate({"targetRootIndex": 0, "folderName": name})


def test_valid_folder_names_are_accepted() -> None:
    for name in ("Qwen3-1.7B-bnb-nf4", "a", "CONSOLE", "COM10", "model_v2.final"):
        assert folder_name_error(name) is None
    request = ModelFlushRequest.model_validate({"targetRootIndex": 1, "folderName": "M-bnb-int8"})
    assert (request.target_root_index, request.folder_name) == (1, "M-bnb-int8")
    with pytest.raises(ValueError):
        ModelFlushRequest.model_validate({"targetRootIndex": -1, "folderName": "M"})


def test_weight_quantization_is_read_from_the_config() -> None:
    assert weight_quantization({}) is None
    assert weight_quantization({"quantization_config": NF4_CONFIG}) == {
        "method": "bitsandbytes",
        "bits": 4,
        "quant_type": "nf4",
    }
    assert weight_quantization(
        {"quantization_config": {"quant_method": "bitsandbytes", "load_in_8bit": True}}
    ) == {"method": "bitsandbytes", "bits": 8, "quant_type": "llm_int8"}
    assert weight_quantization({"quantization_config": {"quant_method": "gptq", "bits": 4}}) == {
        "method": "gptq",
        "bits": 4,
        "quant_type": None,
    }


def _prequantized_model(root: Path, safetensors_writer: Any) -> Path:
    model = root / "Tiny-bnb-nf4"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "dtype": "bfloat16",
                "max_position_embeddings": 1024,
                "quantization_config": NF4_CONFIG,
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}), "utf-8")
    safetensors_writer(
        model / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", [8, 4]),
            "model.layers.0.mlp.up_proj.weight": ("U8", [64, 1]),
            "model.layers.0.mlp.up_proj.weight.absmax": ("U8", [4]),
            "model.layers.0.mlp.up_proj.weight.quant_map": ("F32", [16]),
        },
    )
    return model


def test_prequantized_checkpoints_are_discovered_with_packed_headers(
    tmp_path: Path, safetensors_writer: Any
) -> None:
    model_dir = _prequantized_model(tmp_path / "root", safetensors_writer)
    (model_dir / "local_ai_doctor_derivation.json").write_text(
        json.dumps(
            {
                "schema": "local-ai-doctor/derivation",
                "source_display_name": "Tiny",
                "source_model_id": "tiny-0123",
                "unexpected": "dropped",
            }
        ),
        encoding="utf-8",
    )
    scanner = ModelScanner([tmp_path / "root"], available_backends=CUDA, quantization_backend=True)
    model = scanner.scan().models[0]

    assert model.loadable
    assert model.metadata["weight_quantization"] == {
        "method": "bitsandbytes",
        "bits": 4,
        "quant_type": "nf4",
    }
    assert model.parameter_count is None
    assert model.metadata["parameter_count_note"] == "packed quantized tensors"
    assert model.weight_dtypes["U8"] == 68
    assert model.dtype == "bfloat16"
    assert "dtype_metadata_mismatch" not in {item.code for item in model.diagnostics}
    support = model.capabilities.support(Capability.WEIGHT_QUANTIZATION)
    assert support.state is CapabilityState.PARTIAL
    assert support.reason and "cannot be re-quantized" in support.reason
    assert model.capabilities.support(Capability.CPU).state is CapabilityState.UNSUPPORTED
    assert model.capabilities.support(Capability.CPU_OFFLOAD).state is CapabilityState.UNSUPPORTED
    public = model.public_dict()
    assert public["root_index"] == 0
    assert public["derivation"] == {
        "schema": "local-ai-doctor/derivation",
        "source_model_id": "tiny-0123",
        "source_display_name": "Tiny",
    }


def test_prequantized_checkpoint_without_bitsandbytes_is_blocked(
    tmp_path: Path, safetensors_writer: Any
) -> None:
    _prequantized_model(tmp_path / "root", safetensors_writer)
    model = (
        ModelScanner([tmp_path / "root"], available_backends=CUDA, quantization_backend=False)
        .scan()
        .models[0]
    )

    blocking = [item for item in model.diagnostics if item.code == "quantization_backend_missing"]
    assert blocking and blocking[0].severity.value == "error"
    assert not model.loadable


def _evidence(task: ModelTask, **values: Any) -> ModelEvidence:
    return ModelEvidence(
        task=task,
        modalities=frozenset({"text"}),
        has_tokenizer=True,
        has_processor=False,
        has_reasoning_delimiters=False,
        is_moe=False,
        architecture_known=True,
        **values,
    )


@pytest.mark.parametrize(
    ("task", "values", "state"),
    [
        (
            ModelTask.TEXT_GENERATION,
            {"available_backends": CUDA, "quantization_backend": True},
            CapabilityState.PARTIAL,
        ),
        (
            ModelTask.TEXT_GENERATION,
            {"available_backends": CUDA},
            CapabilityState.UNAVAILABLE_ON_BACKEND,
        ),
        (
            ModelTask.TEXT_GENERATION,
            {"quantization_backend": True},
            CapabilityState.UNAVAILABLE_ON_BACKEND,
        ),
        (
            ModelTask.EMBEDDING,
            {"available_backends": CUDA, "quantization_backend": True},
            CapabilityState.UNSUPPORTED,
        ),
        (
            ModelTask.ENCODER_DECODER_GENERATION,
            {"available_backends": CUDA, "quantization_backend": True},
            CapabilityState.UNSUPPORTED,
        ),
        (
            ModelTask.TEXT_GENERATION,
            {"weight_quantization": "bitsandbytes"},
            CapabilityState.UNAVAILABLE_ON_BACKEND,
        ),
        (
            ModelTask.TEXT_GENERATION,
            {"available_backends": CUDA, "weight_quantization": "gptq"},
            CapabilityState.UNSUPPORTED,
        ),
    ],
)
def test_weight_quantization_capability_states(
    task: ModelTask, values: dict[str, Any], state: CapabilityState
) -> None:
    support = build_capability_matrix(_evidence(task, **values)).support(
        Capability.WEIGHT_QUANTIZATION
    )
    assert support.state is state
    assert support.reason


class _Queue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


def _runtime() -> WorkerRuntime:
    return WorkerRuntime(cast(Any, _Queue()), cast(Any, _Queue()), object())


def _load_command(model_dir: Path, *, device: str, quantization: str) -> dict[str, Any]:
    return {
        "model_key": "k",
        "model": {"id": "m", "path": str(model_dir), "task": "text_generation"},
        "runtime": {"device": device, "dtype": "bfloat16", "quantization": quantization},
    }


def test_worker_rejects_quantization_it_cannot_run_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    runtime = _runtime()

    with pytest.raises(ValueError, match="CUDA only"):
        runtime._load(_load_command(tmp_path, device="cpu", quantization="bitsandbytes-4bit"))
    monkeypatch.setattr("local_ai_doctor.domain.quantization.find_spec", lambda name: None)
    with pytest.raises(ValueError, match="not installed"):
        runtime._load(_load_command(tmp_path, device="cuda:0", quantization="bitsandbytes-8bit"))
    with pytest.raises(ValueError, match="legacy"):
        runtime._load(_load_command(tmp_path, device="cuda:0", quantization="int4"))
    assert runtime.residents == {}


class _SavingModel:
    def save_pretrained(self, directory: Path, **options: Any) -> None:
        assert options == {"safe_serialization": True, "max_shard_size": "2GB"}
        (Path(directory) / "config.json").write_text('{"quantization_config": {}}', "utf-8")
        (Path(directory) / "model.safetensors").write_bytes(bytes(32))


class _SavingTokenizer:
    def save_pretrained(self, directory: Path) -> None:
        (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")
        (Path(directory) / "chat_template.jinja").write_text("saved", encoding="utf-8")


def test_worker_flush_saves_the_resident_and_writes_path_free_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.50.2")
    source = tmp_path / "source"
    source.mkdir()
    for name in ("LICENSE", "NOTICE.md", "USE_POLICY.md", "chat_template.jinja", "README.md"):
        (source / name).write_text(f"source {name}", encoding="utf-8")
    staging = tmp_path / "root" / ".lad-staging-1"
    staging.mkdir(parents=True)
    runtime = _runtime()
    runtime.residents["k"] = ResidentModel(
        key="k",
        model=_SavingModel(),
        tokenizer=_SavingTokenizer(),
        quantization="bitsandbytes-4bit",
        placement="gpu",
        quantization_config={"load_in_4bit": True},
    )

    result = runtime._flush(
        {
            "model_key": "k",
            "staging_dir": str(staging),
            "source_dir": str(source),
            "derivation": {"source_model_id": "m", "source_display_name": "Source"},
        }
    )

    assert sorted(result["files"]) == [
        "LICENSE",
        "NOTICE.md",
        "USE_POLICY.md",
        "chat_template.jinja",
        "config.json",
        "local_ai_doctor_derivation.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    assert result["bytes_written"] == sum(path.stat().st_size for path in staging.iterdir())
    # The saved template wins; the source copy only fills a gap.
    assert (staging / "chat_template.jinja").read_text("utf-8") == "saved"
    text = (staging / "local_ai_doctor_derivation.json").read_text("utf-8")
    derivation = json.loads(text)
    assert derivation["source_display_name"] == "Source"
    assert derivation["quantization"] == "bitsandbytes-4bit"
    assert derivation["quantization_config"] == {"load_in_4bit": True}
    assert derivation["software"]["bitsandbytes"] == "0.50.2"
    for host_path in (str(tmp_path), json.dumps(str(tmp_path))[1:-1]):
        assert host_path not in text


def test_worker_flush_requires_a_quantized_gpu_resident(tmp_path: Path) -> None:
    runtime = _runtime()
    runtime.residents["k"] = ResidentModel(key="k", model=_SavingModel(), placement="gpu")
    command = {"model_key": "k", "staging_dir": str(tmp_path), "source_dir": str(tmp_path)}

    with pytest.raises(WorkerReportedError) as caught:
        runtime._flush(command)
    assert caught.value.error["code"] == "flush_requires_quantized_resident"
    assert list(tmp_path.iterdir()) == []
