from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Write a structurally valid, tiny SafeTensors fixture without the package."""

    dtype_sizes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U8": 1, "I8": 1}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        end = offset + elements * dtype_sizes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


@pytest.fixture
def safetensors_writer() -> Any:
    """``write_safetensors`` for test modules, which cannot import this conftest by name."""

    return write_safetensors


@pytest.fixture
def causal_model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "Dense-Reasoner"
    root.mkdir()
    write_json(
        root / "config.json",
        {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 8192,
            "sliding_window": 4096,
            "use_sliding_window": False,
            "bos_token_id": 1,
        },
    )
    write_json(
        root / "tokenizer_config.json",
        {"model_max_length": 16384, "chat_template": "{{ bos_token }}<think>\n</think>"},
    )
    write_json(root / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"a": 0}}})
    write_json(root / "generation_config.json", {"bos_token_id": 2, "eos_token_id": 3})
    write_safetensors(
        root / "model.safetensors",
        {
            "model.layers.0.mlp.gate_proj.weight": ("BF16", [2, 3]),
            "lm_head.weight": ("BF16", [4, 3]),
        },
    )
    return root


@pytest.fixture
def base_model_dir(tmp_path: Path) -> Path:
    """Mirror of Llama-3.2-1B: no chat template, plus Meta's original-format copy."""

    root = tmp_path / "Base-Llama"
    write_json(
        root / "config.json",
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
        },
    )
    write_json(root / "tokenizer_config.json", {"bos_token": "<|begin_of_text|>"})
    write_json(root / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"a": 0}}})
    write_safetensors(root / "model.safetensors", {"lm_head.weight": ("BF16", [4, 3])})
    (root / "original").mkdir()
    (root / "original" / "consolidated.00.pth").write_bytes(bytes(64))
    write_json(root / "original" / "params.json", {"dim": 3})
    (root / ".cache" / "huggingface").mkdir(parents=True)
    return root


@pytest.fixture
def pickle_only_model_dir(tmp_path: Path) -> Path:
    """Mirror of Krutrim-2-instruct: sharded `pytorch_model-*.bin` weights only."""

    root = tmp_path / "Pickle-Mistral"
    write_json(
        root / "config.json",
        {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 1024000,
        },
    )
    write_json(root / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
    write_json(root / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"a": 0}}})
    shards = ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin")
    for index, name in enumerate(shards, 1):
        (root / name).write_bytes(bytes(100 * index))
    (root / "training_args.bin").write_bytes(bytes(7))
    write_json(
        root / "pytorch_model.bin.index.json",
        {"weight_map": {"a.weight": shards[0], "b.weight": shards[1]}},
    )
    return root


@pytest.fixture
def reviewed_code_model_dir(tmp_path: Path) -> Path:
    """Mirror of Krutrim-1-instruct: MPT grouped-query attention needing bundled code.

    The bundled modules reproduce the real checkpoint's import failure modes against
    Transformers 4.57: a removed Llama helper, a package that is not installed, and
    optional accelerators that are only imported behind guards.
    """

    root = tmp_path / "Krutrim-1-instruct"
    write_json(
        root / "config.json",
        {
            "architectures": ["MPTForCausalLM"],
            "model_type": "mpt",
            "max_seq_len": 4096,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.37.2",
            "attn_config": {"attn_type": "grouped_query_attention", "kv_n_heads": 8},
            "auto_map": {
                "AutoConfig": "configuration_mpt.MPTConfig",
                "AutoModelForCausalLM": "modeling_mpt.MPTForCausalLM",
            },
        },
    )
    write_json(root / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"a": 0}}})
    (root / "configuration_mpt.py").write_text(
        "from transformers import PretrainedConfig\nfrom .attention import helper\n",
        encoding="utf-8",
    )
    (root / "attention.py").write_text(
        "import math\n"
        "import lad_fixture_package_that_is_not_installed\n"
        "try:\n    import flash_attn\nexcept ImportError:\n    flash_attn = None\n"
        "def helper():\n    import another_missing_optional_package\n",
        encoding="utf-8",
    )
    (root / "modeling_mpt.py").write_text(
        "import torch\n"
        "from transformers.modeling_outputs import CausalLMOutputWithPast\n"
        "from transformers.models.llama.modeling_llama import (\n"
        "    LlamaDynamicNTKScalingRotaryEmbedding,\n"
        "    LlamaRotaryEmbedding,\n"
        ")\n"
        "from .attention import helper\n"
        "from .configuration_mpt import MPTConfig\n",
        encoding="utf-8",
    )
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    write_safetensors(root / shards[0], {"a.weight": ("BF16", [2, 3])})
    write_safetensors(root / shards[1], {"b.weight": ("BF16", [3, 3])})
    write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"a.weight": shards[0], "b.weight": shards[1]}},
    )
    return root


@pytest.fixture
def embedding_model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "Vision-Embedder"
    root.mkdir()
    write_json(
        root / "config.json",
        {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "dtype": "bfloat16",
            "text_config": {"max_position_embeddings": 262144},
            "vision_config": {"hidden_size": 8},
            "video_token_id": 10,
        },
    )
    write_json(root / "modules.json", {"modules": ["Pooling", "Normalize"]})
    write_json(root / "config_sentence_transformers.json", {"model_type": "SentenceTransformer"})
    write_json(root / "sentence_bert_config.json", {"transformer_task": "feature-extraction"})
    write_json(root / "tokenizer_config.json", {"model_max_length": 262144})
    write_json(root / "tokenizer.json", {"model": {"type": "BPE", "vocab": {"a": 0}}})
    write_json(root / "preprocessor_config.json", {"processor_class": "Qwen3VLProcessor"})
    write_json(root / "video_preprocessor_config.json", {"fps": 2})
    write_json(
        root / "1_Pooling" / "config.json",
        {"pooling_mode_lasttoken": True, "embedding_dimension": 128},
    )
    (root / "2_Normalize").mkdir()
    write_safetensors(
        root / "model.safetensors",
        {"model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8])},
    )
    return root
