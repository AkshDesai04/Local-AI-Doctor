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

    dtype_sizes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
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
    write_json(root / "1_Pooling" / "config.json", {"pooling_mode_lasttoken": True})
    (root / "2_Normalize").mkdir()
    write_safetensors(
        root / "model.safetensors",
        {"model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8])},
    )
    return root
