from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_ai_doctor.workers.memory import (
    available_ram,
    available_vram,
    checkpoint_tensor_shapes,
    device_map_kwargs,
    estimate_load_bytes,
    kv_reserve_bytes,
    placement_of,
    vram_cap_fraction,
)

TENSORS = {
    "model.embed_tokens.weight": [10, 4],
    "model.layers.0.mlp.up_proj.weight": [8, 4],
    "model.norm.weight": [4],
    "lm_head.weight": [10, 4],
}
CONFIG = {
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_size": 16,
}


def test_weights_count_at_the_compute_dtype_plus_a_kv_reserve() -> None:
    estimate = estimate_load_bytes(TENSORS, CONFIG, compute_dtype="bfloat16", kv_reserve_tokens=10)

    # 116 parameters at 2 bytes; KV = 2 x 2 layers x 2 kv heads x head_dim 4 x 2 bytes x 10.
    assert estimate == {"weights": 232, "kv_reserve": 640, "total": 872}
    assert estimate_load_bytes(TENSORS, CONFIG, compute_dtype="float32")["weights"] == 464


def test_quantized_linear_weights_skip_embeddings_and_the_lm_head() -> None:
    nf4 = estimate_load_bytes(
        TENSORS, CONFIG, quantization="bitsandbytes-4bit", compute_dtype="bfloat16"
    )
    int8 = estimate_load_bytes(
        TENSORS, CONFIG, quantization="bitsandbytes-8bit", compute_dtype="bfloat16"
    )

    assert nf4["weights"] == 185  # ceil(32 x 0.5 x 1.03 + 84 x 2)
    assert int8["weights"] == 200


def test_kv_reserve_reads_nested_text_config_and_family_aliases() -> None:
    nested = {"text_config": {**CONFIG, "head_dim": 8}, "vision_config": {"depth": 9}}
    assert kv_reserve_bytes(nested, dtype_bytes=2, tokens=1) == 2 * 2 * 2 * 8 * 2
    t5 = {"num_layers": 6, "num_heads": 8, "d_kv": 64}
    assert kv_reserve_bytes(t5, dtype_bytes=4, tokens=2) == 2 * 6 * 8 * 64 * 4 * 2
    assert kv_reserve_bytes({}, dtype_bytes=2, tokens=100) == 0
    assert kv_reserve_bytes(CONFIG, dtype_bytes=2, tokens=0) == 0


def test_checkpoint_shapes_follow_the_loader_shards_only(
    tmp_path: Path, safetensors_writer: Any
) -> None:
    write_safetensors = safetensors_writer
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a.weight": ("BF16", [2, 3])})
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b.weight": ("F32", [4])})
    write_safetensors(tmp_path / "consolidated.safetensors", {"c.weight": ("BF16", [99, 99])})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert checkpoint_tensor_shapes(tmp_path) == {"a.weight": [2, 3], "b.weight": [4]}


def test_available_memory_subtracts_the_margin_and_respects_budgets() -> None:
    assert available_vram(free=1000, reserved=300, allocated=200, margin=100) == 1000
    assert available_vram(free=1000, reserved=300, allocated=200, margin=100, budget=700) == 500
    assert available_vram(free=10, reserved=0, allocated=0, margin=100) == 0
    assert available_ram(system_available=5000, process_rss=1000, margin=500) == 4500
    assert available_ram(system_available=5000, process_rss=1000, margin=500, budget=3000) == 2000


def test_vram_cap_fraction_never_drops_below_reserved_memory() -> None:
    assert vram_cap_fraction(reserved=2000, free=4000, total=10000, margin=512) == pytest.approx(
        0.5488
    )
    assert vram_cap_fraction(reserved=2000, free=100, total=10000, margin=512) == 0.2
    assert vram_cap_fraction(reserved=0, free=0, total=0, margin=0) == 1.0


def test_device_map_kwargs_per_placement() -> None:
    assert device_map_kwargs("gpu_only", "cuda:0") == {"device_map": {"": "cuda:0"}}
    assert device_map_kwargs("cpu", "cuda:0") == {"device_map": {"": "cpu"}}
    assert device_map_kwargs("offload", "cuda:1", gpu_bytes=10, cpu_bytes=20) == {
        "device_map": "auto",
        "max_memory": {1: 10, "cpu": 20},
    }
    with pytest.raises(ValueError, match="unknown model placement"):
        device_map_kwargs("disk", "cuda:0")


def test_placement_is_read_back_from_the_produced_device_map() -> None:
    assert placement_of({"": "cuda:0"}, "gpu") == "gpu"
    assert placement_of({"model.embed": 0, "model.layers.9": "cpu"}, "gpu") == "offload"
    assert placement_of({"": "cpu"}, "gpu") == "cpu"
    assert placement_of(None, "gpu") == "gpu"
