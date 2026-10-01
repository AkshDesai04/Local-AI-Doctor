"""Pure memory arithmetic for placing checkpoints; torch is never imported here."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..discovery.fingerprint import _loader_weight_files
from ..discovery.safetensors import read_safetensors_header
from ..domain.quantization import VISION_MODULES

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
# Bytes per parameter of 2-D Linear weights once quantized; embeddings and the LM
# head stay at the compute dtype. NF4 carries ~3% of block scales and constants.
_QUANTIZED_BYTES = {
    "bitsandbytes-4bit": 0.5 * 1.03,
    "int4": 0.5 * 1.03,
    "bitsandbytes-8bit": 1.0,
    "int8": 1.0,
}
_UNQUANTIZED_PARTS = ("embed", "lm_head", "wte", "wpe", "shared", *VISION_MODULES)


def _loader_files(model_dir: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in model_dir.iterdir()
        if path.suffix.lower() == ".safetensors" and path.is_file() and not path.is_symlink()
    )
    return _loader_weight_files(model_dir, candidates)


def checkpoint_stored_bytes(model_dir: Path) -> int:
    """Size of the SafeTensors files ``from_pretrained`` reads."""

    return sum(path.stat().st_size for path in _loader_files(model_dir))


def checkpoint_tensor_shapes(model_dir: Path) -> dict[str, list[int]]:
    """Tensor shapes from the SafeTensors headers ``from_pretrained`` reads; no weights."""

    shapes: dict[str, list[int]] = {}
    for path in _loader_files(model_dir):
        for name, record in read_safetensors_header(path).items():
            if name != "__metadata__":
                shapes[name] = list(record["shape"])
    return shapes


def _first_positive(config: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def kv_reserve_bytes(config: Mapping[str, Any], *, dtype_bytes: int, tokens: int) -> int:
    """``2 x layers x kv_heads x head_dim x dtype_bytes x tokens`` (text config when nested)."""

    nested = config.get("text_config")
    text = nested if isinstance(nested, Mapping) else config
    layers = _first_positive(text, "num_hidden_layers", "num_layers", "n_layer")
    heads = _first_positive(text, "num_attention_heads", "num_heads", "n_head")
    kv_heads = _first_positive(text, "num_key_value_heads") or heads
    head_dim = _first_positive(text, "head_dim", "d_kv")
    if head_dim is None:
        hidden = _first_positive(text, "hidden_size", "d_model", "n_embd")
        head_dim = hidden // heads if hidden and heads else None
    if not (layers and kv_heads and head_dim) or tokens <= 0:
        return 0
    return 2 * layers * kv_heads * head_dim * dtype_bytes * tokens


def estimate_load_bytes(
    tensors: Mapping[str, Sequence[int]],
    config: Mapping[str, Any],
    *,
    quantization: str = "none",
    compute_dtype: str = "float32",
    kv_reserve_tokens: int = 0,
    stored_weight_bytes: int | None = None,
) -> dict[str, int]:
    """Resident bytes of a checkpoint: weights at the compute dtype plus a KV reserve.

    ``stored_weight_bytes`` is set for a pre-quantized checkpoint, whose packed
    tensors load exactly as stored, so the file bytes are the weight estimate.
    """

    size = DTYPE_BYTES.get(compute_dtype, 4)
    quantized = _QUANTIZED_BYTES.get(quantization)
    weights = 0.0
    for name, shape in tensors.items():
        parameters = math.prod(shape)
        if (
            quantized is not None
            and len(shape) == 2
            and name.endswith(".weight")
            and not any(part in name for part in _UNQUANTIZED_PARTS)
        ):
            weights += parameters * quantized
        else:
            weights += parameters * size
    weight_bytes = math.ceil(weights) if stored_weight_bytes is None else stored_weight_bytes
    kv = kv_reserve_bytes(config, dtype_bytes=size, tokens=kv_reserve_tokens)
    return {"weights": weight_bytes, "kv_reserve": kv, "total": weight_bytes + kv}


def available_vram(
    *, free: int, reserved: int, allocated: int, margin: int, budget: int | None = None
) -> int:
    """Free device memory plus the allocator's unused cache, less the safety margin.

    ``budget`` caps everything this process allocates, so it is compared with the
    bytes already allocated rather than with the device's free memory.
    """

    available = free + (reserved - allocated) - margin
    if budget is not None:
        available = min(available, budget - allocated)
    return max(0, available)


def available_ram(
    *, system_available: int, process_rss: int, margin: int, budget: int | None = None
) -> int:
    available = system_available - margin
    if budget is not None:
        available = min(available, budget - process_rss)
    return max(0, available)


def vram_cap_fraction(*, reserved: int, free: int, total: int, margin: int) -> float:
    """Allocator cap that lets the process grow into free memory less the margin, never below
    what it already holds."""

    if total <= 0:
        return 1.0
    return min(1.0, max(reserved, reserved + free - margin) / total)


def device_map_kwargs(
    placement: str, device: str, *, gpu_bytes: int = 0, cpu_bytes: int = 0
) -> dict[str, Any]:
    """``from_pretrained`` placement kwargs; a device map is always used, never ``.to()``."""

    if placement == "gpu_only":
        return {"device_map": {"": device}}
    if placement == "cpu":
        return {"device_map": {"": "cpu"}}
    if placement == "offload":
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return {"device_map": "auto", "max_memory": {index: gpu_bytes, "cpu": cpu_bytes}}
    raise ValueError(f"unknown model placement: {placement!r}")


def placement_of(device_map: Mapping[str, Any] | None, fallback: str) -> str:
    """``gpu``, ``cpu``, or ``offload`` for the device map a load actually produced."""

    if not device_map:
        return fallback
    targets = {str(value) for value in device_map.values()}
    on_host = bool(targets & {"cpu", "disk"})
    on_gpu = bool(targets - {"cpu", "disk"})
    if on_gpu and on_host:
        return "offload"
    return "gpu" if on_gpu else "cpu"
