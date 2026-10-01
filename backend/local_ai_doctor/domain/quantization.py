"""Load-time weight quantization rules shared by discovery, the registry, and the worker.

Pure: nothing here imports torch, Transformers, or bitsandbytes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, Final

BITSANDBYTES_MODES: Final[dict[str, int]] = {"bitsandbytes-4bit": 4, "bitsandbytes-8bit": 8}
LEGACY_MODES: Final[frozenset[str]] = frozenset({"int4", "int8"})
# Vision towers and projectors stay at the compute dtype; only the language model is quantized.
VISION_MODULES: Final[tuple[str, ...]] = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
)
DERIVATION_FILE: Final[str] = "local_ai_doctor_derivation.json"

_FOLDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def bitsandbytes_available() -> bool:
    """Whether the bitsandbytes package is installed; it is not imported here."""

    return find_spec("bitsandbytes") is not None


def weight_quantization(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """``config.quantization_config`` as ``{method, bits, quant_type}``, or None."""

    raw = config.get("quantization_config")
    if not isinstance(raw, Mapping):
        return None
    method = str(raw.get("quant_method") or "unknown")
    bits = raw.get("bits")
    if raw.get("load_in_4bit") is True:
        bits = 4
    elif raw.get("load_in_8bit") is True:
        bits = 8
    quant_type = raw.get("bnb_4bit_quant_type") if bits == 4 else raw.get("quant_type")
    if method == "bitsandbytes" and bits == 8:
        quant_type = "llm_int8"
    return {
        "method": method,
        "bits": bits if isinstance(bits, int) and not isinstance(bits, bool) else None,
        "quant_type": str(quant_type) if quant_type else None,
    }


def is_bitsandbytes_checkpoint(config: Mapping[str, Any]) -> bool:
    found = weight_quantization(config)
    return found is not None and found["method"] == "bitsandbytes"


def quantization_rejection(
    quantization: str,
    *,
    task: str,
    cuda: bool,
    prequantized: bool,
    backend_available: bool,
) -> str | None:
    """Why a load with ``quantization`` cannot run, or None when it can.

    ``prequantized`` means the checkpoint itself stores bitsandbytes weights.
    """

    if quantization in LEGACY_MODES:
        return (
            f"quantization {quantization!r} is a legacy value with no adapter; "
            "use bitsandbytes-4bit or bitsandbytes-8bit"
        )
    if quantization == "none":
        if prequantized and not cuda:
            return "pre-quantized bitsandbytes checkpoints run on CUDA only"
        return None
    if quantization not in BITSANDBYTES_MODES:
        return f"unknown quantization {quantization!r}"
    if prequantized:
        return (
            "the checkpoint is already bitsandbytes-quantized; it loads as-is and cannot be "
            "re-quantized (load it with quantization none)"
        )
    if task != "text_generation":
        return "load-time weight quantization is implemented only for decoder-only text generation"
    if not cuda:
        return "bitsandbytes quantization runs on CUDA only"
    if not backend_available:
        return "the bitsandbytes package is not installed (install the CUDA extra)"
    return None


def bitsandbytes_config(
    quantization: str, *, compute_dtype: str, placement: str, media: bool = False
) -> dict[str, Any]:
    """``BitsAndBytesConfig`` keyword arguments (JSON-safe); empty for ``none``.

    The default skip list (the LM head) is kept for text models, so the final
    projection runs at the compute dtype exactly as before. Multimodal models list
    it explicitly together with their vision modules.
    """

    offload = placement == "offload"
    if quantization == "bitsandbytes-4bit":
        args: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_quant_storage": "uint8",
            "llm_int8_enable_fp32_cpu_offload": offload,
        }
    elif quantization == "bitsandbytes-8bit":
        args = {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_enable_fp32_cpu_offload": offload,
        }
    else:
        return {}
    if media:
        args["llm_int8_skip_modules"] = ["lm_head", *VISION_MODULES]
    return args


def folder_name_error(name: str) -> str | None:
    """Why ``name`` cannot be a new model folder, or None when it is acceptable."""

    if not _FOLDER_NAME.fullmatch(name):
        return (
            "use 1-100 letters, digits, dots, underscores, or hyphens, starting with a "
            "letter or digit"
        )
    if name.endswith((".", " ")):
        return "the folder name must not end with a dot or a space"
    if name.split(".", 1)[0].upper() in _RESERVED_NAMES:
        return "the folder name is reserved on Windows"
    return None
