"""Native support facts read from the installed Transformers auto mappings.

Discovery must never hard-code the list of model families this workbench can load:
that list goes stale with every Transformers release, which shows up as a usable
checkpoint being reported as unsupported. The auto mappings are plain
``model_type -> class name`` dictionaries, so reading them stays correct across
upgrades, imports no torch, and touches no network.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from ..domain.models import ModelTask

_SAFE_TEXT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)
_GENERATION_MAPPINGS: Final[tuple[tuple[str, ModelTask], ...]] = (
    ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", ModelTask.TEXT_GENERATION),
    ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", ModelTask.TEXT_GENERATION),
    ("MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", ModelTask.TEXT_GENERATION),
    ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", ModelTask.ENCODER_DECODER_GENERATION),
)


@lru_cache(maxsize=1)
def _native_generation_tasks() -> dict[str, ModelTask]:
    try:
        from transformers.models.auto import modeling_auto
    except ImportError:
        return {}
    result: dict[str, ModelTask] = {}
    for attribute, task in _GENERATION_MAPPINGS:
        mapping = getattr(modeling_auto, attribute, None)
        if not isinstance(mapping, dict):
            continue
        for model_type in mapping:
            result.setdefault(str(model_type), task)
    return result


@lru_cache(maxsize=1)
def _native_config_types() -> frozenset[str]:
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return frozenset()
    return frozenset(str(key) for key in CONFIG_MAPPING_NAMES)


def native_generation_task(model_type: str | None) -> ModelTask | None:
    """Return the generation task the installed build maps this model type to."""

    return _native_generation_tasks().get(model_type) if model_type else None


def builtin_config_error(directory: Path, model_type: str | None) -> str | None:
    """Return why the reviewed built-in path cannot read this checkpoint, or None.

    A model type appearing in the auto mapping is not proof: Krutrim-1 declares
    ``mpt`` but uses an ``attn_type`` only its bundled code accepts. Parsing the
    configuration with ``trust_remote_code=False`` is the actual question, and it
    reads metadata only, so no weights, code, or network are touched.
    """

    if not model_type or model_type not in _native_config_types():
        return "no built-in implementation is registered for this model type"
    try:
        from transformers import AutoConfig
    except ImportError:  # pragma: no cover - ML extras are optional at scan time
        return "the transformers package is not installed"
    try:
        AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    except Exception as exc:  # any parse failure means the built-in path cannot read it
        return _rejection_reason(exc)
    return None


def _rejection_reason(exc: BaseException) -> str:
    """Describe a rejection without ever quoting filesystem detail.

    Configuration validation raises value/type errors whose text names only the
    offending metadata field, which is exactly what an operator needs to see.
    Environment and filesystem errors carry paths, so those report the class only.
    """

    detail = str(exc).strip().splitlines()[0][:200] if isinstance(exc, _SAFE_TEXT_ERRORS) else ""
    suffix = f": {detail}" if detail else f" ({type(exc).__name__})"
    return f"the built-in configuration rejected this checkpoint{suffix}"
