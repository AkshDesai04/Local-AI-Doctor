from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_ai_doctor.discovery import ModelScanner
from local_ai_doctor.domain import Capability, CapabilityState, ModelTask


@pytest.mark.real_model
def test_configured_real_models_are_classified_without_loading_weights() -> None:
    configured_root = os.environ.get("LAD_REAL_MODEL_ROOT")
    if not configured_root:
        pytest.skip("set LAD_REAL_MODEL_ROOT to opt in to the read-only real-model scan")
    root = Path(configured_root)
    if not root.is_dir():
        pytest.skip("the configured real-model root is unavailable")

    report = ModelScanner([root]).scan()
    models = {model.display_name: model for model in report.models}
    expected = {"DeepSeek-R1-Distill-Qwen-1.5B", "Qwen3-VL-Embedding-2B"}
    assert expected <= models.keys()

    deepseek = models["DeepSeek-R1-Distill-Qwen-1.5B"]
    assert deepseek.task is ModelTask.TEXT_GENERATION
    assert deepseek.reasoning_delimiters == ("<think>", "</think>")
    assert not deepseek.is_moe
    assert (
        deepseek.capabilities.support(Capability.MOE_ROUTING).state is CapabilityState.UNSUPPORTED
    )

    qwen = models["Qwen3-VL-Embedding-2B"]
    assert qwen.task is ModelTask.MULTIMODAL_EMBEDDING
    assert {"text", "image", "video"} <= qwen.modalities
    assert "audio" not in qwen.modalities
    assert not qwen.is_moe
    assert qwen.capabilities.support(Capability.EMBEDDINGS).state is CapabilityState.FULL
    assert (
        qwen.capabilities.support(Capability.TEXT_GENERATION).state is CapabilityState.UNSUPPORTED
    )
