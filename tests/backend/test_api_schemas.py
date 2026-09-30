from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from local_ai_doctor.api.schemas import GenerationRunCreate

_BASE: dict[str, Any] = {"chatId": "chat", "modelId": "model", "content": "hello"}


def test_browser_settings_fill_only_fields_that_are_not_already_set() -> None:
    request = GenerationRunCreate.model_validate(
        {
            **_BASE,
            "settings": {
                "device": "cuda",
                "dtype": "float16",
                "instrumentation": "full",
                "reasoning": True,
                "seed": "42",
                "deterministic": True,
                "temperature": 0.2,
                "alternatives": 3,
                "maxOutputTokens": 64,
                "topK": 5,
                "topP": 0.5,
                "minP": 0.1,
                "repetitionPenalty": 1.1,
                "frequencyPenalty": 0.2,
                "presencePenalty": 0.3,
                "stopSequences": ["END"],
            },
        }
    )

    assert (request.chat_id, request.model_id, request.prompt) == ("chat", "model", "hello")
    assert request.device is not None and request.device.value == "cuda"
    assert request.seed == 42
    assert request.reasoning is True
    assert request.deterministic_reference_mode is True
    assert request.sampling.model_dump() == {
        "max_output_tokens": 64,
        "temperature": 0.2,
        "top_k": 5,
        "top_p": 0.5,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "stop_sequences": ["END"],
        "alternatives": 3,
    }


def test_explicit_sampling_is_not_overwritten_by_browser_settings() -> None:
    request = GenerationRunCreate.model_validate(
        {
            **_BASE,
            "sampling": {"temperature": 0.0, "max_output_tokens": 7},
            "settings": {"temperature": 0.9, "topK": 3, "maxOutputTokens": 99},
        }
    )

    assert request.sampling.temperature == 0.0
    assert request.sampling.max_output_tokens == 7
    assert request.sampling.top_k == 3


def test_top_level_deterministic_mode_wins_over_browser_settings() -> None:
    request = GenerationRunCreate.model_validate(
        {
            **_BASE,
            "deterministic_reference_mode": False,
            "settings": {"deterministic": True},
        }
    )
    assert request.deterministic_reference_mode is False

    fallback = GenerationRunCreate.model_validate({**_BASE, "settings": {"deterministic": True}})
    assert fallback.deterministic_reference_mode is True


@pytest.mark.parametrize(
    "settings",
    [
        {"top_k": 3},
        {"typo": 1, "temperature": 0.5},
        {"seed": "not-a-number"},
        {"seed": ["1"]},
        "not-an-object",
    ],
)
def test_unknown_or_malformed_browser_settings_are_rejected(settings: Any) -> None:
    with pytest.raises(ValidationError):
        GenerationRunCreate.model_validate({**_BASE, "settings": settings})


def test_canonical_snake_case_request_is_unchanged() -> None:
    request = GenerationRunCreate.model_validate(
        {
            "chat_id": "chat",
            "model_id": "model",
            "prompt": "hello",
            "seed": None,
            "deterministic_reference_mode": True,
            "sampling": {"top_k": 9},
        }
    )
    assert request.seed is None
    assert request.deterministic_reference_mode is True
    assert request.sampling.top_k == 9


def test_browser_strict_vram_and_quantization_map_to_canonical_fields() -> None:
    from local_ai_doctor.api.schemas import ModelLoadRequest
    from local_ai_doctor.config import Quantization

    request = GenerationRunCreate.model_validate(
        {**_BASE, "settings": {"strictVram": False, "quantization": "none"}}
    )
    assert request.strict_vram is False
    assert request.quantization is Quantization.NONE
    assert GenerationRunCreate.model_validate(_BASE).strict_vram is None
    top_level = GenerationRunCreate.model_validate(
        {**_BASE, "strict_vram": True, "settings": {"strictVram": False}}
    )
    assert top_level.strict_vram is True
    assert ModelLoadRequest.model_validate({"strictVram": False}).strict_vram is False
    assert ModelLoadRequest.model_validate({"strict_vram": True}).strict_vram is True
    with pytest.raises(ValidationError):
        ModelLoadRequest.model_validate({"placement": "offload"})
