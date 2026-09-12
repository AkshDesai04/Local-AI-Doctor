import pytest
from pydantic import ValidationError

from local_ai_doctor.api.schemas import EmbeddingItemRequest, GenerationRunCreate
from local_ai_doctor.config import AppSettings
from local_ai_doctor.services.runs import _effective_deterministic_mode, _effective_sampling


def test_configured_generation_defaults_fill_only_omitted_request_fields() -> None:
    settings = AppSettings.model_validate(
        {
            "inference": {
                "deterministic_reference_mode": True,
                "defaults": {
                    "max_output_tokens": 91,
                    "temperature": 0.4,
                    "top_k": 13,
                    "top_p": 0.8,
                    "min_p": 0.1,
                    "repetition_penalty": 1.2,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.2,
                    "alternatives": 6,
                },
            }
        }
    )
    request = GenerationRunCreate.model_validate(
        {
            "chat_id": "chat",
            "model_id": "model",
            "prompt": "hello",
            "sampling": {"temperature": 0.0, "top_k": 0},
        }
    )

    sampling = _effective_sampling(settings, request.sampling)

    assert sampling["max_output_tokens"] == 91
    assert sampling["temperature"] == 0.0
    assert sampling["top_k"] == 0
    assert sampling["alternatives"] == 6
    assert _effective_deterministic_mode(settings, request) is True


def test_explicit_false_overrides_configured_deterministic_mode() -> None:
    settings = AppSettings.model_validate({"inference": {"deterministic_reference_mode": True}})
    request = GenerationRunCreate.model_validate(
        {
            "chat_id": "chat",
            "model_id": "model",
            "prompt": "hello",
            "deterministic_reference_mode": False,
        }
    )

    assert _effective_deterministic_mode(settings, request) is False


@pytest.mark.parametrize(
    "payload",
    [
        {"modality": "text"},
        {"modality": "text", "text": "hello", "attachment_id": "media"},
        {"modality": "image"},
        {"modality": "image", "attachment_id": "media", "text": "caption"},
        {"modality": "mixed", "text": "caption"},
        {"modality": "mixed", "attachment_id": "media"},
    ],
)
def test_embedding_input_fields_must_match_declared_modality(payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingItemRequest.model_validate(payload)


def test_embedding_input_accepts_coherent_text_media_and_mixed_shapes() -> None:
    assert EmbeddingItemRequest(modality="text", text="hello").attachment_id is None
    assert EmbeddingItemRequest(modality="image", attachment_id="media").text is None
    assert (
        EmbeddingItemRequest(modality="mixed", text="caption", attachment_id="media").modality
        == "mixed"
    )
