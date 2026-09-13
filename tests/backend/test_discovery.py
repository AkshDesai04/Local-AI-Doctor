from __future__ import annotations

import json
import struct
from pathlib import Path

from local_ai_doctor.discovery import ModelScanner, fingerprint_model_directory
from local_ai_doctor.discovery.capabilities import ModelEvidence, build_capability_matrix
from local_ai_doctor.domain import Capability, CapabilityState, ModelTask
from local_ai_doctor.hardware.models import BackendKind


def test_dense_generation_discovery_uses_metadata_not_gate_name(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    assert model.task is ModelTask.TEXT_GENERATION
    assert model.parameter_count == 18
    assert model.weight_dtypes == {"BF16": 18}
    assert not model.is_moe  # Dense SwiGLU gate_proj is not a router.
    assert model.reasoning_delimiters == ("<think>", "</think>")
    assert model.effective_context_limit == 4096
    assert model.capabilities.support(Capability.MOE_ROUTING).state is CapabilityState.UNSUPPORTED
    assert model.capabilities.support(Capability.REASONING_CHANNEL).state is CapabilityState.FULL
    assert model.capabilities.support(Capability.ATTENTION_CAPTURE).state is CapabilityState.PARTIAL
    assert (
        model.capabilities.support(Capability.HIDDEN_STATE_CAPTURE).state
        is CapabilityState.UNSUPPORTED
    )
    assert (
        model.capabilities.support(Capability.EXTRACTED_TEXT_FILE_INPUT).state
        is CapabilityState.UNSUPPORTED
    )
    codes = {item.code for item in model.diagnostics}
    assert {"conflicting_context_metadata", "conflicting_bos_token_id"} <= codes
    assert model.loadable


def test_sentence_transformer_evidence_overrides_conditional_generation_label(
    embedding_model_dir: Path,
) -> None:
    model = ModelScanner([embedding_model_dir.parent]).scan().models[0]
    assert model.task is ModelTask.MULTIMODAL_EMBEDDING
    assert model.modalities == frozenset({"text", "image", "video"})
    assert model.capabilities.support(Capability.EMBEDDINGS).state is CapabilityState.FULL
    assert (
        model.capabilities.support(Capability.TEXT_GENERATION).state is CapabilityState.UNSUPPORTED
    )
    assert model.capabilities.support(Capability.VISION).state is CapabilityState.FULL
    assert model.capabilities.support(Capability.VIDEO).state is CapabilityState.FULL
    assert model.capabilities.support(Capability.AUDIO).state is CapabilityState.UNSUPPORTED
    assert model.capabilities.support(Capability.MOE_ROUTING).reason.startswith("not applicable")
    assert model.metadata["normalization"] is True
    assert model.metadata["embedding_dimension"] == 128
    assert model.metadata["minimum_embedding_dimension"] == 64
    assert model.metadata["supports_dimension_truncation"] is True
    assert model.metadata["pooling"] == "last-token"
    assert model.metadata["joint_embedding_space"] is True


def test_sentence_transformer_normalization_is_read_from_modules_manifest(
    embedding_model_dir: Path,
) -> None:
    (embedding_model_dir / "2_Normalize").rmdir()
    (embedding_model_dir / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.Transformer"},
                {
                    "idx": 1,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize",
                },
            ]
        ),
        encoding="utf-8",
    )

    model = ModelScanner([embedding_model_dir.parent]).scan().models[0]

    assert model.metadata["normalization"] is True


def test_generic_sentence_transformer_does_not_inherit_qwen_matryoshka_claims(
    embedding_model_dir: Path,
) -> None:
    config_path = embedding_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"architectures": ["BertModel"], "model_type": "bert"})
    config.pop("vision_config", None)
    config.pop("video_token_id", None)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (embedding_model_dir / "preprocessor_config.json").unlink()
    (embedding_model_dir / "video_preprocessor_config.json").unlink()

    model = ModelScanner([embedding_model_dir.parent]).scan().models[0]

    assert model.task is ModelTask.EMBEDDING
    assert model.metadata["pooling"] == "last-token"
    assert model.metadata["supports_dimension_truncation"] is False
    assert model.metadata["minimum_embedding_dimension"] is None
    assert model.metadata["joint_embedding_space"] is False


def test_corrupt_shard_is_registered_with_actionable_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "broken"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"]}), encoding="utf-8"
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(struct.pack("<Q", 1000) + b"{}")
    model = ModelScanner([tmp_path]).scan().models[0]
    assert not model.loadable
    assert any(item.code == "safetensors_header_invalid" for item in model.diagnostics)


def test_fingerprint_is_stable_and_changes_with_metadata(causal_model_dir: Path) -> None:
    first = fingerprint_model_directory(causal_model_dir)
    second = fingerprint_model_directory(causal_model_dir)
    assert first == second
    config_path = causal_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_position_embeddings"] = 16384
    config_path.write_text(json.dumps(config), encoding="utf-8")
    changed = fingerprint_model_directory(causal_model_dir)
    assert changed.value != first.value
    assert changed.weights_digest == first.weights_digest


def test_unknown_architecture_fails_gracefully(tmp_path: Path) -> None:
    model_dir = tmp_path / "unknown"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["FutureModel"]}), encoding="utf-8"
    )
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(4)
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.task is ModelTask.UNKNOWN
    assert not model.loadable
    assert any(item.code == "unsupported_architecture" for item in model.diagnostics)


def test_encoder_decoder_capabilities_do_not_overclaim_prompt_scoring(tmp_path: Path) -> None:
    model_dir = tmp_path / "encoder-decoder"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["T5ForConditionalGeneration"],
                "is_encoder_decoder": True,
                "decoder_start_token_id": 0,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {"encoder.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(4)
    )

    model = ModelScanner([tmp_path]).scan().models[0]

    assert model.task is ModelTask.ENCODER_DECODER_GENERATION
    assert (
        model.capabilities.support(Capability.ENCODER_DECODER_GENERATION).state
        is CapabilityState.FULL
    )
    prompt_scoring = model.capabilities.support(Capability.PROMPT_SCORING)
    assert prompt_scoring.state is CapabilityState.UNSUPPORTED
    assert "source and target" in (prompt_scoring.reason or "")


def test_public_descriptor_redacts_model_root(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    public = model.public_dict()
    assert str(causal_model_dir.parent) not in public["path"]
    assert public["path"].endswith(causal_model_dir.name)


def test_unimplemented_accelerator_adapters_are_never_advertised_as_full() -> None:
    matrix = build_capability_matrix(
        ModelEvidence(
            task=ModelTask.TEXT_GENERATION,
            modalities=frozenset({"text"}),
            has_tokenizer=True,
            has_processor=False,
            has_reasoning_delimiters=False,
            is_moe=False,
            architecture_known=True,
            available_backends=frozenset({BackendKind.CPU, BackendKind.ROCM, BackendKind.MPS}),
        )
    )

    assert matrix.support(Capability.CPU).state is CapabilityState.FULL
    assert matrix.support(Capability.ROCM).state is CapabilityState.UNAVAILABLE_ON_BACKEND
    assert matrix.support(Capability.MPS).state is CapabilityState.UNAVAILABLE_ON_BACKEND


def test_moe_checkpoint_does_not_claim_routing_without_an_instrumentation_adapter() -> None:
    matrix = build_capability_matrix(
        ModelEvidence(
            task=ModelTask.TEXT_GENERATION,
            modalities=frozenset({"text"}),
            has_tokenizer=True,
            has_processor=False,
            has_reasoning_delimiters=False,
            is_moe=True,
            architecture_known=True,
        )
    )

    routing = matrix.support(Capability.MOE_ROUTING)
    assert routing.state is CapabilityState.UNSUPPORTED
    assert "no production router" in (routing.reason or "")
