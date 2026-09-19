from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from local_ai_doctor.discovery import ModelScanner, fingerprint_model_directory
from local_ai_doctor.discovery.capabilities import ModelEvidence, build_capability_matrix
from local_ai_doctor.domain import Capability, CapabilityState, ModelTask
from local_ai_doctor.domain.models import DiagnosticSeverity, TrustDecision
from local_ai_doctor.hardware.models import BackendKind


def test_dense_generation_discovery_uses_metadata_not_gate_name(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    assert model.task is ModelTask.TEXT_GENERATION
    assert model.parameter_count == 18
    assert model.weight_dtypes == {"BF16": 18}
    assert not model.is_moe  # Dense SwiGLU gate_proj is not a router.
    assert model.reasoning_delimiters == ("<think>", "</think>")
    assert model.effective_context_limit == 8192  # The declared capacity, not the app default.
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


def _minimal_checkpoint(directory: Path, config: dict[str, object]) -> Path:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}}}), encoding="utf-8"
    )
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(4)
    )
    return directory


def test_sliding_window_is_not_reported_as_the_context_limit(tmp_path: Path) -> None:
    """Gemma 3 declares a 512-token attention span with a 32768-token context."""

    _minimal_checkpoint(
        tmp_path / "sliding",
        {
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "max_position_embeddings": 32768,
            "sliding_window": 512,
            "sliding_window_pattern": 6,
        },
    )
    model = ModelScanner([tmp_path], conservative_context_limit=8192).scan().models[0]
    assert model.effective_context_limit == 32768
    assert all("sliding_window" not in item.source for item in model.context_values)


def test_declared_capacity_outranks_the_application_fallback(tmp_path: Path) -> None:
    """Selecting the minimum used to floor every checkpoint to the portable default."""

    _minimal_checkpoint(
        tmp_path / "long",
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 131072,
        },
    )
    model = ModelScanner([tmp_path], conservative_context_limit=4096).scan().models[0]
    assert model.effective_context_limit == 131072
    fallback = next(item for item in model.context_values if item.source.startswith("application."))
    assert fallback.value == 4096  # Still recorded as evidence, just not selected.
    assert not any(item.code == "conflicting_context_metadata" for item in model.diagnostics)


def test_rope_scaling_base_length_is_evidence_but_never_selected(tmp_path: Path) -> None:
    """Llama 3.2 declares an 8192 pre-scaling base and reaches 131072 through RoPE."""

    _minimal_checkpoint(
        tmp_path / "scaled",
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 131072,
            "rope_scaling": {"rope_type": "llama3", "original_max_position_embeddings": 8192},
        },
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.effective_context_limit == 131072
    assert any(
        item.source.endswith("original_max_position_embeddings") and item.value == 8192
        for item in model.context_values
    )
    assert not any(item.code == "conflicting_context_metadata" for item in model.diagnostics)


def test_family_specific_capacity_keys_are_read(tmp_path: Path) -> None:
    """MPT checkpoints such as Krutrim-1 spell the capacity `max_seq_len`."""

    _minimal_checkpoint(
        tmp_path / "mpt",
        {"architectures": ["MPTForCausalLM"], "model_type": "mpt", "max_seq_len": 4096},
    )
    model = ModelScanner([tmp_path], conservative_context_limit=1024).scan().models[0]
    assert model.effective_context_limit == 4096
    assert any(item.source == "config.max_seq_len" for item in model.context_values)


def test_sentence_transformer_truncation_outranks_positional_capacity(tmp_path: Path) -> None:
    """XLM-R allocates 514 positions while the pipeline truncates at 128."""

    directory = _minimal_checkpoint(
        tmp_path / "embedder",
        {
            "architectures": ["XLMRobertaModel"],
            "model_type": "xlm-roberta",
            "max_position_embeddings": 514,
        },
    )
    (directory / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 128}), encoding="utf-8"
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 128}), encoding="utf-8"
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.effective_context_limit == 128


def test_tokenizer_sentinel_does_not_displace_the_declared_capacity(tmp_path: Path) -> None:
    """Gemma 3 ships `model_max_length` as an unbounded float sentinel."""

    directory = _minimal_checkpoint(
        tmp_path / "sentinel",
        {
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "max_position_embeddings": 32768,
        },
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 1000000000000000019884624838656}), encoding="utf-8"
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.effective_context_limit == 32768
    assert any(item.code == "context_sentinel_ignored" for item in model.diagnostics)
    assert not any(item.code == "conflicting_context_metadata" for item in model.diagnostics)


def test_application_fallback_applies_only_when_nothing_is_declared(tmp_path: Path) -> None:
    _minimal_checkpoint(
        tmp_path / "bare",
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
    )
    model = ModelScanner([tmp_path], conservative_context_limit=2048).scan().models[0]
    assert model.effective_context_limit == 2048


def test_auto_map_is_not_blocking_when_transformers_implements_the_family(tmp_path: Path) -> None:
    """Phi-3 ships a pre-upstreaming auto_map that the built-in class supersedes."""

    _minimal_checkpoint(
        tmp_path / "superseded",
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 4096,
            "auto_map": {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"},
        },
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.trust_decision is TrustDecision.BUILTIN_ONLY
    assert model.loadable
    assert any(item.code == "bundled_custom_code_superseded" for item in model.diagnostics)


def test_auto_map_still_blocks_when_no_built_in_implementation_reads_it(tmp_path: Path) -> None:
    _minimal_checkpoint(
        tmp_path / "custom",
        {
            "architectures": ["FutureForCausalLM"],
            "model_type": "future-private-family",
            "auto_map": {"AutoModelForCausalLM": "modeling_future.FutureForCausalLM"},
        },
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.trust_decision is TrustDecision.REJECTED_CUSTOM_CODE
    assert not model.loadable
    assert any(item.code == "custom_code_not_trusted" for item in model.diagnostics)


def test_rejected_custom_code_reports_the_configuration_field_at_fault(tmp_path: Path) -> None:
    """Krutrim-1 declares `mpt` but a grouped-query `attn_type` only its code accepts."""

    _minimal_checkpoint(
        tmp_path / "grouped",
        {
            "architectures": ["MPTForCausalLM"],
            "model_type": "mpt",
            "max_seq_len": 4096,
            "attn_config": {"attn_type": "grouped_query_attention", "kv_n_heads": 8},
            "auto_map": {"AutoModelForCausalLM": "modeling_mpt.MPTForCausalLM"},
        },
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert not model.loadable
    rejection = next(item for item in model.diagnostics if item.code == "custom_code_not_trusted")
    assert "attn_type" in rejection.message
    assert str(tmp_path) not in rejection.message


def test_fingerprint_pinned_review_admits_bundled_code_that_builtin_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reviewed entry is the only way past the same rejection above.

    It is pinned to one manifest fingerprint and one directory name, so it
    cannot be satisfied by an unrelated checkpoint that merely shares a
    `model_type`, and it must stop applying the moment either changes.
    """

    from local_ai_doctor.discovery import reviewed_bundled_code
    from local_ai_doctor.discovery import scanner as scanner_module
    from local_ai_doctor.discovery.fingerprint import fingerprint_model_directory

    directory = _minimal_checkpoint(
        tmp_path / "Krutrim-1-instruct",
        {
            "architectures": ["MPTForCausalLM"],
            "model_type": "mpt",
            "max_seq_len": 4096,
            "attn_config": {"attn_type": "grouped_query_attention", "kv_n_heads": 8},
            "auto_map": {"AutoModelForCausalLM": "modeling_mpt.MPTForCausalLM"},
        },
    )
    fingerprint = fingerprint_model_directory(directory)
    reviewed = (
        reviewed_bundled_code.ReviewedCheckpoint(
            fingerprint=fingerprint.value,
            directory_name="Krutrim-1-instruct",
            reason="test-only synthetic review",
        ),
    )
    monkeypatch.setattr(reviewed_bundled_code, "_REVIEWED", reviewed)
    monkeypatch.setattr(scanner_module, "reviewed_reason", reviewed_bundled_code.reviewed_reason)

    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.trust_decision is TrustDecision.REVIEWED_BUNDLED_CODE
    assert model.loadable
    approved = next(
        item for item in model.diagnostics if item.code == "bundled_custom_code_reviewed"
    )
    assert approved.severity is DiagnosticSeverity.INFO
    assert not any(item.code == "custom_code_not_trusted" for item in model.diagnostics)

    # Renaming the directory changes nothing about the bytes, but the review is
    # pinned by name too: a copy at a different path must not inherit it.
    renamed = tmp_path / "Krutrim-1-instruct-copy"
    directory.rename(renamed)
    renamed_model = ModelScanner([tmp_path]).scan().models[0]
    assert renamed_model.trust_decision is TrustDecision.REJECTED_CUSTOM_CODE
    assert not renamed_model.loadable


def test_task_falls_back_to_the_installed_transformers_mapping(tmp_path: Path) -> None:
    """An unfamiliar architecture name must not hide a natively supported family."""

    _minimal_checkpoint(
        tmp_path / "mapped",
        {"architectures": ["Qwen3VLThinkingModel"], "model_type": "qwen3_vl"},
    )
    model = ModelScanner([tmp_path]).scan().models[0]
    assert model.task is ModelTask.TEXT_GENERATION
    assert model.loadable
