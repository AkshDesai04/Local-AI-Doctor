from __future__ import annotations

import json
import struct
from pathlib import Path

from local_ai_doctor.discovery import ModelScanner, fingerprint_model_directory
from local_ai_doctor.domain import Capability, CapabilityState, ModelTask


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


def test_public_descriptor_redacts_model_root(causal_model_dir: Path) -> None:
    model = ModelScanner([causal_model_dir.parent]).scan().models[0]
    public = model.public_dict()
    assert str(causal_model_dir.parent) not in public["path"]
    assert public["path"].endswith(causal_model_dir.name)
