"""Secure, read-only scanning of configured model roots."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from ..domain.models import (
    ContextValue,
    Diagnostic,
    DiagnosticSeverity,
    ModelComponents,
    ModelDescriptor,
    ModelTask,
    TrustDecision,
)
from ..domain.quantization import (
    DERIVATION_FILE,
    bitsandbytes_available,
    weight_quantization,
)
from ..hardware.models import BackendKind
from .bundled_imports import check_bundled_imports
from .capabilities import ModelEvidence, build_capability_matrix
from .fingerprint import FingerprintPolicy, fingerprint_model_directory
from .reviewed_bundled_code import reviewed_reason
from .safetensors import SafeTensorSummary, inspect_safetensors
from .transformers_support import builtin_config_error, native_generation_task

_MAX_JSON_BYTES: Final[int] = 32 * 1024**2
_KNOWN_GENERATION_ARCHITECTURE_PARTS: Final[tuple[str, ...]] = (
    "forcausallm",
    "forconditionalgeneration",
    "forseq2seqlm",
    "encoderdecodermodel",
)
_KNOWN_EMBEDDING_ARCHITECTURE_PARTS: Final[tuple[str, ...]] = (
    "embedding",
    "sentence",
    "featureextraction",
)
_CHAT_TEMPLATE_FILES: Final[tuple[str, ...]] = ("chat_template.jinja", "chat_template.json")
# Positional-capacity keys, in the order runtimes honour them. Families that predate
# `max_position_embeddings` still ship their own spelling: MPT declares `max_seq_len`,
# GPT-2 derivatives `n_positions`, ChatGLM/Baichuan `seq_length`.
_CONTEXT_CAPACITY_KEYS: Final[tuple[str, ...]] = (
    "max_position_embeddings",
    "max_seq_len",
    "n_positions",
    "seq_length",
    "max_sequence_length",
    "n_ctx",
)
# Tokenizers routinely encode "unbounded" as a float sentinel (1e30) rather than omitting
# the field, so any value beyond this ceiling is evidence, never a selectable length.
_CONTEXT_SENTINEL_CEILING: Final[int] = 10_000_000
_SAFETENSORS_DTYPE_NAMES: Final[dict[str, str]] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
}
# Pickle-based checkpoint formats. They are recognized only to explain why the
# folder cannot load; the workbench never unpickles weights.
_PICKLE_WEIGHT_SUFFIXES: Final[frozenset[str]] = frozenset({".bin", ".pt", ".pth", ".ckpt"})
_MOE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_routed_experts",
        "moe_num_experts",
    }
)


class RootScanResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: Path
    discovered_count: int = Field(ge=0)
    diagnostics: tuple[Diagnostic, ...] = ()


class ModelScanReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[ModelDescriptor, ...]
    roots: tuple[RootScanResult, ...]

    @property
    def by_id(self) -> dict[str, ModelDescriptor]:
        return {model.id: model for model in self.models}

    def public_roots(self) -> list[dict[str, Any]]:
        """Root results without host paths; diagnostics carry only relative names."""

        return [
            {
                "root": f"<model-root:{index}>",
                "discovered_count": result.discovered_count,
                "diagnostics": [item.model_dump(mode="json") for item in result.diagnostics],
            }
            for index, result in enumerate(self.roots)
        ]


def _safe_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if path.is_symlink():
        return {}, f"{path.name} is a symbolic link; linked metadata is not read"
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return {}, f"{path.name} exceeds the {_MAX_JSON_BYTES}-byte metadata limit"
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"{path.name} could not be parsed ({type(exc).__name__})"
    if not isinstance(value, dict):
        return {}, f"{path.name} must contain a JSON object"
    return value, None


def _nested_value(mapping: Mapping[str, Any], *parts: str) -> Any:
    value: Any = mapping
    for part in parts:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _sentence_transformer_normalizes(directory: Path) -> bool:
    """Recognize Normalize modules even when they have no configuration folder."""

    modules_path = directory / "modules.json"
    if not _regular_file(modules_path):
        return False
    try:
        if modules_path.stat().st_size > _MAX_JSON_BYTES:
            return False
        modules = json.loads(modules_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    entries = (
        modules
        if isinstance(modules, list)
        else modules.get("modules", [])
        if isinstance(modules, Mapping)
        else []
    )
    return any(
        isinstance(entry, Mapping)
        and "normalize" in str(entry.get("type", entry.get("name", ""))).lower()
        for entry in entries
    )


def _sentence_transformer_embedding_dimension(directory: Path) -> int | None:
    """Read the declared pooling width without importing checkpoint code."""

    pooling, error = _safe_json(directory / "1_Pooling" / "config.json")
    if error is not None:
        return None
    # Classic sentence-transformers Pooling configs (XLM-R, BERT) name the width
    # `word_embedding_dimension`; newer packaging (Qwen3-VL-Embedding) uses the short key.
    return _positive_int(pooling.get("embedding_dimension")) or _positive_int(
        pooling.get("word_embedding_dimension")
    )


def _sentence_transformer_pooling(directory: Path) -> str | None:
    """Return the explicitly configured pooling strategy, if unambiguous."""

    pooling, error = _safe_json(directory / "1_Pooling" / "config.json")
    if error is not None:
        return None
    direct = pooling.get("pooling_mode")
    if isinstance(direct, str) and direct.strip():
        normalized = direct.strip().casefold().replace("_", "-")
        return {"lasttoken": "last-token", "weightedmean": "weighted-mean"}.get(
            normalized, normalized
        )
    candidates = {
        "cls": pooling.get("pooling_mode_cls_token"),
        "max": pooling.get("pooling_mode_max_tokens"),
        "mean": pooling.get("pooling_mode_mean_tokens"),
        "mean-sqrt-length": pooling.get("pooling_mode_mean_sqrt_len_tokens"),
        "weighted-mean": pooling.get("pooling_mode_weightedmean_tokens"),
        "last-token": pooling.get("pooling_mode_lasttoken"),
    }
    enabled = [name for name, value in candidates.items() if value is True]
    return enabled[0] if len(enabled) == 1 else None


def _sentence_transformer_max_seq_length(directory: Path) -> int | None:
    """Read the hard truncation length the sentence-transformers pipeline applies."""

    payload, error = _safe_json(directory / "sentence_bert_config.json")
    if error is not None:
        return None
    return _positive_int(payload.get("max_seq_length"))


def _context_values(
    config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    conservative_default: int,
    sentence_transformer_max_seq_length: int | None = None,
) -> tuple[tuple[ContextValue, ...], int, list[Diagnostic]]:
    """Select the context length the checkpoint itself declares.

    Sources are ranked by authority instead of reduced to a minimum. Taking the minimum
    silently floored every model to the application default, so a 131,072-token
    checkpoint reported 4,096. Ranking matters because the sources mean different
    things: the sentence-transformers pipeline truncates hard at its configured length,
    the architecture's positional capacity is the field every runtime honours, and
    `model_max_length` is a tokenizer hint that is frequently a sentinel or a stale copy.
    Every discovered value is still recorded so disagreements stay visible.
    """

    ranked: tuple[tuple[int | None, str, Any, str | None], ...] = (
        (
            0,
            "sentence_bert_config.max_seq_length",
            sentence_transformer_max_seq_length,
            "the sentence-transformers pipeline truncates every input at this length",
        ),
        *((1, f"config.{key}", config.get(key), None) for key in _CONTEXT_CAPACITY_KEYS),
        *(
            (1, f"config.text_config.{key}", _nested_value(config, "text_config", key), None)
            for key in _CONTEXT_CAPACITY_KEYS
        ),
        # `sliding_window` is a per-layer attention span, not a sequence limit: Gemma 3
        # declares 512 with a 32768 context and Phi-3-mini-4k declares 2047 with 4096.
        # Treating it as a context candidate under-reported the usable context badly.
        # `original_max_position_embeddings` is the pre-scaling base for the same reason:
        # Llama 3.2 declares 8192 there and reaches 131072 through RoPE scaling.
        (
            None,
            "config.rope_scaling.original_max_position_embeddings",
            _nested_value(config, "rope_scaling", "original_max_position_embeddings"),
            "pre-scaling base length; RoPE scaling extends it to the declared capacity",
        ),
        (2, "tokenizer_config.model_max_length", tokenizer.get("model_max_length"), None),
    )

    diagnostics: list[Diagnostic] = []
    candidates: list[ContextValue] = []
    declared: set[int] = set()
    selected: int | None = None
    selected_source: str | None = None
    selected_tier: int | None = None
    for tier, source, raw_value, note in ranked:
        value = _positive_int(raw_value)
        if value is None:
            continue
        if value > _CONTEXT_SENTINEL_CEILING:
            candidates.append(ContextValue(source=source, value=value, note=note))
            diagnostics.append(
                Diagnostic(
                    code="context_sentinel_ignored",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"{source} looks like an unbounded tokenizer sentinel and was not selected",
                    evidence={"source": source, "value": value},
                )
            )
            continue
        candidates.append(ContextValue(source=source, value=value, note=note))
        if tier is None:
            continue
        declared.add(value)
        if selected_tier is None or tier < selected_tier:
            selected_tier, selected, selected_source = tier, value, source

    candidates.append(
        ContextValue(
            source="application.conservative_context_limit",
            value=conservative_default,
            note="portable fallback used only when the checkpoint declares no context length",
        )
    )
    if selected is None:
        selected = conservative_default
        selected_source = "application.conservative_context_limit"
    if len(declared) > 1:
        diagnostics.append(
            Diagnostic(
                code="conflicting_context_metadata",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "model files declare conflicting context lengths; the most authoritative "
                    "source was selected"
                ),
                evidence={
                    "discovered_values": sorted(declared),
                    "selected": selected,
                    "selected_source": selected_source,
                },
            )
        )
    return tuple(candidates), selected, diagnostics


def _detect_modalities(config: Mapping[str, Any], directory: Path) -> frozenset[str]:
    modalities = {"text"}
    if isinstance(config.get("vision_config"), Mapping) or _regular_file(
        directory / "preprocessor_config.json"
    ):
        modalities.add("image")
    if (
        isinstance(config.get("video_config"), Mapping)
        or _regular_file(directory / "video_preprocessor_config.json")
        or _positive_int(config.get("video_token_id")) is not None
    ):
        modalities.add("video")
    if (
        isinstance(config.get("audio_config"), Mapping)
        or _regular_file(directory / "audio_preprocessor_config.json")
        or _positive_int(config.get("audio_token_id")) is not None
    ):
        modalities.add("audio")
    return frozenset(modalities)


def _detect_moe(config: Mapping[str, Any], tensor_names: Iterable[str]) -> bool:
    def walk(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        for key, item in value.items():
            if key in _MOE_KEYS and (_positive_int(item) or 0) > 1:
                return True
            if isinstance(item, Mapping) and walk(item):
                return True
        return False

    if walk(config):
        return True
    # `gate_proj` is intentionally absent: dense SwiGLU layers use that name.
    return any(".experts." in name or ".router." in name for name in tensor_names)


def _detect_task(
    config: Mapping[str, Any],
    architectures: tuple[str, ...],
    directory: Path,
    modalities: frozenset[str],
) -> tuple[ModelTask, bool, str]:
    architecture_text = " ".join(architectures).lower()
    sentence_transformer = (
        _regular_file(directory / "modules.json")
        or _regular_file(directory / "config_sentence_transformers.json")
        or _regular_file(directory / "sentence_bert_config.json")
    )
    if sentence_transformer or any(
        part in architecture_text for part in _KNOWN_EMBEDDING_ARCHITECTURE_PARTS
    ):
        task = ModelTask.MULTIMODAL_EMBEDDING if len(modalities) > 1 else ModelTask.EMBEDDING
        return task, True, "SentenceTransformers/embedding metadata"
    encoder_decoder = config.get("is_encoder_decoder") is True or any(
        part in architecture_text for part in ("seq2seq", "encoderdecoder")
    )
    if encoder_decoder:
        return ModelTask.ENCODER_DECODER_GENERATION, True, "encoder-decoder architecture metadata"
    if any(part in architecture_text for part in _KNOWN_GENERATION_ARCHITECTURE_PARTS):
        return ModelTask.TEXT_GENERATION, True, "generation architecture metadata"
    native = native_generation_task(str(config.get("model_type") or "") or None)
    if native is not None:
        return native, True, "installed Transformers native architecture mapping"
    return ModelTask.UNKNOWN, False, "no recognized task metadata"


def _components(
    directory: Path,
    config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> ModelComponents:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "spiece.model",
    )
    processor_files = (
        "processor_config.json",
        "preprocessor_config.json",
        "image_processor_config.json",
        "video_preprocessor_config.json",
        "audio_preprocessor_config.json",
    )
    custom_code = any(
        path.is_file() and not path.is_symlink()
        for path in directory.rglob("*.py")
        if len(path.relative_to(directory).parts) <= 3
    )
    return ModelComponents(
        config=_regular_file(directory / "config.json"),
        safetensors=any(_regular_file(path) for path in directory.glob("*.safetensors")),
        tokenizer=any(_regular_file(directory / name) for name in tokenizer_files),
        chat_template=bool(tokenizer.get("chat_template"))
        or any(_regular_file(directory / name) for name in _CHAT_TEMPLATE_FILES),
        generation_config=_regular_file(directory / "generation_config.json"),
        processor=any(_regular_file(directory / name) for name in processor_files),
        pooling=_regular_file(directory / "modules.json")
        or _regular_file(directory / "1_Pooling" / "config.json"),
        custom_code=custom_code,
    )


def _template_strings(value: Any) -> list[str]:
    """Normalize a chat template field: one string, or a list of named templates."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [
            item["template"]
            for item in value
            if isinstance(item, Mapping) and isinstance(item.get("template"), str)
        ]
    return []


def _reasoning_delimiters(tokenizer: Mapping[str, Any], directory: Path) -> tuple[str, str] | None:
    # Only chat-template text proves the model emits reasoning markers. The rest of
    # tokenizer_config.json (notably `added_tokens_decoder`) lists `<think>` for every
    # tokenizer in a family, including embedding checkpoints that never generate.
    templates = _template_strings(tokenizer.get("chat_template"))
    jinja_path = directory / "chat_template.jinja"
    if _regular_file(jinja_path) and jinja_path.stat().st_size <= _MAX_JSON_BYTES:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            templates.append(jinja_path.read_text(encoding="utf-8"))
    if _regular_file(directory / "chat_template.json"):
        payload, error = _safe_json(directory / "chat_template.json")
        if error is None:
            templates.extend(_template_strings(payload.get("chat_template")))
    if any("<think>" in text and "</think>" in text for text in templates):
        return ("<think>", "</think>")
    return None


def _metadata_diagnostics(
    config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    bos_values: dict[str, int] = {}
    for source, value in (
        ("config.json", config.get("bos_token_id")),
        ("config.text_config", _nested_value(config, "text_config", "bos_token_id")),
        ("generation_config.json", generation.get("bos_token_id")),
    ):
        parsed = _positive_int(value)
        if parsed is not None:
            bos_values[source] = parsed
    if len(set(bos_values.values())) > 1:
        diagnostics.append(
            Diagnostic(
                code="conflicting_bos_token_id",
                severity=DiagnosticSeverity.WARNING,
                message="bundled files disagree about the BOS token; the adapter must use the loaded tokenizer and generation configuration",
                evidence={"values": bos_values},
            )
        )
    if not tokenizer:
        diagnostics.append(
            Diagnostic(
                code="tokenizer_metadata_missing",
                severity=DiagnosticSeverity.INFO,
                message="tokenizer_config.json is absent or unreadable",
            )
        )
    return diagnostics


def _skipped_directory_diagnostic(directory: Path, root: Path) -> Diagnostic | None:
    """Explain a folder that looks like a model location but can never be a candidate.

    Without this an empty download or a GGUF-only export was silently invisible, so
    the registry gave no hint why a model the user placed in the root did not appear.
    """

    try:
        entries = [item for item in directory.iterdir() if not item.name.startswith(".")]
    except OSError:
        return None  # The caller reports unreadable directories when it lists them.
    relative = directory.relative_to(root).as_posix()
    if not entries:
        return Diagnostic(
            code="empty_model_directory",
            severity=DiagnosticSeverity.WARNING,
            message=f"model directory {relative!r} is empty",
            hint="Finish the download or remove the folder from the model root.",
            evidence={"directory": relative},
        )
    if any(item.suffix.lower() == ".gguf" and _regular_file(item) for item in entries):
        return Diagnostic(
            code="gguf_only_directory",
            severity=DiagnosticSeverity.WARNING,
            message=f"model directory {relative!r} contains GGUF weights, which this workbench does not load",
            hint="Use the original Transformers SafeTensors release of the model (config.json, tokenizer, *.safetensors).",
            evidence={"directory": relative},
        )
    return None


def _size_note(weight_bytes: int) -> str:
    gib = weight_bytes / 1024**3
    if gib <= 8:
        return ""
    return f" For reference, the weights total {gib:.1f} GiB, more than a typical 8 GiB GPU holds."


def _declared_dtype(config: Mapping[str, Any]) -> str | None:
    value = (
        config.get("dtype")
        or config.get("torch_dtype")
        or _nested_value(config, "text_config", "dtype")
        or _nested_value(config, "text_config", "torch_dtype")
    )
    return str(value) if value else None


def _stored_dtype(dtype_parameter_counts: Mapping[str, int]) -> str | None:
    """Name the dtype that stores most parameters, in the configuration's spelling."""

    if not dtype_parameter_counts:
        return None
    dominant = max(dtype_parameter_counts.items(), key=lambda item: item[1])[0]
    return _SAFETENSORS_DTYPE_NAMES.get(dominant, dominant.lower())


_DERIVATION_KEYS: Final[tuple[str, ...]] = (
    "schema",
    "schema_version",
    "source_model_id",
    "source_fingerprint",
    "source_display_name",
    "quantization",
    "quantization_config",
    "software",
    "created_at",
)


def _derivation(directory: Path) -> dict[str, Any] | None:
    """The path-free provenance a Flush to storage wrote beside the weights, if any."""

    path = directory / DERIVATION_FILE
    if not _regular_file(path):
        return None
    raw, error = _safe_json(path)
    if error:
        return None
    return {key: raw[key] for key in _DERIVATION_KEYS if key in raw}


def _safe_identifier(name: str, fingerprint: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "model"
    return f"{slug[:80]}-{fingerprint[:12]}"


class ModelScanner:
    """Scan configured roots without following links or modifying model files."""

    def __init__(
        self,
        roots: Iterable[Path],
        *,
        conservative_context_limit: int = 4096,
        max_depth: int = 2,
        fingerprint_policy: FingerprintPolicy | None = None,
        available_backends: frozenset[BackendKind] = frozenset({BackendKind.CPU}),
        quantization_backend: bool | None = None,
    ) -> None:
        if conservative_context_limit <= 0:
            raise ValueError("conservative_context_limit must be positive")
        if max_depth < 0 or max_depth > 16:
            raise ValueError("max_depth must be between 0 and 16")
        self._roots = tuple(roots)
        self._conservative_context_limit = conservative_context_limit
        self._max_depth = max_depth
        self._fingerprint_policy = fingerprint_policy or FingerprintPolicy()
        self._available_backends = available_backends
        self._quantization_backend = (
            bitsandbytes_available() if quantization_backend is None else quantization_backend
        )

    @staticmethod
    def _is_candidate(directory: Path) -> bool:
        return _regular_file(directory / "config.json") or any(
            _regular_file(path) for path in directory.glob("*.safetensors")
        )

    def _candidate_directories(self, root: Path) -> tuple[list[Path], list[Diagnostic]]:
        candidates: list[Path] = []
        diagnostics: list[Diagnostic] = []

        def visit(directory: Path, depth: int) -> None:
            if self._is_candidate(directory):
                candidates.append(directory)
                return
            if depth > 0:
                skipped = _skipped_directory_diagnostic(directory, root)
                if skipped is not None:
                    diagnostics.append(skipped)
                    return
            if depth >= self._max_depth:
                return
            try:
                children = sorted(
                    (
                        item
                        for item in directory.iterdir()
                        # Dot-directories are VCS, cache, or staging state (`.git`,
                        # `.cache`), never a model folder.
                        if item.is_dir() and not item.is_symlink() and not item.name.startswith(".")
                    ),
                    key=lambda item: item.name.casefold(),
                )
            except OSError as exc:
                diagnostics.append(
                    Diagnostic(
                        code="model_directory_unreadable",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"could not inspect model directory {directory.name!r}",
                        evidence={"directory": directory.name, "reason": type(exc).__name__},
                    )
                )
                return
            for child in children:
                try:
                    child.resolve(strict=True).relative_to(root)
                except (OSError, ValueError):
                    diagnostics.append(
                        Diagnostic(
                            code="model_symlink_escape_blocked",
                            severity=DiagnosticSeverity.WARNING,
                            message=f"skipped directory {child.name!r} because it escapes the configured root",
                        )
                    )
                    continue
                visit(child, depth + 1)

        visit(root, 0)
        return candidates, diagnostics

    def _describe(self, directory: Path, root_index: int | None = None) -> ModelDescriptor:
        config, config_error = (
            _safe_json(directory / "config.json")
            if (directory / "config.json").is_file()
            else ({}, "config.json is missing")
        )
        tokenizer, tokenizer_error = (
            _safe_json(directory / "tokenizer_config.json")
            if (directory / "tokenizer_config.json").is_file()
            else ({}, None)
        )
        generation, generation_error = (
            _safe_json(directory / "generation_config.json")
            if (directory / "generation_config.json").is_file()
            else ({}, None)
        )
        diagnostics: list[Diagnostic] = []
        metadata_errors = (
            (config_error, DiagnosticSeverity.ERROR),
            (tokenizer_error, DiagnosticSeverity.WARNING),
            (generation_error, DiagnosticSeverity.WARNING),
        )
        for error, severity in metadata_errors:
            if error:
                diagnostics.append(
                    Diagnostic(
                        code="model_metadata_invalid",
                        severity=severity,
                        message=error,
                        hint="Restore the original metadata file or re-export the model folder.",
                    )
                )

        weights = tuple(
            sorted(
                (path for path in directory.glob("*.safetensors") if _regular_file(path)),
                key=lambda item: item.name.casefold(),
            )
        )
        summary: SafeTensorSummary = inspect_safetensors(weights)
        # Computed early: the fingerprint pins reviewed bundled code below, and its
        # loader-weight byte total sizes the informational notes in blocking hints.
        fingerprint = fingerprint_model_directory(directory, self._fingerprint_policy)
        pickle_weights = sorted(
            path.name
            for path in directory.iterdir()
            if path.suffix.lower() in _PICKLE_WEIGHT_SUFFIXES and _regular_file(path)
        )
        if not weights and pickle_weights:
            diagnostics.append(
                Diagnostic(
                    code="pickle_weights_only",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"weights are stored only as pickle checkpoints ({len(pickle_weights)} "
                        f"file(s), for example {pickle_weights[0]}); the workbench never "
                        "unpickles weights because loading a pickle can execute code"
                    ),
                    hint=(
                        "Convert the checkpoint to SafeTensors offline with a trusted tool in an "
                        "isolated environment, write the result to a new folder with the same "
                        "config and tokenizer files, and add that folder to a model root. The "
                        "original folder is never modified."
                        + _size_note(fingerprint.total_weight_bytes)
                    ),
                    evidence={"pickle_files": len(pickle_weights)},
                )
            )
        elif not weights:
            diagnostics.append(
                Diagnostic(
                    code="safetensors_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="no SafeTensors weight file was found",
                    hint="This workbench intentionally discovers SafeTensors checkpoints only.",
                )
            )
        for error in summary.errors:
            diagnostics.append(
                Diagnostic(
                    code="safetensors_header_invalid",
                    severity=DiagnosticSeverity.ERROR,
                    message=error,
                    hint="Verify the checkpoint download without modifying the model in place.",
                )
            )

        raw_architectures = config.get("architectures", [])
        architectures = (
            tuple(str(item) for item in raw_architectures)
            if isinstance(raw_architectures, list)
            else ()
        )
        modalities = _detect_modalities(config, directory)
        task, architecture_known, task_evidence = _detect_task(
            config, architectures, directory, modalities
        )
        components = _components(directory, config, tokenizer)
        delimiters = _reasoning_delimiters(tokenizer, directory)
        is_moe = _detect_moe(config, summary.tensor_names)
        quantized = weight_quantization(config)
        bitsandbytes = quantized is not None and quantized["method"] == "bitsandbytes"
        if bitsandbytes and not self._quantization_backend:
            diagnostics.append(
                Diagnostic(
                    code="quantization_backend_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="the checkpoint stores bitsandbytes-quantized weights, but bitsandbytes is not installed",
                    hint="Install the CUDA extra (bitsandbytes) on a CUDA host to load this checkpoint.",
                )
            )

        if (
            task in {ModelTask.TEXT_GENERATION, ModelTask.ENCODER_DECODER_GENERATION}
            and not components.tokenizer
        ):
            diagnostics.append(
                Diagnostic(
                    code="generation_tokenizer_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="generation model is missing tokenizer vocabulary files",
                    hint="Restore tokenizer.json/tokenizer.model and its bundled configuration.",
                )
            )
        if len(modalities) > 1 and not components.processor:
            diagnostics.append(
                Diagnostic(
                    code="multimodal_processor_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="multimodal metadata was found but no processor configuration is available",
                )
            )
        if task is ModelTask.UNKNOWN:
            diagnostics.append(
                Diagnostic(
                    code="unsupported_architecture",
                    severity=DiagnosticSeverity.ERROR,
                    message="the model task could not be identified from local metadata",
                    hint="Add a reviewed ModelAdapter rather than enabling arbitrary repository code.",
                    evidence={"architectures": architectures},
                )
            )

        # The reviewed-bundled-code exception below is pinned to the fingerprint computed
        # above: a changed checkpoint at the same path must not silently inherit a review
        # that covered different bytes.
        model_type = str(config.get("model_type")) if config.get("model_type") else None
        auto_map = config.get("auto_map")
        trust_decision = TrustDecision.BUILTIN_ONLY
        if isinstance(auto_map, Mapping) and auto_map:
            # Many published checkpoints still carry an `auto_map` written before the
            # family was upstreamed (Phi-3). When the reviewed built-in configuration
            # reads the checkpoint, the bundled code is redundant rather than blocking.
            builtin_error = builtin_config_error(directory, model_type)
            if builtin_error is None:
                diagnostics.append(
                    Diagnostic(
                        code="bundled_custom_code_superseded",
                        severity=DiagnosticSeverity.INFO,
                        message="metadata points at bundled custom code; the reviewed built-in Transformers implementation is used instead",
                        evidence={"model_type": model_type},
                    )
                )
            else:
                review_reason = reviewed_reason(fingerprint.value, directory.name)
                if review_reason is not None:
                    trust_decision = TrustDecision.REVIEWED_BUNDLED_CODE
                    diagnostics.append(
                        Diagnostic(
                            code="bundled_custom_code_reviewed",
                            severity=DiagnosticSeverity.INFO,
                            message="the built-in Transformers implementation cannot read this checkpoint; a fingerprint-pinned review approved the bundled code instead",
                            evidence={
                                "model_type": model_type,
                                "builtin_reason": builtin_error,
                                "review": review_reason,
                            },
                        )
                    )
                    imports = check_bundled_imports(directory, auto_map)
                    if not imports.ok:
                        declared = config.get("transformers_version")
                        written_for = (
                            f" (it was written for Transformers {declared})" if declared else ""
                        )
                        diagnostics.append(
                            Diagnostic(
                                code="bundled_code_incompatible",
                                severity=DiagnosticSeverity.ERROR,
                                message=(
                                    "the reviewed bundled code cannot be imported with the "
                                    f"installed packages: {imports.describe()}"
                                ),
                                hint=(
                                    "The workbench does not patch checkpoint code or install "
                                    f"packages for it{written_for}. Use a SafeTensors export of "
                                    "this model for an architecture the installed Transformers "
                                    "implements, or review an updated revision of the bundled "
                                    "code pinned to its new fingerprint."
                                    + _size_note(fingerprint.total_weight_bytes)
                                ),
                                evidence={
                                    "missing_files": imports.missing_files,
                                    "missing_modules": imports.missing_modules,
                                    "missing_names": imports.missing_names,
                                },
                            )
                        )
                else:
                    trust_decision = TrustDecision.REJECTED_CUSTOM_CODE
                    diagnostics.append(
                        Diagnostic(
                            code="custom_code_not_trusted",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"model metadata requires custom Python code that has not been reviewed: {builtin_error}",
                            hint="Review it and enable a narrow isolated adapter for this exact fingerprint.",
                            evidence={"model_type": model_type, "builtin_reason": builtin_error},
                        )
                    )
        elif components.custom_code:
            diagnostics.append(
                Diagnostic(
                    code="bundled_scripts_ignored",
                    severity=DiagnosticSeverity.INFO,
                    message="bundled Python helper scripts were found and will not be imported automatically",
                )
            )

        context_values, context_limit, context_diagnostics = _context_values(
            config,
            tokenizer,
            self._conservative_context_limit,
            _sentence_transformer_max_seq_length(directory),
        )
        diagnostics.extend(context_diagnostics)
        diagnostics.extend(_metadata_diagnostics(config, tokenizer, generation))

        capabilities = build_capability_matrix(
            ModelEvidence(
                task=task,
                modalities=modalities,
                has_tokenizer=components.tokenizer,
                has_processor=components.processor,
                has_reasoning_delimiters=delimiters is not None,
                is_moe=is_moe,
                architecture_known=architecture_known,
                available_backends=self._available_backends,
                weight_quantization=quantized["method"] if quantized else None,
                quantization_backend=self._quantization_backend,
                blocking_diagnostics=tuple(
                    dict.fromkeys(
                        item.code
                        for item in diagnostics
                        if item.severity is DiagnosticSeverity.ERROR
                    )
                ),
            )
        )
        # Report what the headers store, not what config.json claims: the two disagree
        # in real checkpoints, and pickle-only folders have no verifiable dtype at all.
        declared_dtype = _declared_dtype(config)
        # Packed bitsandbytes tensors (U8/I8) do not describe the compute dtype.
        dtype = _stored_dtype(
            {
                name: count
                for name, count in summary.dtype_parameter_counts.items()
                if not (bitsandbytes and name in {"U8", "I8"})
            }
        )
        if declared_dtype and dtype and declared_dtype != dtype:
            diagnostics.append(
                Diagnostic(
                    code="dtype_metadata_mismatch",
                    severity=DiagnosticSeverity.INFO,
                    message=f"config.json declares {declared_dtype} but the SafeTensors headers store {dtype}",
                    evidence={"declared": declared_dtype, "stored": dtype},
                )
            )
        qwen_vl_embedding = (
            task is ModelTask.MULTIMODAL_EMBEDDING and config.get("model_type") == "qwen3_vl"
        )
        metadata: dict[str, Any] = {
            "task_evidence": task_evidence,
            "declared_dtype": declared_dtype,
            "transformers_version": config.get("transformers_version"),
            "vocab_size": config.get("vocab_size")
            or _nested_value(config, "text_config", "vocab_size"),
            "pooling": _sentence_transformer_pooling(directory),
            "normalization": (directory / "2_Normalize").is_dir()
            or _sentence_transformer_normalizes(directory),
            "embedding_dimension": _sentence_transformer_embedding_dimension(directory),
            "supports_dimension_truncation": qwen_vl_embedding,
            "minimum_embedding_dimension": 64 if qwen_vl_embedding else None,
            "joint_embedding_space": qwen_vl_embedding,
            "weight_quantization": quantized,
        }
        if bitsandbytes:
            metadata["parameter_count_note"] = "packed quantized tensors"
        return ModelDescriptor(
            id=_safe_identifier(directory.name, fingerprint.value),
            display_name=directory.name,
            path=directory,
            fingerprint=fingerprint,
            task=task,
            architectures=architectures,
            model_type=model_type,
            dtype=dtype,
            # Packed quantized tensors hold several parameters per stored element.
            parameter_count=None if bitsandbytes else summary.parameter_count or None,
            weight_dtypes=summary.dtype_parameter_counts,
            modalities=modalities,
            components=components,
            capabilities=capabilities,
            context_values=context_values,
            effective_context_limit=context_limit,
            reasoning_delimiters=delimiters,
            is_moe=is_moe,
            trust_decision=trust_decision,
            diagnostics=tuple(diagnostics),
            metadata=metadata,
            root_index=root_index,
            derivation=_derivation(directory),
        )

    def scan(self) -> ModelScanReport:
        models: list[ModelDescriptor] = []
        root_results: list[RootScanResult] = []
        seen_paths: set[Path] = set()
        for root_index, configured_root in enumerate(self._roots):
            root_diagnostics: list[Diagnostic] = []
            try:
                root = configured_root.resolve(strict=True)
            except OSError as exc:
                root_results.append(
                    RootScanResult(
                        root=configured_root,
                        discovered_count=0,
                        diagnostics=(
                            Diagnostic(
                                code="model_root_unavailable",
                                severity=DiagnosticSeverity.WARNING,
                                message=f"model root {configured_root.name!r} is unavailable",
                                hint="Create it or update paths.model_roots in the local configuration.",
                                evidence={
                                    "root": configured_root.name,
                                    "reason": type(exc).__name__,
                                },
                            ),
                        ),
                    )
                )
                continue
            if not root.is_dir():
                root_results.append(
                    RootScanResult(
                        root=root,
                        discovered_count=0,
                        diagnostics=(
                            Diagnostic(
                                code="model_root_not_directory",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"configured model root {root.name!r} is not a directory",
                            ),
                        ),
                    )
                )
                continue
            candidates, scan_diagnostics = self._candidate_directories(root)
            root_diagnostics.extend(scan_diagnostics)
            before = len(models)
            for directory in candidates:
                resolved = directory.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                try:
                    models.append(self._describe(resolved, root_index))
                except (OSError, ValueError) as exc:
                    root_diagnostics.append(
                        Diagnostic(
                            code="model_discovery_failed",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"model directory {directory.name!r} could not be inspected",
                            evidence={"directory": directory.name, "reason": type(exc).__name__},
                        )
                    )
            root_results.append(
                RootScanResult(
                    root=root,
                    discovered_count=len(models) - before,
                    diagnostics=tuple(root_diagnostics),
                )
            )

        models.sort(key=lambda item: (item.display_name.casefold(), item.id))
        return ModelScanReport(models=tuple(models), roots=tuple(root_results))
