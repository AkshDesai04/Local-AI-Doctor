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
from ..hardware.models import BackendKind
from .capabilities import ModelEvidence, build_capability_matrix
from .fingerprint import FingerprintPolicy, fingerprint_model_directory
from .safetensors import SafeTensorSummary, inspect_safetensors

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


def _context_values(
    config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    conservative_default: int,
) -> tuple[tuple[ContextValue, ...], int, list[Diagnostic]]:
    candidates: list[ContextValue] = []
    sources = (
        ("config.max_position_embeddings", config.get("max_position_embeddings"), None),
        (
            "config.text_config.max_position_embeddings",
            _nested_value(config, "text_config", "max_position_embeddings"),
            None,
        ),
        (
            "config.sliding_window",
            config.get("sliding_window"),
            "declared but disabled" if config.get("use_sliding_window") is False else None,
        ),
        (
            "config.rope_scaling.original_max_position_embeddings",
            _nested_value(config, "rope_scaling", "original_max_position_embeddings"),
            None,
        ),
        ("tokenizer_config.model_max_length", tokenizer.get("model_max_length"), None),
    )
    diagnostics: list[Diagnostic] = []
    for source, raw_value, note in sources:
        value = _positive_int(raw_value)
        if value is None:
            continue
        if value > 10_000_000:
            diagnostics.append(
                Diagnostic(
                    code="context_sentinel_ignored",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"{source} looks like an unbounded tokenizer sentinel and was not selected",
                    evidence={"source": source, "value": value},
                )
            )
        candidates.append(ContextValue(source=source, value=value, note=note))
    candidates.append(
        ContextValue(
            source="application.conservative_context_limit",
            value=conservative_default,
            note="portable default until a larger limit is validated on this backend",
        )
    )
    plausible = [item.value for item in candidates if item.value <= 10_000_000]
    selected = min(plausible)
    discovered = {item.value for item in candidates if not item.source.startswith("application.")}
    if len(discovered) > 1:
        diagnostics.append(
            Diagnostic(
                code="conflicting_context_metadata",
                severity=DiagnosticSeverity.WARNING,
                message="model files declare conflicting context limits; the conservative minimum is selected",
                evidence={"discovered_values": sorted(discovered), "selected": selected},
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
        or _regular_file(directory / "chat_template.jinja"),
        generation_config=_regular_file(directory / "generation_config.json"),
        processor=any(_regular_file(directory / name) for name in processor_files),
        pooling=_regular_file(directory / "modules.json")
        or _regular_file(directory / "1_Pooling" / "config.json"),
        custom_code=custom_code,
    )


def _reasoning_delimiters(tokenizer: Mapping[str, Any], directory: Path) -> tuple[str, str] | None:
    text = json.dumps(tokenizer, ensure_ascii=False)
    template_path = directory / "chat_template.jinja"
    if _regular_file(template_path) and template_path.stat().st_size <= _MAX_JSON_BYTES:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            text += template_path.read_text(encoding="utf-8")
    if "<think>" in text and "</think>" in text:
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
            if depth >= self._max_depth:
                return
            try:
                children = sorted(
                    (
                        item
                        for item in directory.iterdir()
                        if item.is_dir() and not item.is_symlink()
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

    def _describe(self, directory: Path) -> ModelDescriptor:
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
        if not weights:
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

        auto_map = config.get("auto_map")
        trust_decision = TrustDecision.BUILTIN_ONLY
        if isinstance(auto_map, Mapping) and auto_map:
            trust_decision = TrustDecision.REJECTED_CUSTOM_CODE
            diagnostics.append(
                Diagnostic(
                    code="custom_code_not_trusted",
                    severity=DiagnosticSeverity.ERROR,
                    message="model metadata requires custom Python code that has not been reviewed",
                    hint="Review it and enable a narrow isolated adapter for this exact fingerprint.",
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
            config, tokenizer, self._conservative_context_limit
        )
        diagnostics.extend(context_diagnostics)
        diagnostics.extend(_metadata_diagnostics(config, tokenizer, generation))

        fingerprint = fingerprint_model_directory(directory, self._fingerprint_policy)
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
            )
        )
        dtype = (
            config.get("dtype")
            or config.get("torch_dtype")
            or _nested_value(config, "text_config", "dtype")
        )
        metadata: dict[str, Any] = {
            "task_evidence": task_evidence,
            "transformers_version": config.get("transformers_version"),
            "vocab_size": config.get("vocab_size")
            or _nested_value(config, "text_config", "vocab_size"),
            "pooling": "sentence-transformers" if components.pooling else None,
            "normalization": (directory / "2_Normalize").is_dir()
            or _sentence_transformer_normalizes(directory),
        }
        return ModelDescriptor(
            id=_safe_identifier(directory.name, fingerprint.value),
            display_name=directory.name,
            path=directory,
            fingerprint=fingerprint,
            task=task,
            architectures=architectures,
            model_type=str(config.get("model_type")) if config.get("model_type") else None,
            dtype=str(dtype) if dtype else None,
            parameter_count=summary.parameter_count or None,
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
        )

    def scan(self) -> ModelScanReport:
        models: list[ModelDescriptor] = []
        root_results: list[RootScanResult] = []
        seen_paths: set[Path] = set()
        for configured_root in self._roots:
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
                    models.append(self._describe(resolved))
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
