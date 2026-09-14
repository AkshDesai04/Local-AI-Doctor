"""Model runtime executed only inside an isolated spawned worker process."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import signal
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from ..reasoning import (
    SegmentClass,
    SegmentedToken,
    TagReasoningSegmenter,
    UnknownReasoningSegmenter,
)

# This must be set before the worker imports torch or initializes CUDA.  It is
# required by cuBLAS when deterministic/reference mode enables deterministic
# algorithms on CUDA 10.2 and newer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# Persisting every attention weight for every generated token grows as
# O(generated_tokens * context_tokens).  The complete prompt-token catalogue is
# emitted once, while each generated token retains the highest-weight source
# positions and the exact amount of attention mass omitted by this bound.
_ATTENTION_SOURCE_LIMIT = 128


def _send(output: Queue[Any], kind: str, **values: Any) -> None:
    output.put({"kind": kind, **values})


def _torch_dtype(torch: Any, name: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name)


def _escaped_bytes(value: str) -> str:
    return "".join(f"\\x{byte:02x}" for byte in value.encode("utf-8", errors="surrogatepass"))


def _token_source(
    tokenizer: Any,
    token_id: int,
    context_index: int,
    *,
    source_kind: str,
    generated_token_index: int | None = None,
    display_text: str | None = None,
) -> dict[str, Any]:
    """Build a serializable context-position descriptor without guessing roles.

    Arbitrary tokenizer chat templates can rewrite and inject text, so a token
    in the rendered prompt cannot in general be mapped back to one message
    without lying.  Absolute context positions remain exact and cover system,
    history, current-prompt, and template/control tokens alike.
    """

    piece = str(tokenizer.convert_ids_to_tokens(token_id))
    if display_text is None:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        display_text = str(decoded)
    source = {
        "context_index": context_index,
        "token_id": token_id,
        "piece": piece,
        "display_text": display_text,
        "source_kind": source_kind,
    }
    if generated_token_index is not None:
        source["generated_token_index"] = generated_token_index
    return source


def _mean_causal_self_attention(
    torch: Any,
    attentions: Any,
    context_tokens: Sequence[Mapping[str, Any]],
    *,
    source_limit: int = _ATTENTION_SOURCE_LIMIT,
) -> tuple[dict[str, Any] | None, str | None]:
    """Aggregate the model's exact attention rows for one next-token decision.

    Each available layer/head contributes its post-softmax row at the final
    query position.  Their arithmetic mean is normalized for floating-point
    drift.  These values are model-internal attention allocations, *not* causal
    effects or probabilities that a source token caused the sampled token.
    """

    if not isinstance(attentions, (list, tuple)) or not attentions:
        return None, "the model did not return causal self-attention tensors"
    if not context_tokens:
        return None, "the attention query has no source-token catalogue"

    context_count = len(context_tokens)
    accumulated: Any | None = None
    captured_layers: list[int] = []
    heads_per_layer: list[int] = []
    for layer_index, layer_attention in enumerate(attentions):
        if layer_attention is None:
            continue
        shape = [int(dimension) for dimension in getattr(layer_attention, "shape", ())]
        if len(shape) != 4 or shape[0] < 1 or shape[1] < 1 or shape[2] < 1:
            continue
        key_count = shape[3]
        if key_count < 1 or key_count > context_count:
            return (
                None,
                "a returned attention tensor could not be aligned to the causal context",
            )
        # Sliding-window layers may expose only a suffix of the full context.
        # Left-padding with exact zeros aligns those keys to absolute positions.
        rows = layer_attention[0, :, -1, :].float()
        layer_sum = rows.sum(dim=0)
        if key_count < context_count:
            layer_sum = torch.cat(
                [
                    torch.zeros(
                        context_count - key_count,
                        device=layer_sum.device,
                        dtype=layer_sum.dtype,
                    ),
                    layer_sum,
                ],
                dim=0,
            )
        accumulated = layer_sum if accumulated is None else accumulated + layer_sum
        captured_layers.append(layer_index)
        heads_per_layer.append(shape[1])

    total_head_rows = sum(heads_per_layer)
    if accumulated is None or total_head_rows == 0:
        return None, "the model returned no usable causal self-attention rows"
    weights = accumulated / total_head_rows
    # Attention is non-negative by construction. Clamp only minute numerical
    # underflow before normalizing the arithmetic mean back to unit mass.
    weights = torch.clamp(weights, min=0.0)
    total = float(weights.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        return None, "the model returned non-finite or empty attention weights"
    weights = weights / total
    host_weights = [float(value) for value in weights.detach().cpu().tolist()]
    retained_indices = sorted(
        sorted(range(context_count), key=lambda index: (-host_weights[index], index))[
            : min(source_limit, context_count)
        ]
    )
    retained_weight = min(
        1.0,
        max(0.0, sum(host_weights[index] for index in retained_indices)),
    )
    sources = [
        {**dict(context_tokens[index]), "weight": host_weights[index]} for index in retained_indices
    ]
    return (
        {
            "method": "mean_causal_self_attention",
            "aggregation": "arithmetic_mean_over_layers_and_heads",
            "semantics": "attention_weights_not_causal_contributions",
            "scope": "decoder_step_context_attention_independent_of_sampled_candidate",
            "query": "final_sequence_position_predicting_selected_token",
            "attention_implementation": "eager",
            "normalized": True,
            "captured_layers": captured_layers,
            "captured_heads": (
                heads_per_layer[0] if len(set(heads_per_layer)) == 1 else max(heads_per_layer)
            ),
            "total_head_rows": total_head_rows,
            "heads_per_layer": heads_per_layer,
            "total_source_count": context_count,
            "retained_source_count": len(sources),
            "retained_weight": retained_weight,
            "omitted_weight": max(0.0, 1.0 - retained_weight),
            "source_limit": source_limit,
            "source_tokens": sources,
        },
        None,
    )


def _safe_error(exc: BaseException) -> dict[str, Any]:
    # Exception messages from model libraries routinely contain absolute model
    # paths, prompt fragments, and environment details. They remain available in
    # the worker-only diagnostic traceback, but never cross the public event/API
    # boundary.
    lowered = str(exc).lower()
    if "decoder_start_token_id" in lowered:
        code = "missing_decoder_start_token_id"
        message = "encoder-decoder model configuration is missing a usable decoder start token"
        hint = (
            "Set decoder_start_token_id (or a valid BOS fallback) in the model's "
            "bundled generation/config metadata."
        )
    elif "out of memory" in lowered:
        code = "model_out_of_memory"
        message = "model worker exhausted available memory"
        hint = "Unload the current model, reduce context or batch size, enable CPU offload, or select CPU."
    elif "cuda" in lowered:
        code = "cuda_runtime_error"
        message = "model worker encountered a CUDA runtime failure"
        hint = "Verify the selected CUDA runtime and retry with CPU fallback if configured."
    else:
        code = "model_worker_error"
        message = "model worker operation failed"
        hint = "Review the model diagnostics and worker runtime dependencies."
    return {"code": code, "message": message, "hint": hint, "exception": type(exc).__name__}


def _render_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    reasoning: bool | None = None,
    reasoning_delimiters: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Render chat messages, with a deterministic plain-text fallback.

    Many encoder-decoder checkpoints have no chat template at all.  A single
    user turn is kept byte-for-byte so translation, summarization, and similar
    instruction models receive the source text they expect.  Multi-turn input
    uses explicit role labels and is reported as a fallback in telemetry.
    """

    if not messages:
        raise ValueError("generation requires at least one message")
    template = getattr(tokenizer, "chat_template", None)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if template and callable(apply_template):
        template_options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if reasoning is not None:
            # Transformers forwards extra keyword arguments to the tokenizer's
            # Jinja template. Thinking-capable templates (for example Qwen3)
            # consume ``enable_thinking``; ordinary templates safely ignore it.
            template_options["enable_thinking"] = reasoning
        rendered = apply_template(list(messages), **template_options)
        if not isinstance(rendered, str):
            raise ValueError("tokenizer chat template did not render text")
        if reasoning is False and reasoning_delimiters is not None:
            opening, closing = reasoning_delimiters
            trimmed = rendered.rstrip()
            if trimmed.endswith(opening):
                history_only: Any = None
                # The preference remains best-effort for unusual custom
                # templates; never rewrite a suffix without proving that it
                # came from the generation prompt.
                with contextlib.suppress(TypeError, ValueError):
                    history_only = apply_template(
                        list(messages),
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                if isinstance(history_only, str):
                    history = history_only.rstrip()
                    prompt_suffix = (
                        trimmed[len(history) :].strip() if trimmed.startswith(history) else ""
                    )
                    # Some reasoning checkpoints hard-code an opening marker
                    # and ignore ``enable_thinking``. Only close it when the
                    # history-only rendering proves the marker was injected by
                    # add_generation_prompt, rather than typed by the user.
                    if prompt_suffix.endswith(opening):
                        rendered = f"{trimmed}\n\n{closing}\n\n"
        return rendered, "chat_template"

    normalized: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        normalized.append((role, content))
    if len(normalized) == 1 and normalized[0][0] == "user":
        return normalized[0][1], "plain_text_fallback"
    labels = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
    lines = [
        f"{labels.get(role, role.title() or 'User')}: {content}" for role, content in normalized
    ]
    if normalized[-1][0] != "assistant":
        lines.append("Assistant:")
    return "\n".join(lines), "plain_text_fallback"


def _valid_token_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _decoder_start_token(model: Any, tokenizer: Any) -> tuple[int, str, str | None]:
    """Resolve a seq2seq decoder start token using Transformers' safe order."""

    generation_config = getattr(model, "generation_config", None)
    model_config = getattr(model, "config", None)
    explicit_candidates = (
        (
            "generation_config.decoder_start_token_id",
            getattr(generation_config, "decoder_start_token_id", None),
        ),
        ("config.decoder_start_token_id", getattr(model_config, "decoder_start_token_id", None)),
    )
    for source, value in explicit_candidates:
        token_id = _valid_token_id(value)
        if token_id is not None:
            return token_id, source, None

    fallback_candidates = (
        ("generation_config.bos_token_id", getattr(generation_config, "bos_token_id", None)),
        ("config.bos_token_id", getattr(model_config, "bos_token_id", None)),
        ("tokenizer.bos_token_id", getattr(tokenizer, "bos_token_id", None)),
    )
    for source, value in fallback_candidates:
        token_id = _valid_token_id(value)
        if token_id is not None:
            return (
                token_id,
                source,
                "decoder_start_token_id is absent; using the bundled BOS token as the decoder start",
            )
    raise ValueError(
        "encoder-decoder model is missing decoder_start_token_id and no valid BOS token fallback exists"
    )


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    values = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    )
    result: set[int] = set()
    for value in values:
        candidates = value if isinstance(value, (list, tuple, set)) else (value,)
        for candidate in candidates:
            token_id = _valid_token_id(candidate)
            if token_id is not None:
                result.add(token_id)
    return result


def _embedding_model_kwargs(model_path: Path, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return loader kwargs required by reviewed built-in embedding architectures.

    Qwen3-VL embedding checkpoints are saved from a task wrapper whose tensors are
    named ``model.*``. SentenceTransformers 5.4 correctly chooses the built-in
    ``Qwen3VLModel`` for feature extraction, but that bare model expects the same
    tensors without the wrapper prefix. Transformers supports an explicit regex
    key mapping for this exact checkpoint-compatibility case. Applying the mapping
    avoids silently accepting a fully randomly initialized model and does not
    import or execute the helper Python bundled with the checkpoint.
    """

    result = dict(kwargs)
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return result
    if isinstance(config, dict) and config.get("model_type") == "qwen3_vl":
        result["key_mapping"] = {r"^model\.": ""}
    return result


def _embedding_dimension_bounds(
    model_type: str | None, native_width: int
) -> tuple[int, int] | None:
    """Return reviewed Matryoshka bounds without generalizing across models."""

    return (64, native_width) if model_type == "qwen3_vl" else None


def _reasoning_segmenter(
    model_info: Mapping[str, Any], rendered_prompt: str
) -> tuple[TagReasoningSegmenter | UnknownReasoningSegmenter, bool]:
    """Build the model-declared reasoning segmenter for one generated stream."""

    configured = model_info.get("reasoning_delimiters")
    if not isinstance(configured, (list, tuple)) or len(configured) != 2:
        return UnknownReasoningSegmenter(), False
    opening_tag, closing_tag = configured
    if not isinstance(opening_tag, str) or not isinstance(closing_tag, str):
        return UnknownReasoningSegmenter(), False
    if not opening_tag or not closing_tag or opening_tag == closing_tag:
        return UnknownReasoningSegmenter(), False
    reasoning_primed = rendered_prompt.endswith(f"{opening_tag}\n")
    return (
        TagReasoningSegmenter(
            opening_tag,
            closing_tag,
            initially_inside=reasoning_primed,
        ),
        reasoning_primed,
    )


def _contains_visible_answer(token: SegmentedToken) -> bool:
    """Return whether an exactly segmented token contains answer text.

    A closing reasoning delimiter can share a decoded token with the first
    answer characters. Inspecting the character slices avoids treating the
    delimiter itself (or trailing whitespace after it) as an answer.
    """

    return any(
        item.classification is SegmentClass.ANSWER
        and not item.delimiter
        and bool(token.text[item.start : item.end].strip())
        for item in token.slices
    )


class WorkerRuntime:
    def __init__(self, commands: Queue[Any], output: Queue[Any], cancel_event: Any) -> None:
        self.commands = commands
        self.output = output
        self.cancel_event = cancel_event
        self.model: Any = None
        self.tokenizer: Any = None
        self.sentence_model: Any = None
        self.model_info: dict[str, Any] | None = None
        self.device = "cpu"
        self.dtype = "float32"
        self.decoder_start_token_id: int | None = None
        self.decoder_start_token_source: str | None = None
        self.decoder_start_warning: str | None = None
        self.loaded_attention_implementation: str | None = None

    def run(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        _send(self.output, "ready")
        while True:
            command = self.commands.get()
            operation = command.get("op")
            request_id = str(command.get("request_id", ""))
            try:
                if operation == "shutdown":
                    self._unload()
                    _send(self.output, "reply", request_id=request_id, ok=True, payload={})
                    return
                if operation == "load":
                    payload = self._load(command)
                    _send(self.output, "reply", request_id=request_id, ok=True, payload=payload)
                elif operation == "unload":
                    payload = self._unload()
                    _send(self.output, "reply", request_id=request_id, ok=True, payload=payload)
                elif operation == "generate":
                    self._generate(command)
                elif operation == "embed":
                    payload = self._embed(command)
                    _send(self.output, "reply", request_id=request_id, ok=True, payload=payload)
                elif operation == "score_prompt":
                    payload = self._score_prompt(command)
                    _send(self.output, "reply", request_id=request_id, ok=True, payload=payload)
                else:
                    raise ValueError(f"unknown worker operation: {operation!r}")
            except BaseException as exc:  # worker boundary must report model/runtime crashes
                # A full-instrumentation run may temporarily select eager
                # attention so Transformers can return attention probabilities.
                # Never let a failed request silently change later run kernels.
                self._restore_attention_implementation()
                error = _safe_error(exc)
                if command.get("run_id"):
                    _send(
                        self.output,
                        "run_event",
                        run_id=command["run_id"],
                        event_type="error",
                        payload=error,
                    )
                _send(self.output, "reply", request_id=request_id, ok=False, error=error)
                # The traceback stays inside the isolated process and is bounded.
                _send(
                    self.output,
                    "diagnostic",
                    request_id=request_id,
                    payload={"traceback": "".join(traceback.format_exception(exc))[-8000:]},
                )

    def _load(self, command: Mapping[str, Any]) -> dict[str, Any]:
        self._unload()
        import torch

        model = dict(command["model"])
        runtime = dict(command.get("runtime", {}))
        model_path = Path(model["path"])
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"configured model directory {model_path.name!r} is unavailable"
            )
        self.device = str(runtime.get("device", "cpu"))
        self.dtype = str(runtime.get("dtype", "float32"))
        torch.set_num_threads(int(runtime.get("cpu_threads", max(1, os.cpu_count() or 1))))
        dtype = _torch_dtype(torch, self.dtype)
        attention = runtime.get("attention_backend")
        attention = (
            None
            if attention in {None, "auto"}
            else attention.replace("flash-attention-2", "flash_attention_2")
        )
        common: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        model_kwargs = dict(common)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attention is not None:
            model_kwargs["attn_implementation"] = attention
        task = str(model["task"])
        started = time.monotonic()
        if task in {"text_generation", "encoder_decoder_generation"}:
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer_loader: Any = AutoTokenizer
            self.tokenizer = tokenizer_loader.from_pretrained(model_path, **common)
            loader: Any = (
                AutoModelForSeq2SeqLM
                if task == "encoder_decoder_generation"
                else AutoModelForCausalLM
            )
            self.model = loader.from_pretrained(
                model_path,
                low_cpu_mem_usage=bool(runtime.get("low_memory_loading", True)),
                **model_kwargs,
            )
            self.model.eval()
            self.model.to(self.device)
            configured_attention = getattr(self.model.config, "_attn_implementation", None)
            self.loaded_attention_implementation = (
                str(configured_attention) if configured_attention is not None else None
            )
            if task == "encoder_decoder_generation":
                try:
                    (
                        self.decoder_start_token_id,
                        self.decoder_start_token_source,
                        self.decoder_start_warning,
                    ) = _decoder_start_token(self.model, self.tokenizer)
                except ValueError:
                    self._unload()
                    raise
        elif task in {"embedding", "multimodal_embedding"}:
            from sentence_transformers import SentenceTransformer

            sentence_kwargs: dict[str, Any] = {
                "device": self.device,
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if model_kwargs:
                embedding_model_kwargs = {
                    key: value
                    for key, value in model_kwargs.items()
                    if key not in {"local_files_only", "trust_remote_code"}
                }
                sentence_kwargs["model_kwargs"] = _embedding_model_kwargs(
                    model_path, embedding_model_kwargs
                )
            self.sentence_model = SentenceTransformer(str(model_path), **sentence_kwargs)
        else:
            raise ValueError(f"no runtime adapter supports task {task!r}")

        self.model_info = model
        memory = self._memory_snapshot(torch)
        return {
            "model_id": model["id"],
            "task": task,
            "device": self.device,
            "dtype": self.dtype,
            "load_seconds": time.monotonic() - started,
            "memory": memory,
            "trust_remote_code": False,
            "local_files_only": True,
            "decoder_start_token_id": self.decoder_start_token_id,
            "decoder_start_token_source": self.decoder_start_token_source,
            "warnings": [self.decoder_start_warning] if self.decoder_start_warning else [],
        }

    def _unload(self) -> dict[str, Any]:
        unloaded = self.model_info["id"] if self.model_info else None
        memory_before: dict[str, Any] = {}
        torch_module: Any | None = None
        if self.model is not None or self.sentence_model is not None or self.model_info is not None:
            try:
                import torch

                torch_module = torch
                memory_before = self._memory_snapshot(torch)
            except ImportError:
                pass
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        self.model_info = None
        self.decoder_start_token_id = None
        self.decoder_start_token_source = None
        self.decoder_start_warning = None
        self.loaded_attention_implementation = None
        gc.collect()
        try:
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
                torch_module.cuda.ipc_collect()
        except RuntimeError:
            pass
        memory_after = self._memory_snapshot(torch_module) if torch_module is not None else {}
        return {
            "unloaded_model_id": unloaded,
            "memory_before": memory_before,
            "memory_after": memory_after,
        }

    @staticmethod
    def _memory_snapshot(torch: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            psutil: Any = __import__("psutil")

            process = psutil.Process()
            result["process_rss_bytes"] = int(process.memory_info().rss)
            result["system_available_bytes"] = int(psutil.virtual_memory().available)
        except (ImportError, OSError):
            pass
        if torch.cuda.is_available():
            result.update(
                {
                    "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
        return result

    def _emit_run(self, run_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        _send(
            self.output,
            "run_event",
            run_id=run_id,
            event_type=event_type,
            payload=dict(payload),
        )

    def _select_attention_implementation(self, implementation: str) -> bool:
        """Best-effort dynamic attention-kernel selection for instrumentation."""

        if self.model is None:
            return False
        current = getattr(getattr(self.model, "config", None), "_attn_implementation", None)
        if current == implementation:
            return True
        setter = getattr(self.model, "set_attn_implementation", None)
        if not callable(setter):
            return False
        with contextlib.suppress(Exception):
            setter(implementation)
        return (
            getattr(getattr(self.model, "config", None), "_attn_implementation", None)
            == implementation
        )

    def _restore_attention_implementation(self) -> None:
        if self.loaded_attention_implementation is not None:
            self._select_attention_implementation(self.loaded_attention_implementation)

    @staticmethod
    def _apply_penalties(
        torch: Any,
        raw: Any,
        token_ids: Sequence[int],
        repetition_penalty: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> Any:
        processed = raw.clone()
        if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
            return processed
        counts = Counter(int(token_id) for token_id in token_ids)
        if counts:
            ids = torch.tensor(tuple(counts), device=processed.device, dtype=torch.long)
            values = processed[ids]
            if repetition_penalty != 1.0:
                processed[ids] = torch.where(
                    values < 0,
                    values * repetition_penalty,
                    values / repetition_penalty,
                )
            if frequency_penalty != 0.0:
                frequencies = torch.tensor(
                    tuple(counts[token_id] for token_id in counts),
                    device=processed.device,
                    dtype=processed.dtype,
                )
                processed[ids] -= frequency_penalty * frequencies
            if presence_penalty != 0.0:
                processed[ids] -= presence_penalty
        return processed

    @staticmethod
    def _filter_distribution(
        torch: Any,
        logits: Any,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> tuple[Any, list[str]]:
        operations: list[str] = []
        filtered = logits.float().clone()
        if temperature > 0:
            filtered /= temperature
            operations.append("temperature")
        if top_k > 0 and top_k < filtered.numel():
            threshold = torch.topk(filtered, top_k).values[-1]
            filtered[filtered < threshold] = -torch.inf
            operations.append("top_k")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, stable=True)
            probabilities = torch.softmax(sorted_logits, dim=-1)
            remove = torch.cumsum(probabilities, dim=-1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = -torch.inf
            operations.append("top_p")
        if min_p > 0.0:
            probabilities = torch.softmax(filtered, dim=-1)
            cutoff = probabilities.max() * min_p
            filtered[probabilities < cutoff] = -torch.inf
            operations.append("min_p")
        return filtered, operations

    def _forward_last(
        self,
        torch: Any,
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        *,
        capture_attention: bool = False,
    ) -> tuple[Any, Any, Any]:
        captured_attentions: list[Any | None] = []
        hook_handles: list[Any] = []
        if capture_attention:
            # Transformers 4.57 no longer propagates per-layer attentions into
            # BaseModelOutput for several decoder architectures (including
            # Qwen2), even though each eager self-attention module still
            # returns its exact post-softmax weights. Capture those module
            # outputs directly and remove every hook before returning.
            named_modules = getattr(self.model, "named_modules", None)
            modules = (
                [
                    module
                    for name, module in named_modules()
                    if name.rsplit(".", 1)[-1] == "self_attn"
                ]
                if callable(named_modules)
                else []
            )
            captured_attentions = [None] * len(modules)

            def capture_layer(layer_index: int) -> Any:
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    if isinstance(output, (list, tuple)) and len(output) > 1:
                        candidate = output[1]
                        if getattr(candidate, "shape", None) is not None:
                            captured_attentions[layer_index] = candidate

                return hook

            hook_handles = [
                module.register_forward_hook(capture_layer(layer_index))
                for layer_index, module in enumerate(modules)
            ]
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": capture_attention,
        }
        # Causal decoder models can avoid materializing prompt-length vocabulary
        # logits by applying the LM head only to the final hidden state.
        body = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        if (
            body is not None
            and head is not None
            and not getattr(self.model.config, "is_encoder_decoder", False)
        ):
            try:
                outputs = body(**kwargs)
                logits = head(outputs.last_hidden_state[:, -1:, :])
                hook_attentions = tuple(captured_attentions)
                attentions = (
                    hook_attentions
                    if any(item is not None for item in hook_attentions)
                    else getattr(outputs, "attentions", None)
                )
                return logits[:, -1, :], outputs.past_key_values, attentions
            finally:
                for handle in hook_handles:
                    handle.remove()
        try:
            try:
                outputs = self.model(**kwargs, logits_to_keep=1)
            except TypeError:
                captured_attentions[:] = [None] * len(captured_attentions)
                outputs = self.model(**kwargs)
            model_attentions = getattr(outputs, "attentions", None)
            hook_attentions = tuple(captured_attentions)
            attentions = (
                hook_attentions
                if any(item is not None for item in hook_attentions)
                else model_attentions
            )
            return outputs.logits[:, -1, :], outputs.past_key_values, attentions
        finally:
            for handle in hook_handles:
                handle.remove()

    def _encode_source(self, input_ids: Any, attention_mask: Any) -> Any:
        """Run an encoder exactly once for a cached encoder-decoder generation."""

        get_encoder = getattr(self.model, "get_encoder", None)
        if not callable(get_encoder):
            raise ValueError("encoder-decoder model does not expose get_encoder()")
        encoder = get_encoder()
        return encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def _forward_encoder_decoder(
        self,
        decoder_input_ids: Any,
        decoder_attention_mask: Any,
        encoder_outputs: Any,
        encoder_attention_mask: Any,
        past_key_values: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Run one decoder step while reusing encoder outputs and decoder cache."""

        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values, None

    def _generate(self, command: Mapping[str, Any]) -> None:
        try:
            self._generate_impl(command)
        finally:
            self._restore_attention_implementation()

    def _generate_impl(self, command: Mapping[str, Any]) -> None:
        if self.model is None or self.tokenizer is None or self.model_info is None:
            raise RuntimeError("no generation model is loaded")
        import torch

        run_id = str(command["run_id"])
        request_id = str(command["request_id"])
        settings = dict(command["sampling"])
        instrumentation = str(command.get("instrumentation", "token"))
        detailed_metrics = instrumentation in {"token", "full", "expert"}
        synchronize = instrumentation in {"full", "expert"} and self.device.startswith("cuda")
        encoder_decoder = self.model_info.get("task") == "encoder_decoder_generation"
        attention_capture_requested = instrumentation in {"full", "expert"}
        attention_capture_active = False
        attention_capture_warning_emitted = False
        captured_attention_token_count = 0
        if attention_capture_requested and not encoder_decoder:
            attention_capture_active = self._select_attention_implementation("eager")
            if not attention_capture_active:
                self._emit_run(
                    run_id,
                    "warning",
                    {
                        "code": "attention_capture_unavailable",
                        "message": (
                            "This model cannot switch to an eager attention implementation, "
                            "so exact attention weights are unavailable for this run."
                        ),
                    },
                )
                attention_capture_warning_emitted = True
        elif attention_capture_requested:
            self._emit_run(
                run_id,
                "warning",
                {
                    "code": "attention_capture_unavailable",
                    "message": (
                        "Encoder-decoder attention is split between decoder self-attention "
                        "and encoder cross-attention; this causal attribution view does not "
                        "combine those incomparable distributions."
                    ),
                },
            )
            attention_capture_warning_emitted = True
        else:
            self._restore_attention_implementation()
        deterministic_reference_mode = bool(command.get("deterministic_reference_mode"))
        # Set this on every request.  torch's switch is process-global, so merely
        # enabling it for reference runs would silently affect every later run.
        torch.use_deterministic_algorithms(
            deterministic_reference_mode,
            warn_only=deterministic_reference_mode,
        )
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            # The reference worker keeps autotuning off for both modes.  This is
            # stable across requests and is reported verbatim below.
            cudnn.benchmark = False
        messages = list(command["messages"])
        if (
            any(item.get("role") == "system" for item in messages)
            and "deepseek" in self.model_info["display_name"].lower()
        ):
            self._emit_run(
                run_id,
                "warning",
                {
                    "code": "model_card_discourages_system_prompt",
                    "message": "This model card recommends avoiding system prompts.",
                },
            )
        template_started = time.monotonic_ns()
        reasoning_requested = command.get("reasoning")
        if reasoning_requested is not None and not isinstance(reasoning_requested, bool):
            raise ValueError("reasoning must be a boolean when provided")
        rendered_prompt, prompt_renderer = _render_messages(
            self.tokenizer,
            messages,
            reasoning=reasoning_requested,
            reasoning_delimiters=self.model_info.get("reasoning_delimiters"),
        )
        template_ended = time.monotonic_ns()
        if prompt_renderer == "plain_text_fallback":
            self._emit_run(
                run_id,
                "warning",
                {
                    "code": "chat_template_unavailable",
                    "message": (
                        "The bundled tokenizer has no chat template; messages were rendered "
                        "with the deterministic plain-text fallback."
                    ),
                },
            )
        if encoder_decoder and self.decoder_start_token_id is None:
            (
                self.decoder_start_token_id,
                self.decoder_start_token_source,
                self.decoder_start_warning,
            ) = _decoder_start_token(self.model, self.tokenizer)
        if encoder_decoder and self.decoder_start_warning:
            self._emit_run(
                run_id,
                "warning",
                {
                    "code": "decoder_start_token_fallback",
                    "message": self.decoder_start_warning,
                    "source": self.decoder_start_token_source,
                },
            )
        tokenization_started = time.monotonic_ns()
        encoded = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        tokenization_ended = time.monotonic_ns()
        prompt_tokens = int(input_ids.shape[-1])
        context_limit = int(self.model_info.get("effective_context_limit") or 4096)
        configured_prompt_limit = int(command.get("max_prompt_tokens", context_limit))
        reserved_output_tokens = int(command.get("reserved_output_tokens", 0))
        effective_prompt_limit = min(
            configured_prompt_limit,
            context_limit if encoder_decoder else max(1, context_limit - reserved_output_tokens),
        )
        if prompt_tokens > effective_prompt_limit:
            raise ValueError(
                "rendered prompt exceeds the effective prompt-token limit "
                f"({prompt_tokens} > {effective_prompt_limit})"
            )
        forced_prefix_token_ids = [int(item) for item in command.get("forced_prefix_token_ids", ())]
        available_output_tokens = (
            max(0, context_limit - 1) if encoder_decoder else max(0, context_limit - prompt_tokens)
        )
        configured_max_output_tokens = int(settings["max_output_tokens"])
        if len(forced_prefix_token_ids) > configured_max_output_tokens:
            raise ValueError("forced generation prefix exceeds the configured output limit")
        if len(forced_prefix_token_ids) > available_output_tokens:
            raise ValueError("forced generation prefix exceeds the available model context")
        max_new_tokens = min(
            configured_max_output_tokens,
            available_output_tokens,
        )
        if max_new_tokens <= 0:
            raise ValueError("rendered prompt consumes the effective context limit")
        # Decode the prompt-token catalogue only after every bounded context
        # check succeeds. Rejected oversized prompts must not trigger an
        # unbounded device-to-CPU copy plus one tokenizer call per token.
        prompt_context_tokens = (
            [
                _token_source(
                    self.tokenizer,
                    int(token_id),
                    context_index,
                    source_kind="prompt",
                )
                for context_index, token_id in enumerate(input_ids[0].tolist())
            ]
            if attention_capture_active
            else []
        )

        configured_delimiters = self.model_info.get("reasoning_delimiters")
        tagged_reasoning_enabled = (
            reasoning_requested is not False
            and isinstance(configured_delimiters, (list, tuple))
            and len(configured_delimiters) == 2
            and all(isinstance(item, str) and item for item in configured_delimiters)
            and configured_delimiters[0] != configured_delimiters[1]
        )
        # ``max_output_tokens`` normally remains a hard limit. A tagged
        # reasoning model is the one exception: if emitted reasoning reaches
        # that boundary before any answer text, it receives one bounded
        # additional output window. Otherwise a valid run can end with only
        # reasoning (or just ``</think>``) and the non-Nerd UI has nothing to
        # display. The allowance is still clipped to the remaining context.
        reasoning_answer_allowance = (
            min(
                configured_max_output_tokens,
                max(0, available_output_tokens - max_new_tokens),
            )
            if tagged_reasoning_enabled
            else 0
        )
        generation_token_limit = max_new_tokens + reasoning_answer_allowance

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(command["effective_seed"]))
        generated: list[int] = []
        context_tokens: list[dict[str, Any]] = list(prompt_context_tokens)
        full_ids = (
            [self.decoder_start_token_id]
            if encoder_decoder and self.decoder_start_token_id is not None
            else input_ids[0].tolist()
        )
        previous_text = ""
        cumulative_logprob = 0.0
        raw_segment_logprobs: dict[str, list[float]] = {
            "reasoning": [],
            "answer": [],
            "unknown": [],
        }
        segment_token_counts = {"reasoning": 0, "answer": 0, "unknown": 0}
        reasoning_segmenter, reasoning_primed = _reasoning_segmenter(
            self.model_info, rendered_prompt
        )
        pending_token_events: dict[int, dict[str, Any]] = {}
        reasoning_observed = reasoning_primed
        visible_answer_observed = False

        def emit_segmented_tokens(tokens: Sequence[SegmentedToken]) -> None:
            nonlocal reasoning_observed, visible_answer_observed
            for token in tokens:
                payload = pending_token_events.pop(token.token_index)
                token_segment = token.classification.value
                payload["segment"] = token_segment
                payload["reasoning_slices"] = [
                    item.model_dump(mode="json") for item in token.slices
                ]
                segment_token_counts[token_segment] += 1
                raw_logprob = payload.get("raw_logprob")
                if isinstance(raw_logprob, float):
                    raw_segment_logprobs[token_segment].append(raw_logprob)
                if any(item.classification is SegmentClass.REASONING for item in token.slices):
                    reasoning_observed = True
                if payload["token_id"] not in eos_set and _contains_visible_answer(token):
                    visible_answer_observed = True
                self._emit_run(run_id, "token", payload)

        prefill_started = time.monotonic_ns()
        self._emit_run(
            run_id,
            "stage",
            {
                "stage": "prefill",
                "rendered_prompt": rendered_prompt,
                "prompt_tokens": prompt_tokens,
                "context_limit": context_limit,
                "effective_prompt_limit": effective_prompt_limit,
                "reserved_output_tokens": reserved_output_tokens,
                "prompt_renderer": prompt_renderer,
                "architecture_mode": "encoder_decoder" if encoder_decoder else "causal",
                "decoder_start_token_id": self.decoder_start_token_id if encoder_decoder else None,
                "decoder_start_token_source": self.decoder_start_token_source
                if encoder_decoder
                else None,
                "reasoning_primed": reasoning_primed,
                "reasoning_requested": reasoning_requested,
                "configured_max_output_tokens": configured_max_output_tokens,
                "generation_token_limit": generation_token_limit,
                "reasoning_answer_allowance": reasoning_answer_allowance,
                "attention_capture_requested": attention_capture_requested,
                "attention_capture_active": attention_capture_active,
                "attention_capture_method": (
                    "mean_causal_self_attention" if attention_capture_active else None
                ),
                "attention_source_limit": (
                    _ATTENTION_SOURCE_LIMIT if attention_capture_active else None
                ),
                "attention_implementation": (
                    getattr(self.model.config, "_attn_implementation", None)
                    if attention_capture_active
                    else self.loaded_attention_implementation
                ),
                "template_ms": (template_ended - template_started) / 1e6,
                "tokenization_ms": (tokenization_ended - tokenization_started) / 1e6,
            },
        )
        if synchronize:
            torch.cuda.synchronize()
        encoder_outputs: Any | None = None
        decoder_attention_mask: Any | None = None
        current_attentions: Any | None = None
        with torch.inference_mode():
            if encoder_decoder:
                encoder_outputs = self._encode_source(input_ids, attention_mask)
                decoder_input_ids = torch.tensor(
                    [[self.decoder_start_token_id]],
                    device=self.device,
                    dtype=torch.long,
                )
                decoder_attention_mask = torch.ones_like(decoder_input_ids)
                logits, past, current_attentions = self._forward_encoder_decoder(
                    decoder_input_ids,
                    decoder_attention_mask,
                    encoder_outputs,
                    attention_mask,
                )
            elif attention_capture_active and prompt_tokens > 1:
                # A normal prompt prefill would materialize an O(context^2)
                # attention matrix. Prefill all but the final prompt token into
                # the KV cache with the configured efficient kernel, then ask
                # eager attention for the single final query row that actually
                # produces generated token zero.
                self._restore_attention_implementation()
                _, prefix_past, _ = self._forward_last(
                    torch,
                    input_ids[:, :-1],
                    attention_mask[:, :-1],
                )
                attention_capture_active = self._select_attention_implementation("eager")
                if not attention_capture_active and not attention_capture_warning_emitted:
                    self._emit_run(
                        run_id,
                        "warning",
                        {
                            "code": "attention_capture_unavailable",
                            "message": (
                                "The model could not re-enable eager attention after prompt "
                                "prefill, so exact attention weights are unavailable."
                            ),
                        },
                    )
                    attention_capture_warning_emitted = True
                logits, past, current_attentions = self._forward_last(
                    torch,
                    input_ids[:, -1:],
                    attention_mask,
                    prefix_past,
                    capture_attention=attention_capture_active,
                )
            else:
                logits, past, current_attentions = self._forward_last(
                    torch,
                    input_ids,
                    attention_mask,
                    capture_attention=attention_capture_active,
                )
        if synchronize:
            torch.cuda.synchronize()
        prefill_ended = time.monotonic_ns()
        prefill_ms = (prefill_ended - prefill_started) / 1e6
        # The prompt forward produces the distribution for generated token 0.
        # Attribute that work to token 0 so the first-token point includes
        # prefill; subsequent token points receive their own decode forward.
        forward_ms = prefill_ms
        generation_started = prefill_started
        previous_emitted_ns: int | None = None
        emission_times_ns: list[int] = []
        finish_reason = "length"
        eos_set = _eos_token_ids(self.model, self.tokenizer)
        reasoning_answer_allowance_active = False
        stop_sequences = tuple(settings.get("stop_sequences", ()))
        operation_order = [
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "renormalize",
            "sample",
        ]
        self._emit_run(
            run_id,
            "metric",
            {
                "sampling_operation_order": operation_order,
                "effective_seed": str(command["effective_seed"]),
                "rng_algorithm": "torch.Generator",
                "generator_device": self.device,
                "seed_affected_token_selection": settings["temperature"] > 0,
                "deterministic_reference_mode": deterministic_reference_mode,
                "torch_use_deterministic_algorithms": deterministic_reference_mode,
                "cudnn_benchmark": False if cudnn is not None else None,
                "prefill_ms": prefill_ms,
                "prompt_tokens_per_second": prompt_tokens
                / max((prefill_ended - prefill_started) / 1e9, 1e-9),
            },
        )

        for token_index in range(generation_token_limit):
            if self.cancel_event.is_set():
                finish_reason = "cancelled"
                break
            sample_started = time.monotonic_ns()
            raw = logits[0].float()
            processed = self._apply_penalties(
                torch,
                raw,
                full_ids,
                float(settings["repetition_penalty"]),
                float(settings["frequency_penalty"]),
                float(settings["presence_penalty"]),
            )
            forced_token_id = (
                forced_prefix_token_ids[token_index]
                if token_index < len(forced_prefix_token_ids)
                else None
            )
            if forced_token_id is not None and not 0 <= forced_token_id < raw.numel():
                raise ValueError("forced generation prefix contains an invalid token ID")
            temperature = float(settings["temperature"])
            if temperature == 0:
                sampled_id = int(torch.argmax(processed).item())
                chosen_id = forced_token_id if forced_token_id is not None else sampled_id
                sampler_logprob: float | None = 0.0 if chosen_id == sampled_id else None
                sampler_probability = 1.0 if chosen_id == sampled_id else 0.0
                sampler_entropy: float | None = 0.0 if detailed_metrics else None
                filtered = torch.full_like(processed, -torch.inf)
                filtered[sampled_id] = 0.0
                filters: list[str] = ["greedy_argmax"]
                sampler_log_probs = filtered
            else:
                filtered, filters = self._filter_distribution(
                    torch,
                    processed,
                    temperature=temperature,
                    top_k=int(settings["top_k"]),
                    top_p=float(settings["top_p"]),
                    min_p=float(settings["min_p"]),
                )
                sampler_log_probs = torch.log_softmax(filtered, dim=-1)
                sampler_probs = torch.exp(sampler_log_probs)
                sampled_id = int(torch.multinomial(sampler_probs, 1, generator=generator).item())
                chosen_id = forced_token_id if forced_token_id is not None else sampled_id
                selected_sampler_logprob = float(sampler_log_probs[chosen_id].item())
                sampler_logprob = (
                    selected_sampler_logprob if math.isfinite(selected_sampler_logprob) else None
                )
                sampler_probability = float(sampler_probs[chosen_id].item())
                if detailed_metrics:
                    finite = torch.isfinite(sampler_log_probs)
                    sampler_entropy = float(
                        -(sampler_probs[finite] * sampler_log_probs[finite]).sum().item()
                    )
                else:
                    sampler_entropy = None
            if forced_token_id is not None:
                filters.append("forced_prefix")
            if detailed_metrics:
                raw_log_probs = torch.log_softmax(raw, dim=-1)
                raw_logprob_value = float(raw_log_probs[chosen_id].item())
                raw_logprob: float | None = raw_logprob_value
                raw_probability: float | None = float(torch.exp(raw_log_probs[chosen_id]).item())
                raw_rank: int | None = int(
                    1
                    + (raw > raw[chosen_id]).sum().item()
                    + (
                        (raw == raw[chosen_id])
                        & (torch.arange(raw.numel(), device=raw.device) < chosen_id)
                    )
                    .sum()
                    .item()
                )
                cumulative_logprob += raw_logprob_value
                running_perplexity: float | None = math.exp(
                    min(700.0, -cumulative_logprob / (token_index + 1))
                )
            else:
                raw_log_probs = None
                raw_logprob = None
                raw_probability = None
                raw_rank = None
                running_perplexity = None
            generated.append(chosen_id)
            full_ids.append(chosen_id)
            piece = str(self.tokenizer.convert_ids_to_tokens(chosen_id))
            current_text = self.tokenizer.decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            common = 0
            for left, right in zip(previous_text, current_text, strict=False):
                if left != right:
                    break
                common += 1
            display_text = current_text[common:]
            attention_attribution: dict[str, Any] | None = None
            if attention_capture_active:
                attention_attribution, capture_error = _mean_causal_self_attention(
                    torch,
                    current_attentions,
                    context_tokens,
                )
                if attention_attribution is None:
                    attention_capture_active = False
                    if not attention_capture_warning_emitted:
                        self._emit_run(
                            run_id,
                            "warning",
                            {
                                "code": "attention_capture_unavailable",
                                "message": (
                                    "The model did not expose usable causal self-attention "
                                    "weights for this run."
                                ),
                                "reason": capture_error,
                            },
                        )
                        attention_capture_warning_emitted = True
                elif token_index == 0:
                    # The complete prompt catalogue is stored once. Subsequent
                    # generated entries are reconstructed from the run's token
                    # stream using prompt_token_count + generated_token_index.
                    attention_attribution["context_tokens"] = prompt_context_tokens
                if attention_attribution is not None:
                    captured_attention_token_count += 1
            alternatives: dict[str, list[dict[str, Any]]] = {"raw": [], "sampling": []}
            alternative_count = int(settings.get("alternatives", 10))
            if detailed_metrics and alternative_count and raw_log_probs is not None:
                top_count = min(alternative_count, raw.numel())
                raw_values, raw_ids = torch.topk(raw, top_count, sorted=True)
                for rank, (candidate_logit, candidate_id) in enumerate(
                    zip(raw_values.tolist(), raw_ids.tolist(), strict=True), 1
                ):
                    candidate = int(candidate_id)
                    alternatives["raw"].append(
                        {
                            "rank": rank,
                            "token_id": candidate,
                            "piece": str(self.tokenizer.convert_ids_to_tokens(candidate)),
                            "logit": float(candidate_logit),
                            "log_probability": float(raw_log_probs[candidate].item()),
                            "probability": float(torch.exp(raw_log_probs[candidate]).item()),
                            "survived_filter": bool(torch.isfinite(filtered[candidate]).item()),
                        }
                    )
                finite_count = int(torch.isfinite(filtered).sum().item())
                if finite_count:
                    sample_count = min(top_count, finite_count)
                    values, ids = torch.topk(sampler_log_probs, sample_count, sorted=True)
                    for rank, (candidate_logprob, candidate_id) in enumerate(
                        zip(values.tolist(), ids.tolist(), strict=True), 1
                    ):
                        candidate = int(candidate_id)
                        alternatives["sampling"].append(
                            {
                                "rank": rank,
                                "token_id": candidate,
                                "piece": str(self.tokenizer.convert_ids_to_tokens(candidate)),
                                "logit": float(filtered[candidate].item()),
                                "log_probability": float(candidate_logprob),
                                "probability": float(math.exp(candidate_logprob)),
                                "survived_filter": True,
                            }
                        )
            sample_ended = time.monotonic_ns()
            emitted_ns = time.monotonic_ns()
            inter_token_ms = (
                (emitted_ns - previous_emitted_ns) / 1e6
                if previous_emitted_ns is not None
                else None
            )
            cumulative_ms = (emitted_ns - generation_started) / 1e6
            instantaneous_tps = (
                1000.0 / inter_token_ms if inter_token_ms and inter_token_ms > 0 else None
            )
            emission_times_ns.append(emitted_ns)
            rolling_window = emission_times_ns[-10:]
            if len(rolling_window) == 1:
                rolling_tps = 1.0 / max((emitted_ns - generation_started) / 1e9, 1e-9)
            else:
                rolling_tps = (len(rolling_window) - 1) / max(
                    (rolling_window[-1] - rolling_window[0]) / 1e9,
                    1e-9,
                )
            pending_token_events[token_index] = {
                "token_index": token_index,
                "token_id": chosen_id,
                "piece": piece,
                "escaped_bytes": _escaped_bytes(display_text),
                "display_text": display_text,
                "replace_from": common,
                "span_start": common,
                "span_end": len(current_text),
                "raw_logit": float(raw[chosen_id].item()),
                "raw_logprob": raw_logprob,
                "raw_probability": raw_probability,
                "raw_rank": raw_rank,
                "processed_logit": float(processed[chosen_id].item()),
                "sample_logprob": sampler_logprob,
                "sample_probability": sampler_probability,
                "entropy": sampler_entropy,
                "surprise": (
                    -sampler_logprob if detailed_metrics and sampler_logprob is not None else None
                ),
                "cumulative_logprob": cumulative_logprob if detailed_metrics else None,
                "running_perplexity": running_perplexity,
                "alternatives": alternatives,
                "decode_ms": forward_ms,
                "includes_prefill": token_index == 0,
                "sample_ms": (sample_ended - sample_started) / 1e6,
                "emit_ms": 0.0,
                "inter_token_ms": inter_token_ms,
                "cumulative_ms": cumulative_ms,
                "instantaneous_tps": instantaneous_tps,
                "rolling_tps": rolling_tps,
                "filters": filters,
                "expert_routing": None,
                "attention_attribution": attention_attribution,
            }
            emit_segmented_tokens(reasoning_segmenter.feed(token_index, display_text))
            previous_emitted_ns = emitted_ns
            previous_text = current_text
            if attention_capture_active:
                context_tokens.append(
                    _token_source(
                        self.tokenizer,
                        chosen_id,
                        prompt_tokens + token_index,
                        source_kind="generated",
                        generated_token_index=token_index,
                        display_text=display_text,
                    )
                )
            if chosen_id in eos_set:
                finish_reason = "eos"
                break
            if stop_sequences and any(current_text.endswith(stop) for stop in stop_sequences):
                finish_reason = "stop_sequence"
                break
            generated_count = token_index + 1
            if generated_count >= max_new_tokens and not reasoning_answer_allowance_active:
                reasoning_without_answer = (
                    tagged_reasoning_enabled
                    and reasoning_observed
                    and isinstance(reasoning_segmenter, TagReasoningSegmenter)
                    and not visible_answer_observed
                )
                if reasoning_answer_allowance and reasoning_without_answer:
                    reasoning_answer_allowance_active = True
                    self._emit_run(
                        run_id,
                        "warning",
                        {
                            "code": "reasoning_answer_allowance_activated",
                            "message": (
                                "Reasoning consumed the configured output limit before visible "
                                "answer text; generation is continuing within a bounded answer "
                                "allowance."
                            ),
                            "configured_max_output_tokens": configured_max_output_tokens,
                            "reasoning_answer_allowance": reasoning_answer_allowance,
                        },
                    )
                else:
                    if reasoning_without_answer:
                        self._emit_run(
                            run_id,
                            "warning",
                            {
                                "code": "reasoning_answer_allowance_unavailable",
                                "message": (
                                    "Reasoning consumed the output budget and the model context "
                                    "has no remaining capacity for an answer."
                                ),
                                "configured_max_output_tokens": configured_max_output_tokens,
                            },
                        )
                    break
            if generated_count >= generation_token_limit:
                break

            selected_tensor = torch.tensor([[chosen_id]], device=self.device)
            if encoder_decoder:
                if decoder_attention_mask is None or encoder_outputs is None:
                    raise RuntimeError("encoder-decoder prefill state was not initialized")
                decoder_attention_mask = torch.cat(
                    [
                        decoder_attention_mask,
                        torch.ones(
                            (1, 1),
                            device=self.device,
                            dtype=decoder_attention_mask.dtype,
                        ),
                    ],
                    dim=-1,
                )
            else:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
                    ],
                    dim=-1,
                )
            if synchronize:
                torch.cuda.synchronize()
            decode_started = time.monotonic_ns()
            with torch.inference_mode():
                if encoder_decoder:
                    logits, past, current_attentions = self._forward_encoder_decoder(
                        selected_tensor,
                        decoder_attention_mask,
                        encoder_outputs,
                        attention_mask,
                        past,
                    )
                else:
                    logits, past, current_attentions = self._forward_last(
                        torch,
                        selected_tensor,
                        attention_mask,
                        past,
                        capture_attention=attention_capture_active,
                    )
            if synchronize:
                torch.cuda.synchronize()
            decode_ended = time.monotonic_ns()
            forward_ms = (decode_ended - decode_started) / 1e6

        emit_segmented_tokens(reasoning_segmenter.finalize())
        if pending_token_events:
            raise RuntimeError("reasoning segmenter did not finalize every generated token")
        for warning in getattr(reasoning_segmenter, "warnings", ()):
            self._emit_run(
                run_id,
                "warning",
                {"code": "reasoning_segmentation_warning", "message": warning},
            )
        if reasoning_answer_allowance_active and not visible_answer_observed:
            self._emit_run(
                run_id,
                "warning",
                {
                    "code": "reasoning_answer_missing",
                    "message": (
                        "The model ended without visible answer text after using its bounded "
                        "reasoning answer allowance."
                    ),
                },
            )

        completed_ns = time.monotonic_ns()
        total_generation_seconds = max((completed_ns - generation_started) / 1e9, 1e-9)
        if len(emission_times_ns) > 1:
            steady_decode_seconds = max(
                (emission_times_ns[-1] - emission_times_ns[0]) / 1e9,
                1e-9,
            )
            decode_tokens_per_second = (len(emission_times_ns) - 1) / steady_decode_seconds
        else:
            decode_tokens_per_second = None
        segment_metrics = {}
        for name, segment_logprobs in raw_segment_logprobs.items():
            segment_metrics[name] = {
                "token_count": segment_token_counts[name],
                "perplexity": math.exp(min(700.0, -sum(segment_logprobs) / len(segment_logprobs)))
                if segment_logprobs
                else None,
                "mean_raw_logprob": sum(segment_logprobs) / len(segment_logprobs)
                if segment_logprobs
                else None,
            }
        status = "cancelled" if finish_reason == "cancelled" else "completed"
        current_attentions = None
        payload = {
            "finish_reason": finish_reason,
            "generated_token_count": len(generated),
            "configured_max_output_tokens": configured_max_output_tokens,
            "reasoning_answer_allowance": reasoning_answer_allowance,
            "reasoning_answer_allowance_used": max(0, len(generated) - max_new_tokens),
            "text": previous_text,
            "conditional_response_perplexity": math.exp(
                min(700.0, -cumulative_logprob / len(generated))
            )
            if generated and detailed_metrics
            else None,
            "segment_metrics": segment_metrics,
            "total_generation_ms": (completed_ns - generation_started) / 1e6,
            "engine_ttft_ms": (
                (emission_times_ns[0] - generation_started) / 1e6 if emission_times_ns else None
            ),
            "prefill_ms": prefill_ms,
            "decode_tokens_per_second": decode_tokens_per_second,
            "end_to_end_tokens_per_second": len(generated) / total_generation_seconds,
            "memory": self._memory_snapshot(torch),
            "expert_routing": {"state": "not_applicable", "reason": "dense model"},
            "attention_capture": {
                "requested": attention_capture_requested,
                "method": "mean_causal_self_attention" if captured_attention_token_count else None,
                "captured_token_count": captured_attention_token_count,
                "semantics": "attention_weights_not_causal_contributions",
                "scope": "decoder_step_context_attention_independent_of_sampled_candidate",
            },
        }
        self._emit_run(run_id, status, payload)
        _send(self.output, "reply", request_id=request_id, ok=True, payload=payload)

    def _embed(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if self.sentence_model is None or self.model_info is None:
            raise RuntimeError("no embedding model is loaded")
        import numpy as np
        import torch

        inputs = list(command["inputs"])
        values: list[Any] = []
        for item in inputs:
            modality = item["modality"]
            content_path = item.get("content_path")
            if content_path and str(content_path).lower().startswith(("http://", "https://")):
                raise ValueError("network media URLs are disabled; upload the file locally")
            if modality == "text":
                values.append(item.get("text") or "")
            elif modality in {"image", "video"}:
                if not content_path:
                    raise ValueError(f"{modality} input requires an uploaded file")
                values.append({modality: content_path})
            elif modality == "mixed":
                if not content_path:
                    raise ValueError("mixed input requires an uploaded image or video")
                media_key = item.get("media_kind", "image")
                values.append({"text": item.get("text") or "", media_key: content_path})
            else:
                raise ValueError(f"unsupported embedding modality: {modality}")
        started = time.monotonic_ns()
        vectors = self.sentence_model.encode(
            values,
            batch_size=min(len(values), int(command.get("batch_size", 1))),
            convert_to_numpy=True,
            normalize_embeddings=bool(command.get("normalize", True)),
            show_progress_bar=False,
        )
        forward_ended = time.monotonic_ns()
        vectors = np.asarray(vectors, dtype=np.float32)
        dimensions = command.get("dimensions")
        if dimensions is not None:
            dimensions = int(dimensions)
            bounds = _embedding_dimension_bounds(
                self.model_info.get("model_type"), int(vectors.shape[1])
            )
            if bounds is None and dimensions != int(vectors.shape[1]):
                raise ValueError(
                    "custom embedding dimensions are not supported by this model adapter"
                )
            minimum, maximum = bounds or (int(vectors.shape[1]), int(vectors.shape[1]))
            if not minimum <= dimensions <= maximum:
                raise ValueError(f"requested dimensions must be between {minimum} and {maximum}")
            vectors = vectors[:, :dimensions]
            if command.get("normalize", True):
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, np.finfo(np.float32).tiny)
        results = []
        for item, vector in zip(inputs, vectors, strict=True):
            results.append(
                {
                    "input_id": item["input_id"],
                    "vector": vector.tolist(),
                    "output_dimension": int(vector.shape[0]),
                    "output_dtype": str(vector.dtype),
                    "normalized": bool(command.get("normalize", True)),
                    "l2_norm": float(np.linalg.norm(vector)),
                    "statistics": {
                        "minimum": float(vector.min()),
                        "maximum": float(vector.max()),
                        "mean": float(vector.mean()),
                        "standard_deviation": float(vector.std()),
                        "finite": bool(np.isfinite(vector).all()),
                    },
                }
            )
        completed = time.monotonic_ns()
        return {
            "results": results,
            "pooling": self.model_info.get("embedding_pooling") or "not_reported",
            "joint_embedding_space": bool(self.model_info.get("joint_embedding_space", False)),
            "preprocessing_ms": None,
            "forward_ms": (forward_ended - started) / 1e6,
            "total_ms": (completed - started) / 1e6,
            "items_per_second": len(results) / max((completed - started) / 1e9, 1e-9),
            "memory": self._memory_snapshot(torch),
        }

    def _score_prompt(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("no generation model is loaded")
        import torch

        text = str(command["text"])
        started = time.monotonic_ns()
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        length = int(input_ids.shape[1])
        if length < 2:
            return {
                "perplexity": None,
                "mean_log_probability": None,
                "included_token_count": 0,
                "excluded_first_token": 1 if length else 0,
                "excluded_masked_tokens": 0,
                "excluded_non_text_tokens": 0,
                "token_log_probabilities": [None] * length,
                "duration_ms": (time.monotonic_ns() - started) / 1e6,
            }
        body = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        token_logprobs: list[float | None] = [None] * length
        included: list[float] = []
        excluded_masked = 0
        with torch.inference_mode():
            if body is not None and head is not None:
                outputs = body(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state[:, :-1, :]
                targets = input_ids[:, 1:]
                for start in range(0, length - 1, 128):
                    end = min(length - 1, start + 128)
                    logits = head(hidden[:, start:end, :]).float()
                    logs = torch.log_softmax(logits, dim=-1)
                    chosen = logs.gather(-1, targets[:, start:end].unsqueeze(-1)).squeeze(-1)
                    for offset, value in enumerate(chosen[0].tolist(), start + 1):
                        if not bool(attention_mask[0, offset].item()):
                            excluded_masked += 1
                            continue
                        selected = float(value)
                        token_logprobs[offset] = selected
                        included.append(selected)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                logs = torch.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
                chosen = logs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                for offset, value in enumerate(chosen[0].tolist(), 1):
                    if not bool(attention_mask[0, offset].item()):
                        excluded_masked += 1
                        continue
                    selected = float(value)
                    token_logprobs[offset] = selected
                    included.append(selected)
        mean = sum(included) / len(included) if included else None
        return {
            "perplexity": math.exp(min(700.0, -mean)) if mean is not None else None,
            "mean_log_probability": mean,
            "included_token_count": len(included),
            "excluded_first_token": 1,
            "excluded_masked_tokens": excluded_masked,
            "excluded_non_text_tokens": 0,
            "token_log_probabilities": token_logprobs,
            "duration_ms": (time.monotonic_ns() - started) / 1e6,
            "definition": "exp(-mean shifted raw full-vocabulary log probability)",
        }


def worker_main(commands: Queue[Any], output: Queue[Any], cancel_event: Any) -> None:
    # The parent coordinates graceful cancellation and shutdown. Ignoring the
    # console interrupt here prevents a duplicate child traceback on Windows.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    WorkerRuntime(commands, output, cancel_event).run()
