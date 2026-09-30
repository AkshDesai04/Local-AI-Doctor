"""Model runtime executed only inside an isolated spawned worker process."""

from __future__ import annotations

import contextlib
import gc
import importlib
import json
import math
import os
import queue
import signal
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import psutil

from ..reasoning import (
    SegmentClass,
    SegmentedToken,
    TagReasoningSegmenter,
    UnknownReasoningSegmenter,
)
from .memory import (
    available_ram,
    available_vram,
    checkpoint_tensor_shapes,
    device_map_kwargs,
    estimate_load_bytes,
    placement_of,
    vram_cap_fraction,
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


class WorkerReportedError(RuntimeError):
    """A failure the worker describes itself with codes and numbers only.

    ``_safe_error`` passes the dict through verbatim, so it must never hold paths,
    prompts, or exception text.
    """

    def __init__(self, error: Mapping[str, Any]) -> None:
        self.error = dict(error)
        super().__init__(str(self.error.get("code")))


def _insufficient_memory(
    memory_kind: str, required: int, available: int, estimate: Mapping[str, int]
) -> WorkerReportedError:
    where = "GPU" if memory_kind == "vram" else "system"
    return WorkerReportedError(
        {
            "code": "insufficient_memory",
            "message": f"the model does not fit the available {where} memory",
            "hint": (
                "Quantize the model, unload another model, or turn off Strict VRAM to allow "
                "system-RAM offload."
                if memory_kind == "vram"
                else "Unload another model or choose a smaller model."
            ),
            "memory_kind": memory_kind,
            "required_bytes": int(required),
            "available_bytes": int(available),
            "estimate": dict(estimate),
        }
    )


def _rope_state_owner(model: Any) -> Any:
    """The module caching multimodal RoPE deltas between prefill and decode, if any."""

    for candidate in (getattr(model, "model", None), model):
        if candidate is not None and hasattr(candidate, "rope_deltas"):
            return candidate
    return None


def _safe_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, WorkerReportedError):
        return dict(exc.error)
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


def _fold_system_into_first_user(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """Return ``messages`` with a leading system prompt prepended to the first user turn."""

    if not messages or messages[0].get("role") != "system":
        return None
    system = str(messages[0].get("content") or "")
    rest = [dict(message) for message in messages[1:]]
    for message in rest:
        if system.strip() and message.get("role") == "user":
            message["content"] = f"{system}\n\n{message.get('content') or ''}"
            return rest
    return None


def _render_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    reasoning: bool | None = None,
    reasoning_delimiters: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Render chat messages; templates without a system role get it folded into the user turn.

    A template that raises on the system role or silently drops its text is
    re-rendered with the system prompt prepended to the first user message, and
    the renderer is reported as ``chat_template_system_merged``.
    """

    options: dict[str, Any] = {"reasoning": reasoning, "reasoning_delimiters": reasoning_delimiters}
    merged = (
        _fold_system_into_first_user(messages)
        if getattr(tokenizer, "chat_template", None)
        else None
    )
    if merged is None:
        return _render_conversation(tokenizer, messages, **options)
    with contextlib.suppress(Exception):  # the folded re-render below reports real failures
        rendered, renderer = _render_conversation(tokenizer, messages, **options)
        if str(messages[0].get("content") or "").strip() in rendered:
            return rendered, renderer
    return _render_conversation(tokenizer, merged, **options)[0], "chat_template_system_merged"


def _render_conversation(
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


def _default_bos_token_id(tokenizer: Any) -> int | None:
    """Return the BOS token the tokenizer inserts by default, if any.

    Prompts are tokenized with ``add_special_tokens=False`` because chat templates
    write their own BOS. The plain-text fallback has no template, so that setting
    silently dropped BOS for base checkpoints such as Llama 3.2, which were trained
    with it. Probing an empty encode answers "does this tokenizer add BOS" without
    guessing from flags that differ between slow and fast tokenizers.
    """

    bos = _valid_token_id(getattr(tokenizer, "bos_token_id", None))
    if bos is None:
        return None
    try:
        probe = list(tokenizer("", add_special_tokens=True)["input_ids"])
    except Exception:  # an exotic tokenizer that cannot encode "" adds nothing here
        return None
    return bos if probe[:1] == [bos] else None


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


def _generation_loader(model_path: Path, common: Mapping[str, Any]) -> Any:
    """Pick the Auto class the installed Transformers build maps this checkpoint to.

    Decoder-only checkpoints are registered under ``AutoModelForCausalLM``, but
    multimodal generators (Qwen3-VL and every other ``*ForConditionalGeneration``
    family) are only registered under the image-text-to-text head. Resolving through
    the loaded configuration keeps new families working without a code change and
    without enabling repository code.
    """

    import transformers

    config = transformers.AutoConfig.from_pretrained(model_path, **common)
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        loader = getattr(transformers, name, None)
        mapping = getattr(loader, "_model_mapping", None)
        if mapping is not None and type(config) in mapping:
            return loader
    # Fall through to the causal head so Transformers raises its own exact message.
    return transformers.AutoModelForCausalLM


_MEDIA_KINDS = frozenset({"image", "video"})


def _processor_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat messages to the content-part form processor chat templates expect.

    Media parts carry no path or bytes: the template only emits placeholder tokens,
    and the processor expands them from the decoded inputs passed alongside the text.
    """

    return [
        {
            "role": str(message.get("role", "user")),
            "content": [
                *({"type": str(item["kind"])} for item in message.get("attachments") or ()),
                {"type": "text", "text": str(message.get("content") or "")},
            ],
        }
        for message in messages
    ]


def _load_media(
    processor: Any, messages: Sequence[Mapping[str, Any]]
) -> tuple[list[Any], list[Any], list[Any]]:
    """Decode attachment files the parent resolved inside the upload store.

    Order follows the conversation so it matches the placeholder order the chat
    template renders. Video frames are sampled with the processor's own policy,
    and the metadata is kept because Qwen3-VL derives frame timestamps from it.
    """

    from PIL import Image

    images: list[Any] = []
    videos: list[Any] = []
    video_metadata: list[Any] = []
    for message in messages:
        for item in message.get("attachments") or ():
            kind = str(item.get("kind"))
            path = Path(str(item.get("path")))
            if kind not in _MEDIA_KINDS:
                raise ValueError(f"unsupported chat media kind: {kind!r}")
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError("a chat attachment file is unavailable")
            if kind == "image":
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            else:
                from transformers import video_utils

                # Local paths only: the parent resolved each file inside the upload
                # store, so the URL branches of this loader are never reached.
                load_video: Any = video_utils.load_video
                frames, metadata = load_video(
                    str(path),
                    backend="pyav",
                    sample_indices_fn=processor.video_processor.sample_frames,
                )
                videos.append(frames)
                video_metadata.append(metadata)
    return images, videos, video_metadata


def _media_token_counts(processor: Any, media_inputs: Mapping[str, Any]) -> dict[str, list[int]]:
    """Placeholder tokens per media item, from the processor's patch grids."""

    counts: dict[str, list[int]] = {}
    for kind, grid_key, component in (
        ("image", "image_grid_thw", "image_processor"),
        ("video", "video_grid_thw", "video_processor"),
    ):
        grid = media_inputs.get(grid_key)
        merge = getattr(getattr(processor, component, None), "merge_size", None)
        if grid is not None and isinstance(merge, int) and merge > 0:
            counts[kind] = [
                int(frames * height * width) // merge**2 for frames, height, width in grid.tolist()
            ]
    return counts


def _media_labels(
    token_ids: Sequence[int],
    placeholder_ids: Mapping[int, str],
    tokens_per_item: Mapping[str, Sequence[int]],
) -> list[dict[str, Any] | None]:
    """Label each prompt position that holds image or video features.

    Placeholder positions are assigned to media items by the per-item token counts
    the processor produced; when those are unknown the item index is omitted.
    """

    boundaries = {
        kind: [sum(counts[: index + 1]) for index in range(len(counts))]
        for kind, counts in tokens_per_item.items()
    }
    totals = {
        kind: sum(1 for token_id in token_ids if placeholder_ids.get(token_id) == kind)
        for kind in set(placeholder_ids.values())
    }
    seen: dict[str, int] = {}
    labels: list[dict[str, Any] | None] = []
    for token_id in token_ids:
        kind = placeholder_ids.get(token_id)
        if kind is None:
            labels.append(None)
            continue
        position = seen.get(kind, 0)
        seen[kind] = position + 1
        label: dict[str, Any] = {"kind": kind}
        limits = boundaries.get(kind)
        if limits and limits[-1] == totals[kind]:
            label["index"] = next(index for index, end in enumerate(limits) if position < end)
        labels.append(label)
    return labels


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


@dataclass(eq=False)
class ResidentModel:
    """One loaded checkpoint and everything the worker needs to run it."""

    key: str
    model: Any = None
    tokenizer: Any = None
    processor: Any = None
    sentence_model: Any = None
    model_info: dict[str, Any] | None = None
    # `device` is the requested device; `input_device` is where input tensors go,
    # which is the GPU whenever any part of the device map uses it.
    device: str = "cpu"
    input_device: str = "cpu"
    dtype: str = "float32"
    quantization: str = "none"
    strict_vram: bool = True
    placement: str = "cpu"
    loaded_attention_implementation: str | None = None
    decoder_start_token_id: int | None = None
    decoder_start_token_source: str | None = None
    decoder_start_warning: str | None = None
    gpu_bytes: int = 0
    cpu_bytes: int = 0
    kv_reserve_bytes: int = 0
    load_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "model_key": self.key,
            "model_id": (self.model_info or {}).get("id"),
            "placement": self.placement,
            "quantization": self.quantization,
            "strict_vram": self.strict_vram,
            "gpu_bytes": self.gpu_bytes,
            "cpu_bytes": self.cpu_bytes,
            "kv_reserve_bytes": self.kv_reserve_bytes,
        }


def _active_attribute(name: str, default: Any) -> Any:
    """Expose one attribute of the active resident as a runtime attribute.

    Setting a value on a runtime with no residents creates a ``default`` resident,
    so single-model callers keep assigning ``runtime.model`` and friends directly.
    """

    def get(runtime: WorkerRuntime) -> Any:
        return default if runtime._active is None else getattr(runtime._active, name)

    def set_(runtime: WorkerRuntime, value: Any) -> None:
        if runtime._active is None:
            if value is None:
                return
            runtime._active = runtime.residents["default"] = ResidentModel(key="default")
        setattr(runtime._active, name, value)
        if name == "device":
            runtime._active.input_device = value

    return property(get, set_)


class WorkerRuntime:
    model = _active_attribute("model", None)
    tokenizer = _active_attribute("tokenizer", None)
    processor = _active_attribute("processor", None)
    sentence_model = _active_attribute("sentence_model", None)
    model_info = _active_attribute("model_info", None)
    device = _active_attribute("device", "cpu")
    dtype = _active_attribute("dtype", "float32")
    decoder_start_token_id = _active_attribute("decoder_start_token_id", None)
    decoder_start_token_source = _active_attribute("decoder_start_token_source", None)
    decoder_start_warning = _active_attribute("decoder_start_warning", None)
    loaded_attention_implementation = _active_attribute("loaded_attention_implementation", None)

    def __init__(self, commands: Queue[Any], output: Queue[Any], cancel_event: Any) -> None:
        self.commands = commands
        self.output = output
        self.cancel_event = cancel_event
        self.residents: dict[str, ResidentModel] = {}
        self._active: ResidentModel | None = None
        self.sessions: list[GenerationSession] = []
        self.max_concurrent_runs = 1
        self._deterministic_state: bool | None = None
        self._vram_cap_fraction = 1.0
        self._safety_margin_bytes = 0

    def _select_resident(self, model_key: Any) -> ResidentModel | None:
        """Make the named resident active; no key means the active resident."""

        if model_key is None:
            return self._active
        resident = self.residents.get(str(model_key))
        if resident is None:
            raise WorkerReportedError(
                {
                    "code": "model_not_resident",
                    "message": "the requested model is not resident in the worker",
                    "hint": "Load the model before running it.",
                }
            )
        self._active = resident
        return resident

    def run(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        _send(self.output, "ready")
        while True:
            # Block only while idle; running sessions poll so a cancel or a second
            # generation reaches the worker between decode steps.
            command = self._next_command(block=not self.sessions)
            if command is not None and self._handle(command):
                return
            if self.sessions:
                self._step_round()

    def _next_command(self, *, block: bool) -> dict[str, Any] | None:
        if block:
            return dict(self.commands.get())
        try:
            return dict(self.commands.get_nowait())
        except queue.Empty:
            return None

    def _handle(self, command: Mapping[str, Any]) -> bool:
        """Run one command; True once the worker should exit."""

        operation = command.get("op")
        request_id = str(command.get("request_id", ""))
        try:
            if operation == "shutdown":
                self._end_sessions(self.sessions)
                self._unload()
                _send(self.output, "reply", request_id=request_id, ok=True, payload={})
                return True
            if operation == "cancel":
                # Fire and forget: an unknown run already finished or never started.
                for session in self.sessions:
                    if session.run_id == str(command.get("run_id")):
                        session.cancel_requested = True
                return False
            if operation == "generate":
                self._start_session(command)
                return False
            if operation == "load":
                payload = self._load(command)
            elif operation == "unload":
                model_key = command.get("model_key")
                payload = self._unload(str(model_key) if model_key else None)
            elif operation == "memory_status":
                payload = {"ledger": self._ledger()}
            elif operation == "embed":
                payload = self._embed(command)
            elif operation == "score_prompt":
                payload = self._score_prompt(command)
            else:
                raise ValueError(f"unknown worker operation: {operation!r}")
            _send(self.output, "reply", request_id=request_id, ok=True, payload=payload)
        except BaseException as exc:  # worker boundary must report model/runtime crashes
            run_id = command.get("run_id")
            self._report_failure(str(run_id) if run_id else None, request_id, exc)
        return False

    def _report_failure(self, run_id: str | None, request_id: str, exc: BaseException) -> None:
        # A full-instrumentation run may temporarily select eager
        # attention so Transformers can return attention probabilities.
        # Never let a failed request silently change later run kernels.
        self._restore_attention_implementation()
        error = _safe_error(exc)
        if run_id:
            _send(self.output, "run_event", run_id=run_id, event_type="error", payload=error)
        _send(self.output, "reply", request_id=request_id, ok=False, error=error)
        # The traceback stays inside the isolated process and is bounded.
        _send(
            self.output,
            "diagnostic",
            request_id=request_id,
            payload={"traceback": "".join(traceback.format_exception(exc))[-8000:]},
        )

    def _start_session(self, command: Mapping[str, Any]) -> None:
        """Prefill and emit the first token at once, so every session gets a clean TTFT."""

        if len(self.sessions) >= self.max_concurrent_runs:
            raise WorkerReportedError(
                {
                    "code": "worker_busy",
                    "message": "the worker is already running its maximum concurrent generations",
                    "hint": "Wait for a running generation or raise runtime.max_concurrent_runs.",
                    "max_concurrent_runs": self.max_concurrent_runs,
                }
            )
        self._apply_vram_cap()
        session = GenerationSession(self, command)
        session.prefill()
        if session.step():
            self.sessions.append(session)
        else:
            self._restore_attention_implementation()

    def _activate(self, session: GenerationSession) -> None:
        """Restore the process-wide state one session expects before it steps."""

        self._active = session.resident
        mode = session.deterministic_reference_mode
        if self._deterministic_state != mode:
            session.torch.use_deterministic_algorithms(mode, warn_only=mode)
            self._deterministic_state = mode
        wanted = session.attention_implementation
        config = getattr(self.model, "config", None)
        if wanted is not None and getattr(config, "_attn_implementation", None) != wanted:
            self._select_attention_implementation(wanted)
        owner = _rope_state_owner(self.model)
        if owner is not None:
            owner.rope_deltas = session.rope_deltas

    def _step_round(self) -> None:
        # ponytail: round-robin one step per session; a long prefill blocks the other
        # session; chunked prefill if it matters
        running = len(self.sessions)
        for session in list(self.sessions):
            session.concurrent_sessions_max = max(session.concurrent_sessions_max, running)
            try:
                self._activate(session)
                alive = session.step()
            except BaseException as exc:  # one failing session must not stop the others
                alive = False
                session.finished = True
                session._release_tensors()
                self._report_failure(session.run_id, session.request_id, exc)
            else:
                if not alive:
                    self._restore_attention_implementation()
            if not alive:
                self.sessions.remove(session)
        if not self.sessions:
            self._empty_cache()

    def _end_sessions(self, sessions: Sequence[GenerationSession]) -> None:
        """Cancel sessions now; each still emits its terminal event and reply."""

        for session in list(sessions):
            session.cancel_requested = True
            try:
                self._activate(session)
                session.step()
                self._restore_attention_implementation()
            except BaseException as exc:
                session._release_tensors()
                self._report_failure(session.run_id, session.request_id, exc)
            if session in self.sessions:
                self.sessions.remove(session)

    @staticmethod
    def _empty_cache() -> None:
        import torch

        if torch.cuda.is_available():
            with contextlib.suppress(RuntimeError):
                torch.cuda.empty_cache()

    def _load(self, command: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        model = dict(command["model"])
        runtime = dict(command.get("runtime", {}))
        model_path = Path(model["path"])
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"configured model directory {model_path.name!r} is unavailable"
            )
        key = str(command.get("model_key") or "default")
        if key in self.residents:
            # Loading an existing key replaces it, e.g. to re-place an offloaded model.
            self._unload(key)
        self.max_concurrent_runs = max(1, int(runtime.get("max_concurrent_runs", 1)))
        device = str(runtime.get("device", "cpu"))
        dtype_name = str(runtime.get("dtype", "float32"))
        cuda_device = device.startswith("cuda")
        placement = str(runtime.get("placement") or ("gpu_only" if cuda_device else "cpu"))
        if placement not in {"gpu_only", "offload", "cpu"}:
            raise ValueError(f"unknown model placement: {placement!r}")
        if placement != "cpu" and not cuda_device:
            raise ValueError("GPU placement requires a CUDA device")
        quantization = str(runtime.get("quantization") or "none")
        if quantization != "none":
            raise ValueError("model-weight quantization has no installed adapter")
        task = str(model["task"])
        generation = task in {"text_generation", "encoder_decoder_generation"}
        if not generation and task not in {"embedding", "multimodal_embedding"}:
            raise ValueError(f"no runtime adapter supports task {task!r}")
        if placement == "offload" and not generation:
            raise ValueError("layer offload is available only for generation models")
        self._safety_margin_bytes = margin = int(runtime.get("safety_margin_bytes", 0))
        torch.set_num_threads(int(runtime.get("cpu_threads", max(1, os.cpu_count() or 1))))

        try:
            config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            config = {}
        estimate = estimate_load_bytes(
            checkpoint_tensor_shapes(model_path),
            config if isinstance(config, dict) else {},
            quantization=quantization,
            compute_dtype=dtype_name,
            kv_reserve_tokens=int(runtime.get("kv_reserve_tokens", 0)) if generation else 0,
        )
        estimate_report = {
            "weights": estimate["weights"],
            "kv_reserve": estimate["kv_reserve"],
            "margin": margin,
        }
        ram_available = available_ram(
            system_available=int(psutil.virtual_memory().available),
            process_rss=int(psutil.Process().memory_info().rss),
            margin=margin,
            budget=runtime.get("ram_budget_bytes"),
        )
        cuda: Any = torch.cuda
        gpu_index = int(device.split(":", 1)[1]) if cuda_device and ":" in device else 0
        vram_available = 0
        if cuda_device:
            free, _total = cuda.mem_get_info(gpu_index)
            vram_available = available_vram(
                free=int(free),
                reserved=int(cuda.memory_reserved(gpu_index)),
                allocated=int(cuda.memory_allocated(gpu_index)),
                margin=margin,
                budget=runtime.get("vram_budget_bytes"),
            )
        # Preflight before from_pretrained: a load that cannot fit fails with numbers
        # instead of filling memory first.
        if placement == "gpu_only" and estimate["total"] > vram_available:
            raise _insufficient_memory("vram", estimate["total"], vram_available, estimate_report)
        if placement == "cpu" and estimate["total"] > ram_available:
            raise _insufficient_memory("ram", estimate["total"], ram_available, estimate_report)
        gpu_weight_budget = max(0, vram_available - estimate["kv_reserve"])
        if placement == "offload" and estimate["weights"] > gpu_weight_budget + ram_available:
            raise _insufficient_memory(
                "ram", estimate["total"], vram_available + ram_available, estimate_report
            )

        resident = ResidentModel(
            key=key,
            device=device,
            input_device=device if placement != "cpu" else "cpu",
            dtype=dtype_name,
            quantization=quantization,
            strict_vram=bool(runtime.get("strict_vram", True)),
            placement={"gpu_only": "gpu", "offload": "offload", "cpu": "cpu"}[placement],
            kv_reserve_bytes=estimate["kv_reserve"],
        )
        # Registered before loading so a strict load already runs under the cap.
        self._active = self.residents[key] = resident
        self._apply_vram_cap()
        allocated_before = int(cuda.memory_allocated(gpu_index)) if cuda_device else 0
        rss_before = int(psutil.Process().memory_info().rss)
        started = time.monotonic()
        out_of_memory = False
        try:
            self._load_resident(
                resident,
                model,
                model_path,
                runtime,
                device_map_kwargs(
                    placement, device, gpu_bytes=gpu_weight_budget, cpu_bytes=ram_available
                ),
            )
        except (torch.OutOfMemoryError, MemoryError):
            out_of_memory = True
        except BaseException:
            self._discard(key)
            raise
        if out_of_memory:
            # Outside the except block, so the traceback no longer pins partial weights.
            self._discard(key)
            free_after = int(cuda.mem_get_info(gpu_index)[0]) if cuda_device else 0
            raise _insufficient_memory(
                "vram" if cuda_device and placement == "gpu_only" else "ram",
                estimate["total"],
                free_after if cuda_device else ram_available,
                estimate_report,
            )
        resident.load_seconds = time.monotonic() - started
        resident.gpu_bytes = (
            max(0, int(cuda.memory_allocated(gpu_index)) - allocated_before) if cuda_device else 0
        )
        resident.cpu_bytes = max(0, int(psutil.Process().memory_info().rss) - rss_before)
        device_map = getattr(resident.model, "hf_device_map", None)
        if isinstance(device_map, Mapping):
            resident.placement = placement_of(device_map, resident.placement)
            if resident.placement == "cpu":
                resident.input_device = "cpu"
        self._apply_vram_cap()
        summary = Counter(
            str(value)
            for value in (device_map if isinstance(device_map, Mapping) else {"": device}).values()
        )
        return {
            "model_id": model["id"],
            "task": task,
            "device": device,
            "dtype": dtype_name,
            "load_seconds": resident.load_seconds,
            "memory": self._memory_snapshot(torch),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
            "local_files_only": True,
            "decoder_start_token_id": resident.decoder_start_token_id,
            "decoder_start_token_source": resident.decoder_start_token_source,
            "warnings": [resident.decoder_start_warning] if resident.decoder_start_warning else [],
            "model_key": key,
            "placement": resident.placement,
            "quantization": quantization,
            "strict_vram": resident.strict_vram,
            "gpu_bytes": resident.gpu_bytes,
            "cpu_bytes": resident.cpu_bytes,
            "kv_reserve_bytes": resident.kv_reserve_bytes,
            "device_map_summary": dict(summary),
            "estimate": estimate_report,
            "ledger": self._ledger(),
        }

    def _load_resident(
        self,
        resident: ResidentModel,
        model: Mapping[str, Any],
        model_path: Path,
        runtime: Mapping[str, Any],
        placement_kwargs: Mapping[str, Any],
    ) -> None:
        import torch

        dtype = _torch_dtype(torch, resident.dtype)
        attention = runtime.get("attention_backend")
        attention = (
            None
            if attention in {None, "auto"}
            else attention.replace("flash-attention-2", "flash_attention_2")
        )
        # Only True when discovery matched this exact manifest fingerprint against a
        # human-reviewed entry in discovery/reviewed_bundled_code.py; never client-set.
        trust_remote_code = bool(model.get("trust_remote_code", False))
        common: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
        }
        model_kwargs = dict(common)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attention is not None:
            model_kwargs["attn_implementation"] = attention
        task = str(model["task"])
        if task in {"text_generation", "encoder_decoder_generation"}:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer_loader: Any = AutoTokenizer
            resident.tokenizer = tokenizer_loader.from_pretrained(model_path, **common)
            if model.get("media_modalities"):
                from transformers import AutoProcessor

                processor_loader: Any = AutoProcessor
                resident.processor = processor_loader.from_pretrained(model_path, **common)
            loader: Any = (
                AutoModelForSeq2SeqLM
                if task == "encoder_decoder_generation"
                else _generation_loader(model_path, common)
            )
            # Always a device map, never `.to(device)`: accelerate-dispatched (offloaded)
            # models reject `.to()`.
            resident.model = loader.from_pretrained(
                model_path,
                low_cpu_mem_usage=bool(runtime.get("low_memory_loading", True)),
                **model_kwargs,
                **placement_kwargs,
            )
            resident.model.eval()
            configured_attention = getattr(resident.model.config, "_attn_implementation", None)
            resident.loaded_attention_implementation = (
                str(configured_attention) if configured_attention is not None else None
            )
            if task == "encoder_decoder_generation":
                (
                    resident.decoder_start_token_id,
                    resident.decoder_start_token_source,
                    resident.decoder_start_warning,
                ) = _decoder_start_token(resident.model, resident.tokenizer)
        else:
            from sentence_transformers import SentenceTransformer

            sentence_kwargs: dict[str, Any] = {
                "device": resident.device,
                "local_files_only": True,
                "trust_remote_code": trust_remote_code,
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
            resident.sentence_model = SentenceTransformer(str(model_path), **sentence_kwargs)
        resident.model_info = dict(model)

    def _discard(self, key: str) -> None:
        """Forget a resident that failed to load and give its memory back."""

        resident = self.residents.pop(key, None)
        if resident is not None and self._active is resident:
            self._active = next(reversed(self.residents.values()), None)
        del resident
        gc.collect()
        self._empty_cache()
        self._apply_vram_cap()

    def _apply_vram_cap(self) -> None:
        """Cap the allocator while any strict CUDA resident exists, so nothing spills.

        Some drivers silently page device allocations into system RAM once VRAM is
        full; the cap turns that into an out-of-memory error instead.
        """

        import torch

        cuda: Any = torch.cuda
        if not cuda.is_available() or not cuda.is_initialized():
            return
        # ponytail: process-wide cap; a non-strict resident next to a strict one is also capped
        strict = any(
            resident.strict_vram and resident.placement != "cpu"
            for resident in self.residents.values()
        )
        fraction = 1.0
        if strict:
            free, total = cuda.mem_get_info(0)
            fraction = vram_cap_fraction(
                reserved=int(cuda.memory_reserved(0)),
                free=int(free),
                total=int(total),
                margin=self._safety_margin_bytes,
            )
        if fraction != self._vram_cap_fraction:
            cuda.set_per_process_memory_fraction(fraction, 0)
            self._vram_cap_fraction = fraction

    def _unload(self, model_key: str | None = None) -> dict[str, Any]:
        """Unload one resident (or all) and report the device memory actually returned."""

        if model_key is not None and model_key not in self.residents:
            raise WorkerReportedError(
                {
                    "code": "model_not_resident",
                    "message": "the requested model is not resident in the worker",
                    "hint": "Refresh the resident model list.",
                }
            )
        keys = [model_key] if model_key is not None else list(self.residents)
        memory_before: dict[str, Any] = {}
        torch_module: Any | None = None
        allocated_before = 0
        if keys:
            try:
                import torch

                torch_module = torch
                memory_before = self._memory_snapshot(torch)
                allocated_before = self._allocated_bytes(torch)
            except ImportError:
                pass
        # A session still using a resident is cancelled first; it keeps its terminal event.
        self._end_sessions([session for session in self.sessions if session.resident.key in keys])
        expected_gpu_bytes = 0
        unloaded_ids: list[str] = []
        for key in keys:
            resident = self.residents.pop(key)
            expected_gpu_bytes += resident.gpu_bytes
            if resident.model_info and resident.model_info.get("id"):
                unloaded_ids.append(str(resident.model_info["id"]))
            if resident.model is not None and getattr(resident.model, "hf_device_map", None):
                with contextlib.suppress(Exception):
                    hooks: Any = importlib.import_module("accelerate.hooks")
                    hooks.remove_hook_from_module(resident.model, recurse=True)
            if self._active is resident:
                self._active = None
            # Memory returns only once every reference (model, caches, outputs) is gone.
            del resident
        if self._active is None and self.residents:
            self._active = next(reversed(self.residents.values()))
        gc.collect()
        try:
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
                torch_module.cuda.ipc_collect()
        except RuntimeError:
            pass
        memory_after = self._memory_snapshot(torch_module) if torch_module is not None else {}
        freed_bytes = 0
        if torch_module is not None:
            freed_bytes = max(0, allocated_before - self._allocated_bytes(torch_module))
            self._apply_vram_cap()
        return {
            "unloaded_model_id": unloaded_ids[-1] if unloaded_ids else None,
            "unloaded_model_ids": unloaded_ids,
            "unloaded_model_keys": keys,
            "freed_bytes": freed_bytes,
            "leaked_bytes": max(0, expected_gpu_bytes - freed_bytes),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "ledger": self._ledger() if torch_module is not None else None,
        }

    @staticmethod
    def _allocated_bytes(torch: Any) -> int:
        cuda: Any = torch.cuda
        if cuda.is_available() and cuda.is_initialized():
            return int(cuda.memory_allocated(0))
        return 0

    @staticmethod
    def _memory_snapshot(torch: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            process = psutil.Process()
            result["process_rss_bytes"] = int(process.memory_info().rss)
            result["system_available_bytes"] = int(psutil.virtual_memory().available)
        except (psutil.Error, OSError):
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

    def _ledger(self) -> dict[str, Any]:
        """Process memory plus every resident's share, without initializing CUDA."""

        import torch

        ledger: dict[str, Any] = {
            "device": "cpu",
            "total_bytes": None,
            "free_bytes": None,
            "torch_allocated_bytes": None,
            "torch_reserved_bytes": None,
            "cap_bytes": None,
            "process_rss_bytes": None,
            "system_available_bytes": None,
            "residents": [resident.summary() for resident in self.residents.values()],
        }
        try:
            memory = psutil.virtual_memory()
            ledger["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
            ledger["system_available_bytes"] = int(memory.available)
            ledger["total_bytes"] = int(memory.total)
            ledger["free_bytes"] = int(memory.available)
        except (psutil.Error, OSError):
            pass
        cuda: Any = torch.cuda
        if cuda.is_available():
            # Querying device memory creates a CUDA context, which costs VRAM on its
            # own; before the first CUDA load the device totals stay unknown.
            ledger.update(device="cuda:0", total_bytes=None, free_bytes=None)
            if cuda.is_initialized():
                free, total = cuda.mem_get_info(0)
                ledger.update(
                    total_bytes=int(total),
                    free_bytes=int(free),
                    torch_allocated_bytes=int(cuda.memory_allocated(0)),
                    torch_reserved_bytes=int(cuda.memory_reserved(0)),
                    cap_bytes=(
                        int(total * self._vram_cap_fraction)
                        if self._vram_cap_fraction < 1.0
                        else None
                    ),
                )
        return ledger

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
        media: Mapping[str, Any] | None = None,
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
            # Processor outputs (pixel values, grids) belong to the prompt forward
            # that contains their placeholder tokens; decode steps never repeat them.
            **(media or {}),
        }
        # The attention mask spans cached and new positions, so the cache holds
        # everything before these input tokens. Families with multimodal RoPE
        # (Qwen3-VL) derive decode positions only from `cache_position`; without it
        # every decoded token was placed at position zero.
        past_length = int(attention_mask.shape[-1]) - int(input_ids.shape[-1])
        optional: dict[str, Any] = {
            "cache_position": torch.arange(
                past_length, past_length + int(input_ids.shape[-1]), device=input_ids.device
            )
        }

        def call(target: Any, extra: dict[str, Any]) -> Any:
            # Older or custom families reject newer keywords; drop them newest first.
            extra = dict(extra)
            while True:
                try:
                    return target(**kwargs, **extra)
                except TypeError:
                    if not extra:
                        raise
                    extra.popitem()
                    captured_attentions[:] = [None] * len(captured_attentions)

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
                outputs = call(body, optional)
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
            outputs = call(self.model, {**optional, "logits_to_keep": 1})
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
        """Run one generation to completion (prefill, then one step per token)."""

        try:
            session = GenerationSession(self, command)
            session.prefill()
            while session.step():
                pass
        finally:
            self._restore_attention_implementation()

    def _embed(self, command: Mapping[str, Any]) -> dict[str, Any]:
        self._select_resident(command.get("model_key"))
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
        self._select_resident(command.get("model_key"))
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


class GenerationSession:
    """One generation request, advanced one sampled token per ``step()``.

    The constructor validates, renders, tokenizes, and emits the ``stage: prefill``
    event; ``prefill()`` runs the forward pass that yields token-0 logits; each
    ``step()`` samples and emits one token and runs the forward pass for the next.
    The session operates on the runtime's active resident, so the scheduler makes
    its resident active before every call.
    """

    def __init__(self, rt: WorkerRuntime, command: Mapping[str, Any]) -> None:
        resident = rt._select_resident(command.get("model_key"))
        if resident is None or rt.model is None or rt.tokenizer is None or rt.model_info is None:
            raise RuntimeError("no generation model is loaded")
        import torch

        self.rt = rt
        self.torch = torch
        self.resident = resident
        self.cancel_requested = False
        self.finished = False
        self.concurrent_sessions_max = len(rt.sessions) + 1
        self.rope_deltas: Any = None
        self.run_id = str(command["run_id"])
        self.request_id = str(command["request_id"])
        run_id = self.run_id
        self.settings = settings = dict(command["sampling"])
        self.command = command
        self.device = device = resident.input_device
        instrumentation = str(command.get("instrumentation", "token"))
        self.detailed_metrics = instrumentation in {"token", "full", "expert"}
        self.synchronize = instrumentation in {"full", "expert"} and device.startswith("cuda")
        self.encoder_decoder = encoder_decoder = (
            rt.model_info.get("task") == "encoder_decoder_generation"
        )
        self.attention_capture_requested = instrumentation in {"full", "expert"}
        self.attention_capture_active = False
        self.attention_capture_warning_emitted = False
        self.captured_attention_token_count = 0
        # The attention kernel this session runs with; the scheduler re-selects it
        # when another session on the same resident switched the kernel.
        self.attention_implementation: str | None = resident.loaded_attention_implementation
        if self.attention_capture_requested and not encoder_decoder:
            self.attention_capture_active = rt._select_attention_implementation("eager")
            if self.attention_capture_active:
                self.attention_implementation = "eager"
            else:
                rt._emit_run(
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
                self.attention_capture_warning_emitted = True
        elif self.attention_capture_requested:
            rt._emit_run(
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
            self.attention_capture_warning_emitted = True
        else:
            rt._restore_attention_implementation()
        self.deterministic_reference_mode = bool(command.get("deterministic_reference_mode"))
        # Set this on every request.  torch's switch is process-global, so merely
        # enabling it for reference runs would silently affect every later run.
        torch.use_deterministic_algorithms(
            self.deterministic_reference_mode,
            warn_only=self.deterministic_reference_mode,
        )
        rt._deterministic_state = self.deterministic_reference_mode
        self.cudnn = cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            # The reference worker keeps autotuning off for both modes.  This is
            # stable across requests and is reported verbatim below.
            cudnn.benchmark = False
        messages = list(command["messages"])
        if (
            any(item.get("role") == "system" for item in messages)
            and "deepseek" in rt.model_info["display_name"].lower()
        ):
            rt._emit_run(
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
        media_requested = any(item.get("attachments") for item in messages)
        if media_requested and (rt.processor is None or encoder_decoder):
            raise ValueError("the loaded model has no chat media processor")
        # Processors render the same chat templates from content-part messages,
        # so media conversations reuse the tokenizer rendering path unchanged.
        rendered_prompt, prompt_renderer = _render_messages(
            rt.processor if media_requested else rt.tokenizer,
            _processor_messages(messages) if media_requested else messages,
            reasoning=reasoning_requested,
            reasoning_delimiters=rt.model_info.get("reasoning_delimiters"),
        )
        template_ended = time.monotonic_ns()
        if prompt_renderer == "chat_template_system_merged":
            rt._emit_run(
                run_id,
                "warning",
                {
                    "code": "system_prompt_merged",
                    "message": (
                        "This model's chat template has no system role; the system prompt "
                        "was prepended to the first user message."
                    ),
                },
            )
        if prompt_renderer == "plain_text_fallback":
            rt._emit_run(
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
        if encoder_decoder and rt.decoder_start_token_id is None:
            (
                rt.decoder_start_token_id,
                rt.decoder_start_token_source,
                rt.decoder_start_warning,
            ) = _decoder_start_token(rt.model, rt.tokenizer)
        if encoder_decoder and rt.decoder_start_warning:
            rt._emit_run(
                run_id,
                "warning",
                {
                    "code": "decoder_start_token_fallback",
                    "message": rt.decoder_start_warning,
                    "source": rt.decoder_start_token_source,
                },
            )
        self.decoder_start_token_id: int | None = rt.decoder_start_token_id
        tokenization_started = time.monotonic_ns()
        media_inputs: dict[str, Any] = {}
        media_summary: dict[str, Any] | None = None
        if media_requested:
            images, videos, video_metadata = _load_media(rt.processor, messages)
            processor_kwargs: dict[str, Any] = {}
            if images:
                processor_kwargs["images"] = images
            if videos:
                processor_kwargs.update(
                    videos=videos, video_metadata=video_metadata, do_sample_frames=False
                )
            encoded = rt.processor(
                text=[rendered_prompt],
                return_tensors="pt",
                add_special_tokens=False,
                **processor_kwargs,
            )
            media_inputs = {
                key: value
                for key, value in encoded.items()
                if key not in {"input_ids", "attention_mask"}
            }
            media_summary = {"images": len(images), "videos": len(videos)}
        else:
            encoded = rt.tokenizer(
                rendered_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        fallback_bos = (
            _default_bos_token_id(rt.tokenizer)
            if prompt_renderer == "plain_text_fallback"
            else None
        )
        if fallback_bos is not None and input_ids[0].tolist()[:1] != [fallback_bos]:
            input_ids = torch.cat(
                [torch.tensor([[fallback_bos]], dtype=input_ids.dtype), input_ids], dim=-1
            )
            attention_mask = torch.cat(
                [torch.ones((1, 1), dtype=attention_mask.dtype), attention_mask], dim=-1
            )
        self.input_ids = input_ids = input_ids.to(device)
        self.attention_mask = attention_mask.to(device)
        self.media_inputs = media_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in media_inputs.items()
        }
        tokenization_ended = time.monotonic_ns()
        self.prompt_tokens = prompt_tokens = int(input_ids.shape[-1])
        media_placeholders = {
            int(token_id): kind
            for kind, token_id in (
                ("image", getattr(rt.model.config, "image_token_id", None)),
                ("video", getattr(rt.model.config, "video_token_id", None)),
            )
            if media_summary is not None and isinstance(token_id, int)
        }
        if media_summary is not None:
            prompt_ids = input_ids[0].tolist()
            media_summary["placeholder_tokens"] = sum(
                1 for token_id in prompt_ids if token_id in media_placeholders
            )
        context_limit = int(rt.model_info.get("effective_context_limit") or 4096)
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
        self.forced_prefix_token_ids = forced_prefix_token_ids = [
            int(item) for item in command.get("forced_prefix_token_ids", ())
        ]
        available_output_tokens = (
            max(0, context_limit - 1) if encoder_decoder else max(0, context_limit - prompt_tokens)
        )
        self.configured_max_output_tokens = configured_max_output_tokens = int(
            settings["max_output_tokens"]
        )
        if len(forced_prefix_token_ids) > configured_max_output_tokens:
            raise ValueError("forced generation prefix exceeds the configured output limit")
        if len(forced_prefix_token_ids) > available_output_tokens:
            raise ValueError("forced generation prefix exceeds the available model context")
        self.max_new_tokens = max_new_tokens = min(
            configured_max_output_tokens,
            available_output_tokens,
        )
        if max_new_tokens <= 0:
            raise ValueError("rendered prompt consumes the effective context limit")
        # Decode the prompt-token catalogue only after every bounded context
        # check succeeds. Rejected oversized prompts must not trigger an
        # unbounded device-to-CPU copy plus one tokenizer call per token.
        self.prompt_context_tokens = prompt_context_tokens = (
            [
                _token_source(
                    rt.tokenizer,
                    int(token_id),
                    context_index,
                    source_kind="prompt",
                )
                for context_index, token_id in enumerate(input_ids[0].tolist())
            ]
            if self.attention_capture_active
            else []
        )
        if prompt_context_tokens and media_placeholders:
            # Media positions stay `prompt` sources for older readers; the extra
            # `media` field says which attached image or video they carry.
            labels = _media_labels(
                [int(item["token_id"]) for item in prompt_context_tokens],
                media_placeholders,
                _media_token_counts(rt.processor, media_inputs),
            )
            for source, label in zip(prompt_context_tokens, labels, strict=True):
                if label is not None:
                    source["media"] = label

        configured_delimiters = rt.model_info.get("reasoning_delimiters")
        self.tagged_reasoning_enabled = (
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
        self.reasoning_answer_allowance = reasoning_answer_allowance = (
            min(
                configured_max_output_tokens,
                max(0, available_output_tokens - max_new_tokens),
            )
            if self.tagged_reasoning_enabled
            else 0
        )
        self.generation_token_limit = generation_token_limit = (
            max_new_tokens + reasoning_answer_allowance
        )

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(command["effective_seed"]))
        self.generated: list[int] = []
        self.context_tokens: list[dict[str, Any]] = list(prompt_context_tokens)
        self.full_ids = (
            [self.decoder_start_token_id]
            if encoder_decoder and self.decoder_start_token_id is not None
            else input_ids[0].tolist()
        )
        self.previous_text = ""
        self.cumulative_logprob = 0.0
        self.raw_segment_logprobs: dict[str, list[float]] = {
            "reasoning": [],
            "answer": [],
            "unknown": [],
        }
        self.segment_token_counts = {"reasoning": 0, "answer": 0, "unknown": 0}
        self.reasoning_segmenter, reasoning_primed = _reasoning_segmenter(
            rt.model_info, rendered_prompt
        )
        self.pending_token_events: dict[int, dict[str, Any]] = {}
        self.reasoning_observed = reasoning_primed
        self.visible_answer_observed = False
        self.eos_set: set[int] = set()
        self.token_index = 0
        self.finish_reason = "length"
        self.reasoning_answer_allowance_active = False
        self.stop_sequences = tuple(settings.get("stop_sequences", ()))
        self.previous_emitted_ns: int | None = None
        self.emission_times_ns: list[int] = []
        self.logits: Any = None
        self.past: Any = None
        self.current_attentions: Any = None
        self.encoder_outputs: Any = None
        self.decoder_attention_mask: Any = None

        self.prefill_started = time.monotonic_ns()
        rt._emit_run(
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
                "decoder_start_token_source": rt.decoder_start_token_source
                if encoder_decoder
                else None,
                "reasoning_primed": reasoning_primed,
                "reasoning_requested": reasoning_requested,
                "configured_max_output_tokens": configured_max_output_tokens,
                "generation_token_limit": generation_token_limit,
                "reasoning_answer_allowance": reasoning_answer_allowance,
                "attention_capture_requested": self.attention_capture_requested,
                "attention_capture_active": self.attention_capture_active,
                "attention_capture_method": (
                    "mean_causal_self_attention" if self.attention_capture_active else None
                ),
                "attention_source_limit": (
                    _ATTENTION_SOURCE_LIMIT if self.attention_capture_active else None
                ),
                "attention_implementation": (
                    getattr(rt.model.config, "_attn_implementation", None)
                    if self.attention_capture_active
                    else rt.loaded_attention_implementation
                ),
                "template_ms": (template_ended - template_started) / 1e6,
                "tokenization_ms": (tokenization_ended - tokenization_started) / 1e6,
                "media": media_summary,
                "model_key": resident.key,
                "placement": resident.placement,
                "quantization": resident.quantization,
                "concurrent_sessions": self.concurrent_sessions_max,
            },
        )

    def _save_rope_state(self) -> None:
        owner = _rope_state_owner(self.rt.model)
        if owner is not None:
            self.rope_deltas = owner.rope_deltas

    def prefill(self) -> None:
        rt = self.rt
        torch = self.torch
        run_id = self.run_id
        input_ids = self.input_ids
        attention_mask = self.attention_mask
        media_inputs = self.media_inputs
        prefill_started = self.prefill_started
        if self.synchronize:
            torch.cuda.synchronize()
        with torch.inference_mode():
            if self.encoder_decoder:
                self.encoder_outputs = rt._encode_source(input_ids, attention_mask)
                decoder_input_ids = torch.tensor(
                    [[self.decoder_start_token_id]],
                    device=self.device,
                    dtype=torch.long,
                )
                self.decoder_attention_mask = torch.ones_like(decoder_input_ids)
                self.logits, self.past, self.current_attentions = rt._forward_encoder_decoder(
                    decoder_input_ids,
                    self.decoder_attention_mask,
                    self.encoder_outputs,
                    attention_mask,
                )
            elif self.attention_capture_active and self.prompt_tokens > 1:
                # A normal prompt prefill would materialize an O(context^2)
                # attention matrix. Prefill all but the final prompt token into
                # the KV cache with the configured efficient kernel, then ask
                # eager attention for the single final query row that actually
                # produces generated token zero.
                rt._restore_attention_implementation()
                _, prefix_past, _ = rt._forward_last(
                    torch,
                    input_ids[:, :-1],
                    attention_mask[:, :-1],
                    media=media_inputs,
                )
                self.attention_capture_active = rt._select_attention_implementation("eager")
                self.attention_implementation = (
                    "eager"
                    if self.attention_capture_active
                    else self.resident.loaded_attention_implementation
                )
                if not self.attention_capture_active and not self.attention_capture_warning_emitted:
                    rt._emit_run(
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
                    self.attention_capture_warning_emitted = True
                self.logits, self.past, self.current_attentions = rt._forward_last(
                    torch,
                    input_ids[:, -1:],
                    attention_mask,
                    prefix_past,
                    capture_attention=self.attention_capture_active,
                )
            else:
                self.logits, self.past, self.current_attentions = rt._forward_last(
                    torch,
                    input_ids,
                    attention_mask,
                    capture_attention=self.attention_capture_active,
                    media=media_inputs,
                )
        self._save_rope_state()
        media_inputs.clear()  # Pixel tensors are consumed by prefill; free them now.
        if self.synchronize:
            torch.cuda.synchronize()
        prefill_ended = time.monotonic_ns()
        self.prefill_ms = prefill_ms = (prefill_ended - prefill_started) / 1e6
        # The prompt forward produces the distribution for generated token 0.
        # Attribute that work to token 0 so the first-token point includes
        # prefill; subsequent token points receive their own decode forward.
        self.forward_ms = prefill_ms
        self.generation_started = prefill_started
        self.eos_set = _eos_token_ids(rt.model, rt.tokenizer)
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
        rt._emit_run(
            run_id,
            "metric",
            {
                "sampling_operation_order": operation_order,
                "effective_seed": str(self.command["effective_seed"]),
                "rng_algorithm": "torch.Generator",
                "generator_device": self.device,
                "seed_affected_token_selection": self.settings["temperature"] > 0,
                "deterministic_reference_mode": self.deterministic_reference_mode,
                "torch_use_deterministic_algorithms": self.deterministic_reference_mode,
                "cudnn_benchmark": False if self.cudnn is not None else None,
                "prefill_ms": prefill_ms,
                "prompt_tokens_per_second": self.prompt_tokens
                / max((prefill_ended - prefill_started) / 1e9, 1e-9),
            },
        )

    def _emit_segmented_tokens(self, tokens: Sequence[SegmentedToken]) -> None:
        for token in tokens:
            payload = self.pending_token_events.pop(token.token_index)
            token_segment = token.classification.value
            payload["segment"] = token_segment
            payload["reasoning_slices"] = [item.model_dump(mode="json") for item in token.slices]
            self.segment_token_counts[token_segment] += 1
            raw_logprob = payload.get("raw_logprob")
            if isinstance(raw_logprob, float):
                self.raw_segment_logprobs[token_segment].append(raw_logprob)
            if any(item.classification is SegmentClass.REASONING for item in token.slices):
                self.reasoning_observed = True
            if payload["token_id"] not in self.eos_set and _contains_visible_answer(token):
                self.visible_answer_observed = True
            self.rt._emit_run(self.run_id, "token", payload)

    def step(self) -> bool:
        """Emit one token and prepare the next; False once the run has ended."""

        if self.finished:
            return False
        rt = self.rt
        torch = self.torch
        run_id = self.run_id
        settings = self.settings
        detailed_metrics = self.detailed_metrics
        token_index = self.token_index
        if token_index >= self.generation_token_limit:
            return self._finish()
        if rt.cancel_event.is_set() or self.cancel_requested:
            self.finish_reason = "cancelled"
            return self._finish()
        sample_started = time.monotonic_ns()
        raw = self.logits[0].float()
        processed = rt._apply_penalties(
            torch,
            raw,
            self.full_ids,
            float(settings["repetition_penalty"]),
            float(settings["frequency_penalty"]),
            float(settings["presence_penalty"]),
        )
        forced_token_id = (
            self.forced_prefix_token_ids[token_index]
            if token_index < len(self.forced_prefix_token_ids)
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
            filtered, filters = rt._filter_distribution(
                torch,
                processed,
                temperature=temperature,
                top_k=int(settings["top_k"]),
                top_p=float(settings["top_p"]),
                min_p=float(settings["min_p"]),
            )
            sampler_log_probs = torch.log_softmax(filtered, dim=-1)
            sampler_probs = torch.exp(sampler_log_probs)
            sampled_id = int(torch.multinomial(sampler_probs, 1, generator=self.generator).item())
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
            self.cumulative_logprob += raw_logprob_value
            running_perplexity: float | None = math.exp(
                min(700.0, -self.cumulative_logprob / (token_index + 1))
            )
        else:
            raw_log_probs = None
            raw_logprob = None
            raw_probability = None
            raw_rank = None
            running_perplexity = None
        self.generated.append(chosen_id)
        self.full_ids.append(chosen_id)
        piece = str(rt.tokenizer.convert_ids_to_tokens(chosen_id))
        current_text = rt.tokenizer.decode(
            self.generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        common = 0
        for left, right in zip(self.previous_text, current_text, strict=False):
            if left != right:
                break
            common += 1
        display_text = current_text[common:]
        attention_attribution: dict[str, Any] | None = None
        if self.attention_capture_active:
            attention_attribution, capture_error = _mean_causal_self_attention(
                torch,
                self.current_attentions,
                self.context_tokens,
            )
            if attention_attribution is None:
                self.attention_capture_active = False
                if not self.attention_capture_warning_emitted:
                    rt._emit_run(
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
                    self.attention_capture_warning_emitted = True
            elif token_index == 0:
                # The complete prompt catalogue is stored once. Subsequent
                # generated entries are reconstructed from the run's token
                # stream using prompt_token_count + generated_token_index.
                attention_attribution["context_tokens"] = self.prompt_context_tokens
            if attention_attribution is not None:
                self.captured_attention_token_count += 1
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
                        "piece": str(rt.tokenizer.convert_ids_to_tokens(candidate)),
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
                            "piece": str(rt.tokenizer.convert_ids_to_tokens(candidate)),
                            "logit": float(filtered[candidate].item()),
                            "log_probability": float(candidate_logprob),
                            "probability": float(math.exp(candidate_logprob)),
                            "survived_filter": True,
                        }
                    )
        sample_ended = time.monotonic_ns()
        emitted_ns = time.monotonic_ns()
        previous_emitted_ns = self.previous_emitted_ns
        inter_token_ms = (
            (emitted_ns - previous_emitted_ns) / 1e6 if previous_emitted_ns is not None else None
        )
        cumulative_ms = (emitted_ns - self.generation_started) / 1e6
        instantaneous_tps = (
            1000.0 / inter_token_ms if inter_token_ms and inter_token_ms > 0 else None
        )
        self.emission_times_ns.append(emitted_ns)
        rolling_window = self.emission_times_ns[-10:]
        if len(rolling_window) == 1:
            rolling_tps = 1.0 / max((emitted_ns - self.generation_started) / 1e9, 1e-9)
        else:
            rolling_tps = (len(rolling_window) - 1) / max(
                (rolling_window[-1] - rolling_window[0]) / 1e9,
                1e-9,
            )
        self.pending_token_events[token_index] = {
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
            "cumulative_logprob": self.cumulative_logprob if detailed_metrics else None,
            "running_perplexity": running_perplexity,
            "alternatives": alternatives,
            "decode_ms": self.forward_ms,
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
        self._emit_segmented_tokens(self.reasoning_segmenter.feed(token_index, display_text))
        self.previous_emitted_ns = emitted_ns
        self.previous_text = current_text
        if self.attention_capture_active:
            self.context_tokens.append(
                _token_source(
                    rt.tokenizer,
                    chosen_id,
                    self.prompt_tokens + token_index,
                    source_kind="generated",
                    generated_token_index=token_index,
                    display_text=display_text,
                )
            )
        if chosen_id in self.eos_set:
            self.finish_reason = "eos"
            return self._finish()
        if self.stop_sequences and any(current_text.endswith(stop) for stop in self.stop_sequences):
            self.finish_reason = "stop_sequence"
            return self._finish()
        generated_count = token_index + 1
        if generated_count >= self.max_new_tokens and not self.reasoning_answer_allowance_active:
            reasoning_without_answer = (
                self.tagged_reasoning_enabled
                and self.reasoning_observed
                and isinstance(self.reasoning_segmenter, TagReasoningSegmenter)
                and not self.visible_answer_observed
            )
            if self.reasoning_answer_allowance and reasoning_without_answer:
                self.reasoning_answer_allowance_active = True
                rt._emit_run(
                    run_id,
                    "warning",
                    {
                        "code": "reasoning_answer_allowance_activated",
                        "message": (
                            "Reasoning consumed the configured output limit before visible "
                            "answer text; generation is continuing within a bounded answer "
                            "allowance."
                        ),
                        "configured_max_output_tokens": self.configured_max_output_tokens,
                        "reasoning_answer_allowance": self.reasoning_answer_allowance,
                    },
                )
            else:
                if reasoning_without_answer:
                    rt._emit_run(
                        run_id,
                        "warning",
                        {
                            "code": "reasoning_answer_allowance_unavailable",
                            "message": (
                                "Reasoning consumed the output budget and the model context "
                                "has no remaining capacity for an answer."
                            ),
                            "configured_max_output_tokens": self.configured_max_output_tokens,
                        },
                    )
                return self._finish()
        if generated_count >= self.generation_token_limit:
            return self._finish()

        device = self.device
        selected_tensor = torch.tensor([[chosen_id]], device=device)
        if self.encoder_decoder:
            if self.decoder_attention_mask is None or self.encoder_outputs is None:
                raise RuntimeError("encoder-decoder prefill state was not initialized")
            self.decoder_attention_mask = torch.cat(
                [
                    self.decoder_attention_mask,
                    torch.ones(
                        (1, 1),
                        device=device,
                        dtype=self.decoder_attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
        else:
            self.attention_mask = torch.cat(
                [
                    self.attention_mask,
                    torch.ones((1, 1), device=device, dtype=self.attention_mask.dtype),
                ],
                dim=-1,
            )
        if self.synchronize:
            torch.cuda.synchronize()
        decode_started = time.monotonic_ns()
        with torch.inference_mode():
            if self.encoder_decoder:
                self.logits, self.past, self.current_attentions = rt._forward_encoder_decoder(
                    selected_tensor,
                    self.decoder_attention_mask,
                    self.encoder_outputs,
                    self.attention_mask,
                    self.past,
                )
            else:
                self.logits, self.past, self.current_attentions = rt._forward_last(
                    torch,
                    selected_tensor,
                    self.attention_mask,
                    self.past,
                    capture_attention=self.attention_capture_active,
                )
        self._save_rope_state()
        if self.synchronize:
            torch.cuda.synchronize()
        decode_ended = time.monotonic_ns()
        self.forward_ms = (decode_ended - decode_started) / 1e6
        self.token_index += 1
        return True

    def _finish(self) -> bool:
        """Emit the terminal event and reply, then drop every tensor this run holds."""

        rt = self.rt
        run_id = self.run_id
        self._emit_segmented_tokens(self.reasoning_segmenter.finalize())
        if self.pending_token_events:
            raise RuntimeError("reasoning segmenter did not finalize every generated token")
        for warning in getattr(self.reasoning_segmenter, "warnings", ()):
            rt._emit_run(
                run_id,
                "warning",
                {"code": "reasoning_segmentation_warning", "message": warning},
            )
        if self.reasoning_answer_allowance_active and not self.visible_answer_observed:
            rt._emit_run(
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

        generated = self.generated
        emission_times_ns = self.emission_times_ns
        generation_started = self.generation_started
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
        for name, segment_logprobs in self.raw_segment_logprobs.items():
            segment_metrics[name] = {
                "token_count": self.segment_token_counts[name],
                "perplexity": math.exp(min(700.0, -sum(segment_logprobs) / len(segment_logprobs)))
                if segment_logprobs
                else None,
                "mean_raw_logprob": sum(segment_logprobs) / len(segment_logprobs)
                if segment_logprobs
                else None,
            }
        status = "cancelled" if self.finish_reason == "cancelled" else "completed"
        # Drop the cache and activations before measuring memory for the payload.
        self._release_tensors()
        payload = {
            "finish_reason": self.finish_reason,
            "generated_token_count": len(generated),
            "configured_max_output_tokens": self.configured_max_output_tokens,
            "reasoning_answer_allowance": self.reasoning_answer_allowance,
            "reasoning_answer_allowance_used": max(0, len(generated) - self.max_new_tokens),
            "text": self.previous_text,
            "conditional_response_perplexity": math.exp(
                min(700.0, -self.cumulative_logprob / len(generated))
            )
            if generated and self.detailed_metrics
            else None,
            "segment_metrics": segment_metrics,
            "total_generation_ms": (completed_ns - generation_started) / 1e6,
            "engine_ttft_ms": (
                (emission_times_ns[0] - generation_started) / 1e6 if emission_times_ns else None
            ),
            "prefill_ms": self.prefill_ms,
            "decode_tokens_per_second": decode_tokens_per_second,
            "end_to_end_tokens_per_second": len(generated) / total_generation_seconds,
            "memory": rt._memory_snapshot(self.torch),
            "expert_routing": {"state": "not_applicable", "reason": "dense model"},
            "attention_capture": {
                "requested": self.attention_capture_requested,
                "method": "mean_causal_self_attention"
                if self.captured_attention_token_count
                else None,
                "captured_token_count": self.captured_attention_token_count,
                "semantics": "attention_weights_not_causal_contributions",
                "scope": "decoder_step_context_attention_independent_of_sampled_candidate",
            },
            "ledger": rt._ledger(),
            "scheduling": {
                "concurrent_sessions_max": self.concurrent_sessions_max,
                "interleaved": self.concurrent_sessions_max > 1,
            },
        }
        self.finished = True
        rt._emit_run(run_id, status, payload)
        _send(rt.output, "reply", request_id=self.request_id, ok=True, payload=payload)
        return False

    def _release_tensors(self) -> None:
        self.logits = self.past = self.current_attentions = None
        self.encoder_outputs = self.decoder_attention_mask = None
        self.input_ids = self.attention_mask = None
        self.media_inputs = {}


def worker_main(commands: Queue[Any], output: Queue[Any], cancel_event: Any) -> None:
    # The parent coordinates graceful cancellation and shutdown. Ignoring the
    # console interrupt here prevents a duplicate child traceback on Windows.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    WorkerRuntime(commands, output, cancel_event).run()
