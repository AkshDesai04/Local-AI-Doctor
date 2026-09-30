"""Generation and embedding orchestration with durable partial results."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..api.schemas import (
    EmbeddingRunCreate,
    GenerationRunCreate,
    PromptScoreRequest,
    SamplingRequest,
    TokenBranchCreate,
)
from ..config import AppSettings, DeviceMode, DType, InstrumentationLevel, Quantization
from ..domain.capabilities import Capability, CapabilityState
from ..domain.models import ModelDescriptor, ModelTask
from ..errors import (
    CapabilityUnavailableError,
    ChatNotFoundError,
    InvalidRequestError,
    RunNotFoundError,
    WorkbenchError,
)
from ..persistence import TelemetryWriter, WorkspaceRepository
from ..workers import InferenceReservation, ModelWorkerSupervisor, WorkerFailure
from .events import EventBroker
from .models import ModelRegistry
from .uploads import UploadStore

logger = logging.getLogger(__name__)

TOKEN_INSERT_SQL = """
INSERT OR REPLACE INTO token_events(
    run_id, token_index, token_id, piece, escaped_bytes, display_text,
    span_start, span_end, raw_logit, raw_logprob, raw_probability, raw_rank,
    processed_logit, sample_logprob, sample_probability, entropy, surprise,
    cumulative_logprob, running_perplexity, decode_ms, sample_ms, emit_ms,
    inter_token_ms, cumulative_ms, instantaneous_tps, rolling_tps, segment,
    reasoning_slices_json, selected_experts_json, attention_attribution_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

ALTERNATIVE_INSERT_SQL = """
INSERT OR REPLACE INTO token_alternatives(
    run_id, token_index, distribution, rank, token_id, piece, logit,
    log_probability, probability, survived_filter
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


_GENERATION_ROLES = frozenset({"system", "user", "assistant", "tool"})
_HISTORICAL_MESSAGE_STATUSES = frozenset({"complete", "cancelled", "failed"})
_CHAT_MEDIA_CAPABILITIES = {"image": Capability.VISION, "video": Capability.VIDEO}


def _generation_messages(
    lineage: list[dict[str, Any]],
    media: Mapping[str, list[dict[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in lineage:
        if (
            message["role"] not in _GENERATION_ROLES
            or message["status"] not in _HISTORICAL_MESSAGE_STATUSES
        ):
            continue
        entry: dict[str, Any] = {"role": str(message["role"]), "content": str(message["content"])}
        attachments = (media or {}).get(str(message["id"]))
        if attachments:
            entry["attachments"] = attachments
        messages.append(entry)
    return messages


def _with_system_prompt(
    messages: list[dict[str, Any]], system_prompt: object
) -> tuple[list[dict[str, Any]], str | None]:
    """Prepend the chat-level system prompt unless the lineage already opens with one.

    Returns the messages plus the prompt that was actually applied (the run snapshot).
    """

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return messages, None
    if messages and messages[0]["role"] == "system":
        return messages, None
    return [{"role": "system", "content": system_prompt}, *messages], system_prompt


def _rendered_history_bytes(messages: list[dict[str, Any]]) -> int:
    """Measure the deterministic conversation envelope used by the fallback renderer."""

    if len(messages) == 1 and messages[0]["role"] == "user":
        rendered = messages[0]["content"]
    else:
        labels = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
        lines = [
            f"{labels.get(message['role'], message['role'].title() or 'User')}: "
            f"{message['content']}"
            for message in messages
        ]
        if messages and messages[-1]["role"] != "assistant":
            lines.append("Assistant:")
        rendered = "\n".join(lines)
    return len(rendered.encode("utf-8"))


def _effective_sampling(settings: AppSettings, request: SamplingRequest) -> dict[str, Any]:
    """Merge only caller-supplied fields over the centralized configured defaults."""

    effective = settings.inference.defaults.model_dump(mode="json")
    effective["stop_sequences"] = []
    effective.update(request.model_dump(mode="json", include=request.model_fields_set))
    return SamplingRequest.model_validate(effective).model_dump(mode="json")


def _effective_deterministic_mode(settings: AppSettings, request: GenerationRunCreate) -> bool:
    if "deterministic_reference_mode" in request.model_fields_set:
        return request.deterministic_reference_mode
    return settings.inference.deterministic_reference_mode


# Reproducibility fields that describe one execution, never copied from a source run.
_EXECUTION_FIELDS = ("placement", "model_key", "scheduling")


@dataclass
class RunContext:
    """Everything one generation run needs, plus its partial output while it streams."""

    run_id: str
    message_id: str
    model_id: str
    messages: list[dict[str, Any]]
    sampling: dict[str, Any]
    seed: int
    instrumentation: str
    deterministic: bool
    reasoning: bool | None
    forced_prefix: list[int]
    device: DeviceMode | None
    dtype: DType | None
    quantization: Quantization | None
    strict_vram: bool | None
    model_key: str | None = None
    stream: AsyncGenerator[dict[str, Any], None] | None = None
    output: str = ""
    token_count: int = 0
    persisted_token_count: int = 0
    persisted_trace_bytes: int = 0
    persistence_limit_reported: bool = False


def _recorded_placement(
    settings: AppSettings, recorded: Mapping[str, Any]
) -> tuple[Quantization, bool]:
    """Quantization and Strict VRAM a source run asked for, else the configured defaults."""

    try:
        quantization = Quantization(recorded.get("quantization") or settings.runtime.quantization)
    except ValueError:
        quantization = settings.runtime.quantization
    strict = recorded.get("strict_vram")
    return quantization, strict if isinstance(strict, bool) else settings.runtime.strict_vram


class RunManager:
    def __init__(
        self,
        settings: AppSettings,
        repository: WorkspaceRepository,
        registry: ModelRegistry,
        worker: ModelWorkerSupervisor,
        events: EventBroker,
        telemetry: TelemetryWriter,
        uploads: UploadStore,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.registry = registry
        self.worker = worker
        self.events = events
        self.telemetry = telemetry
        self.uploads = uploads
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._reservations: dict[str, InferenceReservation] = {}

    async def _chat_media(
        self, descriptor: ModelDescriptor, attachment_ids: Sequence[str]
    ) -> list[dict[str, str]]:
        """Resolve stored attachments the selected generator can read natively.

        Paths come only from the upload store, which confines them to its root; the
        worker decodes nothing else. Kinds without a usable capability are rejected
        before any chat or run state is written.
        """

        media: list[dict[str, str]] = []
        for attachment_id in attachment_ids:
            attachment, path = await self.uploads.resolve(attachment_id)
            kind = str(attachment["media_type"]).split("/", maxsplit=1)[0]
            capability = _CHAT_MEDIA_CAPABILITIES.get(kind)
            support = descriptor.capabilities.support(capability) if capability else None
            if support is None or support.state not in {
                CapabilityState.FULL,
                CapabilityState.PARTIAL,
            }:
                raise CapabilityUnavailableError(
                    "the selected generation adapter does not expose native media input",
                    hint="Choose a generation model whose vision or video capability is available, or remove the attachment.",
                    details={
                        "attachment_id": attachment_id,
                        "media_kind": kind,
                        "reason": support.reason
                        if support is not None
                        else "only image and video attachments can accompany a chat message",
                    },
                )
            media.append({"kind": kind, "path": str(path)})
        return media

    async def _lineage_media(
        self, descriptor: ModelDescriptor, lineage: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, str]]]:
        """Media attached to earlier messages, so re-rendered history keeps it."""

        linked = await self.repository.list_message_attachments(
            [str(message["id"]) for message in lineage]
        )
        return {
            message_id: await self._chat_media(
                descriptor, [str(item["id"]) for item in attachments]
            )
            for message_id, attachments in linked.items()
            if attachments
        }

    def _release_generation_reservation(
        self,
        run_id: str,
        reservation: InferenceReservation,
    ) -> None:
        self._tasks.pop(run_id, None)
        self._reservations.pop(run_id, None)
        self.worker.forget_run(run_id)
        reservation.release()

    async def create_generation(self, request: GenerationRunCreate) -> dict[str, Any]:
        descriptor = self.registry.get(request.model_id)
        if descriptor.task not in {
            ModelTask.TEXT_GENERATION,
            ModelTask.ENCODER_DECODER_GENERATION,
        }:
            raise CapabilityUnavailableError(
                "the selected model does not expose text generation",
                details={"model_id": request.model_id, "task": descriptor.task.value},
            )
        if len(request.attachment_ids) > self.settings.limits.attachment_count:
            raise CapabilityUnavailableError(
                "attachment count exceeds the configured per-request limit",
                details={"limit": self.settings.limits.attachment_count},
            )
        quantization = request.quantization or self.settings.runtime.quantization
        if quantization is not Quantization.NONE:
            raise CapabilityUnavailableError(
                "the requested model-weight quantization has no installed compatible adapter",
                hint="Use quantization none.",
                details={"quantization": quantization.value},
            )
        new_media = await self._chat_media(descriptor, request.attachment_ids)
        chat = await self.repository.get_chat(request.chat_id)
        if chat is None:
            raise ChatNotFoundError("chat not found", details={"chat_id": request.chat_id})
        lineage: list[dict[str, Any]] = []
        if request.parent_message_id is not None:
            lineage = await self.repository.get_message_lineage(request.parent_message_id)
            if not lineage:
                raise InvalidRequestError(
                    "parent message was not found",
                    details={"parent_message_id": request.parent_message_id},
                )
            if any(message["chat_id"] != request.chat_id for message in lineage):
                raise InvalidRequestError(
                    "parent message lineage must belong to the selected chat",
                    hint="Choose a parent message from the same chat.",
                    details={
                        "chat_id": request.chat_id,
                        "parent_message_id": request.parent_message_id,
                    },
                )
        messages = _generation_messages(lineage, await self._lineage_media(descriptor, lineage))
        user_turn: dict[str, Any] = {"role": "user", "content": request.prompt}
        if new_media:
            user_turn["attachments"] = new_media
        messages.append(user_turn)
        messages, system_prompt = _with_system_prompt(messages, chat.get("system_prompt"))
        rendered_bytes = _rendered_history_bytes(messages)
        if rendered_bytes > self.settings.limits.prompt_bytes:
            raise InvalidRequestError(
                "rendered conversation exceeds the configured prompt byte limit",
                hint="Shorten the selected branch or raise limits.prompt_bytes in local configuration.",
                details={
                    "maximum_bytes": self.settings.limits.prompt_bytes,
                    "rendered_history_bytes": rendered_bytes,
                    "message_count": len(messages),
                },
            )

        run_id = str(uuid.uuid4())
        reservation = self.registry.reserve_inference(run_id, "generation")
        self._reservations[run_id] = reservation
        try:
            return await self._create_generation_admitted(
                request,
                descriptor,
                chat,
                messages,
                system_prompt,
                run_id,
                reservation,
            )
        except BaseException:
            self._release_generation_reservation(run_id, reservation)
            raise

    async def _create_generation_admitted(
        self,
        request: GenerationRunCreate,
        descriptor: ModelDescriptor,
        chat: Mapping[str, Any],
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        run_id: str,
        reservation: InferenceReservation,
    ) -> dict[str, Any]:
        selection = self.registry.choose_hardware(device=request.device, dtype=request.dtype)
        effective_seed = request.seed if request.seed is not None else secrets.randbits(64)
        sampling = _effective_sampling(self.settings, request.sampling)
        instrumentation = request.instrumentation or self.settings.inference.instrumentation
        deterministic_reference_mode = _effective_deterministic_mode(self.settings, request)
        reasoning = request.reasoning
        quantization = request.quantization or self.settings.runtime.quantization
        strict_vram = (
            self.settings.runtime.strict_vram
            if request.strict_vram is None
            else request.strict_vram
        )
        reproducibility = {
            "requested_seed": None if request.seed is None else str(request.seed),
            "effective_seed": str(effective_seed),
            "rng_algorithm": "torch.Generator",
            "generator_device": selection.device_identifier,
            "sampling": sampling,
            "model_fingerprint": descriptor.fingerprint.model_dump(mode="json"),
            "backend": "transformers-reference-loop",
            "device": selection.model_dump(mode="json"),
            "dtype": selection.effective_dtype.value,
            "quantization": quantization.value,
            "strict_vram": strict_vram,
            "attention_implementation": self.settings.runtime.attention_backend.value,
            "deterministic_reference_mode": deterministic_reference_mode,
            "reasoning": reasoning,
            "deterministic_kernels": {
                "torch_use_deterministic_algorithms": deterministic_reference_mode,
                "cublas_workspace_config": ":4096:8" if deterministic_reference_mode else None,
                "cudnn_benchmark": False,
            },
            "seed_affected_token_selection": sampling["temperature"] > 0,
            "batching": {"batch_size": 1, "concurrent_runs": 1},
            "software_versions": (
                self.registry.hardware.software_versions if self.registry.hardware else {}
            ),
        }
        run_record = {
            "id": run_id,
            "model_id": descriptor.id,
            "kind": "generation",
            "status": "queued",
            "requested_seed": request.seed,
            "effective_seed": effective_seed,
            "rng_algorithm": "torch.Generator",
            "generator_device": selection.device_identifier,
            "settings": {
                "sampling": sampling,
                "instrumentation": instrumentation.value,
                "deterministic_reference_mode": deterministic_reference_mode,
                "reasoning": reasoning,
                "system_prompt": system_prompt,
                "quantization": quantization.value,
                "strict_vram": strict_vram,
            },
            "effective_config": self.settings.inference_snapshot(),
            "reproducibility": reproducibility,
            "model_fingerprint": descriptor.fingerprint.value,
        }
        hardware_snapshot = (
            self.registry.hardware.model_dump(mode="json") if self.registry.hardware else {}
        )
        title = " ".join(request.prompt.strip().split())[:60]
        setup = await self.repository.create_generation_setup(
            chat_id=request.chat_id,
            user_content=request.prompt,
            parent_message_id=request.parent_message_id,
            run=run_record,
            hardware=hardware_snapshot,
            software=hardware_snapshot.get("software_versions", {}),
            backend={
                "name": "transformers-reference-loop",
                "selection": selection.model_dump(mode="json"),
                "attention_implementation": self.settings.runtime.attention_backend.value,
                "quantization": quantization.value,
                "strict_vram": strict_vram,
            },
            chat_title=(title or "New chat") if chat["title"] == "New chat" else None,
            attachment_ids=request.attachment_ids,
        )
        user_message = setup["user_message"]
        assistant_message = setup["assistant_message"]
        run = setup["run"]
        task = asyncio.create_task(
            self._execute_generation(
                run_id=run_id,
                message_id=assistant_message["id"],
                model_id=descriptor.id,
                messages=messages,
                sampling=sampling,
                effective_seed=effective_seed,
                instrumentation=instrumentation.value,
                deterministic_reference_mode=deterministic_reference_mode,
                reasoning=reasoning,
                device=request.device,
                dtype=request.dtype,
                reservation=reservation,
                quantization=quantization,
                strict_vram=strict_vram,
            ),
            name=f"generation-{run_id}",
        )
        self._tasks[run_id] = task
        task.add_done_callback(
            lambda _task: self._release_generation_reservation(run_id, reservation)
        )
        return {"run": run, "user_message": user_message, "assistant_message": assistant_message}

    async def _persist_token(self, run_id: str, token: Mapping[str, Any]) -> None:
        now = _now()
        await self.telemetry.submit(
            TOKEN_INSERT_SQL,
            [
                (
                    run_id,
                    token["token_index"],
                    token["token_id"],
                    token["piece"],
                    token["escaped_bytes"],
                    token["display_text"],
                    token.get("span_start"),
                    token.get("span_end"),
                    token.get("raw_logit"),
                    token.get("raw_logprob"),
                    token.get("raw_probability"),
                    token.get("raw_rank"),
                    token.get("processed_logit"),
                    token.get("sample_logprob"),
                    token.get("sample_probability"),
                    token.get("entropy"),
                    token.get("surprise"),
                    token.get("cumulative_logprob"),
                    token.get("running_perplexity"),
                    token.get("decode_ms"),
                    token.get("sample_ms"),
                    token.get("emit_ms"),
                    token.get("inter_token_ms"),
                    token.get("cumulative_ms"),
                    token.get("instantaneous_tps"),
                    token.get("rolling_tps"),
                    token.get("segment", "unknown"),
                    json.dumps(token.get("reasoning_slices", [])),
                    json.dumps(token.get("expert_routing")),
                    json.dumps(token.get("attention_attribution")),
                    now,
                )
            ],
        )
        alternatives: list[tuple[Any, ...]] = []
        for distribution, values in token.get("alternatives", {}).items():
            for alternative in values:
                alternatives.append(
                    (
                        run_id,
                        token["token_index"],
                        distribution,
                        alternative["rank"],
                        alternative["token_id"],
                        alternative["piece"],
                        alternative.get("logit"),
                        alternative.get("log_probability"),
                        alternative.get("probability"),
                        int(bool(alternative.get("survived_filter"))),
                    )
                )
        await self.telemetry.submit(ALTERNATIVE_INSERT_SQL, alternatives)

    async def _execute_generation(
        self,
        *,
        run_id: str,
        message_id: str,
        model_id: str,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        effective_seed: int,
        instrumentation: str,
        deterministic_reference_mode: bool,
        device: Any,
        dtype: Any,
        reservation: InferenceReservation,
        forced_prefix_token_ids: list[int] | None = None,
        reasoning: bool | None = None,
        quantization: Quantization | None = None,
        strict_vram: bool | None = None,
    ) -> None:
        ctx = RunContext(
            run_id=run_id,
            message_id=message_id,
            model_id=model_id,
            messages=messages,
            sampling=sampling,
            seed=effective_seed,
            instrumentation=instrumentation,
            deterministic=deterministic_reference_mode,
            reasoning=reasoning,
            forced_prefix=list(forced_prefix_token_ids or ()),
            device=device,
            dtype=dtype,
            quantization=quantization,
            strict_vram=strict_vram,
        )
        try:
            await self.events.publish(
                run_id,
                "run_created",
                {"status": "queued", "effective_seed": str(effective_seed)},
            )
            await reservation.acquire()
            await self._load_side(ctx, reservation, frozenset())
            await self._stream_generation(ctx)
        except Exception as exc:
            await self._fail_run(ctx, exc)
        finally:
            # Close the worker stream before freeing the lease so a failure in the
            # loop above stops the worker at once instead of when the generator is
            # garbage collected.
            if ctx.stream is not None:
                with suppress(Exception):
                    await ctx.stream.aclose()
            reservation.release()

    async def _load_side(
        self,
        ctx: RunContext,
        reservation: InferenceReservation,
        pinned: frozenset[str],
    ) -> None:
        """Make the run's model resident and record where it actually landed."""

        await self.repository.update_run(ctx.run_id, status="loading", queue_exited_at=_now())
        await self.events.publish(ctx.run_id, "stage", {"stage": "model_loading"})
        loaded = await self.registry.load_reserved(
            reservation,
            ctx.model_id,
            device=ctx.device,
            dtype=ctx.dtype,
            quantization=ctx.quantization,
            strict_vram=ctx.strict_vram,
            pinned=pinned,
        )
        model_key = loaded.get("model_key")
        ctx.model_key = str(model_key) if model_key else None
        placement = {
            field: loaded.get(field)
            for field in ("model_key", "placement", "quantization", "strict_vram")
            if loaded.get(field) is not None
        }
        if placement:
            await self.repository.merge_run_reproducibility(ctx.run_id, placement)
        await self.events.publish(ctx.run_id, "model_loaded", loaded)

    async def _stream_generation(self, ctx: RunContext) -> None:
        """Run the worker stream, persisting partial output and telemetry as it arrives."""

        run_id = ctx.run_id
        message_id = ctx.message_id
        started = _now()
        await self.repository.update_run(run_id, status="running", started_at=started)
        await self.repository.update_message(message_id, content="", status="streaming")
        ctx.stream = self.worker.generate(
            run_id=run_id,
            messages=ctx.messages,
            sampling=ctx.sampling,
            effective_seed=ctx.seed,
            instrumentation=ctx.instrumentation,
            deterministic_reference_mode=ctx.deterministic,
            max_prompt_tokens=self.settings.inference.max_prompt_tokens,
            reserved_output_tokens=self.settings.inference.reserved_output_tokens,
            timeout_seconds=self.settings.workers.inference_timeout_seconds,
            forced_prefix_token_ids=ctx.forced_prefix,
            reasoning=ctx.reasoning,
            model_key=ctx.model_key,
        )
        async for worker_event in ctx.stream:
            event_type = str(worker_event["event_type"])
            payload = dict(worker_event.get("payload", {}))
            if event_type == "stage" and payload.get("stage") == "prefill":
                current_message = await self.repository.get_message(message_id)
                await self.repository.update_message(
                    message_id,
                    content=ctx.output,
                    status="streaming",
                    metadata={
                        **dict((current_message or {}).get("metadata") or {}),
                        "reasoning_primed": bool(payload.get("reasoning_primed")),
                    },
                )
                await self.repository.update_run(
                    run_id,
                    rendered_prompt=payload.get("rendered_prompt"),
                    prompt_token_count=payload.get("prompt_tokens"),
                )
                await self.repository.upsert_phase_metric(
                    run_id,
                    "chat_template",
                    duration_ms=payload.get("template_ms"),
                    details={"rendered_prompt_tokens": payload.get("prompt_tokens")},
                )
                await self.repository.upsert_phase_metric(
                    run_id,
                    "tokenization",
                    duration_ms=payload.get("tokenization_ms"),
                    details={
                        "prompt_tokens": payload.get("prompt_tokens"),
                        "context_limit": payload.get("context_limit"),
                        "attention_capture_requested": payload.get("attention_capture_requested"),
                        "attention_capture_active": payload.get("attention_capture_active"),
                        "attention_capture_method": payload.get("attention_capture_method"),
                        "attention_source_limit": payload.get("attention_source_limit"),
                        "attention_implementation": payload.get("attention_implementation"),
                    },
                )
            elif event_type == "metric" and payload.get("prefill_ms") is not None:
                await self.repository.upsert_phase_metric(
                    run_id,
                    "prefill",
                    duration_ms=payload.get("prefill_ms"),
                    details={
                        "prompt_tokens_per_second": payload.get("prompt_tokens_per_second"),
                        "sampling_operation_order": payload.get("sampling_operation_order"),
                    },
                )
            if event_type == "token":
                if ctx.token_count == 0:
                    await self.repository.update_run(run_id, first_token_at=_now())
                replace_from = int(payload.get("replace_from", len(ctx.output)))
                ctx.output = ctx.output[:replace_from] + str(payload.get("display_text", ""))
                ctx.token_count += 1
                if self.settings.telemetry.persist_token_events:
                    payload_bytes = len(
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                            "utf-8"
                        )
                    )
                    within_event_limit = (
                        ctx.persisted_token_count < self.settings.limits.telemetry_events_per_run
                    )
                    within_byte_limit = (
                        ctx.persisted_trace_bytes + payload_bytes
                        <= self.settings.limits.trace_bytes_per_run
                    )
                    if within_event_limit and within_byte_limit:
                        await self._persist_token(run_id, payload)
                        ctx.persisted_token_count += 1
                        ctx.persisted_trace_bytes += payload_bytes
                    elif not ctx.persistence_limit_reported:
                        ctx.persistence_limit_reported = True
                        await self.events.publish(
                            run_id,
                            "warning",
                            {
                                "code": "telemetry_persistence_truncated",
                                "message": "Per-run telemetry persistence reached a configured limit; live generation continues.",
                                "persisted_token_events": ctx.persisted_token_count,
                                "persisted_trace_bytes": ctx.persisted_trace_bytes,
                            },
                        )
                if ctx.token_count % 8 == 0:
                    await self.repository.update_message(
                        message_id, content=ctx.output, status="streaming"
                    )
            if event_type in {"completed", "cancelled"}:
                status = "complete" if event_type == "completed" else "cancelled"
                await self.repository.upsert_phase_metric(
                    run_id,
                    "generation",
                    duration_ms=payload.get("total_generation_ms"),
                    details={
                        "decode_tokens_per_second": payload.get("decode_tokens_per_second"),
                        "end_to_end_tokens_per_second": payload.get("end_to_end_tokens_per_second"),
                        "engine_ttft_ms": payload.get("engine_ttft_ms"),
                        "prefill_ms": payload.get("prefill_ms"),
                        "conditional_response_perplexity": payload.get(
                            "conditional_response_perplexity"
                        ),
                        "segment_metrics": payload.get("segment_metrics"),
                        "memory": payload.get("memory"),
                        "ledger": payload.get("ledger"),
                        "expert_routing": payload.get("expert_routing"),
                        "attention_capture": payload.get("attention_capture"),
                    },
                )
                if isinstance(payload.get("scheduling"), Mapping):
                    await self.repository.merge_run_reproducibility(
                        run_id, {"scheduling": payload["scheduling"]}
                    )
                # A terminal run status promises that every accepted token and
                # alternative is durable, including to polling API clients.
                await self.telemetry.flush()
                await self.repository.update_message(message_id, content=ctx.output, status=status)
                await self.repository.update_run(
                    run_id,
                    status=status,
                    generated_token_count=ctx.token_count,
                    finish_reason=payload.get("finish_reason"),
                    completed_at=_now(),
                )
            elif event_type == "error":
                await self.telemetry.flush()
                await self.repository.update_message(
                    message_id, content=ctx.output, status="failed"
                )
                await self.repository.update_run(
                    run_id,
                    status="failed",
                    generated_token_count=ctx.token_count,
                    error_code=payload.get("code"),
                    error_message=payload.get("message"),
                    completed_at=_now(),
                )
            await self.events.publish(run_id, event_type, payload)

    async def _fail_run(self, ctx: RunContext, exc: Exception) -> None:
        """Persist a failed run with its partial output and publish exactly one error."""

        logger.error("generation run %s failed (%s)", ctx.run_id, type(exc).__name__)
        if isinstance(exc, WorkerFailure):
            error = exc.error
        elif isinstance(exc, WorkbenchError):
            error = exc.to_dict()
        else:
            error = {
                "code": "internal_error",
                "message": "generation failed at an application boundary",
                "hint": "Review local application logs for the private diagnostic details.",
            }
        # Establish the same durability barrier used by normal terminal
        # events before making the failed state visible to polling clients.
        with suppress(Exception):
            await self.telemetry.flush()
        await self.repository.update_message(ctx.message_id, content=ctx.output, status="failed")
        await self.repository.update_run(
            ctx.run_id,
            status="failed",
            generated_token_count=ctx.token_count,
            error_code=error.get("code"),
            error_message=error.get("message"),
            completed_at=_now(),
        )
        # EventBroker fans out before surfacing an optional persistence
        # failure, so suppressing here avoids an unhandled background task
        # while connected clients still receive exactly one terminal error.
        with suppress(Exception):
            await self.events.publish(ctx.run_id, "error", error)

    async def cancel(self, run_id: str) -> bool:
        run = await self.repository.get_run(run_id)
        if run is None:
            raise RunNotFoundError("run not found", details={"run_id": run_id})
        if run["status"] not in {"queued", "loading", "running"}:
            return False
        reservation = self._reservations.get(run_id)
        if reservation is not None and reservation.cancel_if_waiting():
            task = self._tasks.get(run_id)
            if task is not None:
                task.cancel()
            await self._mark_cancelled_before_start(run_id, run)
            return True
        self.worker.cancel(run_id)
        return True

    async def _mark_cancelled_before_start(self, run_id: str, run: Mapping[str, Any]) -> None:
        message_id = run.get("message_id")
        if message_id is not None:
            await self.repository.update_message(str(message_id), content="", status="cancelled")
        await self.repository.update_run(
            run_id,
            status="cancelled",
            finish_reason="cancelled_before_start",
            completed_at=_now(),
        )
        await self.events.publish(
            run_id,
            "cancelled",
            {"finish_reason": "cancelled_before_start", "generated_token_count": 0},
        )

    async def replay_generation(self, parent_run_id: str) -> dict[str, Any]:
        """Replay a completed generation as a new assistant branch."""

        source = await self.repository.get_run(parent_run_id)
        if source is None:
            raise CapabilityUnavailableError(
                "the source run does not exist", details={"run_id": parent_run_id}
            )
        if source["kind"] != "generation" or source["status"] != "complete":
            raise CapabilityUnavailableError(
                "only completed generation runs can be replayed",
                details={"run_id": parent_run_id, "status": source["status"]},
            )
        if source.get("model_id") is None:
            raise CapabilityUnavailableError(
                "the recorded model is no longer registered",
                hint="Restore the original checkpoint and refresh the model registry.",
            )
        descriptor = self.registry.get(str(source["model_id"]))
        if source.get("model_fingerprint") != descriptor.fingerprint.value:
            raise CapabilityUnavailableError(
                "the recorded model fingerprint no longer matches the registered checkpoint",
                details={"run_id": parent_run_id, "model_id": descriptor.id},
            )

        source_message_id = source.get("message_id")
        source_message = (
            await self.repository.get_message(str(source_message_id))
            if source_message_id is not None
            else None
        )
        if source_message is None or source_message["role"] != "assistant":
            raise CapabilityUnavailableError(
                "the source generation is not attached to an assistant message"
            )
        branch_parent_id = source_message.get("parent_id")
        if branch_parent_id is None:
            raise CapabilityUnavailableError(
                "the source generation has no replayable parent message"
            )
        branch = await self.repository.get_message_lineage(str(branch_parent_id))
        if not branch or branch[-1]["role"] != "user":
            raise CapabilityUnavailableError(
                "the source generation branch does not end in a user message"
            )
        settings = dict(source.get("settings") or {})
        messages, system_prompt = _with_system_prompt(
            _generation_messages(branch, await self._lineage_media(descriptor, branch)),
            settings.get("system_prompt"),
        )
        settings["system_prompt"] = system_prompt
        rendered_bytes = _rendered_history_bytes(messages)
        if rendered_bytes > self.settings.limits.prompt_bytes:
            raise InvalidRequestError(
                "rendered conversation exceeds the configured prompt byte limit",
                hint="Shorten the selected branch or raise limits.prompt_bytes in local configuration.",
                details={
                    "maximum_bytes": self.settings.limits.prompt_bytes,
                    "rendered_history_bytes": rendered_bytes,
                    "message_count": len(messages),
                },
            )

        try:
            sampling = SamplingRequest.model_validate(settings.get("sampling", {})).model_dump(
                mode="json"
            )
            instrumentation = InstrumentationLevel(
                settings.get("instrumentation", self.settings.inference.instrumentation.value)
            )
            effective_seed = int(source["effective_seed"])
            if not 0 <= effective_seed <= 2**64 - 1:
                raise ValueError("seed is outside the unsigned 64-bit range")
        except (TypeError, ValueError) as exc:
            raise CapabilityUnavailableError(
                "the recorded generation settings are not replayable",
                details={"run_id": parent_run_id},
            ) from exc
        deterministic_reference_mode = bool(settings.get("deterministic_reference_mode", False))
        recorded_reasoning = settings.get("reasoning")
        reasoning = recorded_reasoning if isinstance(recorded_reasoning, bool) else None
        recorded_reproducibility = {
            key: value
            for key, value in dict(source.get("reproducibility") or {}).items()
            if key not in _EXECUTION_FIELDS
        }
        quantization, strict_vram = _recorded_placement(self.settings, settings)
        settings.update(quantization=quantization.value, strict_vram=strict_vram)
        recorded_device = recorded_reproducibility.get("device")
        device: DeviceMode | None = None
        if isinstance(recorded_device, Mapping):
            requested_device = recorded_device.get("requested")
            try:
                device = DeviceMode(requested_device) if isinstance(requested_device, str) else None
            except ValueError:
                device = None
        dtype: DType | None = None
        recorded_dtype = recorded_reproducibility.get("dtype")
        try:
            dtype = DType(recorded_dtype) if isinstance(recorded_dtype, str) else None
        except ValueError:
            dtype = None
        selection = self.registry.choose_hardware(device=device, dtype=dtype)

        run_id = str(uuid.uuid4())
        reservation = self.registry.reserve_inference(run_id, "generation_replay")
        self._reservations[run_id] = reservation
        try:
            return await self._replay_generation_admitted(
                parent_run_id=parent_run_id,
                source=source,
                descriptor=descriptor,
                source_message=source_message,
                branch_parent_id=str(branch_parent_id),
                messages=messages,
                settings=settings,
                sampling=sampling,
                instrumentation=instrumentation,
                effective_seed=effective_seed,
                deterministic_reference_mode=deterministic_reference_mode,
                reasoning=reasoning,
                recorded_reproducibility=recorded_reproducibility,
                quantization=quantization,
                strict_vram=strict_vram,
                selection=selection,
                device=device,
                dtype=dtype,
                run_id=run_id,
                reservation=reservation,
            )
        except BaseException:
            self._release_generation_reservation(run_id, reservation)
            raise

    async def _replay_generation_admitted(
        self,
        *,
        parent_run_id: str,
        source: Mapping[str, Any],
        descriptor: ModelDescriptor,
        source_message: Mapping[str, Any],
        branch_parent_id: str,
        messages: list[dict[str, Any]],
        settings: dict[str, Any],
        sampling: dict[str, Any],
        instrumentation: InstrumentationLevel,
        effective_seed: int,
        deterministic_reference_mode: bool,
        reasoning: bool | None,
        recorded_reproducibility: dict[str, Any],
        quantization: Quantization,
        strict_vram: bool,
        selection: Any,
        device: DeviceMode | None,
        dtype: DType | None,
        run_id: str,
        reservation: InferenceReservation,
    ) -> dict[str, Any]:

        assistant_message = await self.repository.create_message(
            chat_id=str(source_message["chat_id"]),
            role="assistant",
            content="",
            parent_id=branch_parent_id,
            status="pending",
            branch_index=await self.repository.next_branch_index(branch_parent_id),
            metadata={"replay_of_run_id": parent_run_id},
        )
        reproducibility = {
            **recorded_reproducibility,
            "reasoning": reasoning,
            "replayed_from_run_id": parent_run_id,
            "replayed_from_configuration_digest": (
                source.get("effective_config", {}).get("configuration_digest")
                if isinstance(source.get("effective_config"), Mapping)
                else None
            ),
            "effective_seed": str(effective_seed),
            "model_fingerprint": descriptor.fingerprint.model_dump(mode="json"),
            "device": selection.model_dump(mode="json"),
            "dtype": selection.effective_dtype.value,
            "quantization": quantization.value,
            "strict_vram": strict_vram,
            "software_versions": (
                self.registry.hardware.software_versions if self.registry.hardware else {}
            ),
        }
        run = await self.repository.create_run(
            {
                "id": run_id,
                "message_id": assistant_message["id"],
                "model_id": descriptor.id,
                "parent_run_id": parent_run_id,
                "kind": "generation",
                "status": "queued",
                "requested_seed": source.get("requested_seed"),
                "effective_seed": effective_seed,
                "rng_algorithm": source["rng_algorithm"],
                "generator_device": selection.device_identifier,
                "settings": {
                    **settings,
                    "sampling": sampling,
                    "instrumentation": instrumentation.value,
                    "deterministic_reference_mode": deterministic_reference_mode,
                    "reasoning": reasoning,
                },
                "effective_config": self.settings.inference_snapshot(),
                "reproducibility": reproducibility,
                "model_fingerprint": descriptor.fingerprint.value,
                "tokenizer_fingerprint": source.get("tokenizer_fingerprint"),
            }
        )
        hardware_snapshot = (
            self.registry.hardware.model_dump(mode="json") if self.registry.hardware else {}
        )
        await self.repository.save_environment_snapshot(
            run_id,
            hardware=hardware_snapshot,
            software=hardware_snapshot.get("software_versions", {}),
            backend={
                "name": "transformers-reference-loop",
                "selection": selection.model_dump(mode="json"),
                "attention_implementation": self.settings.runtime.attention_backend.value,
                "quantization": quantization.value,
                "strict_vram": strict_vram,
                "replayed_from_run_id": parent_run_id,
            },
        )
        task = asyncio.create_task(
            self._execute_generation(
                run_id=run_id,
                message_id=assistant_message["id"],
                model_id=descriptor.id,
                messages=messages,
                sampling=sampling,
                effective_seed=effective_seed,
                instrumentation=instrumentation.value,
                deterministic_reference_mode=deterministic_reference_mode,
                reasoning=reasoning,
                device=device,
                dtype=dtype,
                reservation=reservation,
                quantization=quantization,
                strict_vram=strict_vram,
            ),
            name=f"generation-replay-{run_id}",
        )
        self._tasks[run_id] = task
        task.add_done_callback(
            lambda _task: self._release_generation_reservation(run_id, reservation)
        )
        return {"run": run, "assistant_message": assistant_message}

    async def branch_generation(
        self, parent_run_id: str, request: TokenBranchCreate
    ) -> dict[str, Any]:
        """Continue a completed generation after replacing one token with a recorded alternative."""

        source = await self.repository.get_run(parent_run_id)
        if source is None:
            raise CapabilityUnavailableError(
                "the source run does not exist", details={"run_id": parent_run_id}
            )
        if source["kind"] != "generation" or source["status"] != "complete":
            raise CapabilityUnavailableError(
                "only completed generation runs can be branched",
                details={"run_id": parent_run_id, "status": source["status"]},
            )
        if source.get("model_id") is None:
            raise CapabilityUnavailableError(
                "the recorded model is no longer registered",
                hint="Restore the original checkpoint and refresh the model registry.",
            )
        descriptor = self.registry.get(str(source["model_id"]))
        if source.get("model_fingerprint") != descriptor.fingerprint.value:
            raise CapabilityUnavailableError(
                "the recorded model fingerprint no longer matches the registered checkpoint",
                details={"run_id": parent_run_id, "model_id": descriptor.id},
            )

        persisted_tokens = await self.repository.list_run_tokens(parent_run_id)
        prefix = [
            token for token in persisted_tokens if int(token["token_index"]) <= request.token_index
        ]
        if len(prefix) != request.token_index + 1 or any(
            int(token["token_index"]) != expected for expected, token in enumerate(prefix)
        ):
            raise InvalidRequestError(
                "the selected token does not have a complete persisted prefix",
                hint="Choose a token whose telemetry is still available.",
                details={"run_id": parent_run_id, "token_index": request.token_index},
            )
        selected = next(
            (
                alternative
                for alternative in prefix[-1].get("alternatives", [])
                if alternative["distribution"] == request.distribution
                and int(alternative["rank"]) == request.rank
                and int(alternative["token_id"]) == request.token_id
            ),
            None,
        )
        if selected is None:
            raise InvalidRequestError(
                "the selected token alternative is not present in persisted telemetry",
                hint="Refresh the run inspector and choose one of the recorded alternatives.",
                details={
                    "run_id": parent_run_id,
                    "token_index": request.token_index,
                    "distribution": request.distribution,
                    "rank": request.rank,
                    "token_id": request.token_id,
                },
            )
        forced_prefix_token_ids = [int(token["token_id"]) for token in prefix[:-1]]
        forced_prefix_token_ids.append(request.token_id)

        source_message_id = source.get("message_id")
        source_message = (
            await self.repository.get_message(str(source_message_id))
            if source_message_id is not None
            else None
        )
        if source_message is None or source_message["role"] != "assistant":
            raise CapabilityUnavailableError(
                "the source generation is not attached to an assistant message"
            )
        branch_parent_id = source_message.get("parent_id")
        if branch_parent_id is None:
            raise CapabilityUnavailableError(
                "the source generation has no branchable parent message"
            )
        lineage = await self.repository.get_message_lineage(str(branch_parent_id))
        if not lineage or lineage[-1]["role"] != "user":
            raise CapabilityUnavailableError(
                "the source generation branch does not end in a user message"
            )
        settings = dict(source.get("settings") or {})
        messages, system_prompt = _with_system_prompt(
            _generation_messages(lineage, await self._lineage_media(descriptor, lineage)),
            settings.get("system_prompt"),
        )
        settings["system_prompt"] = system_prompt
        rendered_bytes = _rendered_history_bytes(messages)
        if rendered_bytes > self.settings.limits.prompt_bytes:
            raise InvalidRequestError(
                "rendered conversation exceeds the configured prompt byte limit",
                hint="Shorten the selected branch or raise limits.prompt_bytes in local configuration.",
                details={
                    "maximum_bytes": self.settings.limits.prompt_bytes,
                    "rendered_history_bytes": rendered_bytes,
                    "message_count": len(messages),
                },
            )
        source_chat = await self.repository.get_chat(str(source_message["chat_id"]))
        if source_chat is None:
            raise CapabilityUnavailableError("the source generation chat no longer exists")

        try:
            sampling = SamplingRequest.model_validate(settings.get("sampling", {})).model_dump(
                mode="json"
            )
            instrumentation = InstrumentationLevel(
                settings.get("instrumentation", self.settings.inference.instrumentation.value)
            )
            effective_seed = int(source["effective_seed"])
            if not 0 <= effective_seed <= 2**64 - 1:
                raise ValueError("seed is outside the unsigned 64-bit range")
        except (TypeError, ValueError) as exc:
            raise CapabilityUnavailableError(
                "the recorded generation settings are not replayable",
                details={"run_id": parent_run_id},
            ) from exc
        deterministic_reference_mode = bool(settings.get("deterministic_reference_mode", False))
        recorded_reasoning = settings.get("reasoning")
        reasoning = recorded_reasoning if isinstance(recorded_reasoning, bool) else None
        recorded_reproducibility = {
            key: value
            for key, value in dict(source.get("reproducibility") or {}).items()
            if key not in _EXECUTION_FIELDS
        }
        quantization, strict_vram = _recorded_placement(self.settings, settings)
        settings.update(quantization=quantization.value, strict_vram=strict_vram)
        recorded_device = recorded_reproducibility.get("device")
        device: DeviceMode | None = None
        if isinstance(recorded_device, Mapping):
            requested_device = recorded_device.get("requested")
            try:
                device = DeviceMode(requested_device) if isinstance(requested_device, str) else None
            except ValueError:
                device = None
        dtype: DType | None = None
        recorded_dtype = recorded_reproducibility.get("dtype")
        try:
            dtype = DType(recorded_dtype) if isinstance(recorded_dtype, str) else None
        except ValueError:
            dtype = None
        selection = self.registry.choose_hardware(device=device, dtype=dtype)

        branch_details = {
            "source_run_id": parent_run_id,
            "source_token_index": request.token_index,
            "source_token_id": int(prefix[-1]["token_id"]),
            "selected_token_id": request.token_id,
            "selected_piece": str(selected["piece"]),
            "distribution": request.distribution,
            "rank": request.rank,
            "forced_prefix_token_count": len(forced_prefix_token_ids),
        }
        reproducibility = {
            **recorded_reproducibility,
            "branched_from": branch_details,
            "requested_seed": (
                None if source.get("requested_seed") is None else str(source["requested_seed"])
            ),
            "effective_seed": str(effective_seed),
            "rng_algorithm": source["rng_algorithm"],
            "generator_device": selection.device_identifier,
            "sampling": sampling,
            "model_fingerprint": descriptor.fingerprint.model_dump(mode="json"),
            "backend": "transformers-reference-loop",
            "device": selection.model_dump(mode="json"),
            "dtype": selection.effective_dtype.value,
            "quantization": quantization.value,
            "strict_vram": strict_vram,
            "attention_implementation": self.settings.runtime.attention_backend.value,
            "deterministic_reference_mode": deterministic_reference_mode,
            "reasoning": reasoning,
            "deterministic_kernels": {
                "torch_use_deterministic_algorithms": deterministic_reference_mode,
                "cublas_workspace_config": ":4096:8" if deterministic_reference_mode else None,
                "cudnn_benchmark": False,
            },
            "seed_affected_token_selection": sampling["temperature"] > 0,
            "batching": {"batch_size": 1, "concurrent_runs": 1},
            "software_versions": (
                self.registry.hardware.software_versions if self.registry.hardware else {}
            ),
        }
        run_id = str(uuid.uuid4())
        reservation = self.registry.reserve_inference(run_id, "generation_token_branch")
        self._reservations[run_id] = reservation
        try:
            hardware_snapshot = (
                self.registry.hardware.model_dump(mode="json") if self.registry.hardware else {}
            )
            source_title = str(source_chat.get("title") or "New chat")
            setup = await self.repository.create_token_branch_setup(
                title=f"{source_title} · Branch",
                lineage=lineage,
                run={
                    "id": run_id,
                    "model_id": descriptor.id,
                    # This run belongs to a self-contained new chat. Its source
                    # provenance lives in token_branch/branched_from so workspace
                    # exports do not contain a dangling cross-chat run parent.
                    "parent_run_id": None,
                    "kind": "generation",
                    "status": "queued",
                    "requested_seed": source.get("requested_seed"),
                    "effective_seed": effective_seed,
                    "rng_algorithm": source["rng_algorithm"],
                    "generator_device": selection.device_identifier,
                    "settings": {
                        **settings,
                        "sampling": sampling,
                        "instrumentation": instrumentation.value,
                        "deterministic_reference_mode": deterministic_reference_mode,
                        "reasoning": reasoning,
                        "token_branch": branch_details,
                    },
                    "effective_config": self.settings.inference_snapshot(),
                    "reproducibility": reproducibility,
                    "model_fingerprint": descriptor.fingerprint.value,
                    "tokenizer_fingerprint": source.get("tokenizer_fingerprint"),
                },
                assistant_metadata={"token_branch": branch_details},
                system_prompt=system_prompt,
                hardware=hardware_snapshot,
                software=hardware_snapshot.get("software_versions", {}),
                backend={
                    "name": "transformers-reference-loop",
                    "selection": selection.model_dump(mode="json"),
                    "attention_implementation": self.settings.runtime.attention_backend.value,
                    "quantization": quantization.value,
                    "strict_vram": strict_vram,
                    "token_branch": branch_details,
                },
            )
            assistant_message = setup["assistant_message"]
            task = asyncio.create_task(
                self._execute_generation(
                    run_id=run_id,
                    message_id=assistant_message["id"],
                    model_id=descriptor.id,
                    messages=messages,
                    sampling=sampling,
                    effective_seed=effective_seed,
                    instrumentation=instrumentation.value,
                    deterministic_reference_mode=deterministic_reference_mode,
                    device=device,
                    dtype=dtype,
                    reservation=reservation,
                    forced_prefix_token_ids=forced_prefix_token_ids,
                    reasoning=reasoning,
                    quantization=quantization,
                    strict_vram=strict_vram,
                ),
                name=f"generation-token-branch-{run_id}",
            )
            self._tasks[run_id] = task
            task.add_done_callback(
                lambda _task: self._release_generation_reservation(run_id, reservation)
            )
            return setup
        except BaseException:
            self._release_generation_reservation(run_id, reservation)
            raise

    async def create_embedding(
        self,
        request: EmbeddingRunCreate,
        resolved_inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        descriptor = self.registry.get(request.model_id)
        if descriptor.task not in {ModelTask.EMBEDDING, ModelTask.MULTIMODAL_EMBEDDING}:
            raise CapabilityUnavailableError("the selected model does not expose embeddings")
        input_text_bytes = sum(
            len(str(item.get("text") or "").encode("utf-8")) for item in resolved_inputs
        )
        if input_text_bytes > self.settings.limits.prompt_bytes:
            raise InvalidRequestError(
                "embedding text exceeds the configured byte limit",
                details={
                    "maximum_bytes": self.settings.limits.prompt_bytes,
                    "input_text_bytes": input_text_bytes,
                },
            )
        for item in resolved_inputs:
            modality = str(item["modality"])
            media_kind = item.get("media_kind")
            if modality == "text":
                continue
            if not isinstance(media_kind, str):
                raise InvalidRequestError(f"{modality} input requires a stored media attachment")
            if modality != "mixed" and modality != media_kind:
                raise InvalidRequestError(
                    "declared embedding modality does not match the attachment",
                    details={"declared_modality": modality, "attachment_modality": media_kind},
                )
            required_modalities = {media_kind}
            if modality == "mixed":
                required_modalities.add("text")
            unsupported = required_modalities.difference(descriptor.modalities)
            if unsupported:
                raise CapabilityUnavailableError(
                    "the selected model does not support every requested input modality",
                    details={
                        "unsupported_modalities": sorted(unsupported),
                        "supported_modalities": sorted(descriptor.modalities),
                    },
                )
        if request.dimensions is not None:
            minimum = descriptor.metadata.get("minimum_embedding_dimension")
            maximum = descriptor.metadata.get("embedding_dimension")
            supports_truncation = descriptor.metadata.get("supports_dimension_truncation") is True
            if isinstance(maximum, int) and request.dimensions == maximum:
                pass
            elif not supports_truncation:
                raise InvalidRequestError(
                    "custom embedding dimensions are not supported by this model adapter",
                    details={"native_dimensions": maximum},
                )
            elif (
                isinstance(minimum, int)
                and isinstance(maximum, int)
                and not (minimum <= request.dimensions <= maximum)
            ):
                raise InvalidRequestError(
                    f"requested dimensions must be between {minimum} and {maximum}",
                    details={
                        "requested_dimensions": request.dimensions,
                        "minimum_dimensions": minimum,
                        "maximum_dimensions": maximum,
                    },
                )
        selection = self.registry.choose_hardware()
        run_id = str(uuid.uuid4())
        reservation = self.registry.reserve_inference(run_id, "embedding")
        try:
            return await self._create_embedding_admitted(
                request=request,
                resolved_inputs=resolved_inputs,
                descriptor=descriptor,
                selection=selection,
                run_id=run_id,
                reservation=reservation,
            )
        except BaseException:
            reservation.release()
            raise

    async def _create_embedding_admitted(
        self,
        *,
        request: EmbeddingRunCreate,
        resolved_inputs: list[dict[str, Any]],
        descriptor: ModelDescriptor,
        selection: Any,
        run_id: str,
        reservation: InferenceReservation,
    ) -> dict[str, Any]:
        await self.repository.create_run(
            {
                "id": run_id,
                "model_id": descriptor.id,
                "kind": "embedding",
                "status": "queued",
                "effective_seed": "not_applicable",
                "rng_algorithm": "not_applicable",
                "generator_device": selection.device_identifier,
                "settings": request.model_dump(mode="json", exclude={"inputs"}),
                "effective_config": self.settings.inference_snapshot(),
                "reproducibility": {
                    "model_fingerprint": descriptor.fingerprint.model_dump(mode="json"),
                    "backend": "sentence-transformers",
                    "device": selection.model_dump(mode="json"),
                    "dtype": selection.effective_dtype.value,
                    "software_versions": (
                        self.registry.hardware.software_versions if self.registry.hardware else {}
                    ),
                    "batching": {"max_batch_size": self.settings.runtime.max_batch_size},
                    "requested_dimensions": request.dimensions,
                    "normalize": request.normalize,
                    "embedding_output_quantization": None,
                },
                "model_fingerprint": descriptor.fingerprint.value,
            }
        )
        hardware_snapshot = (
            self.registry.hardware.model_dump(mode="json") if self.registry.hardware else {}
        )
        await self.repository.save_environment_snapshot(
            run_id,
            hardware=hardware_snapshot,
            software=hardware_snapshot.get("software_versions", {}),
            backend={
                "name": "sentence-transformers",
                "selection": selection.model_dump(mode="json"),
                "embedding_output_quantization": None,
            },
        )
        try:
            await reservation.acquire()
            await self.repository.update_run(run_id, status="loading", queue_exited_at=_now())
            loaded = await self.registry.load_reserved(reservation, descriptor.id)
            await self.repository.update_run(run_id, status="running", started_at=_now())
            result = await self.worker.embed(
                inputs=resolved_inputs,
                dimensions=request.dimensions,
                normalize=request.normalize,
                batch_size=self.settings.runtime.max_batch_size,
                timeout_seconds=self.settings.workers.inference_timeout_seconds,
            )
            now = _now()
            async with self.repository.database.transaction() as connection:
                await connection.execute(
                    """
                    INSERT INTO embedding_runs(
                        id, pooling, normalized, requested_dimensions, output_dimensions,
                        output_dtype, joint_space, preprocessing_ms, forward_ms, total_ms,
                        items_per_second, summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result["pooling"],
                        int(request.normalize),
                        request.dimensions,
                        result["results"][0]["output_dimension"],
                        result["results"][0]["output_dtype"],
                        int(bool(result.get("joint_embedding_space"))),
                        result.get("preprocessing_ms"),
                        result.get("forward_ms"),
                        result.get("total_ms"),
                        result.get("items_per_second"),
                        json.dumps({"memory": result.get("memory", {})}),
                    ),
                )
                for ordinal, (source, embedded) in enumerate(
                    zip(resolved_inputs, result["results"], strict=True)
                ):
                    await connection.execute(
                        """
                        INSERT INTO embedding_inputs(
                            id, run_id, ordinal, input_type, content_preview, attachment_id,
                            vector_json, l2_norm, statistics_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"{run_id}:{embedded['input_id']}",
                            run_id,
                            ordinal,
                            source["modality"],
                            (source.get("text") or "")[:200],
                            source.get("attachment_id"),
                            json.dumps(embedded["vector"]) if request.persist_vectors else None,
                            embedded["l2_norm"],
                            json.dumps(embedded["statistics"]),
                        ),
                    )
            await self.repository.update_run(
                run_id,
                status="complete",
                finish_reason="embedding_complete",
                completed_at=now,
            )
            for phase, duration_key in (
                ("preprocessing", "preprocessing_ms"),
                ("embedding_forward", "forward_ms"),
                ("embedding_total", "total_ms"),
            ):
                await self.repository.upsert_phase_metric(
                    run_id,
                    phase,
                    duration_ms=result.get(duration_key),
                    details={
                        "items_per_second": result.get("items_per_second"),
                        "memory": result.get("memory"),
                    },
                )
            return {"run_id": run_id, "loaded": loaded, **result}
        except Exception as exc:
            logger.error("embedding run %s failed (%s)", run_id, type(exc).__name__)
            if isinstance(exc, WorkerFailure):
                error = exc.error
            elif isinstance(exc, WorkbenchError):
                error = exc.to_dict()
            else:
                error = {
                    "code": "embedding_failed",
                    "message": "embedding failed at an application boundary",
                }
            await self.repository.update_run(
                run_id,
                status="failed",
                error_code=error.get("code"),
                error_message=error.get("message"),
                completed_at=_now(),
            )
            raise
        finally:
            reservation.release()

    async def score_prompt(self, request: PromptScoreRequest) -> dict[str, Any]:
        descriptor = self.registry.get(request.model_id)
        if descriptor.task is not ModelTask.TEXT_GENERATION:
            raise CapabilityUnavailableError(
                "prompt perplexity currently requires a causal generation model",
                hint=(
                    "Encoder-decoder scoring needs separate source and target text, which this endpoint "
                    "does not accept."
                ),
            )
        if len(request.text.encode("utf-8")) > self.settings.limits.prompt_bytes:
            raise CapabilityUnavailableError("prompt exceeds the configured byte limit")
        selection = self.registry.choose_hardware(device=request.device, dtype=request.dtype)
        run_id = str(uuid.uuid4())
        reservation = self.registry.reserve_inference(run_id, "prompt_score")
        try:
            return await self._score_prompt_admitted(
                request=request,
                descriptor=descriptor,
                selection=selection,
                run_id=run_id,
                reservation=reservation,
            )
        except BaseException:
            reservation.release()
            raise

    async def _score_prompt_admitted(
        self,
        *,
        request: PromptScoreRequest,
        descriptor: ModelDescriptor,
        selection: Any,
        run_id: str,
        reservation: InferenceReservation,
    ) -> dict[str, Any]:
        await self.repository.create_run(
            {
                "id": run_id,
                "model_id": descriptor.id,
                "kind": "prompt_score",
                "status": "queued",
                "effective_seed": "not_applicable",
                "rng_algorithm": "not_applicable",
                "generator_device": selection.device_identifier,
                "settings": {"teacher_forced": True},
                "effective_config": self.settings.inference_snapshot(),
                "reproducibility": {
                    "model_fingerprint": descriptor.fingerprint.model_dump(mode="json"),
                    "definition": "shifted raw full-vocabulary log probability",
                    "backend": "transformers-reference-loop",
                    "device": selection.model_dump(mode="json"),
                    "dtype": selection.effective_dtype.value,
                    "software_versions": (
                        self.registry.hardware.software_versions if self.registry.hardware else {}
                    ),
                },
                "model_fingerprint": descriptor.fingerprint.value,
            }
        )
        hardware_snapshot = (
            self.registry.hardware.model_dump(mode="json") if self.registry.hardware else {}
        )
        await self.repository.save_environment_snapshot(
            run_id,
            hardware=hardware_snapshot,
            software=hardware_snapshot.get("software_versions", {}),
            backend={
                "name": "transformers-reference-loop",
                "selection": selection.model_dump(mode="json"),
                "operation": "teacher_forced_prompt_score",
            },
        )
        try:
            await reservation.acquire()
            await self.repository.update_run(run_id, status="loading", queue_exited_at=_now())
            loaded = await self.registry.load_reserved(
                reservation, descriptor.id, device=request.device, dtype=request.dtype
            )
            await self.repository.update_run(run_id, status="running", started_at=_now())
            result = await self.worker.score_prompt(
                request.text,
                timeout_seconds=self.settings.workers.inference_timeout_seconds,
            )
            await self.repository.update_run(
                run_id,
                status="complete",
                rendered_prompt=request.text,
                prompt_token_count=len(result["token_log_probabilities"]),
                finish_reason="prompt_scored",
                completed_at=_now(),
            )
            await self.repository.upsert_phase_metric(
                run_id,
                "prompt_scoring",
                duration_ms=result.get("duration_ms"),
                details={
                    "perplexity": result.get("perplexity"),
                    "included_token_count": result.get("included_token_count"),
                    "excluded_first_token": result.get("excluded_first_token"),
                    "excluded_masked_tokens": result.get("excluded_masked_tokens"),
                    "excluded_non_text_tokens": result.get("excluded_non_text_tokens"),
                    "definition": result.get("definition"),
                },
            )
            return {"run_id": run_id, "loaded": loaded, **result}
        except Exception as exc:
            logger.error("prompt score run %s failed (%s)", run_id, type(exc).__name__)
            if isinstance(exc, WorkerFailure):
                error = exc.error
            elif isinstance(exc, WorkbenchError):
                error = exc.to_dict()
            else:
                error = {
                    "code": "prompt_scoring_failed",
                    "message": "prompt scoring failed at an application boundary",
                }
            await self.repository.update_run(
                run_id,
                status="failed",
                error_code=error.get("code"),
                error_message=error.get("message"),
                completed_at=_now(),
            )
            raise
        finally:
            reservation.release()

    async def close(self) -> None:
        entries = tuple(self._tasks.items())
        if not entries:
            return

        queued_run_ids: list[str] = []
        for run_id, task in entries:
            reservation = self._reservations.get(run_id)
            if reservation is not None and reservation.cancel_if_waiting():
                task.cancel()
                queued_run_ids.append(run_id)
            else:
                self.worker.cancel(run_id)
        queued_runs = await asyncio.gather(
            *(self.repository.get_run(run_id) for run_id in queued_run_ids),
            return_exceptions=True,
        )
        queued = [
            (run_id, run)
            for run_id, run in zip(queued_run_ids, queued_runs, strict=True)
            if isinstance(run, Mapping)
        ]
        await asyncio.gather(
            *(self._mark_cancelled_before_start(run_id, run) for run_id, run in queued),
            return_exceptions=True,
        )
        await asyncio.gather(*(task for _, task in entries), return_exceptions=True)
