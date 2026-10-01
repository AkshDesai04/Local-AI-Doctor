"""On-demand token influence: which earlier positions one generated token depended on.

Two stated methods, both recomputed from a persisted run's prefix and cached per
(run, token, method, parameters): post-softmax attention allocation, and gradient x
input at the residual stream entering the first decoder layer. Neither is causal
attribution.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from ..api.schemas import TokenInfluenceRequest
from ..config import AppSettings, DeviceMode, DType
from ..domain.capabilities import Capability, CapabilityState
from ..domain.models import ModelDescriptor, ModelTask
from ..errors import (
    InfluenceSequenceTooLongError,
    InfluenceUnavailableError,
    InvalidRequestError,
    ModelFingerprintChangedError,
    ModelNotFoundError,
    RunNotFoundError,
)
from ..persistence import WorkspaceRepository
from ..workers import ModelWorkerSupervisor
from .models import ModelRegistry
from .runs import RunManager, _generation_messages, _recorded_placement


def _recorded_selection(run: Mapping[str, Any]) -> tuple[DeviceMode | None, DType | None]:
    """The device and dtype the run asked for, as replay re-selects them."""

    reproducibility = run.get("reproducibility") or {}
    device: DeviceMode | None = None
    recorded_device = reproducibility.get("device")
    if isinstance(recorded_device, Mapping) and isinstance(recorded_device.get("requested"), str):
        with suppress(ValueError):
            device = DeviceMode(recorded_device["requested"])
    dtype: DType | None = None
    if isinstance(reproducibility.get("dtype"), str):
        with suppress(ValueError):
            dtype = DType(reproducibility["dtype"])
    return device, dtype


class InfluenceService:
    def __init__(
        self,
        settings: AppSettings,
        repository: WorkspaceRepository,
        registry: ModelRegistry,
        worker: ModelWorkerSupervisor,
        runs: RunManager,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.registry = registry
        self.worker = worker
        self.runs = runs

    async def analyze(
        self, run_id: str, token_index: int, request: TokenInfluenceRequest
    ) -> dict[str, Any]:
        run = await self.repository.get_run(run_id)
        if run is None:
            raise RunNotFoundError("run not found", details={"run_id": run_id})
        if run["kind"] != "generation":
            raise InfluenceUnavailableError(
                "token influence applies only to generation runs",
                details={"run_id": run_id, "kind": run["kind"]},
            )
        if run["status"] not in {"complete", "cancelled"}:
            raise InfluenceUnavailableError(
                "token influence needs a complete or cancelled run",
                hint="Wait for the run to finish, then analyze its tokens.",
                details={"run_id": run_id, "status": run["status"]},
            )
        descriptor = self._recorded_model(run)
        support = descriptor.capabilities.support(Capability.TOKEN_INFLUENCE)
        if descriptor.task is not ModelTask.TEXT_GENERATION or support.state not in {
            CapabilityState.FULL,
            CapabilityState.PARTIAL,
        }:
            raise InfluenceUnavailableError(
                support.reason or "token influence is unavailable for this model",
                details={"model_id": descriptor.id, "task": descriptor.task.value},
            )
        rendered_prompt = run.get("rendered_prompt")
        prompt_tokens = run.get("prompt_token_count")
        if not isinstance(rendered_prompt, str) or not isinstance(prompt_tokens, int):
            raise InfluenceUnavailableError(
                "the run has no persisted rendered prompt to re-run",
                details={"run_id": run_id},
            )
        prefix = await self.repository.list_run_token_prefix(run_id, token_index)
        if (
            token_index < 0
            or len(prefix) != token_index + 1
            or any(int(row["token_index"]) != expected for expected, row in enumerate(prefix))
        ):
            raise InvalidRequestError(
                "the selected token does not have a complete persisted prefix",
                hint="Choose a token whose telemetry is still available.",
                details={"run_id": run_id, "token_index": token_index},
            )
        target = prefix[-1]
        target_id = int(target["token_id"])
        alternative = request.alternative_token_id
        if alternative is not None and alternative == target_id:
            raise InvalidRequestError(
                "the alternative token must differ from the chosen token",
                details={"token_id": target_id},
            )
        sequence_tokens = prompt_tokens + token_index
        limit = self.settings.inference.influence_max_gradient_tokens
        if request.method == "gradient_x_input" and sequence_tokens > limit:
            raise InfluenceSequenceTooLongError(
                "the prefix is longer than the gradient x input limit",
                hint="Raise inference.influence_max_gradient_tokens or use the attention method.",
                details={"sequence_tokens": sequence_tokens, "max_sequence_tokens": limit},
            )

        fingerprint = descriptor.fingerprint.value
        parameters = {
            "layers": request.layers,
            "alternative_token_id": alternative,
            "source_limit": request.source_limit,
        }
        cached = await self.repository.get_token_influence(
            run_id, token_index, request.method, parameters, fingerprint
        )
        if cached is not None:
            return {**cached, "cached": True}

        media = await self._run_media(run, descriptor)
        renderer = await self._prompt_renderer(run_id)
        device, dtype = _recorded_selection(run)
        quantization, strict_vram = _recorded_placement(self.settings, run.get("settings") or {})
        reservation = self.registry.reserve_inference(
            f"influence-{uuid.uuid4()}", "influence_analysis"
        )
        async with reservation:
            loaded = await self.registry.load_reserved(
                reservation,
                descriptor.id,
                device=device,
                dtype=dtype,
                quantization=quantization,
                strict_vram=strict_vram,
            )
            result = await self.worker.analyze_influence(
                {
                    "model_key": loaded.get("model_key"),
                    "method": request.method,
                    "layers": request.layers,
                    "alternative_token_id": alternative,
                    "source_limit": request.source_limit,
                    "rendered_prompt": rendered_prompt,
                    "prompt_renderer": renderer,
                    "media": media,
                    "expected_prompt_token_count": prompt_tokens,
                    "generated_token_ids": [int(row["token_id"]) for row in prefix[:-1]],
                    "target_token_id": target_id,
                    "max_gradient_tokens": limit,
                },
                timeout_seconds=self.settings.workers.inference_timeout_seconds,
            )
        response = self._response(run_id, token_index, prefix, alternative, fingerprint, result)
        await self.repository.save_token_influence(
            run_id,
            token_index,
            request.method,
            parameters,
            fingerprint,
            response,
            response.get("duration_ms"),
        )
        return {**response, "cached": False}

    def _recorded_model(self, run: Mapping[str, Any]) -> ModelDescriptor:
        """The registered checkpoint, only while it still has the run's fingerprint."""

        hint = "Restore the original checkpoint and refresh the model registry."
        model_id = run.get("model_id")
        if model_id is None:
            raise ModelFingerprintChangedError(
                "the model that produced this run is no longer registered",
                hint=hint,
                details={"run_id": run["id"]},
            )
        try:
            descriptor = self.registry.get(str(model_id))
        except ModelNotFoundError:
            # Model IDs derive from the fingerprint, so a changed checkpoint has a new ID.
            raise ModelFingerprintChangedError(
                "the model that produced this run is no longer registered with its fingerprint",
                hint=hint,
                details={"run_id": run["id"], "model_id": model_id},
            ) from None
        if run.get("model_fingerprint") != descriptor.fingerprint.value:
            raise ModelFingerprintChangedError(
                "the recorded model fingerprint no longer matches the registered checkpoint",
                hint=hint,
                details={"run_id": run["id"], "model_id": descriptor.id},
            )
        return descriptor

    async def _run_media(
        self, run: Mapping[str, Any], descriptor: ModelDescriptor
    ) -> list[dict[str, str]]:
        """Attachments of the run's conversation, in the order generation rendered them."""

        message_id = run.get("message_id")
        message = await self.repository.get_message(str(message_id)) if message_id else None
        parent_id = (message or {}).get("parent_id")
        if not parent_id:
            return []
        lineage = await self.repository.get_message_lineage(str(parent_id))
        messages = _generation_messages(
            lineage, await self.runs._lineage_media(descriptor, lineage)
        )
        return [item for message in messages for item in message.get("attachments") or ()]

    async def _prompt_renderer(self, run_id: str) -> str | None:
        for phase in await self.repository.list_phase_metrics(run_id):
            if phase.get("phase") == "tokenization":
                renderer = (phase.get("details") or {}).get("prompt_renderer")
                return renderer if isinstance(renderer, str) else None
        return None

    @staticmethod
    def _response(
        run_id: str,
        token_index: int,
        prefix: list[dict[str, Any]],
        alternative: int | None,
        fingerprint: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        rows = {int(row["token_index"]): row for row in prefix}

        def persisted_text(source: Mapping[str, Any]) -> dict[str, Any]:
            # Generated tokens keep the exact text the run streamed, which a single-token
            # decode cannot reproduce for tokens that split a multi-byte character.
            index = source.get("generated_token_index")
            row = rows.get(index) if isinstance(index, int) else None
            if row is None:
                return dict(source)
            return {**source, "piece": row["piece"], "display_text": row["display_text"]}

        layers = result.get("layers")
        target = rows[token_index]
        return {
            **{key: value for key, value in result.items() if key != "alternative_piece"},
            "run_id": run_id,
            "token_index": token_index,
            "target": {
                "token_id": int(target["token_id"]),
                "piece": target["piece"],
                "display_text": target["display_text"],
                "alternative_token_id": alternative,
                "alternative_piece": result.get("alternative_piece"),
            },
            "sources": [persisted_text(source) for source in result.get("sources") or ()],
            "layers": None
            if layers is None
            else [
                {**layer, "sources": [persisted_text(source) for source in layer["sources"]]}
                for layer in layers
            ],
            "model_fingerprint": fingerprint,
        }
