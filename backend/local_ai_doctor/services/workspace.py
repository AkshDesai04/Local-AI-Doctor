"""Portable, path-free chat workspace export and import."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from ..api.schemas import ChatWorkspaceDocument
from ..errors import InvalidRequestError
from ..persistence import WorkspaceRepository


def _unique_ids(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise InvalidRequestError(
            f"workspace contains duplicate {label} identifiers",
            details={"collection": label},
        )


def _validate_parent_graph(items: Sequence[Mapping[str, Any]], *, label: str) -> None:
    parents = {str(item["id"]): item.get("parent_id") for item in items}
    for item_id, parent_id in parents.items():
        if parent_id is not None and parent_id not in parents:
            raise InvalidRequestError(
                f"workspace {label} parent does not exist",
                details={"collection": label, "id": item_id},
            )
    complete: set[str] = set()
    for item_id in parents:
        chain: set[str] = set()
        cursor: str | None = item_id
        while cursor is not None and cursor not in complete:
            if cursor in chain:
                raise InvalidRequestError(
                    f"workspace {label} graph contains a cycle",
                    details={"collection": label, "id": item_id},
                )
            chain.add(cursor)
            next_value = parents.get(cursor)
            cursor = str(next_value) if next_value is not None else None
        complete.update(chain)


class WorkspaceService:
    """Build and restore versioned workspace documents without filesystem references."""

    def __init__(self, repository: WorkspaceRepository) -> None:
        self.repository = repository

    async def export_chat(self, chat_id: str) -> dict[str, Any] | None:
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            return None
        runs = await self.repository.list_chat_runs(chat_id)
        exported_runs: list[dict[str, Any]] = []
        for run in runs:
            exported_tokens: list[dict[str, Any]] = []
            for token in await self.repository.list_run_tokens(str(run["id"])):
                selected_experts: Any = None
                raw_selected = token.get("selected_experts_json")
                if isinstance(raw_selected, str):
                    try:
                        selected_experts = json.loads(raw_selected)
                    except json.JSONDecodeError:
                        selected_experts = None
                alternatives = [
                    {
                        "distribution": alternative["distribution"],
                        "rank": alternative["rank"],
                        "token_id": alternative["token_id"],
                        "piece": alternative["piece"],
                        "logit": alternative.get("logit"),
                        "log_probability": alternative.get("log_probability"),
                        "probability": alternative.get("probability"),
                        "survived_filter": bool(alternative.get("survived_filter")),
                    }
                    for alternative in token.get("alternatives", [])
                ]
                token_fields = {
                    "token_index",
                    "token_id",
                    "piece",
                    "escaped_bytes",
                    "display_text",
                    "span_start",
                    "span_end",
                    "raw_logit",
                    "raw_logprob",
                    "raw_probability",
                    "raw_rank",
                    "processed_logit",
                    "sample_logprob",
                    "sample_probability",
                    "entropy",
                    "surprise",
                    "cumulative_logprob",
                    "running_perplexity",
                    "decode_ms",
                    "sample_ms",
                    "emit_ms",
                    "inter_token_ms",
                    "cumulative_ms",
                    "instantaneous_tps",
                    "rolling_tps",
                    "segment",
                    "reasoning_slices",
                }
                exported_tokens.append(
                    {
                        **{name: token.get(name) for name in token_fields},
                        "selected_experts": selected_experts,
                        "alternatives": alternatives,
                    }
                )
            run_fields = {
                "id",
                "message_id",
                "parent_run_id",
                "model_id",
                "kind",
                "status",
                "requested_seed",
                "effective_seed",
                "rng_algorithm",
                "generator_device",
                "settings",
                "effective_config",
                "reproducibility",
                "model_fingerprint",
                "tokenizer_fingerprint",
                "rendered_prompt",
                "prompt_token_count",
                "generated_token_count",
                "finish_reason",
                "error_code",
                "error_message",
            }
            exported_runs.append(
                {**{name: run.get(name) for name in run_fields}, "tokens": exported_tokens}
            )

        document = ChatWorkspaceDocument.model_validate(
            {
                "schema": "local-ai-doctor/chat-workspace",
                "schema_version": 1,
                "chat": {
                    "title": chat["title"],
                    "pinned": bool(chat["pinned"]),
                    "archived": bool(chat["archived"]),
                },
                "messages": [
                    {
                        "id": message["id"],
                        "parent_id": message.get("parent_id"),
                        "role": message["role"],
                        "content": message["content"],
                        "status": message["status"],
                        "branch_index": message["branch_index"],
                        "metadata": message.get("metadata", {}),
                    }
                    for message in chat["messages"]
                ],
                "runs": exported_runs,
            }
        )
        return document.model_dump(mode="json", by_alias=True)

    async def import_chat(self, document: ChatWorkspaceDocument) -> dict[str, Any]:
        messages = [item.model_dump(mode="python") for item in document.messages]
        runs = [item.model_dump(mode="python") for item in document.runs]
        _unique_ids([str(item["id"]) for item in messages], "message")
        _unique_ids([str(item["id"]) for item in runs], "run")
        _validate_parent_graph(messages, label="message")

        run_ids = {str(item["id"]) for item in runs}
        message_ids = {str(item["id"]) for item in messages}
        run_parent_graph = [
            {"id": run["id"], "parent_id": run.get("parent_run_id")} for run in runs
        ]
        _validate_parent_graph(run_parent_graph, label="run")
        for run in runs:
            if run.get("message_id") is not None and run["message_id"] not in message_ids:
                raise InvalidRequestError(
                    "workspace run references a message outside the document",
                    details={"run_id": run["id"]},
                )
            token_indices = [int(token["token_index"]) for token in run.get("tokens", [])]
            if len(token_indices) != len(set(token_indices)):
                raise InvalidRequestError(
                    "workspace run contains duplicate token indices",
                    details={"run_id": run["id"]},
                )
            for token in run.get("tokens", []):
                alternative_keys = [
                    (alternative["distribution"], int(alternative["rank"]))
                    for alternative in token.get("alternatives", [])
                ]
                if len(alternative_keys) != len(set(alternative_keys)):
                    raise InvalidRequestError(
                        "workspace token contains duplicate alternatives",
                        details={"run_id": run["id"], "token_index": token["token_index"]},
                    )

        chat_id = str(uuid.uuid4())
        message_map = {source_id: str(uuid.uuid4()) for source_id in message_ids}
        run_map = {source_id: str(uuid.uuid4()) for source_id in run_ids}
        remapped_messages: list[dict[str, Any]] = []
        for message in messages:
            source_parent = message.get("parent_id")
            metadata = dict(message.get("metadata", {}))
            metadata["imported_from_message_id"] = message["id"]
            cloned_run_id = metadata.get("cloned_from_run_id")
            if isinstance(cloned_run_id, str):
                if cloned_run_id in run_map:
                    metadata["cloned_from_run_id"] = run_map[cloned_run_id]
                else:
                    metadata.pop("cloned_from_run_id")
                    metadata["imported_from_cloned_run_id"] = cloned_run_id
            remapped_messages.append(
                {
                    **message,
                    "id": message_map[message["id"]],
                    "parent_id": message_map[source_parent] if source_parent is not None else None,
                    "status": "failed"
                    if message["status"] in {"pending", "streaming"}
                    else message["status"],
                    "metadata": metadata,
                }
            )
        remapped_runs: list[dict[str, Any]] = []
        for run in runs:
            source_parent = run.get("parent_run_id")
            source_message = run.get("message_id")
            status = run["status"]
            reproducibility = dict(run.get("reproducibility", {}))
            reproducibility["imported_from_run_id"] = run["id"]
            normalized = {
                **run,
                "id": run_map[run["id"]],
                "message_id": message_map[source_message] if source_message is not None else None,
                "parent_run_id": run_map[source_parent] if source_parent is not None else None,
                "reproducibility": reproducibility,
            }
            if status in {"queued", "loading", "running"}:
                normalized.update(
                    {
                        "status": "failed",
                        "finish_reason": "imported_incomplete_snapshot",
                        "error_code": "imported_incomplete_snapshot",
                        "error_message": "The exported run was incomplete when imported.",
                    }
                )
            remapped_runs.append(normalized)

        imported = await self.repository.import_chat_workspace(
            chat={
                "id": chat_id,
                "title": document.chat.title,
                "pinned": document.chat.pinned,
                "archived": document.chat.archived,
            },
            messages=remapped_messages,
            runs=remapped_runs,
        )
        return {
            "chat": imported,
            "imported": {
                "messages": len(remapped_messages),
                "runs": len(remapped_runs),
                "tokens": sum(len(run.get("tokens", [])) for run in remapped_runs),
            },
        }
