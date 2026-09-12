"""Parameterized persistence operations for chats, models, runs, and telemetry."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ActiveRunConflictError, InvalidRequestError
from .database import Database, utc_now


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_json_columns(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for key, value in row.items():
        if key.endswith("_json") and isinstance(value, str):
            try:
                decoded[key.removesuffix("_json")] = json.loads(value)
            except json.JSONDecodeError:
                decoded[key.removesuffix("_json")] = None
    return decoded


_INSERT_RUN_SQL = """
INSERT INTO inference_runs(
    id, message_id, model_id, parent_run_id, kind, status, requested_seed,
    effective_seed, rng_algorithm, generator_device, settings_json,
    effective_config_json, reproducibility_json, model_fingerprint,
    tokenizer_fingerprint, received_at, queue_entered_at, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _run_values(run: Mapping[str, Any], run_id: str, now: str) -> tuple[Any, ...]:
    return (
        run_id,
        run.get("message_id"),
        run.get("model_id"),
        run.get("parent_run_id"),
        run.get("kind", "generation"),
        run.get("status", "queued"),
        None if run.get("requested_seed") is None else str(run["requested_seed"]),
        str(run["effective_seed"]),
        run.get("rng_algorithm", "torch.Generator"),
        run.get("generator_device", "cpu"),
        _json(run.get("settings", {})),
        _json(run.get("effective_config", {})),
        _json(run.get("reproducibility", {})),
        run.get("model_fingerprint"),
        run.get("tokenizer_fingerprint"),
        run.get("received_at", now),
        run.get("queue_entered_at", now),
        now,
        now,
    )


class WorkspaceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def upsert_model(self, model: Mapping[str, Any]) -> None:
        now = utc_now()
        existing = await self.database.fetch_one(
            "SELECT id FROM models WHERE canonical_path = ?", (str(model["canonical_path"]),)
        )
        if existing is not None and existing["id"] != str(model["id"]):
            # IDs intentionally include the fingerprint. A changed checkpoint
            # is a new model identity; historical runs retain their persisted
            # fingerprint while the stale registry row is safely replaced.
            await self.database.execute("DELETE FROM models WHERE id = ?", (existing["id"],))
        await self.database.execute(
            """
            INSERT INTO models(
                id, canonical_path, display_name, architecture, task, fingerprint,
                fingerprint_algorithm, parameter_count, weight_bytes, metadata_json,
                diagnostics_json, discovered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_path) DO UPDATE SET
                display_name=excluded.display_name,
                architecture=excluded.architecture,
                task=excluded.task,
                fingerprint=excluded.fingerprint,
                fingerprint_algorithm=excluded.fingerprint_algorithm,
                parameter_count=excluded.parameter_count,
                weight_bytes=excluded.weight_bytes,
                metadata_json=excluded.metadata_json,
                diagnostics_json=excluded.diagnostics_json,
                updated_at=excluded.updated_at
            """,
            (
                str(model["id"]),
                str(model["canonical_path"]),
                str(model["display_name"]),
                model.get("architecture"),
                str(model.get("task", "unknown")),
                str(model["fingerprint"]),
                str(model.get("fingerprint_algorithm", "sha256-manifest-v1")),
                model.get("parameter_count"),
                int(model.get("weight_bytes", 0)),
                _json(model.get("metadata", {})),
                _json(model.get("diagnostics", [])),
                str(model.get("discovered_at", now)),
                now,
            ),
        )

    async def replace_capabilities(
        self, model_id: str, capabilities: Mapping[str, Mapping[str, Any]]
    ) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "DELETE FROM model_capabilities WHERE model_id = ?", (model_id,)
            )
            await connection.executemany(
                """
                INSERT INTO model_capabilities(model_id, capability, state, reason, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        model_id,
                        name,
                        value["state"],
                        value.get("reason"),
                        _json(value.get("details", {})),
                    )
                    for name, value in capabilities.items()
                ],
            )

    async def list_models(self) -> list[dict[str, Any]]:
        models = await self.database.fetch_all(
            "SELECT * FROM models ORDER BY display_name COLLATE NOCASE"
        )
        result: list[dict[str, Any]] = []
        for model in models:
            row = _decode_json_columns(model)
            capabilities = await self.database.fetch_all(
                "SELECT capability, state, reason, details_json FROM model_capabilities WHERE model_id = ? ORDER BY capability",
                (model["id"],),
            )
            row["capabilities"] = {
                capability["capability"]: {
                    "state": capability["state"],
                    "reason": capability["reason"],
                    "details": json.loads(capability["details_json"]),
                }
                for capability in capabilities
            }
            result.append(row)
        return result

    async def create_chat(self, title: str = "New chat") -> dict[str, Any]:
        chat_id = str(uuid.uuid4())
        now = utc_now()
        await self.database.execute(
            "INSERT INTO chats(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, title.strip()[:200] or "New chat", now, now),
        )
        chat = await self.get_chat(chat_id)
        assert chat is not None
        return chat

    async def list_chats(
        self, *, search: str | None = None, archived: bool = False, limit: int = 100
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return await self.database.fetch_all(
                """
                SELECT * FROM chats
                WHERE archived = ? AND title LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY pinned DESC, updated_at DESC LIMIT ?
                """,
                (int(archived), f"%{escaped}%", limit),
            )
        return await self.database.fetch_all(
            """
            SELECT * FROM chats WHERE archived = ?
            ORDER BY pinned DESC, updated_at DESC LIMIT ?
            """,
            (int(archived), limit),
        )

    async def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        chat = await self.database.fetch_one("SELECT * FROM chats WHERE id = ?", (chat_id,))
        if chat is None:
            return None
        messages = await self.database.fetch_all(
            """
            SELECT messages.*,
                   (SELECT inference_runs.id FROM inference_runs
                    WHERE inference_runs.message_id = messages.id
                    ORDER BY inference_runs.created_at DESC LIMIT 1) AS run_id
            FROM messages WHERE chat_id = ? ORDER BY created_at, branch_index
            """,
            (chat_id,),
        )
        decoded_messages = [_decode_json_columns(message) for message in messages]
        message_ids = [str(message["id"]) for message in decoded_messages]
        attachments_by_message: dict[str, list[dict[str, Any]]] = {
            message_id: [] for message_id in message_ids
        }
        if message_ids:
            placeholders = ",".join("?" for _ in message_ids)
            attachment_rows = await self.database.fetch_all(
                f"""
                SELECT association.message_id, association.ordinal,
                       attachment.id, attachment.original_name, attachment.media_type,
                       attachment.size_bytes, attachment.metadata_json,
                       attachment.preprocessing_json, attachment.created_at
                FROM message_attachments AS association
                JOIN attachments AS attachment ON attachment.id = association.attachment_id
                WHERE association.message_id IN ({placeholders})
                ORDER BY association.message_id, association.ordinal
                """,
                tuple(message_ids),
            )
            for attachment in attachment_rows:
                decoded = _decode_json_columns(attachment)
                message_id = str(decoded.pop("message_id"))
                decoded.pop("ordinal", None)
                attachments_by_message.setdefault(message_id, []).append(decoded)
        for message in decoded_messages:
            message["attachments"] = attachments_by_message.get(str(message["id"]), [])
        chat["messages"] = decoded_messages
        return chat

    async def list_chat_runs(self, chat_id: str) -> list[dict[str, Any]]:
        rows = await self.database.fetch_all(
            """
            SELECT inference_runs.*
            FROM inference_runs
            JOIN messages ON messages.id = inference_runs.message_id
            WHERE messages.chat_id = ?
            ORDER BY inference_runs.created_at, inference_runs.id
            """,
            (chat_id,),
        )
        return [_decode_json_columns(row) for row in rows]

    async def get_message_lineage(self, message_id: str) -> list[dict[str, Any]]:
        rows = await self.database.fetch_all(
            """
            WITH RECURSIVE lineage(id, chat_id, parent_id, role, content, status,
                                   branch_index, metadata_json, created_at, updated_at, depth) AS (
                SELECT id, chat_id, parent_id, role, content, status, branch_index,
                       metadata_json, created_at, updated_at, 0
                FROM messages WHERE id = ?
                UNION ALL
                SELECT parent.id, parent.chat_id, parent.parent_id, parent.role,
                       parent.content, parent.status, parent.branch_index,
                       parent.metadata_json, parent.created_at, parent.updated_at,
                       lineage.depth + 1
                FROM messages AS parent
                JOIN lineage ON parent.id = lineage.parent_id
                WHERE lineage.depth < 10000
            )
            SELECT id, chat_id, parent_id, role, content, status, branch_index,
                   metadata_json, created_at, updated_at
            FROM lineage ORDER BY depth DESC
            """,
            (message_id,),
        )
        return [_decode_json_columns(row) for row in rows]

    async def next_branch_index(self, parent_id: str) -> int:
        row = await self.database.fetch_one(
            "SELECT COALESCE(MAX(branch_index), -1) + 1 AS value FROM messages WHERE parent_id = ?",
            (parent_id,),
        )
        return int(row["value"]) if row else 0

    async def update_chat(
        self,
        chat_id: str,
        *,
        title: str | None = None,
        pinned: bool | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None:
        current = await self.database.fetch_one("SELECT * FROM chats WHERE id = ?", (chat_id,))
        if current is None:
            return None
        await self.database.execute(
            "UPDATE chats SET title = ?, pinned = ?, archived = ?, updated_at = ? WHERE id = ?",
            (
                (title.strip()[:200] or "New chat") if title is not None else current["title"],
                int(pinned) if pinned is not None else current["pinned"],
                int(archived) if archived is not None else current["archived"],
                utc_now(),
                chat_id,
            ),
        )
        return await self.get_chat(chat_id)

    async def delete_chat(self, chat_id: str) -> bool:
        async with self.database.transaction() as connection:
            before = await connection.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
            if await before.fetchone() is None:
                return False
            active = await connection.execute(
                """
                SELECT inference_runs.id
                FROM inference_runs
                JOIN messages ON messages.id = inference_runs.message_id
                WHERE messages.chat_id = ?
                  AND inference_runs.kind = 'generation'
                  AND inference_runs.status IN ('queued', 'loading', 'running')
                ORDER BY inference_runs.created_at, inference_runs.id
                """,
                (chat_id,),
            )
            active_run_ids = [str(row[0]) for row in await active.fetchall()]
            if active_run_ids:
                raise ActiveRunConflictError(
                    "chat cannot be deleted while it owns an active generation",
                    hint="Cancel the generation or wait for it to finish, then retry.",
                    details={"chat_id": chat_id, "run_ids": active_run_ids},
                )
            await connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return True

    async def clear_chats(self, *, include_archived: bool = False) -> int:
        condition = "1 = 1" if include_archived else "archived = 0"
        async with self.database.transaction() as connection:
            active = await connection.execute(
                f"""
                SELECT inference_runs.id, messages.chat_id
                FROM inference_runs
                JOIN messages ON messages.id = inference_runs.message_id
                JOIN chats ON chats.id = messages.chat_id
                WHERE {condition}
                  AND inference_runs.kind = 'generation'
                  AND inference_runs.status IN ('queued', 'loading', 'running')
                ORDER BY inference_runs.created_at, inference_runs.id
                """
            )
            active_rows = await active.fetchall()
            if active_rows:
                raise ActiveRunConflictError(
                    "chats cannot be cleared while they own active generations",
                    hint="Cancel the generations or wait for them to finish, then retry.",
                    details={
                        "run_ids": [str(row[0]) for row in active_rows],
                        "chat_ids": sorted({str(row[1]) for row in active_rows}),
                    },
                )
            count_cursor = await connection.execute(
                f"SELECT COUNT(*) AS count FROM chats WHERE {condition}"
            )
            count_row = await count_cursor.fetchone()
            await connection.execute(f"DELETE FROM chats WHERE {condition}")
        return int(count_row[0]) if count_row else 0

    async def create_message(
        self,
        *,
        chat_id: str,
        role: str,
        content: str,
        parent_id: str | None = None,
        status: str = "complete",
        branch_index: int = 0,
        metadata: Mapping[str, Any] | None = None,
        attachment_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        message_id = str(uuid.uuid4())
        now = utc_now()
        async with self.database.transaction() as connection:
            chat_cursor = await connection.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
            if await chat_cursor.fetchone() is None:
                raise InvalidRequestError("chat not found", details={"chat_id": chat_id})
            if parent_id is not None:
                parent_cursor = await connection.execute(
                    "SELECT chat_id FROM messages WHERE id = ?", (parent_id,)
                )
                parent = await parent_cursor.fetchone()
                if parent is None:
                    raise InvalidRequestError(
                        "parent message was not found", details={"parent_id": parent_id}
                    )
                if str(parent[0]) != chat_id:
                    raise InvalidRequestError(
                        "parent message must belong to the selected chat",
                        details={"chat_id": chat_id, "parent_id": parent_id},
                    )
            if len(set(attachment_ids)) != len(attachment_ids):
                raise InvalidRequestError("attachment IDs must be unique")
            if attachment_ids:
                placeholders = ",".join("?" for _ in attachment_ids)
                attachment_cursor = await connection.execute(
                    f"SELECT id FROM attachments WHERE id IN ({placeholders})",
                    tuple(attachment_ids),
                )
                found = {str(row[0]) for row in await attachment_cursor.fetchall()}
                missing = [
                    attachment_id for attachment_id in attachment_ids if attachment_id not in found
                ]
                if missing:
                    raise InvalidRequestError(
                        "one or more attachments were not found",
                        details={"attachment_ids": missing},
                    )
            await connection.execute(
                """
                INSERT INTO messages(
                    id, chat_id, parent_id, role, content, status, branch_index,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    chat_id,
                    parent_id,
                    role,
                    content,
                    status,
                    branch_index,
                    _json(metadata or {}),
                    now,
                    now,
                ),
            )
            if attachment_ids:
                await connection.executemany(
                    """
                    INSERT INTO message_attachments(message_id, attachment_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (message_id, attachment_id, ordinal)
                        for ordinal, attachment_id in enumerate(attachment_ids)
                    ],
                )
            await connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        row = await self.database.fetch_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        assert row is not None
        return _decode_json_columns(row)

    async def update_message(
        self,
        message_id: str,
        *,
        content: str,
        status: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if metadata is None:
            await self.database.execute(
                "UPDATE messages SET content = ?, status = ?, updated_at = ? WHERE id = ?",
                (content, status, utc_now(), message_id),
            )
        else:
            await self.database.execute(
                "UPDATE messages SET content = ?, status = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
                (content, status, _json(metadata), utc_now(), message_id),
            )

    async def get_message(self, message_id: str) -> dict[str, Any] | None:
        row = await self.database.fetch_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        return _decode_json_columns(row) if row else None

    async def create_run(self, run: Mapping[str, Any]) -> dict[str, Any]:
        now = utc_now()
        run_id = str(run.get("id") or uuid.uuid4())
        await self.database.execute(_INSERT_RUN_SQL, _run_values(run, run_id, now))
        result = await self.get_run(run_id)
        assert result is not None
        return result

    async def create_generation_setup(
        self,
        *,
        chat_id: str,
        user_content: str,
        parent_message_id: str | None,
        run: Mapping[str, Any],
        hardware: Mapping[str, Any],
        software: Mapping[str, Any],
        backend: Mapping[str, Any],
        chat_title: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Create a generation's messages, run, and snapshot atomically."""

        now = utc_now()
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        run_id = str(run.get("id") or uuid.uuid4())
        run_record = {**run, "message_id": assistant_message_id}
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO messages(
                    id, chat_id, parent_id, role, content, status, branch_index,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'user', ?, 'complete', 0, '{}', ?, ?)
                """,
                (user_message_id, chat_id, parent_message_id, user_content, now, now),
            )
            await connection.execute(
                """
                INSERT INTO messages(
                    id, chat_id, parent_id, role, content, status, branch_index,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'assistant', '', 'pending', 0, '{}', ?, ?)
                """,
                (assistant_message_id, chat_id, user_message_id, now, now),
            )
            await connection.execute(
                _INSERT_RUN_SQL,
                _run_values(run_record, run_id, now),
            )
            await connection.execute(
                """
                INSERT INTO environment_snapshots(
                    run_id, hardware_json, software_json, backend_json, captured_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, _json(hardware), _json(software), _json(backend), now),
            )
            if chat_title is None:
                await connection.execute(
                    "UPDATE chats SET updated_at = ? WHERE id = ?",
                    (now, chat_id),
                )
            else:
                await connection.execute(
                    "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                    (chat_title.strip()[:200] or "New chat", now, chat_id),
                )

            user_cursor = await connection.execute(
                "SELECT * FROM messages WHERE id = ?", (user_message_id,)
            )
            assistant_cursor = await connection.execute(
                "SELECT * FROM messages WHERE id = ?", (assistant_message_id,)
            )
            run_cursor = await connection.execute(
                "SELECT * FROM inference_runs WHERE id = ?", (run_id,)
            )
            user_row = await user_cursor.fetchone()
            assistant_row = await assistant_cursor.fetchone()
            run_row = await run_cursor.fetchone()
            assert user_row is not None and assistant_row is not None and run_row is not None

        return {
            "user_message": _decode_json_columns(dict(user_row)),
            "assistant_message": _decode_json_columns(dict(assistant_row)),
            "run": _decode_json_columns(dict(run_row)),
        }

    async def import_chat_workspace(
        self,
        *,
        chat: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        runs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Insert a validated, ID-remapped workspace document atomically."""

        now = utc_now()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO chats(id, title, pinned, archived, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chat["id"],
                    chat["title"],
                    int(bool(chat.get("pinned"))),
                    int(bool(chat.get("archived"))),
                    now,
                    now,
                ),
            )
            for message in messages:
                await connection.execute(
                    """
                    INSERT INTO messages(
                        id, chat_id, parent_id, role, content, status, branch_index,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message["id"],
                        chat["id"],
                        message["role"],
                        message["content"],
                        message["status"],
                        message["branch_index"],
                        _json(message.get("metadata", {})),
                        now,
                        now,
                    ),
                )
            for message in messages:
                if message.get("parent_id") is not None:
                    await connection.execute(
                        "UPDATE messages SET parent_id = ? WHERE id = ?",
                        (message["parent_id"], message["id"]),
                    )

            cursor = await connection.execute("SELECT id FROM models")
            available_models = {str(row[0]) for row in await cursor.fetchall()}
            for run in runs:
                model_id = run.get("model_id")
                if model_id not in available_models:
                    model_id = None
                await connection.execute(
                    """
                    INSERT INTO inference_runs(
                        id, message_id, model_id, parent_run_id, kind, status,
                        requested_seed, effective_seed, rng_algorithm, generator_device,
                        settings_json, effective_config_json, reproducibility_json,
                        model_fingerprint, tokenizer_fingerprint, rendered_prompt,
                        prompt_token_count, generated_token_count, finish_reason,
                        error_code, error_message, received_at, queue_entered_at,
                        completed_at, created_at, updated_at
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run["id"],
                        run.get("message_id"),
                        model_id,
                        run["kind"],
                        run["status"],
                        run.get("requested_seed"),
                        run["effective_seed"],
                        run["rng_algorithm"],
                        run["generator_device"],
                        _json(run.get("settings", {})),
                        _json(run.get("effective_config", {})),
                        _json(run.get("reproducibility", {})),
                        run.get("model_fingerprint"),
                        run.get("tokenizer_fingerprint"),
                        run.get("rendered_prompt"),
                        run.get("prompt_token_count"),
                        run.get("generated_token_count", 0),
                        run.get("finish_reason"),
                        run.get("error_code"),
                        run.get("error_message"),
                        now,
                        now,
                        now,
                        now,
                        now,
                    ),
                )
            for run in runs:
                if run.get("parent_run_id") is not None:
                    await connection.execute(
                        "UPDATE inference_runs SET parent_run_id = ? WHERE id = ?",
                        (run["parent_run_id"], run["id"]),
                    )
                for token in run.get("tokens", []):
                    await connection.execute(
                        """
                        INSERT INTO token_events(
                            run_id, token_index, token_id, piece, escaped_bytes, display_text,
                            span_start, span_end, raw_logit, raw_logprob, raw_probability,
                            raw_rank, processed_logit, sample_logprob, sample_probability,
                            entropy, surprise, cumulative_logprob, running_perplexity,
                            decode_ms, sample_ms, emit_ms, inter_token_ms, cumulative_ms,
                            instantaneous_tps, rolling_tps, segment, selected_experts_json,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run["id"],
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
                            _json(token.get("selected_experts")),
                            now,
                        ),
                    )
                    for alternative in token.get("alternatives", []):
                        await connection.execute(
                            """
                            INSERT INTO token_alternatives(
                                run_id, token_index, distribution, rank, token_id, piece,
                                logit, log_probability, probability, survived_filter
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run["id"],
                                token["token_index"],
                                alternative["distribution"],
                                alternative["rank"],
                                alternative["token_id"],
                                alternative["piece"],
                                alternative.get("logit"),
                                alternative.get("log_probability"),
                                alternative.get("probability"),
                                int(bool(alternative.get("survived_filter"))),
                            ),
                        )
        result = await self.get_chat(str(chat["id"]))
        assert result is not None
        return result

    async def delete_run_telemetry_before(self, before: str) -> dict[str, int]:
        """Delete terminal runs older than a UTC cutoff and their cascading telemetry."""

        terminal = "('complete', 'cancelled', 'failed', 'disconnected')"
        counts: dict[str, int] = {}
        joined_tables = {
            "environment_snapshots": "run_id",
            "phase_metrics": "run_id",
            "token_events": "run_id",
            "token_alternatives": "run_id",
            "router_events": "run_id",
            "router_aggregates": "run_id",
            "embedding_runs": "id",
            "embedding_inputs": "run_id",
            "raw_events": "run_id",
        }
        async with self.database.transaction() as connection:
            for table, run_column in joined_tables.items():
                cursor = await connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table} AS telemetry
                    JOIN inference_runs AS run ON run.id = telemetry.{run_column}
                    WHERE run.status IN {terminal}
                      AND run.completed_at IS NOT NULL AND run.completed_at < ?
                    """,
                    (before,),
                )
                row = await cursor.fetchone()
                counts[table] = int(row[0]) if row else 0
            cursor = await connection.execute(
                f"""
                SELECT COUNT(*) FROM inference_runs
                WHERE status IN {terminal}
                  AND completed_at IS NOT NULL AND completed_at < ?
                """,
                (before,),
            )
            row = await cursor.fetchone()
            counts["runs"] = int(row[0]) if row else 0
            await connection.execute(
                f"""
                DELETE FROM inference_runs
                WHERE status IN {terminal}
                  AND completed_at IS NOT NULL AND completed_at < ?
                """,
                (before,),
            )
        return counts

    async def update_run(self, run_id: str, **changes: Any) -> None:
        allowed = {
            "status",
            "rendered_prompt",
            "prompt_token_count",
            "generated_token_count",
            "finish_reason",
            "error_code",
            "error_message",
            "queue_exited_at",
            "started_at",
            "first_token_at",
            "completed_at",
        }
        selected = {key: value for key, value in changes.items() if key in allowed}
        if not selected:
            return
        selected["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in selected)
        await self.database.execute(
            f"UPDATE inference_runs SET {assignments} WHERE id = ?",
            (*selected.values(), run_id),
        )

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = await self.database.fetch_one("SELECT * FROM inference_runs WHERE id = ?", (run_id,))
        return _decode_json_columns(row) if row else None

    async def upsert_phase_metric(
        self,
        run_id: str,
        phase: str,
        *,
        duration_ms: float | None = None,
        started_ns: int | None = None,
        ended_ns: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        await self.database.execute(
            """
            INSERT INTO phase_metrics(
                run_id, phase, started_ns, ended_ns, duration_ms, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, phase) DO UPDATE SET
                started_ns=COALESCE(excluded.started_ns, phase_metrics.started_ns),
                ended_ns=COALESCE(excluded.ended_ns, phase_metrics.ended_ns),
                duration_ms=COALESCE(excluded.duration_ms, phase_metrics.duration_ms),
                details_json=excluded.details_json
            """,
            (run_id, phase, started_ns, ended_ns, duration_ms, _json(details or {})),
        )

    async def list_phase_metrics(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self.database.fetch_all(
            "SELECT * FROM phase_metrics WHERE run_id = ? ORDER BY phase",
            (run_id,),
        )
        return [_decode_json_columns(row) for row in rows]

    async def save_environment_snapshot(
        self,
        run_id: str,
        *,
        hardware: Mapping[str, Any],
        software: Mapping[str, Any],
        backend: Mapping[str, Any],
    ) -> None:
        await self.database.execute(
            """
            INSERT OR REPLACE INTO environment_snapshots(
                run_id, hardware_json, software_json, backend_json, captured_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, _json(hardware), _json(software), _json(backend), utc_now()),
        )

    async def get_environment_snapshot(self, run_id: str) -> dict[str, Any] | None:
        row = await self.database.fetch_one(
            "SELECT * FROM environment_snapshots WHERE run_id = ?",
            (run_id,),
        )
        return _decode_json_columns(row) if row else None

    async def recover_incomplete_runs(self) -> list[str]:
        rows = await self.database.fetch_all(
            "SELECT id, message_id FROM inference_runs WHERE status IN ('queued', 'loading', 'running')"
        )
        if not rows:
            return []
        now = utc_now()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                UPDATE inference_runs
                SET status = 'failed', finish_reason = 'application_restart',
                    error_code = 'interrupted_by_restart',
                    error_message = 'The application restarted before this run completed.',
                    completed_at = ?, updated_at = ?
                WHERE status IN ('queued', 'loading', 'running')
                """,
                (now, now),
            )
            message_ids = [row["message_id"] for row in rows if row.get("message_id")]
            if message_ids:
                placeholders = ",".join("?" for _ in message_ids)
                await connection.execute(
                    f"UPDATE messages SET status = 'failed', updated_at = ? WHERE id IN ({placeholders})",
                    (now, *message_ids),
                )
        return [str(row["id"]) for row in rows]

    async def create_attachment(self, attachment: Mapping[str, Any]) -> dict[str, Any]:
        now = utc_now()
        await self.database.execute(
            """
            INSERT INTO attachments(
                id, sha256, storage_name, original_name, media_type, size_bytes,
                metadata_json, preprocessing_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO NOTHING
            """,
            (
                attachment["id"],
                attachment["sha256"],
                attachment["storage_name"],
                attachment["original_name"],
                attachment["media_type"],
                attachment["size_bytes"],
                _json(attachment.get("metadata", {})),
                _json(attachment.get("preprocessing", {})),
                now,
            ),
        )
        row = await self.database.fetch_one(
            "SELECT * FROM attachments WHERE sha256 = ?", (attachment["sha256"],)
        )
        assert row is not None
        return _decode_json_columns(row)

    async def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        row = await self.database.fetch_one(
            "SELECT * FROM attachments WHERE id = ?", (attachment_id,)
        )
        return _decode_json_columns(row) if row else None

    async def list_run_tokens(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self.database.fetch_all(
            "SELECT * FROM token_events WHERE run_id = ? ORDER BY token_index", (run_id,)
        )
        alternatives = await self.database.fetch_all(
            """
            SELECT * FROM token_alternatives
            WHERE run_id = ?
            ORDER BY token_index, distribution, rank
            """,
            (run_id,),
        )
        alternatives_by_token: dict[int, list[dict[str, Any]]] = {}
        for alternative in alternatives:
            alternatives_by_token.setdefault(int(alternative["token_index"]), []).append(
                alternative
            )
        for row in rows:
            row["alternatives"] = alternatives_by_token.get(int(row["token_index"]), [])
        return rows

    async def append_raw_event(self, event: Mapping[str, Any]) -> None:
        await self.database.execute(
            """
            INSERT OR REPLACE INTO raw_events(
                run_id, sequence, event_type, protocol_version, monotonic_ns,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["run_id"],
                event["sequence"],
                event["type"],
                event.get("version", 1),
                event["monotonic_ns"],
                _json(event.get("payload", {})),
                utc_now(),
            ),
        )

    async def get_raw_events(self, run_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        rows = await self.database.fetch_all(
            """
            SELECT run_id, sequence, event_type AS type, protocol_version AS version,
                   monotonic_ns, payload_json, created_at
            FROM raw_events WHERE run_id = ? AND sequence > ? ORDER BY sequence
            """,
            (run_id, max(0, after_sequence)),
        )
        return [_decode_json_columns(row) for row in rows]

    async def database_stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for table in ("chats", "messages", "inference_runs", "token_events", "attachments"):
            row = await self.database.fetch_one(f"SELECT COUNT(*) AS count FROM {table}")
            counts[table] = int(row["count"]) if row else 0
        return {"size_bytes": self.database.path.stat().st_size, "counts": counts}
