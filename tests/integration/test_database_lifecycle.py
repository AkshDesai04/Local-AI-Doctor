from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from local_ai_doctor.errors import ActiveRunConflictError
from local_ai_doctor.persistence import Database, TelemetryWriter, WorkspaceRepository
from local_ai_doctor.persistence.database import utc_now


@pytest.mark.asyncio
async def test_existing_database_is_backed_up_migrated_once_and_uses_wal(tmp_path: Path) -> None:
    database_path = tmp_path / "state" / "workbench.sqlite3"
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as legacy:
        legacy.execute("CREATE TABLE legacy_data(value TEXT NOT NULL)")
        legacy.execute("INSERT INTO legacy_data(value) VALUES ('preserved')")

    backups = tmp_path / "backups"
    database = Database(database_path, backups)
    await database.initialize()

    journal = await database.fetch_one("PRAGMA journal_mode")
    foreign_keys = await database.fetch_one("PRAGMA foreign_keys")
    migrations = await database.fetch_all("SELECT version FROM schema_migrations ORDER BY version")
    legacy_row = await database.fetch_one("SELECT value FROM legacy_data")
    await database.close()

    assert journal == {"journal_mode": "wal"}
    assert foreign_keys == {"foreign_keys": 1}
    assert migrations == [
        {"version": "0001_initial"},
        {"version": "0002_token_reasoning_slices"},
        {"version": "0003_token_attention_attribution"},
    ]
    assert legacy_row == {"value": "preserved"}

    backup_paths = list(backups.glob("*.sqlite3"))
    assert len(backup_paths) == 1
    with sqlite3.connect(backup_paths[0]) as backup:
        assert backup.execute("SELECT value FROM legacy_data").fetchone() == ("preserved",)

    reopened = Database(database_path, backups)
    await reopened.initialize()
    await reopened.close()
    assert list(backups.glob("*.sqlite3")) == backup_paths


@pytest.mark.asyncio
async def test_deleting_a_chat_cascades_run_telemetry_but_retains_attachment(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.sqlite3", tmp_path / "backups")
    await database.initialize()
    repository = WorkspaceRepository(database)
    try:
        chat = await repository.create_chat("cascade")
        message = await repository.create_message(
            chat_id=chat["id"], role="assistant", content="partial", status="streaming"
        )
        attachment = await repository.create_attachment(
            {
                "id": "attachment-1",
                "sha256": "a" * 64,
                "storage_name": f"{'a' * 64}.txt",
                "original_name": "note.txt",
                "media_type": "text/plain",
                "size_bytes": 4,
            }
        )
        await database.execute(
            "INSERT INTO message_attachments(message_id, attachment_id, ordinal) VALUES (?, ?, ?)",
            (message["id"], attachment["id"], 0),
        )
        run = await repository.create_run(
            {
                "id": "run-cascade",
                "message_id": message["id"],
                "status": "complete",
                "effective_seed": 0,
            }
        )
        now = utc_now()
        await database.execute(
            "INSERT INTO environment_snapshots(run_id, hardware_json, software_json, backend_json, captured_at) VALUES (?, '{}', '{}', '{}', ?)",
            (run["id"], now),
        )
        await database.execute(
            "INSERT INTO phase_metrics(run_id, phase, details_json) VALUES (?, 'prefill', '{}')",
            (run["id"],),
        )
        await database.execute(
            "INSERT INTO token_events(run_id, token_index, token_id, piece, escaped_bytes, display_text, created_at) VALUES (?, 0, 7, 'x', '\\x78', 'x', ?)",
            (run["id"], now),
        )
        await database.execute(
            "INSERT INTO token_alternatives(run_id, token_index, distribution, rank, token_id, piece, survived_filter) VALUES (?, 0, 'raw', 1, 7, 'x', 1)",
            (run["id"],),
        )
        await database.execute(
            "INSERT INTO router_events(run_id, token_index, layer_index, selected_json, executed_json) VALUES (?, 0, 0, '[]', '[]')",
            (run["id"],),
        )
        await database.execute(
            "INSERT INTO router_aggregates(run_id, layer_index, expert_id, activation_count) VALUES (?, 0, 0, 1)",
            (run["id"],),
        )
        await repository.append_raw_event(
            {
                "run_id": run["id"],
                "sequence": 1,
                "type": "token",
                "monotonic_ns": 1,
                "payload": {"text": "x"},
            }
        )

        assert await repository.delete_chat(chat["id"])

        for table in (
            "chats",
            "messages",
            "message_attachments",
            "inference_runs",
            "environment_snapshots",
            "phase_metrics",
            "token_events",
            "token_alternatives",
            "router_events",
            "router_aggregates",
            "raw_events",
        ):
            count = await database.fetch_one(f"SELECT COUNT(*) AS count FROM {table}")
            assert count == {"count": 0}, table
        assert await repository.get_attachment(attachment["id"]) is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_generation_setup_rolls_back_messages_when_run_insert_fails(tmp_path: Path) -> None:
    database = Database(tmp_path / "atomic.sqlite3", tmp_path / "backups")
    await database.initialize()
    repository = WorkspaceRepository(database)
    try:
        chat = await repository.create_chat("New chat")
        await database.execute(
            """
            CREATE TRIGGER reject_test_run
            BEFORE INSERT ON inference_runs
            BEGIN
                SELECT RAISE(ABORT, 'forced run insert failure');
            END
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="forced run insert failure"):
            await repository.create_generation_setup(
                chat_id=chat["id"],
                user_content="must roll back",
                parent_message_id=None,
                run={"id": "failed-setup", "effective_seed": 0},
                hardware={},
                software={},
                backend={},
                chat_title="must not survive",
            )

        persisted_chat = await repository.get_chat(chat["id"])
        assert persisted_chat is not None
        assert persisted_chat["title"] == "New chat"
        assert persisted_chat["messages"] == []
        assert await repository.get_run("failed-setup") is None
        assert await database.fetch_one("SELECT COUNT(*) AS count FROM environment_snapshots") == {
            "count": 0
        }
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sibling_user_messages_get_increasing_branch_indexes(tmp_path: Path) -> None:
    database = Database(tmp_path / "siblings.sqlite3", tmp_path / "backups")
    await database.initialize()
    repository = WorkspaceRepository(database)
    try:
        chat = await repository.create_chat("siblings")
        other_chat = await repository.create_chat("other")

        async def send(
            chat_id: str, parent_id: str | None, run_id: str
        ) -> dict[str, dict[str, Any]]:
            return await repository.create_generation_setup(
                chat_id=chat_id,
                user_content=run_id,
                parent_message_id=parent_id,
                run={"id": run_id, "effective_seed": 0},
                hardware={},
                software={},
                backend={},
            )

        # Two roots in one chat, then two edits under the same parent.
        first_root = await send(chat["id"], None, "root-1")
        second_root = await send(chat["id"], None, "root-2")
        await send(other_chat["id"], None, "other-root")
        parent_id = first_root["assistant_message"]["id"]
        first_edit = await send(chat["id"], parent_id, "edit-1")
        second_edit = await send(chat["id"], parent_id, "edit-2")

        assert first_root["user_message"]["branch_index"] == 0
        assert second_root["user_message"]["branch_index"] == 1
        assert first_edit["user_message"]["branch_index"] == 0
        assert second_edit["user_message"]["branch_index"] == 1
        # The assistant reply to a fresh user message is always that message's first child.
        assert second_edit["assistant_message"]["branch_index"] == 0
        assert (await repository.next_branch_index(parent_id)) == 2
        other = await repository.get_chat(other_chat["id"])
        assert other is not None
        assert other["messages"][0]["branch_index"] == 0
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_restart_recovery_preserves_partial_output_and_is_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    backups = tmp_path / "backups"
    first_database = Database(database_path, backups)
    await first_database.initialize()
    first_repository = WorkspaceRepository(first_database)
    chat = await first_repository.create_chat("restart")
    interrupted_message = await first_repository.create_message(
        chat_id=chat["id"],
        role="assistant",
        content="already streamed",
        status="streaming",
    )
    completed_message = await first_repository.create_message(
        chat_id=chat["id"], role="assistant", content="done", status="complete"
    )
    interrupted_run = await first_repository.create_run(
        {
            "id": "interrupted",
            "message_id": interrupted_message["id"],
            "status": "running",
            "effective_seed": 0,
        }
    )
    completed_run = await first_repository.create_run(
        {
            "id": "completed",
            "message_id": completed_message["id"],
            "status": "complete",
            "effective_seed": 1,
        }
    )
    await first_database.close()

    restarted_database = Database(database_path, backups)
    await restarted_database.initialize()
    restarted_repository = WorkspaceRepository(restarted_database)
    try:
        assert await restarted_repository.recover_incomplete_runs() == [interrupted_run["id"]]
        assert await restarted_repository.recover_incomplete_runs() == []

        recovered_run = await restarted_repository.get_run(interrupted_run["id"])
        recovered_message = await restarted_repository.get_message(interrupted_message["id"])
        untouched_run = await restarted_repository.get_run(completed_run["id"])
        untouched_message = await restarted_repository.get_message(completed_message["id"])

        assert recovered_run is not None
        assert recovered_run["status"] == "failed"
        assert recovered_run["finish_reason"] == "application_restart"
        assert recovered_run["error_code"] == "interrupted_by_restart"
        assert recovered_run["completed_at"] is not None
        assert recovered_message is not None
        assert recovered_message["content"] == "already streamed"
        assert recovered_message["status"] == "failed"
        assert untouched_run is not None and untouched_run["status"] == "complete"
        assert untouched_message is not None and untouched_message["status"] == "complete"
    finally:
        await restarted_database.close()


@pytest.mark.asyncio
async def test_telemetry_batches_preserve_foreign_key_submission_order(tmp_path: Path) -> None:
    database = Database(tmp_path / "telemetry.sqlite3", tmp_path / "backups")
    await database.initialize()
    await database.execute("CREATE TABLE batch_parent(id INTEGER PRIMARY KEY)")
    await database.execute(
        "CREATE TABLE batch_child(id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL REFERENCES batch_parent(id))"
    )
    writer = TelemetryWriter(
        database,
        batch_size=3,
        flush_interval_seconds=60.0,
    )
    writer.start()
    try:
        await writer.submit(
            "INSERT INTO batch_parent(id) VALUES (?)",
            [(1,), (10,), (11,)],
        )
        await writer.flush()

        # This deliberately starts a flush window with a child statement and
        # repeats that SQL after a new parent statement. SQL-wide regrouping
        # would execute child 2 before parent 2 and violate the foreign key.
        await writer.submit("INSERT INTO batch_child(id, parent_id) VALUES (?, ?)", [(1, 1)])
        await writer.submit("INSERT INTO batch_parent(id) VALUES (?)", [(2,)])
        await writer.submit("INSERT INTO batch_child(id, parent_id) VALUES (?, ?)", [(2, 2)])
        await writer.flush()

        assert await database.fetch_one("SELECT COUNT(*) AS count FROM batch_child") == {"count": 2}
    finally:
        await writer.stop()
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_status", ["queued", "loading", "running"])
async def test_active_generation_blocks_chat_deletion_and_clear(
    tmp_path: Path, run_status: str
) -> None:
    database = Database(tmp_path / "active.sqlite3", tmp_path / "backups")
    await database.initialize()
    repository = WorkspaceRepository(database)
    try:
        chat = await repository.create_chat("active")
        assistant = await repository.create_message(
            chat_id=chat["id"], role="assistant", content="", status="streaming"
        )
        run = await repository.create_run(
            {
                "id": f"active-generation-{run_status}",
                "message_id": assistant["id"],
                "kind": "generation",
                "status": run_status,
                "effective_seed": 0,
            }
        )

        with pytest.raises(ActiveRunConflictError) as deleting:
            await repository.delete_chat(chat["id"])
        assert deleting.value.details == {
            "chat_id": chat["id"],
            "run_ids": [run["id"]],
        }

        with pytest.raises(ActiveRunConflictError) as clearing:
            await repository.clear_chats()
        assert clearing.value.details == {
            "run_ids": [run["id"]],
            "chat_ids": [chat["id"]],
        }
        assert await repository.get_chat(chat["id"]) is not None

        await repository.update_run(run["id"], status="cancelled", completed_at=utc_now())
        assert await repository.delete_chat(chat["id"])
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_telemetry_writer_surfaces_background_failure_without_hanging(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "failed-writer.sqlite3", tmp_path / "backups")
    await database.initialize()
    await database.execute("CREATE TABLE writer_parent(id INTEGER PRIMARY KEY)")
    await database.execute(
        "CREATE TABLE writer_child(parent_id INTEGER REFERENCES writer_parent(id))"
    )
    writer = TelemetryWriter(
        database,
        max_queue_size=1,
        batch_size=1,
        flush_interval_seconds=60.0,
    )
    writer.start()
    try:
        await writer.submit("INSERT INTO writer_child(parent_id) VALUES (?)", [(404,)])

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await asyncio.wait_for(writer.flush(), timeout=1.0)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await asyncio.wait_for(
                writer.submit("INSERT INTO writer_parent(id) VALUES (?)", [(1,)]),
                timeout=1.0,
            )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await asyncio.wait_for(writer.stop(), timeout=1.0)
    finally:
        with contextlib.suppress(sqlite3.IntegrityError):
            await asyncio.wait_for(writer.stop(), timeout=1.0)
        await database.close()
