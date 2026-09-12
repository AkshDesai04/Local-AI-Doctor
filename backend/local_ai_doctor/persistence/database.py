"""Async SQLite lifecycle, migrations, and batched telemetry writes."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite


class DatabaseError(RuntimeError):
    """Raised when a durable database operation cannot be completed."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    """Owns one WAL-mode connection and applies append-only SQL migrations."""

    def __init__(self, path: Path, backup_directory: Path) -> None:
        self.path = path
        self.backup_directory = backup_directory
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise DatabaseError("Database has not been initialized")
        return self._connection

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists() and self.path.stat().st_size > 0
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        self._connection = connection
        try:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA synchronous = NORMAL")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            await connection.commit()
            await self._apply_migrations(existed)
        except Exception as exc:
            await connection.close()
            self._connection = None
            raise DatabaseError(f"Unable to initialize database at {self.path}: {exc}") from exc

    async def _apply_migrations(self, database_existed: bool) -> None:
        migration_dir = Path(__file__).with_name("migrations")
        migrations = sorted(migration_dir.glob("*.sql"))
        cursor = await self.connection.execute("SELECT version FROM schema_migrations")
        applied = {str(row[0]) for row in await cursor.fetchall()}
        pending = [migration for migration in migrations if migration.stem not in applied]
        if pending and database_existed:
            await self._backup()
        for migration in pending:
            script = migration.read_text(encoding="utf-8")
            try:
                await self.connection.executescript(script)
                await self.connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (migration.stem, utc_now()),
                )
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

    async def _backup(self) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.backup_directory / f"{self.path.stem}-{timestamp}.sqlite3"
        target = sqlite3.connect(destination)
        try:
            await self.connection.backup(target)
        finally:
            target.close()
        return destination

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def execute(self, sql: str, parameters: Sequence[Any] = ()) -> None:
        async with self._write_lock:
            try:
                await self.connection.execute(sql, parameters)
                await self.connection.commit()
            except Exception as exc:
                await self.connection.rollback()
                raise DatabaseError(str(exc)) from exc

    async def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        values = list(rows)
        if not values:
            return
        async with self._write_lock:
            try:
                await self.connection.executemany(sql, values)
                await self.connection.commit()
            except Exception as exc:
                await self.connection.rollback()
                raise DatabaseError(str(exc)) from exc

    async def fetch_one(self, sql: str, parameters: Sequence[Any] = ()) -> dict[str, Any] | None:
        cursor = await self.connection.execute(sql, parameters)
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def fetch_all(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cursor = await self.connection.execute(sql, parameters)
        return [dict(row) for row in await cursor.fetchall()]

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._write_lock:
            await self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                await self.connection.rollback()
                raise
            else:
                await self.connection.commit()


@dataclass(slots=True)
class _WriteBatch:
    sql: str
    rows: list[Sequence[Any]]


@dataclass(slots=True)
class _FlushRequest:
    completed: asyncio.Future[None]


class TelemetryWriter:
    """Moves bounded telemetry persistence away from the inference hot path."""

    def __init__(
        self,
        database: Database,
        *,
        max_queue_size: int = 4096,
        batch_size: int = 64,
        flush_interval_seconds: float = 0.1,
    ) -> None:
        self.database = database
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self._queue: asyncio.Queue[_WriteBatch | _FlushRequest | None] = asyncio.Queue(
            max_queue_size
        )
        self._task: asyncio.Task[None] | None = None
        self._failure: Exception | None = None

    def start(self) -> None:
        self._raise_if_failed()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="telemetry-writer")
            self._task.add_done_callback(self._remember_failure)

    def _remember_failure(self, task: asyncio.Task[None]) -> None:
        """Retrieve and retain background failures for every public operation."""

        if task.cancelled():
            if self._failure is None:
                self._failure = DatabaseError("Telemetry writer was cancelled")
            return
        failure = task.exception()
        if failure is not None:
            if isinstance(failure, Exception):
                self._failure = failure
            else:  # pragma: no cover - task cancellation is handled above
                self._failure = DatabaseError("Telemetry writer stopped abnormally")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._task is not None and self._task.done():
            self._remember_failure(self._task)
            if self._failure is not None:
                raise self._failure

    def _running_task(self) -> asyncio.Task[None]:
        self._raise_if_failed()
        if self._task is None:
            raise DatabaseError("Telemetry writer has not been started")
        if self._task.done():
            raise DatabaseError("Telemetry writer stopped unexpectedly")
        return self._task

    async def _enqueue(
        self,
        item: _WriteBatch | _FlushRequest | None,
        *,
        allow_clean_stop: bool = False,
    ) -> None:
        """Enqueue without waiting forever if the sole consumer has failed."""

        writer_task = self._running_task()
        put_task = asyncio.create_task(self._queue.put(item))
        try:
            done, _pending = await asyncio.wait(
                {put_task, writer_task}, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            if not put_task.done():
                put_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await put_task
            raise
        if writer_task in done:
            if not put_task.done():
                put_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await put_task
            self._raise_if_failed()
            if allow_clean_stop and put_task.done() and not put_task.cancelled():
                return
            raise DatabaseError("Telemetry writer stopped unexpectedly")
        await put_task
        self._raise_if_failed()

    async def submit(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            self._raise_if_failed()
            return
        await self._enqueue(_WriteBatch(sql=sql, rows=list(rows)))

    async def stop(self) -> None:
        if self._task is None:
            self._raise_if_failed()
            return
        writer_task = self._task
        try:
            if not writer_task.done():
                await self._enqueue(None, allow_clean_stop=True)
            await asyncio.shield(writer_task)
        except asyncio.CancelledError:
            if writer_task.cancelled():
                failure = DatabaseError("Telemetry writer was cancelled")
                self._failure = failure
                raise failure from None
            raise
        except Exception as exc:
            self._failure = exc
            raise
        finally:
            # If this coroutine itself is cancelled, the shielded writer may still
            # be draining. Keep the handle so a later stop can await it safely.
            if writer_task.done():
                self._task = None

    async def flush(self) -> None:
        if self._task is None:
            self._raise_if_failed()
            return
        completed = asyncio.get_running_loop().create_future()
        await self._enqueue(_FlushRequest(completed))
        writer_task = self._running_task()
        done, _pending = await asyncio.wait(
            {completed, writer_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if writer_task in done:
            self._raise_if_failed()
            raise DatabaseError("Telemetry writer stopped unexpectedly")
        await completed
        self._raise_if_failed()

    async def _run(self) -> None:
        pending: list[_WriteBatch] = []
        stopping = False
        while not stopping:
            timed_out = False
            flush_request: _FlushRequest | None = None
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self.flush_interval_seconds
                )
                if item is None:
                    stopping = True
                elif isinstance(item, _FlushRequest):
                    flush_request = item
                else:
                    pending.append(item)
            except TimeoutError:
                timed_out = True

            row_count = sum(len(batch.rows) for batch in pending)
            if pending and (
                stopping or timed_out or flush_request is not None or row_count >= self.batch_size
            ):
                try:
                    # Submission order is semantically significant: token alternatives
                    # reference their token row, and a flush may begin between those two
                    # submissions. Grouping all rows by SQL can therefore move a later
                    # child batch ahead of its parent and violate foreign keys.
                    async with self.database.transaction() as connection:
                        for batch in pending:
                            await connection.executemany(batch.sql, batch.rows)
                    pending.clear()
                except Exception as exc:
                    if flush_request is not None and not flush_request.completed.done():
                        flush_request.completed.set_exception(exc)
                    raise
            if flush_request is not None and not flush_request.completed.done():
                flush_request.completed.set_result(None)
