from __future__ import annotations

from typing import Any, cast

import pytest

from local_ai_doctor.persistence.database import Database
from local_ai_doctor.persistence.repository import WorkspaceRepository


class _TokenQueryDatabase:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetch_all(self, sql: str, _parameters: tuple[object, ...]) -> list[dict[str, Any]]:
        self.queries.append(sql)
        if "FROM token_events" in sql:
            return [
                {"run_id": "run", "token_index": 0, "token_id": 10},
                {"run_id": "run", "token_index": 1, "token_id": 11},
            ]
        return [
            {"run_id": "run", "token_index": 0, "distribution": "raw", "rank": 1},
            {"run_id": "run", "token_index": 1, "distribution": "sampling", "rank": 1},
        ]


@pytest.mark.asyncio
async def test_run_token_alternatives_are_loaded_in_one_bounded_query() -> None:
    database = _TokenQueryDatabase()
    repository = WorkspaceRepository(cast(Database, database))

    tokens = await repository.list_run_tokens("run")

    assert len(database.queries) == 2
    assert tokens[0]["alternatives"][0]["distribution"] == "raw"
    assert tokens[1]["alternatives"][0]["distribution"] == "sampling"
