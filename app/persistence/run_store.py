from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from app.agent.state import AgentState
from app.api.schemas import RunStatus

DEFAULT_DB_PATH = Path("data/runs.sqlite")


class StoredRun(TypedDict):
    run_id: str
    status: RunStatus
    state: AgentState
    created_at: str
    updated_at: str


class SQLiteRunStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_run(self, run_id: str, state: AgentState) -> StoredRun:
        now = datetime.now(UTC).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (run_id, status, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, "created", json.dumps(state), now, now),
            )

        return {
            "run_id": run_id,
            "status": "created",
            "state": state,
            "created_at": now,
            "updated_at": now,
        }

    def get_run(self, run_id: str) -> StoredRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, status, state_json, created_at, updated_at
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "run_id": str(row["run_id"]),
            "status": cast(RunStatus, row["status"]),
            "state": cast(AgentState, json.loads(row["state_json"])),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        state: AgentState | None = None,
    ) -> StoredRun | None:
        existing = self.get_run(run_id)
        if existing is None:
            return None

        next_status = status if status is not None else existing["status"]
        next_state = state if state is not None else existing["state"]
        updated_at = datetime.now(UTC).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, state_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (next_status, json.dumps(next_state), updated_at, run_id),
            )

        return {
            "run_id": run_id,
            "status": next_status,
            "state": next_state,
            "created_at": existing["created_at"],
            "updated_at": updated_at,
        }
