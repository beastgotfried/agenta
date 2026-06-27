from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DEFAULT_CHECKPOINT_DB_PATH = Path("data/checkpoints.sqlite")


def thread_config(run_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": run_id}}


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver()


@asynccontextmanager
async def sqlite_checkpointer(
    db_path: Path = DEFAULT_CHECKPOINT_DB_PATH,
) -> AsyncIterator[AsyncSqliteSaver]:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver


async def checkpoint_exists(checkpointer: Any, run_id: str) -> bool:
    checkpoint = await checkpointer.aget_tuple(thread_config(run_id))
    return checkpoint is not None
