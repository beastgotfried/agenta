from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from app.persistence.checkpointer import sqlite_checkpointer, thread_config


class CounterState(TypedDict):
    count: int


def increment(state: CounterState) -> CounterState:
    return {"count": state["count"] + 1}


def build_counter_graph(checkpointer):
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_sqlite_checkpointer_persists_state_across_connections(tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    config = thread_config("run-1")

    async with sqlite_checkpointer(db_path) as checkpointer:
        graph = build_counter_graph(checkpointer)
        result = await graph.ainvoke({"count": 1}, config)

    async with sqlite_checkpointer(db_path) as checkpointer:
        graph = build_counter_graph(checkpointer)
        snapshot = await graph.aget_state(config)

    assert result == {"count": 2}
    assert snapshot.values == {"count": 2}
