from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.state import AgentState
from app.api import main
from app.persistence.run_store import SQLiteRunStore


class FakeGraph:
    async def astream(
        self,
        state: AgentState,
        *,
        stream_mode: str,
    ) -> AsyncIterator[dict[str, dict[str, Any]]]:
        assert stream_mode == "updates"

        yield {"plan": {"tasks": ["fake task"], "loop_count": 0, "current_analysis": None}}
        yield {"pick_task": {"current_task": "fake task", "tasks": [], "current_analysis": None}}
        yield {
            "analyze": {
                "current_analysis": {
                    "reasoning": "Use fake tool.",
                    "action": "conclude",
                    "arg": "done",
                }
            }
        }
        yield {
            "execute": {
                "completed_tasks": ["fake task"],
                "results": ["fake result"],
                "loop_count": 1,
            }
        }
        yield {"pick_task": {"current_task": None, "current_analysis": None}}
        yield {"summarize": {"summary": "Final API summary."}}


@pytest.fixture
def api_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, SQLiteRunStore]:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    monkeypatch.setattr(main, "RUN_STORE", store)
    return TestClient(main.app), store


def test_create_run_stores_initial_state(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, store = api_client

    response = client.post("/runs", json={"goal": "Explain LangGraph", "max_loops": 3})

    assert response.status_code == 201
    body = response.json()
    record = store.get_run(body["run_id"])

    assert body["status"] == "created"
    assert record is not None
    assert record["state"]["goal"] == "Explain LangGraph"
    assert record["state"]["max_loops"] == 3


def test_stream_run_executes_graph_and_emits_progress(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, store = api_client
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph())

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}/stream")
    record = store.get_run(run_id)

    assert response.status_code == 200
    assert 'event: status\r\ndata: {"status": "running"' in response.text
    assert 'event: node\r\ndata: {"node": "plan"}' in response.text
    assert 'event: plan\r\ndata: {"tasks": ["fake task"]}' in response.text
    assert 'event: task\r\ndata: {"task": "fake task"}' in response.text
    assert 'event: node\r\ndata: {"node": "analyze"}' in response.text
    assert 'event: analysis\r\ndata: {"task": "fake task"' in response.text
    assert (
        'event: task_done\r\ndata: {"task": "fake task", "result": "fake result"'
        in response.text
    )
    assert 'event: summary\r\ndata: {"text": "Final API summary."}' in response.text
    assert "event: done" in response.text

    assert record is not None
    assert record["status"] == "completed"
    assert record["state"]["completed_tasks"] == ["fake task"]
    assert record["state"]["results"] == ["fake result"]
    assert record["state"]["summary"] == "Final API summary."


def test_stream_run_returns_cached_summary_for_completed_run(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _store = api_client
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph())

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    first_response = client.get(f"/runs/{run_id}/stream")
    second_response = client.get(f"/runs/{run_id}/stream")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert 'event: status\r\ndata: {"status": "running"' not in second_response.text
    assert 'event: summary\r\ndata: {"text": "Final API summary."}' in second_response.text
    assert "event: done" in second_response.text


def test_stream_run_returns_404_for_unknown_run(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, _store = api_client

    response = client.get("/runs/missing/stream")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"
