from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.state import AgentState
from app.api import main


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


def test_create_run_stores_initial_state() -> None:
    main.RUNS.clear()
    client = TestClient(main.app)

    response = client.post("/runs", json={"goal": "Explain LangGraph", "max_loops": 3})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "created"
    assert body["run_id"] in main.RUNS
    assert main.RUNS[body["run_id"]]["state"]["goal"] == "Explain LangGraph"
    assert main.RUNS[body["run_id"]]["state"]["max_loops"] == 3


def test_stream_run_executes_graph_and_emits_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    main.RUNS.clear()
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph())
    client = TestClient(main.app)

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}/stream")

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

    assert main.RUNS[run_id]["status"] == "completed"
    assert main.RUNS[run_id]["state"]["completed_tasks"] == ["fake task"]
    assert main.RUNS[run_id]["state"]["results"] == ["fake result"]
    assert main.RUNS[run_id]["state"]["summary"] == "Final API summary."


def test_stream_run_returns_cached_summary_for_completed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main.RUNS.clear()
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph())
    client = TestClient(main.app)

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    first_response = client.get(f"/runs/{run_id}/stream")
    second_response = client.get(f"/runs/{run_id}/stream")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert 'event: status\r\ndata: {"status": "running"' not in second_response.text
    assert 'event: summary\r\ndata: {"text": "Final API summary."}' in second_response.text
    assert "event: done" in second_response.text


def test_stream_run_returns_404_for_unknown_run() -> None:
    main.RUNS.clear()
    client = TestClient(main.app)

    response = client.get("/runs/missing/stream")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"
