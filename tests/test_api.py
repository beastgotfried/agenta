from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.state import AgentState
from app.api import main


class FakeGraph:
    async def ainvoke(self, state: AgentState) -> AgentState:
        return {
            "goal": state["goal"],
            "language": state["language"],
            "max_loops": state["max_loops"],
            "user_id": state["user_id"],
            "user_context": state["user_context"],
            "tasks": [],
            "current_task": None,
            "current_analysis": None,
            "completed_tasks": ["fake task"],
            "results": ["fake result"],
            "loop_count": 1,
            "summary": "Final API summary.",
        }


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


def test_stream_run_executes_graph_and_emits_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    main.RUNS.clear()
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph())
    client = TestClient(main.app)

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}/stream")

    assert response.status_code == 200
    assert 'event: status\r\ndata: {"status": "running"' in response.text
    assert 'event: summary\r\ndata: {"text": "Final API summary."}' in response.text
    assert "event: done" in response.text
    assert main.RUNS[run_id]["status"] == "completed"
    assert main.RUNS[run_id]["state"]["summary"] == "Final API summary."


def test_stream_run_returns_404_for_unknown_run() -> None:
    main.RUNS.clear()
    client = TestClient(main.app)

    response = client.get("/runs/missing/stream")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"
