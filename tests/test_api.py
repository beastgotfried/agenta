from __future__ import annotations

import json
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


def sse_payloads(response_text: str) -> list[dict[str, Any]]:
    payloads = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


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
    payloads = sse_payloads(response.text)

    assert response.status_code == 200
    assert [payload["sequence"] for payload in payloads] == list(range(1, len(payloads) + 1))
    assert {payload["run_id"] for payload in payloads} == {run_id}
    assert [payload["type"] for payload in payloads] == [
        "status",
        "node",
        "plan",
        "node",
        "task",
        "node",
        "analysis",
        "node",
        "task_done",
        "node",
        "node",
        "summary",
        "done",
    ]
    assert payloads[0]["payload"] == {"status": "running"}
    assert payloads[1]["payload"] == {"node": "plan"}
    assert payloads[2]["payload"] == {"tasks": ["fake task"]}
    assert payloads[4]["payload"] == {"task": "fake task"}
    assert payloads[6]["payload"] == {
        "task": "fake task",
        "analysis": {
            "reasoning": "Use fake tool.",
            "action": "conclude",
            "arg": "done",
        },
    }
    assert payloads[8]["payload"] == {
        "task": "fake task",
        "result": "fake result",
        "loop_count": 1,
    }
    assert payloads[11]["payload"] == {"text": "Final API summary."}
    assert payloads[-1]["payload"] == {}

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
    second_payloads = sse_payloads(second_response.text)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert [payload["type"] for payload in second_payloads] == ["summary", "done"]
    assert [payload["sequence"] for payload in second_payloads] == [1, 2]
    assert {payload["run_id"] for payload in second_payloads} == {run_id}
    assert second_payloads[0]["payload"] == {"text": "Final API summary."}
    assert second_payloads[1]["payload"] == {}


def test_stream_run_returns_404_for_unknown_run(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, _store = api_client

    response = client.get("/runs/missing/stream")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"
