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
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str,
    ) -> AsyncIterator[dict[str, dict[str, Any]]]:
        assert state is not None
        assert config is not None
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
    monkeypatch.setattr(main, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
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


def test_get_run_returns_persisted_run_details(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, _store = api_client

    create_response = client.post("/runs", json={"goal": "Explain LangGraph", "max_loops": 3})
    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}")
    body = response.json()

    assert response.status_code == 200
    assert body["run_id"] == run_id
    assert body["status"] == "created"
    assert body["created_at"]
    assert body["updated_at"]
    assert body["state"]["goal"] == "Explain LangGraph"
    assert body["state"]["max_loops"] == 3
    assert body["state"]["summary"] is None


def test_stream_run_executes_graph_and_emits_progress(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, store = api_client
    monkeypatch.setattr(main, "build_graph", lambda *, checkpointer=None: FakeGraph())

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


def test_get_run_returns_completed_run_details_after_stream(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _store = api_client
    monkeypatch.setattr(main, "build_graph", lambda *, checkpointer=None: FakeGraph())

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    stream_response = client.get(f"/runs/{run_id}/stream")
    detail_response = client.get(f"/runs/{run_id}")
    body = detail_response.json()

    assert stream_response.status_code == 200
    assert detail_response.status_code == 200
    assert body["run_id"] == run_id
    assert body["status"] == "completed"
    assert body["state"]["completed_tasks"] == ["fake task"]
    assert body["state"]["results"] == ["fake result"]
    assert body["state"]["summary"] == "Final API summary."


def test_startup_marks_stale_running_runs_as_failed(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    _client, store = api_client

    store.create_run("stale-run", main.build_initial_state(main.CreateRunRequest(goal="Resume me")))
    store.update_run("stale-run", status="running")

    with TestClient(main.app) as client:
        response = client.get("/runs/stale-run")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_stream_run_returns_cached_summary_for_completed_run(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _store = api_client
    monkeypatch.setattr(main, "build_graph", lambda *, checkpointer=None: FakeGraph())

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


def test_get_run_returns_404_for_unknown_run(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, _store = api_client

    response = client.get("/runs/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_pause_and_resume_update_run_status(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, store = api_client

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]
    store.update_run(run_id, status="running")

    pause_response = client.post(f"/runs/{run_id}/pause")
    paused_record = store.get_run(run_id)
    resume_response = client.post(f"/runs/{run_id}/resume")
    resumed_record = store.get_run(run_id)

    assert pause_response.status_code == 200
    assert pause_response.json() == {"run_id": run_id, "status": "paused"}
    assert paused_record is not None
    assert paused_record["status"] == "paused"
    assert resume_response.status_code == 200
    assert resume_response.json() == {"run_id": run_id, "status": "running"}
    assert resumed_record is not None
    assert resumed_record["status"] == "running"


def test_pause_rejects_non_running_run(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, _store = api_client

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    response = client.post(f"/runs/{run_id}/pause")

    assert response.status_code == 409
    assert response.json()["detail"] == "Cannot pause a run with status 'created'"


def test_cancel_updates_run_status(
    api_client: tuple[TestClient, SQLiteRunStore],
) -> None:
    client, store = api_client

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    response = client.post(f"/runs/{run_id}/cancel")
    record = store.get_run(run_id)

    assert response.status_code == 200
    assert response.json() == {"run_id": run_id, "status": "cancelled"}
    assert record is not None
    assert record["status"] == "cancelled"


def test_stream_run_stops_when_run_is_paused(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, store = api_client

    class PausingGraph:
        async def astream(
            self,
            state: AgentState,
            config: dict[str, Any] | None = None,
            *,
            stream_mode: str,
        ) -> AsyncIterator[dict[str, dict[str, Any]]]:
            assert state is not None
            assert config is not None
            assert stream_mode == "updates"

            store.update_run(config["configurable"]["thread_id"], status="paused")
            yield {"plan": {"tasks": ["fake task"], "loop_count": 0, "current_analysis": None}}
            yield {"pick_task": {"current_task": "fake task", "tasks": []}}

    monkeypatch.setattr(main, "build_graph", lambda *, checkpointer=None: PausingGraph())

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}/stream")
    payloads = sse_payloads(response.text)
    record = store.get_run(run_id)

    assert response.status_code == 200
    assert [payload["type"] for payload in payloads] == [
        "status",
        "node",
        "plan",
        "status",
        "done",
    ]
    assert payloads[-2]["payload"] == {"status": "paused"}
    assert record is not None
    assert record["status"] == "paused"
    assert record["state"]["tasks"] == ["fake task"]


def test_stream_run_skips_cached_checkpoint_updates(
    api_client: tuple[TestClient, SQLiteRunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, store = api_client

    class CachedResumeGraph:
        async def astream(
            self,
            state: AgentState | None,
            config: dict[str, Any] | None = None,
            *,
            stream_mode: str,
        ) -> AsyncIterator[dict[str, Any]]:
            assert state is None
            assert config is not None
            assert stream_mode == "updates"

            yield {
                "execute": {
                    "completed_tasks": ["old task"],
                    "results": ["old result"],
                    "loop_count": 1,
                },
                "__metadata__": {"cached": True},
            }
            yield {"summarize": {"summary": "Resumed summary."}}

    async def fake_checkpoint_exists(checkpointer: Any, run_id: str) -> bool:
        return True

    monkeypatch.setattr(main, "build_graph", lambda *, checkpointer=None: CachedResumeGraph())
    monkeypatch.setattr(main, "checkpoint_exists", fake_checkpoint_exists)

    create_response = client.post("/runs", json={"goal": "Explain LangGraph"})
    run_id = create_response.json()["run_id"]
    state = main.build_initial_state(main.CreateRunRequest(goal="Explain LangGraph"))
    state["completed_tasks"] = ["old task"]
    state["results"] = ["old result"]
    store.update_run(run_id, status="running", state=state)

    response = client.get(f"/runs/{run_id}/stream")
    record = store.get_run(run_id)

    assert response.status_code == 200
    assert record is not None
    assert record["status"] == "completed"
    assert record["state"]["completed_tasks"] == ["old task"]
    assert record["state"]["results"] == ["old result"]
    assert record["state"]["summary"] == "Resumed summary."
