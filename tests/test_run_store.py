from pathlib import Path

from app.agent.state import AgentState
from app.persistence.run_store import SQLiteRunStore


def make_state() -> AgentState:
    return {
        "goal": "Explain LangGraph",
        "language": "English",
        "max_loops": 3,
        "user_id": "local",
        "user_context": "",
        "tasks": [],
        "current_task": None,
        "current_analysis": None,
        "completed_tasks": [],
        "results": [],
        "loop_count": 0,
        "summary": None,
    }


def test_run_store_persists_run_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.sqlite"

    store = SQLiteRunStore(db_path)
    store.create_run("run-1", make_state())

    updated_state = make_state()
    updated_state["summary"] = "Finished."
    store.update_run("run-1", status="completed", state=updated_state)

    reloaded_store = SQLiteRunStore(db_path)
    record = reloaded_store.get_run("run-1")

    assert record is not None
    assert record["run_id"] == "run-1"
    assert record["status"] == "completed"
    assert record["state"]["summary"] == "Finished."
