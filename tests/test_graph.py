from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.agent import nodes
from app.agent.graph import build_graph, route_after_execute, route_after_pick
from app.agent.schemas import FollowupTask, Plan, ToolChoice
from app.agent.state import AgentState


def make_state(
    *,
    current_task: str | None = "task",
    loop_count: int = 0,
    max_loops: int = 5,
    expand_tasks: bool = False,
) -> AgentState:
    return {
        "goal": "Write a compact report.",
        "language": "English",
        "max_loops": max_loops,
        "expand_tasks": expand_tasks,
        "user_id": "local",
        "user_context": "",
        "tasks": [],
        "current_task": current_task,
        "current_analysis": None,
        "completed_tasks": [],
        "results": [],
        "loop_count": loop_count,
        "summary": None,
    }


def test_route_after_pick_routes_to_analyze_for_active_task() -> None:
    assert route_after_pick(make_state()) == "analyze"


def test_route_after_pick_routes_to_summary_when_no_task() -> None:
    assert route_after_pick(make_state(current_task=None)) == "summarize"


def test_route_after_pick_routes_to_summary_at_loop_limit() -> None:
    assert route_after_pick(make_state(loop_count=2, max_loops=2)) == "summarize"


def test_route_after_execute_routes_to_pick_task_when_expansion_disabled() -> None:
    assert route_after_execute(make_state(expand_tasks=False)) == "pick_task"


def test_route_after_execute_routes_to_create_tasks_when_expansion_enabled() -> None:
    assert route_after_execute(make_state(expand_tasks=True, loop_count=1, max_loops=3)) == (
        "create_tasks"
    )


def test_route_after_execute_routes_to_pick_task_at_loop_limit() -> None:
    assert route_after_execute(make_state(expand_tasks=True, loop_count=3, max_loops=3)) == (
        "pick_task"
    )


class FakeStructuredModel:
    def __init__(self, schema: type[Any]) -> None:
        self.schema = schema

    async def ainvoke(self, prompt: str) -> Plan | ToolChoice:
        if self.schema is Plan:
            return Plan(tasks=["research topic", "write summary"])
        if self.schema is ToolChoice:
            return ToolChoice(reasoning="Use the fake tool.", action="conclude", arg="done")
        raise AssertionError(f"Unexpected schema: {self.schema}")


class FakeModel:
    def with_structured_output(self, schema: type[Any]) -> FakeStructuredModel:
        return FakeStructuredModel(schema)

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content="Final fake summary.")


class ExpandingStructuredModel:
    def __init__(self, schema: type[Any], parent: ExpandingModel) -> None:
        self.schema = schema
        self.parent = parent

    async def ainvoke(self, prompt: str) -> Plan | ToolChoice | FollowupTask:
        if self.schema is Plan:
            return Plan(tasks=["research topic"])
        if self.schema is ToolChoice:
            return ToolChoice(reasoning="Use the fake tool.", action="conclude", arg="done")
        if self.schema is FollowupTask:
            self.parent.followup_calls += 1
            if self.parent.followup_calls == 1:
                return FollowupTask(task="verify source")
            return FollowupTask(task=None)
        raise AssertionError(f"Unexpected schema: {self.schema}")


class ExpandingModel:
    def __init__(self) -> None:
        self.followup_calls = 0

    def with_structured_output(self, schema: type[Any]) -> ExpandingStructuredModel:
        return ExpandingStructuredModel(schema, self)

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content="Final expanded summary.")


class FakeTool:
    name = "conclude"
    description = "Fake test tool."
    arg_description = "Fake input."

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    async def run(
        self,
        *,
        goal: str,
        task: str,
        arg: str,
        language: str,
        state: dict[str, Any],
    ) -> str:
        self.calls.append({"goal": goal, "task": task, "arg": arg, "language": language})
        return f"{task} -> {arg}"


@pytest.mark.asyncio
async def test_graph_runs_plan_analyze_execute_loop_with_fake_model_and_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tool = FakeTool()
    monkeypatch.setattr(nodes, "make_model", lambda: FakeModel())
    monkeypatch.setattr(nodes, "get_tool", lambda name: fake_tool)

    final_state = await build_graph().ainvoke(make_state(current_task=None))

    assert final_state["completed_tasks"] == ["research topic", "write summary"]
    assert final_state["results"] == ["research topic -> done", "write summary -> done"]
    assert final_state["loop_count"] == 2
    assert final_state["summary"] == "Final fake summary."
    assert [call["task"] for call in fake_tool.calls] == ["research topic", "write summary"]


@pytest.mark.asyncio
async def test_create_tasks_node_adds_new_followup_task(monkeypatch: pytest.MonkeyPatch) -> None:
    class FollowupModel:
        def with_structured_output(self, schema: type[Any]) -> FollowupModel:
            assert schema is FollowupTask
            return self

        async def ainvoke(self, prompt: str) -> FollowupTask:
            return FollowupTask(task="verify source")

    monkeypatch.setattr(nodes, "make_model", lambda: FollowupModel())

    result = await nodes.create_tasks_node(
        make_state(
            current_task="research topic",
            expand_tasks=True,
            loop_count=1,
            max_loops=3,
        )
        | {
            "tasks": ["write summary"],
            "completed_tasks": ["research topic"],
            "results": ["found one source"],
        }
    )

    assert result == {"tasks": ["write summary", "verify source"]}


@pytest.mark.asyncio
async def test_create_tasks_node_deduplicates_followup_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DuplicateModel:
        def with_structured_output(self, schema: type[Any]) -> DuplicateModel:
            assert schema is FollowupTask
            return self

        async def ainvoke(self, prompt: str) -> FollowupTask:
            return FollowupTask(task="WRITE SUMMARY")

    monkeypatch.setattr(nodes, "make_model", lambda: DuplicateModel())

    result = await nodes.create_tasks_node(
        make_state(
            current_task="research topic",
            expand_tasks=True,
            loop_count=1,
            max_loops=3,
        )
        | {
            "tasks": ["write summary"],
            "completed_tasks": ["research topic"],
            "results": ["found one source"],
        }
    )

    assert result == {}


@pytest.mark.asyncio
async def test_graph_can_expand_tasks_with_fake_model_and_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tool = FakeTool()
    expanding_model = ExpandingModel()
    monkeypatch.setattr(nodes, "make_model", lambda: expanding_model)
    monkeypatch.setattr(nodes, "get_tool", lambda name: fake_tool)

    final_state = await build_graph().ainvoke(
        make_state(current_task=None, expand_tasks=True, max_loops=3)
    )

    assert final_state["completed_tasks"] == ["research topic", "verify source"]
    assert final_state["results"] == ["research topic -> done", "verify source -> done"]
    assert final_state["loop_count"] == 2
    assert final_state["summary"] == "Final expanded summary."
