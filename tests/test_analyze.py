from __future__ import annotations

from typing import Any

import pytest

from app.agent import nodes
from app.agent.schemas import ToolChoice
from app.agent.state import AgentState


def make_state(*, current_task: str | None = "research LangGraph") -> AgentState:
    return {
        "goal": "Write a short LangGraph guide.",
        "language": "English",
        "max_loops": 5,
        "expand_tasks": False,
        "user_id": "local",
        "user_context": "",
        "tasks": [],
        "current_task": current_task,
        "current_analysis": None,
        "completed_tasks": [],
        "results": [],
        "loop_count": 0,
        "summary": None,
    }


class SelectedToolModel:
    def with_structured_output(self, schema: type[Any]) -> SelectedToolModel:
        return self

    async def ainvoke(self, prompt: str) -> ToolChoice:
        return ToolChoice(
            reasoning="The task needs current information.",
            action="search",
            arg="LangGraph latest docs",
        )


class BrokenStructuredModel:
    def with_structured_output(self, schema: type[Any]) -> BrokenStructuredModel:
        return self

    async def ainvoke(self, prompt: str) -> ToolChoice:
        raise ValueError("bad structured output")


@pytest.mark.asyncio
async def test_analyze_node_returns_model_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "make_model", lambda: SelectedToolModel())

    result = await nodes.analyze_node(make_state())

    assert result == {
        "current_analysis": {
            "reasoning": "The task needs current information.",
            "action": "search",
            "arg": "LangGraph latest docs",
        }
    }


@pytest.mark.asyncio
async def test_analyze_node_falls_back_to_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "make_model", lambda: BrokenStructuredModel())

    result = await nodes.analyze_node(make_state(current_task="summarize local notes"))

    assert result == {
        "current_analysis": {
            "reasoning": "Defaulting to plain reasoning.",
            "action": "reason",
            "arg": "summarize local notes",
        }
    }


@pytest.mark.asyncio
async def test_analyze_node_skips_when_no_current_task() -> None:
    result = await nodes.analyze_node(make_state(current_task=None))

    assert result == {"current_analysis": None}
