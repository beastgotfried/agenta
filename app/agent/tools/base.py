from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.agent.schemas import ToolName


class Tool(Protocol):
    """A single executable agent capability."""

    name: ToolName
    description: str
    arg_description: str

    def available(self) -> bool:
        """Whether this tool is globally available."""
        ...

    async def run(
        self,
        *,
        goal: str,
        task: str,
        arg: str,
        language: str,
        state: dict,
    ) -> str:
        """Run the tool and return text that can be stored as a task result."""
        ...


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Register a tool instance and return it."""

    _REGISTRY[tool.name] = tool
    return tool


def available_tools(*, enabled_tools: Sequence[str] | None = None) -> list[Tool]:
    """Return tools that are enabled and available."""

    enabled = set(enabled_tools) if enabled_tools is not None else None
    return [
        tool
        for tool in _REGISTRY.values()
        if tool.available() and (enabled is None or tool.name in enabled)
    ]


def get_tool(name: str | None, *, enabled_tools: Sequence[str] | None = None) -> Tool:
    """Find a tool by name, falling back to reason."""

    enabled = set(enabled_tools) if enabled_tools is not None else None
    tool = _REGISTRY.get(name or "")

    if tool is not None and tool.available() and (enabled is None or tool.name in enabled):
        return tool

    return _REGISTRY["reason"]


def tool_descriptions(*, enabled_tools: Sequence[str] | None = None) -> str:
    """Render tool descriptions for the analyze prompt."""

    lines = []
    for tool in available_tools(enabled_tools=enabled_tools):
        lines.append(f'- "{tool.name}": {tool.description} Argument: {tool.arg_description}')
    return "\n".join(lines)
