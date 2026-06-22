from typing import Literal

from pydantic import BaseModel, Field

# One place that names every tool. The schema AND the 2b registry import THIS,
# so the model can never pick a tool the registry doesn't have. (conclude/search/
# code arrive over 2b-2c, but naming them now keeps the enum stable.)
ToolName = Literal["reason", "search", "code", "conclude"]


class Plan(BaseModel):
    """The set of tasks the agent will work through to reach the goal."""

    tasks: list[str] = Field(
        description="Up to 5 concrete, ordered tasks (each like a focused search query) "
        "that together accomplish the user's goal."
    )


class ToolChoice(BaseModel):
    """The analyze node's decision: which tool to run for the current task."""

    reasoning: str = Field(description="Why this tool fits the task (in the user's language).")
    action: ToolName = Field(description="The single tool to run.")
    arg: str = Field(description="Input to the tool — e.g. the search query, or the task text.")


def default_tool_choice(task: str) -> ToolChoice:
    """Safe fallback when the model fails to return a valid choice.

    AgentGPT defaulted to `search`; we default to `reason` because reason never
    needs the network and so can never itself fail — a more robust fallback for a
    free, flaky-search stack. (Flip to "search" here if you'd rather mirror AgentGPT.)
    """
    return ToolChoice(reasoning="Defaulting to plain reasoning.", action="reason", arg=task)
