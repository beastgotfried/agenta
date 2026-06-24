import operator
from typing import Annotated, TypedDict

from app.agent.schemas import ToolName


class Analysis(TypedDict):
    reasoning: str
    action: ToolName
    arg: str


class AgentState(TypedDict):
    # --- inputs / config (set once at the start, then read-only) ---
    goal: str
    language: str
    max_loops: int
    user_id: str
    user_context: str

    # --- the working queue ---
    tasks: list[str]              # tasks still to do (shrinks as we work)
    current_task: str | None      # the one we're on right now (None when none left)
    current_analysis: Analysis | None

    # --- things that ACCUMULATE (append, never overwrite) ---
    completed_tasks: Annotated[list[str], operator.add]
    results: Annotated[list[str], operator.add]

    # --- control / output ---
    loop_count: int               # how many tasks we've executed
    summary: str | None           # filled in at the very end

