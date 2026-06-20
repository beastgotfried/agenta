import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict):
    # --- inputs / config (set once at the start, then read-only) ---
    goal: str
    language: str
    max_loops: int

    # --- the working queue ---
    tasks: list[str]              # tasks still to do (shrinks as we work)
    current_task: Optional[str]   # the one we're on right now (None when none left)

    # --- things that ACCUMULATE (append, never overwrite) ---
    completed_tasks: Annotated[list[str], operator.add]
    results: Annotated[list[str], operator.add]

    # --- control / output ---
    loop_count: int               # how many tasks we've executed
    summary: Optional[str]        # filled in at the very end
