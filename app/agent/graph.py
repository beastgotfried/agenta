from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    analyze_node,
    create_tasks_node,
    execute_node,
    pick_task_node,
    plan_node,
    summarize_node,
)
from app.agent.state import AgentState


def route_after_pick(state: AgentState) -> str:
    """Route to analysis or summary after picking a task."""

    if state["current_task"] is None or state["loop_count"] >= state["max_loops"]:
        return "summarize"
    return "analyze"


def route_after_execute(state: AgentState) -> str:
    """Route to optional task expansion or directly to the next task."""

    if state.get("expand_tasks", False) and state["loop_count"] < state["max_loops"]:
        return "create_tasks"
    return "pick_task"


def build_graph(*, checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_node)
    builder.add_node("pick_task", pick_task_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("execute", execute_node)
    builder.add_node("create_tasks", create_tasks_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "pick_task")
    builder.add_conditional_edges(
        "pick_task",
        route_after_pick,
        {"analyze": "analyze", "summarize": "summarize"},
    )
    builder.add_edge("analyze", "execute")
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {"create_tasks": "create_tasks", "pick_task": "pick_task"},
    )
    builder.add_edge("create_tasks", "pick_task")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)
