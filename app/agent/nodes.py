from app.agent.models import make_model
from app.agent.prompts import ANALYZE_TASK, CREATE_TASKS, START_GOAL, SUMMARIZE
from app.agent.schemas import FollowupTask, Plan, ToolChoice, default_tool_choice
from app.agent.state import AgentState
from app.agent.tools import get_tool, tool_descriptions


async def plan_node(state: AgentState) -> dict:
    model = make_model().with_structured_output(Plan)
    prompt = START_GOAL.format(
        goal=state["goal"],
        language=state["language"],
        user_context=str(state.get("user_context", "")),
    )
    plan_result = await model.ainvoke(prompt)
    plan = plan_result if isinstance(plan_result, Plan) else Plan.model_validate(plan_result)
    return {"tasks": plan.tasks, "loop_count": 0, "current_analysis": None}


def pick_task_node(state: AgentState) -> dict:
    """Take the next task off the front of the queue."""

    tasks = state["tasks"]
    if not tasks:
        return {"current_task": None, "current_analysis": None}
    return {"current_task": tasks[0], "tasks": tasks[1:], "current_analysis": None}


async def analyze_node(state: AgentState) -> dict:
    """Choose the best tool for the current task."""

    current_task = state["current_task"]
    if current_task is None:
        return {"current_analysis": None}
    prompt = ANALYZE_TASK.format(
        goal=state["goal"],
        task=current_task,
        language=state["language"],
        user_context=str(state.get("user_context", "")),
        tool_descriptions=tool_descriptions(),
    )
    try:
        model = make_model().with_structured_output(ToolChoice)
        analysis_result = await model.ainvoke(prompt)
        analysis = (
            analysis_result
            if isinstance(analysis_result, ToolChoice)
            else ToolChoice.model_validate(analysis_result)
        )
    except Exception:
        analysis = default_tool_choice(current_task)

    return {
        "current_analysis": {
            "reasoning": analysis.reasoning,
            "action": analysis.action,
            "arg": analysis.arg,
        }
    }


async def execute_node(state: AgentState) -> dict:
    """Run the current task through the selected tool."""

    current_task = state["current_task"]
    if current_task is None:
        return {}

    analysis = state["current_analysis"]
    if analysis is None:
        analysis_choice = default_tool_choice(current_task)
        analysis = {
            "reasoning": analysis_choice.reasoning,
            "action": analysis_choice.action,
            "arg": analysis_choice.arg,
        }

    tool = get_tool(analysis["action"])
    result = await tool.run(
        goal=state["goal"],
        task=current_task,
        arg=analysis["arg"] or current_task,
        language=state["language"],
        state=dict(state),
    )

    return {
        "results": [result],
        "completed_tasks": [current_task],
        "loop_count": state["loop_count"] + 1,
    }


def normalize_task_key(task: str) -> str:
    return " ".join(task.casefold().split())


def is_duplicate_task(task: str, state: AgentState) -> bool:
    task_key = normalize_task_key(task)
    existing_tasks = [
        *state["tasks"],
        *state["completed_tasks"],
    ]

    if state["current_task"]:
        existing_tasks.append(state["current_task"])

    return task_key in {normalize_task_key(existing) for existing in existing_tasks}


async def create_tasks_node(state: AgentState) -> dict:
    """Optionally add one useful follow-up task after execution."""

    if not state.get("expand_tasks", False):
        return {}
    if state["loop_count"] >= state["max_loops"]:
        return {}
    if state["current_task"] is None or not state["results"]:
        return {}

    prompt = CREATE_TASKS.format(
        goal=state["goal"],
        task=state["current_task"],
        result=state["results"][-1],
        language=state["language"],
        user_context=str(state.get("user_context", "")),
        queued_tasks=state["tasks"] or "None",
        completed_tasks=state["completed_tasks"] or "None",
    )

    try:
        model = make_model().with_structured_output(FollowupTask)
        followup_result = await model.ainvoke(prompt)
        followup = (
            followup_result
            if isinstance(followup_result, FollowupTask)
            else FollowupTask.model_validate(followup_result)
        )
    except Exception:
        return {}

    if followup.task is None or is_duplicate_task(followup.task, state):
        return {}

    return {"tasks": [*state["tasks"], followup.task]}


async def summarize_node(state: AgentState) -> dict:
    """Combine all task results into one final answer."""

    model = make_model()
    joined = "\n\n".join(state["results"])
    prompt = SUMMARIZE.format(
        goal=state["goal"],
        language=state["language"],
        text=joined,
        user_context=str(state.get("user_context", "")),
    )
    response = await model.ainvoke(prompt)
    return {"summary": response.content}
