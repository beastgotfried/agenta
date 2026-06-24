from app.agent.models import make_model
from app.agent.prompts import START_GOAL, SUMMARIZE
from app.agent.schemas import Plan
from app.agent.state import AgentState
from app.agent.tools import get_tool


async def plan_node(state: AgentState) -> dict:
    model = make_model().with_structured_output(Plan)
    prompt = START_GOAL.format(
        goal=state["goal"],
        language=state["language"],
        user_context=str(state.get("user_context", "")),
    )
    plan_result = await model.ainvoke(prompt)
    plan = plan_result if isinstance(plan_result, Plan) else Plan.model_validate(plan_result)
    return {"tasks": plan.tasks, "loop_count": 0}


def pick_task_node(state: AgentState) -> dict:
    """Take the next task off the front of the queue."""

    tasks = state["tasks"]
    if not tasks:
        return {"current_task": None}
    return {"current_task": tasks[0], "tasks": tasks[1:]}


async def execute_node(state: AgentState) -> dict:
    """Run the current task through the default reason tool."""

    current_task = state["current_task"]
    if current_task is None:
        return {}

    tool = get_tool("reason")
    result = await tool.run(
        goal=state["goal"],
        task=current_task,
        arg=current_task,
        language=state["language"],
        state=dict(state),
    )

    return {
        "results": [result],
        "completed_tasks": [current_task],
        "loop_count": state["loop_count"] + 1,
    }


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
