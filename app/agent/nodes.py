from app.agent.models import make_model
from app.agent.schemas import Plan
from app.agent.state import AgentState

async def plan_node(state: AgentState) -> dict:
    model = make_model().with_structured_output(Plan)
    prompt =(
         "You are a planning assistant. Break the goal below into a short list of "
          f"concrete tasks (at most 5). Reply in {state['language']}.\n\n"
          f"Goal: {state['goal']}"
    )
    plan_result = await model.ainvoke(prompt)
    plan = plan_result if isinstance(plan_result, Plan) else Plan.model_validate(plan_result)
    return {"tasks": plan.tasks, "loop_count": 0}

def pick_task_node(state: AgentState) -> dict:
    """take the next task off the front of the queue and set it as the current task"""
    tasks=state["tasks"]
    if not tasks:
        return {"current_task": None}
    return {"current_task": tasks[0], "tasks": tasks[1:]}

async def execute_node(state: AgentState) -> dict:
    """do the current task with a plain LLM call( the skeletons only skill: reasoning and planning are structured)"""
    model= make_model()
    prompt=(        
            f"You are working toward this overall goal: {state['goal']}\n"
            f"Complete this one task and give a useful, detailed result. "
            f"Reply in {state['language']}.\n\n"
            f"Task: {state['current_task']}"
      )
    response = await model.ainvoke([prompt])
    return {"results": [response.content],
            "completed_tasks": [state["current_task"]],
            "loop_count": state["loop_count"] + 1
            }
    
async def summarize_node(state: AgentState) -> dict:
    """comibine all task results into one final answer."""
    model=make_model()
    joined = "\n\n".join(state["results"])
    prompt= (
          f"Combine the task results below into one cohesive, well-organized answer "
          f"for the goal '{state['goal']}'. Reply in {state['language']}.\n\n{joined}"      
    )
    response = await model.ainvoke([prompt])
    return {"summary": response.content}



    