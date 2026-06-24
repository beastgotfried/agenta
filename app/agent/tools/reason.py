from app.agent.models import make_model
from app.agent.prompts import EXECUTE_TASK
from app.agent.schemas import ToolName
from app.agent.tools.base import register


class ReasonTool:
    name: ToolName = "reason"
    description = (
        "A tool that uses the LLM to reason about the task and produce a detailed answer. "
        "This tool does not use the network and is always available."
    )
    arg_description = "The task to reason about."

    def available(self) -> bool:
        return True

    async def run(
        self,
        *,
        goal: str,
        task: str,
        arg: str,
        language: str,
        state: dict,
    ) -> str:
        model = make_model()
        user_context = str(state.get("user_context", ""))

        prompt = EXECUTE_TASK.format(
            goal=goal,
            task=arg or task,
            user_context=user_context,
            language=language,
        )
        response = await model.ainvoke(prompt)
        return str(response.content)


reason_tool = register(ReasonTool())
