from app.agent.schemas import ToolName
from app.agent.tools.base import register


class ConcludeTool:
    name: ToolName = "conclude"
    description = "Use when the current task is already complete or no further action is needed."
    arg_description = "A short reason why the task can be considered complete."

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
        if arg:
            return f"Concluded: {arg}"
        return "Concluded."


conclude_tool = register(ConcludeTool())
