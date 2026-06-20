from pydantic import BaseModel, Field


class Plan(BaseModel):
    """The set of tasks the agent will work through to reach the goal."""

    tasks: list[str] = Field(
        description="Up to 5 concrete, ordered tasks (each like a focused search query) "
        "that together accomplish the user's goal."
    )
