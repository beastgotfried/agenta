from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator

from app.agent.schemas import ToolName
from app.agent.state import AgentState

RunStatus: TypeAlias = Literal[
    "created",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ToolInfo(BaseModel):
    name: ToolName
    description: str
    arg_description: str


class CreateRunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    language: str | None = Field(default=None, min_length=1, max_length=100)
    max_loops: int | None = Field(default=None, ge=1, le=25)

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Goal must not be blank")
        return value


class CreateRunResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunDetailResponse(BaseModel):
    run_id: str
    status: RunStatus
    state: AgentState
    created_at: str
    updated_at: str
