from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.agent.state import AgentState
from app.agent.tools import available_tools
from app.api.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    HealthResponse,
    RunStatus,
    ToolInfo,
)
from app.settings import get_settings


class RunRecord(TypedDict):
    status: RunStatus
    state: AgentState
    created_at: str


RUNS: dict[str, RunRecord] = {}

app = FastAPI(title="agentmake API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def build_initial_state(request: CreateRunRequest) -> AgentState:
    settings = get_settings()

    return {
        "goal": request.goal,
        "language": request.language or settings.language or "English",
        "max_loops": request.max_loops or settings.max_loops,
        "user_id": "local",
        "user_context": "",
        "tasks": [],
        "current_task": None,
        "completed_tasks": [],
        "current_analysis": None,
        "results": [],
        "loop_count": 0,
        "summary": None,
    }


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/tools", response_model=list[ToolInfo])
async def list_tools() -> list[ToolInfo]:
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            arg_description=tool.arg_description,
        )
        for tool in available_tools()
    ]


@app.post("/runs", response_model=CreateRunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(request: CreateRunRequest) -> CreateRunResponse:
    run_id = str(uuid4())

    RUNS[run_id] = {
        "status": "created",
        "state": build_initial_state(request),
        "created_at": datetime.now(UTC).isoformat(),
    }

    return CreateRunResponse(run_id=run_id, status="created")
