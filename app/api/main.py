import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, TypedDict, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import build_graph
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


def sse_event(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {
        "event": event,
        "data": json.dumps(data),
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


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> EventSourceResponse:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        record = RUNS[run_id]

        if record["status"] == "completed":
            yield sse_event("summary", {"text": record["state"].get("summary") or ""})
            yield sse_event("done", {})
            return

        record["status"] = "running"
        yield sse_event("status", {"status": "running", "run_id": run_id})

        try:
            graph = build_graph()
            final_state = cast(AgentState, await graph.ainvoke(record["state"]))

            record["state"] = final_state
            record["status"] = "completed"

            yield sse_event("summary", {"text": final_state.get("summary") or ""})
            yield sse_event("done", {})
        except Exception as error:
            record["status"] = "failed"
            yield sse_event("error", {"message": str(error)})
            yield sse_event("done", {})

    return EventSourceResponse(event_generator())
