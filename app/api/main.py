import json
from collections.abc import AsyncIterator
from typing import Any, cast
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
    ToolInfo,
)
from app.persistence.run_store import SQLiteRunStore
from app.settings import get_settings

RUN_STORE = SQLiteRunStore()
APPEND_FIELDS = {"completed_tasks", "results"}

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
        "current_analysis": None,
        "completed_tasks": [],
        "results": [],
        "loop_count": 0,
        "summary": None,
    }


def sse_event(
    *,
    run_id: str,
    sequence: int,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    data = {
        "run_id": run_id,
        "sequence": sequence,
        "type": event_type,
        "payload": payload,
    }

    return {
        "event": event_type,
        "data": json.dumps(data),
    }


def apply_state_update(state: AgentState, update: dict[str, Any]) -> AgentState:
    next_state = dict(state)

    for key, value in update.items():
        if key in APPEND_FIELDS:
            previous = cast(list[str], next_state.get(key, []))
            incoming = cast(list[str], value)
            next_state[key] = [*previous, *incoming]
        else:
            next_state[key] = value

    return cast(AgentState, next_state)


def progress_events(
    *,
    run_id: str,
    sequence_start: int,
    node_name: str,
    state: AgentState,
    update: dict[str, Any],
) -> tuple[list[dict[str, str]], int]:
    sequence = sequence_start
    events = [
        sse_event(
            run_id=run_id,
            sequence=sequence,
            event_type="node",
            payload={"node": node_name},
        )
    ]
    sequence += 1

    if node_name == "plan" and "tasks" in update:
        events.append(
            sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="plan",
                payload={"tasks": update["tasks"]},
            )
        )
        sequence += 1

    if node_name == "pick_task" and update.get("current_task"):
        events.append(
            sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="task",
                payload={"task": update["current_task"]},
            )
        )
        sequence += 1

    if node_name == "analyze" and update.get("current_analysis"):
        events.append(
            sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="analysis",
                payload={
                    "task": state.get("current_task"),
                    "analysis": update["current_analysis"],
                },
            )
        )
        sequence += 1

    if node_name == "execute":
        completed_tasks = cast(list[str], update.get("completed_tasks", []))
        results = cast(list[str], update.get("results", []))

        if completed_tasks and results:
            events.append(
                sse_event(
                    run_id=run_id,
                    sequence=sequence,
                    event_type="task_done",
                    payload={
                        "task": completed_tasks[-1],
                        "result": results[-1],
                        "loop_count": update.get("loop_count"),
                    },
                )
            )
            sequence += 1

    if node_name == "summarize" and update.get("summary"):
        events.append(
            sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="summary",
                payload={"text": update["summary"]},
            )
        )
        sequence += 1

    return events, sequence


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
    RUN_STORE.create_run(run_id, build_initial_state(request))

    return CreateRunResponse(run_id=run_id, status="created")


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> EventSourceResponse:
    record = RUN_STORE.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        sequence = 1

        if record["status"] == "completed":
            yield sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="summary",
                payload={"text": record["state"].get("summary") or ""},
            )
            sequence += 1
            yield sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="done",
                payload={},
            )
            return

        RUN_STORE.update_run(run_id, status="running")
        yield sse_event(
            run_id=run_id,
            sequence=sequence,
            event_type="status",
            payload={"status": "running"},
        )
        sequence += 1

        try:
            graph = build_graph()
            current_state = record["state"]
            summary_sent = False

            async for chunk in graph.astream(current_state, stream_mode="updates"):
                for node_name, update in chunk.items():
                    current_state = apply_state_update(current_state, update)
                    RUN_STORE.update_run(run_id, state=current_state)

                    events, sequence = progress_events(
                        run_id=run_id,
                        sequence_start=sequence,
                        node_name=node_name,
                        state=current_state,
                        update=update,
                    )

                    for event in events:
                        if event["event"] == "summary":
                            summary_sent = True
                        yield event

            RUN_STORE.update_run(run_id, status="completed", state=current_state)

            if not summary_sent:
                yield sse_event(
                    run_id=run_id,
                    sequence=sequence,
                    event_type="summary",
                    payload={"text": current_state.get("summary") or ""},
                )
                sequence += 1

            yield sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="done",
                payload={},
            )
        except Exception as error:
            RUN_STORE.update_run(run_id, status="failed")
            yield sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="error",
                payload={"message": str(error)},
            )
            sequence += 1
            yield sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="done",
                payload={},
            )

    return EventSourceResponse(event_generator())
