import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.agent.tools import available_tools
from app.api.schemas import (
    ChatMessageResponse,
    ChatRequest,
    CreateRunRequest,
    CreateRunResponse,
    HealthResponse,
    RunDetailResponse,
    RunStatusResponse,
    ToolInfo,
)
from app.chat import answer_run_question
from app.persistence.checkpointer import (
    checkpoint_exists,
    sqlite_checkpointer,
    thread_config,
)
from app.persistence.run_store import SQLiteRunStore
from app.settings import get_settings

settings = get_settings()
RUN_STORE = SQLiteRunStore(settings.run_db_path)
CHECKPOINT_DB_PATH = settings.checkpoint_db_path
APPEND_FIELDS = {"completed_tasks", "results"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    RUN_STORE.mark_running_as_failed()
    yield


app = FastAPI(title="agentmake API", version="0.1.0", lifespan=lifespan)


def allowed_cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in settings.frontend_origins.split(",")
        if origin.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins(),
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
        "expand_tasks": request.expand_tasks,
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

    if node_name == "create_tasks" and update.get("tasks"):
        tasks = cast(list[str], update["tasks"])
        events.append(
            sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="task_created",
                payload={"task": tasks[-1], "queue_size": len(tasks)},
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


def get_run_or_404(run_id: str) -> RunDetailResponse:
    record = RUN_STORE.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunDetailResponse(**record)


def status_response(run_id: str, status_value: str) -> RunStatusResponse:
    return RunStatusResponse(run_id=run_id, status=cast(Any, status_value))


def update_run_status(run_id: str, status_value: str) -> RunStatusResponse:
    record = RUN_STORE.update_run(run_id, status=cast(Any, status_value))
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunStatusResponse(run_id=run_id, status=record["status"])


def control_event_if_needed(run_id: str, sequence: int) -> tuple[dict[str, str], int] | None:
    record = RUN_STORE.get_run(run_id)
    if record is None or record["status"] not in {"paused", "cancelled"}:
        return None

    return (
        sse_event(
            run_id=run_id,
            sequence=sequence,
            event_type="status",
            payload={"status": record["status"]},
        ),
        sequence + 1,
    )


async def persist_pause_checkpoint(
    *,
    graph: Any,
    config: dict[str, Any],
    node_name: str | None,
    update: dict[str, Any] | None,
) -> None:
    if node_name is None or update is None:
        return

    await graph.aupdate_state(config, update, as_node=node_name)


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


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str) -> RunDetailResponse:
    return get_run_or_404(run_id)


@app.post("/runs/{run_id}/pause", response_model=RunStatusResponse)
async def pause_run(run_id: str) -> RunStatusResponse:
    record = get_run_or_404(run_id)

    if record.status == "paused":
        return status_response(run_id, "paused")
    if record.status != "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot pause a run with status '{record.status}'",
        )

    return update_run_status(run_id, "paused")


@app.post("/runs/{run_id}/resume", response_model=RunStatusResponse)
async def resume_run(run_id: str) -> RunStatusResponse:
    record = get_run_or_404(run_id)

    if record.status == "running":
        return status_response(run_id, "running")
    if record.status != "paused":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot resume a run with status '{record.status}'",
        )

    return update_run_status(run_id, "running")


@app.post("/runs/{run_id}/cancel", response_model=RunStatusResponse)
async def cancel_run(run_id: str) -> RunStatusResponse:
    record = get_run_or_404(run_id)

    if record.status == "cancelled":
        return status_response(run_id, "cancelled")
    if record.status in {"completed", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel a run with status '{record.status}'",
        )

    return update_run_status(run_id, "cancelled")


@app.get("/runs/{run_id}/chat", response_model=list[ChatMessageResponse])
async def list_run_chat(run_id: str) -> list[ChatMessageResponse]:
    get_run_or_404(run_id)

    return [ChatMessageResponse(**message) for message in RUN_STORE.list_chat_messages(run_id)]


@app.post("/runs/{run_id}/chat", response_model=ChatMessageResponse)
async def chat_with_run(run_id: str, request: ChatRequest) -> ChatMessageResponse:
    record = RUN_STORE.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot chat with a run with status '{record['status']}'",
        )

    state = record["state"]
    answer = await answer_run_question(
        goal=state["goal"],
        results=state["results"],
        question=request.question,
        language=state["language"],
    )
    message = RUN_STORE.add_chat_message(
        run_id,
        question=request.question,
        answer=answer,
    )

    return ChatMessageResponse(**message)


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

        if record["status"] in {"paused", "failed", "cancelled"}:
            yield sse_event(
                run_id=run_id,
                sequence=sequence,
                event_type="status",
                payload={"status": record["status"]},
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
            current_state = record["state"]
            summary_sent = False

            async with sqlite_checkpointer(CHECKPOINT_DB_PATH) as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                graph_input = (
                    None
                    if await checkpoint_exists(checkpointer, run_id)
                    else current_state
                )
                config = thread_config(run_id)

                async for chunk in graph.astream(
                    graph_input,
                    config,
                    stream_mode="updates",
                ):
                    metadata = chunk.get("__metadata__", {})
                    if isinstance(metadata, dict) and metadata.get("cached"):
                        continue

                    node_updates = {
                        key: value
                        for key, value in chunk.items()
                        if key != "__metadata__"
                    }

                    last_node_name = None
                    last_update = None
                    for node_name, update in node_updates.items():
                        last_node_name = node_name
                        last_update = update
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

                    control_event = control_event_if_needed(run_id, sequence)
                    if control_event is not None:
                        await persist_pause_checkpoint(
                            graph=graph,
                            config=config,
                            node_name=last_node_name,
                            update=last_update,
                        )
                        event, sequence = control_event
                        yield event
                        yield sse_event(
                            run_id=run_id,
                            sequence=sequence,
                            event_type="done",
                            payload={},
                        )
                        return

            current_record = RUN_STORE.get_run(run_id)
            if current_record is not None and current_record["status"] in TERMINAL_STATUSES:
                yield sse_event(
                    run_id=run_id,
                    sequence=sequence,
                    event_type="status",
                    payload={"status": current_record["status"]},
                )
                sequence += 1
                yield sse_event(
                    run_id=run_id,
                    sequence=sequence,
                    event_type="done",
                    payload={},
                )
                return

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
