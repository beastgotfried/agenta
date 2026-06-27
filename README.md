# agentmake

agentmake is a Python rebuild of the AgentGPT-style agent loop using LangGraph
and LangChain. The goal is to move the autonomous agent brain out of the browser
and into one server-side state machine that can later be streamed through an API.

Right now, agentmake can run as a headless CLI agent or as a FastAPI service.
It can take a goal, plan tasks, choose a tool for each task, execute those
tasks, stream progress events, and summarize the results.

## Current Status

Implemented through **M3 SQLite run persistence**:

- A LangGraph `StateGraph` owns the agent loop.
- Goals are converted into structured task lists.
- Each task is picked from a queue and analyzed.
- The analyzer chooses one registered tool:
  - `reason`
  - `search`
  - `code`
  - `conclude`
- The executor runs the selected tool and stores the result.
- The summarizer combines all task results into a final answer.
- The CLI can run the full loop end to end.
- FastAPI exposes:
  - `GET /healthz`
  - `GET /tools`
  - `POST /runs`
  - `GET /runs/{run_id}/stream`
- The stream endpoint emits SSE progress events from graph updates.
- Run state is stored in SQLite at `data/runs.sqlite`.
- Focused tests cover tools, graph routing, analyze fallback, and API streaming.

Not implemented yet:

- Checkpoint persistence and pause/resume
- Task expansion
- Chat over completed results
- Durable user memory
- Frontend

## Architecture At A Glance

![agentmake LangGraph loop](docs/assets/agent_graph_mermaid.png)

The current graph is:

```text
START
  -> plan
  -> pick_task
  -> analyze
  -> execute
  -> pick_task
  -> ...
  -> summarize
  -> END
```

The important loop is:

```text
pick_task -> analyze -> execute -> pick_task
```

That means each task gets its own tool decision before execution.

## How The Agent Works

1. `scripts/run_cli.py` creates the initial state from your goal.
2. `app/agent/graph.py` builds the LangGraph workflow.
3. `plan_node` creates a task list from the goal.
4. `pick_task_node` selects the next task.
5. `analyze_node` chooses the best tool with structured `ToolChoice` output.
6. `execute_node` runs the selected tool.
7. The graph loops until no tasks remain or `max_loops` is reached.
8. `summarize_node` writes the final summary.

## Project Structure

```text
app/
  settings.py              # Provider, model, language, loop settings
  agent/
    graph.py               # LangGraph wiring
    nodes.py               # plan, pick_task, analyze, execute, summarize
    state.py               # AgentState and Analysis shape
    schemas.py             # Plan, ToolChoice, ToolName
    prompts.py             # Fixed agent prompt templates
    models.py              # LLM factory
    tools/
      base.py              # Tool protocol and registry
      reason.py            # General LLM reasoning tool
      search.py            # DuckDuckGo search with cited summary
      code.py              # Code-focused LLM tool
      conclude.py          # Task conclusion tool
  api/
    main.py                # FastAPI app, CORS, run creation, SSE stream
    schemas.py             # API request/response schemas
  persistence/
    run_store.py           # SQLite storage for run records and agent state
scripts/
  run_cli.py               # Headless CLI entrypoint
tests/
  test_analyze.py          # Analyze-node behavior and fallback tests
  test_api.py              # FastAPI and SSE stream tests
  test_graph.py            # Graph routing and fake-model loop tests
  test_tools.py            # Tool registry and tool behavior tests
docs/
  assets/
    agent_graph_mermaid.png
```

## Run The CLI

Install dependencies with `uv`, then set your LLM key in `.env`.

Example `.env`:

```text
GROQ_API_KEY=your_key_here
```

Run a goal:

```bash
uv run python scripts/run_cli.py "Search for LangGraph documentation and write a tiny Python example"
```

## Run The API

Start the FastAPI server from the `code/` directory:

```bash
uv run uvicorn app.api.main:app --reload --port 8000
```

Open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

List available tools:

```bash
curl http://127.0.0.1:8000/tools
```

Create a run:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"goal":"Write a short explanation of LangGraph"}'
```

Copy the returned `run_id`, then stream that run:

```bash
curl -N http://127.0.0.1:8000/runs/YOUR_RUN_ID/stream
```

The stream currently emits these SSE event types:

```text
status
node
plan
task
analysis
task_done
summary
error
done
```

Run state is stored in SQLite:

```text
data/runs.sqlite
```

This stores run status and serialized `AgentState`, so completed runs can be
loaded again after a server restart. LangGraph checkpointing is still a later
M3 step; the current SQLite store is basic run persistence, not pause/resume.

## Quality Checks

Run quality checks:

```bash
uv run ruff check .
uv run mypy --cache-dir /tmp/agentmake-mypy-cache .
uv run pytest -q -p no:cacheprovider
```

To run only API tests:

```bash
uv run pytest -q tests/test_api.py -p no:cacheprovider
```

Do not run test files directly with `python tests/test_api.py`; pytest reads the
project `pythonpath` config from `pyproject.toml`, while direct Python execution
does not.

## Tool System

Tools register themselves in `app/agent/tools/base.py`.

The analyzer sees tool descriptions from the registry, then returns:

```python
{
    "reasoning": "Why this tool fits the task",
    "action": "search",
    "arg": "query or tool input",
}
```

The executor then runs:

```python
tool = get_tool(analysis["action"])
result = await tool.run(...)
```

If the model returns an invalid tool choice, the system falls back to `reason`.

## Current Milestones

```text
M1   complete: headless graph with plan -> execute -> summarize
M2a  complete: prompts and ToolChoice schema
M2b  complete: tool registry, reason, conclude
M2c  complete: search and code tools
M2d  complete: analyze_node and selected-tool execution
M3a  complete: FastAPI shell with health, tools, and run creation
M3b  complete: SSE run stream endpoint
M3c  complete: graph progress events over SSE
M3d  complete: SQLite run persistence
```

## Next Steps

Recommended next work:

1. Improve streaming robustness and event shapes.
2. Add LangGraph checkpoint persistence.
3. Add pause/resume endpoints.
4. Prepare Railway deployment.

M3 target API:

```text
POST /runs
GET  /runs/{run_id}/stream
GET  /tools
GET  /healthz
```

The long-term product goal is a Railway-hosted Python brain with a Vercel
frontend that streams agent progress from the server-side LangGraph run.
