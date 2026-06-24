# agentmake

agentmake is a Python rebuild of the AgentGPT-style agent loop using LangGraph
and LangChain. The goal is to move the autonomous agent brain out of the browser
and into one server-side state machine that can later be streamed through an API.

Right now, agentmake is a headless CLI agent. It can take a goal, plan tasks,
choose a tool for each task, execute those tasks, and summarize the results.

## Current Status

Implemented through **M2 2d**:

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

Not implemented yet:

- FastAPI/SSE run API
- Checkpoint persistence and pause/resume
- Task expansion
- Chat over completed results
- Durable user memory
- Frontend
- Automated tests

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
scripts/
  run_cli.py               # Headless CLI entrypoint
docs/
  assets/
    agent_graph_mermaid.png
```

## Run It

Install dependencies with `uv`, then set your LLM key in `.env`.

Example `.env`:

```text
GROQ_API_KEY=your_key_here
```

Run a goal:

```bash
uv run python scripts/run_cli.py "Search for LangGraph documentation and write a tiny Python example"
```

Run quality checks:

```bash
uv run ruff check .
uv run mypy --cache-dir /tmp/agentmake-mypy-cache .
```

There are currently no automated tests, so `pytest` reports zero collected
tests.

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
```

## Next Steps

Recommended next work:

1. Add focused tests for tools, graph routing, and analyze fallback.
2. Add optional debug transcript output so tool choices are visible in CLI runs.
3. Start M3: FastAPI run API and SSE streaming.

M3 target API:

```text
POST /runs
GET  /runs/{run_id}/stream
GET  /tools
GET  /healthz
```

The long-term product goal is a Railway-hosted Python brain with a Vercel
frontend that streams agent progress from the server-side LangGraph run.
