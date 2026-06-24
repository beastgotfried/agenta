import asyncio
import sys
from pathlib import Path

# Running "python scripts/run_cli.py" only puts the scripts/ folder on Python's import
# path, not the project root (code/). Add the root so the "app" package can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import build_graph
from app.settings import get_settings


async def main():
    settings = get_settings()
    goal = " ".join(sys.argv[1:]) or "Write a haiku about AI."

    initial_state = {
        "goal": goal,
        "language": settings.language or "English",
        "max_loops": settings.max_loops,
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

    compiled_graph = build_graph()
    final_state = await compiled_graph.ainvoke(initial_state)

    if final_state and isinstance(final_state, dict):
        summary = final_state.get("summary")
        if summary:
            print(summary)


if __name__ == "__main__":
    asyncio.run(main())
