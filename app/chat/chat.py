from __future__ import annotations

from app.agent.models import make_model
from app.agent.prompts import CHAT

NO_RUN_RESULTS_MESSAGE = "I don't have any completed run results to answer from."


def format_results(results: list[str]) -> str:
    return "\n\n".join(f"Result {index}:\n{result}" for index, result in enumerate(results, 1))


async def answer_run_question(
    *,
    goal: str,
    results: list[str],
    question: str,
    language: str,
) -> str:
    if not results:
        return NO_RUN_RESULTS_MESSAGE

    prompt = CHAT.format(
        goal=goal,
        results=format_results(results),
        question=question,
        language=language,
    )
    response = await make_model().ainvoke(prompt)
    return str(response.content).strip()
