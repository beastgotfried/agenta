from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.chat import chat
from app.chat.chat import NO_RUN_RESULTS_MESSAGE


class FakeChatModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content="Grounded answer.")


def test_format_results_numbers_each_result() -> None:
    assert chat.format_results(["first", "second"]) == "Result 1:\nfirst\n\nResult 2:\nsecond"


@pytest.mark.asyncio
async def test_answer_run_question_uses_completed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeChatModel()
    monkeypatch.setattr(chat, "make_model", lambda: fake_model)

    answer = await chat.answer_run_question(
        goal="Explain LangGraph",
        results=["LangGraph has checkpoints."],
        question="What does it support?",
        language="English",
    )

    assert answer == "Grounded answer."
    assert len(fake_model.prompts) == 1
    assert "LangGraph has checkpoints." in fake_model.prompts[0]
    assert "What does it support?" in fake_model.prompts[0]


@pytest.mark.asyncio
async def test_answer_run_question_returns_no_results_message_without_results() -> None:
    answer = await chat.answer_run_question(
        goal="Explain LangGraph",
        results=[],
        question="What does it support?",
        language="English",
    )

    assert answer == NO_RUN_RESULTS_MESSAGE
