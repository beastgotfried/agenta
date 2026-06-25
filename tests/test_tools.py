import pytest

from app.agent.tools import available_tools, get_tool, tool_descriptions
from app.agent.tools.conclude import ConcludeTool


def test_registry_exposes_expected_tools() -> None:
    tool_names = {tool.name for tool in available_tools()}

    assert {"reason", "search", "code", "conclude"} <= tool_names


def test_get_tool_falls_back_to_reason_for_unknown_or_disabled_tool() -> None:
    assert get_tool("missing").name == "reason"
    assert get_tool("search", enabled_tools=["reason"]).name == "reason"


def test_tool_descriptions_respects_enabled_tools() -> None:
    descriptions = tool_descriptions(enabled_tools=["reason"])

    assert '"reason"' in descriptions
    assert '"search"' not in descriptions
    assert '"code"' not in descriptions
    assert '"conclude"' not in descriptions


@pytest.mark.asyncio
async def test_conclude_tool_returns_explicit_reason() -> None:
    result = await ConcludeTool().run(
        goal="write a report",
        task="finish",
        arg="Enough evidence has been collected.",
        language="English",
        state={},
    )

    assert result == "Concluded: Enough evidence has been collected."
