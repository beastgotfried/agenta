from __future__ import annotations

import asyncio
from typing import Any

from ddgs import DDGS

from app.agent.models import make_model
from app.agent.prompts import SUMMARIZE_WITH_SOURCES
from app.agent.schemas import ToolName
from app.agent.tools.base import get_tool, register

MAX_RESULT=5

def _search_duckduckgo(query: str) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        return ddgs.text(query,max_results= MAX_RESULT)
    
def _format_result(index: int, result:dict[str, Any]) -> str | None:
    title= str(result.get("title","")).strip()
    url = str(result.get("href","")).strip()
    body= str(
        result.get("body")
        or result.get("snippet")
        or result.get("description")
        or ""
    ).strip()
    if not title and not url and not body:
        return None
    parts= [f"[{index}]"]
    if title:
        parts.append(f"**{title}**")
    if url:
        parts.append(f"<{url}>")
    if body:
        parts.append(f"{body}")
    return " ".join(parts)

def _format_results(results: list[dict[str, Any]]) -> str:
    snippets= []
    for index,result in enumerate(results,start=1):
        formatted= _format_result(index,result)
        if formatted:
            snippets.append(formatted)
    return "\n\n".join(snippets)

class SearchTool:
    name: ToolName=  "search"
    description=(
        "search the web using ddg, best for current facts, recent events"
    )
    arg_description= "A concise web search query"
    
    def available(self) -> bool:
        return True
    
    async def run(
        self,
        *,
        goal: str,
        task: str,
        arg: str,
        language: str,
        state: dict,
    ) -> str:
        query= (arg or task).strip()
        if not query:
            return await self._fallback_to_reason(
                goal= goal,
                task= task,
                language= language,
                state= state,
                arg=arg
            )
        try:
            results= await asyncio.to_thread(_search_duckduckgo,query)
            snippets= _format_results(results)

            if not snippets:
                raise ValueError("No results found")
            
            model= make_model()
            prompt= SUMMARIZE_WITH_SOURCES.format(
                query=query,
                snippets= snippets,
                language= language,
                user_context= str(state.get("user_context","")),
            )
            response= await model.ainvoke(prompt)
            return str(response.content)
        except Exception as e:
            return await self._fallback_to_reason(
                goal= goal,
                task= task,
                language= language,
                state= state,
                arg= str(e),
            )
    async def _fallback_to_reason(
        self,
        *,
        goal: str,
        task: str,
        arg: str,
        language: str,
        state: dict,
    ) -> str:
        reason_tool= get_tool("reason")
        return await reason_tool.run(
            goal= goal,
            task= task,
            arg= arg,
            language= language,
            state= state,
        )
        
search_tool= register(SearchTool())
