"""tool abstraction + a tiny template engine. Adding a capability = registering a single tool 
the analyse prompt will pick it up itself at O(1) time. The tool will be available to the user in the prompt."""

from __future__ import annotations
from loguru import logger

class Tool:
    """Base class for a toll. Subclasses set name/description and implement run()"""
    
    name: str= "tool"
    description: str= ""
    
    async def run(
        self, *, goal: str, task:str, arg: str, language: str, user_context:str="")-> str:
        raise NotImplementedError
    
