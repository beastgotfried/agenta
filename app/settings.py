from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    provider: str = "groq"
    groq_api_key: Optional[str] = None
    default_model: str = "llama-3.3-70b-versatile"
    language: str = "English"
    max_loops: int = 25

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="UTF-8")
    
@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it."""
    return Settings()