from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    provider: str = "groq"
    groq_api_key: str | None = None
    default_model: str = "llama-3.3-70b-versatile"
    language: str = "English"
    max_loops: int = 25
    run_db_path: Path = Path("data/runs.sqlite")
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")
    frontend_origins: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://localhost:5174,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:5173,"
        "http://127.0.0.1:5174"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="UTF-8")


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it."""

    return Settings()
