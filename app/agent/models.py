from langchain.chat_models import init_chat_model

from app.settings import get_settings


def make_model(model: str | None = None, *, temperature: float = 0.7, streaming: bool = False):
    settings = get_settings()
    model_id = model or settings.default_model
    api_key = settings.groq_api_key

    return init_chat_model(
        f"{settings.provider}:{model_id}",
        api_key=api_key,
        temperature=temperature,
        streaming=streaming,
    )
