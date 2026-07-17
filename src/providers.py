import mlflow.pydantic_ai
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.settings import settings

# Configure MLflow tracking and autologging exactly once for the whole app.
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.pydantic_ai.autolog()

_provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)


def build_model(model_name: str) -> OpenAIResponsesModel:
    """Build an OpenAI Responses model bound to the shared provider."""
    return OpenAIResponsesModel(model_name, provider=_provider)


agent_model: OpenAIResponsesModel = build_model(settings.MODEL_NAME)

default_model_settings: dict = {"temperature": settings.MODEL_TEMPERATURE}
