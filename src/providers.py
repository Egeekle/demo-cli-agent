import mlflow.pydantic_ai
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.settings import Settings, settings

# Configure MLflow tracking and autologging exactly once for the whole app.
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.pydantic_ai.autolog()


OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def build_provider(s: Settings) -> OpenAIProvider:
    """Build a provider for any OpenAI-compatible service (None base URL = OpenAI)."""
    # Pass the default explicitly: with base_url=None the OpenAI client falls back
    # to os.environ["OPENAI_BASE_URL"], which may be exported as an empty string.
    return OpenAIProvider(
        base_url=s.LLM_BASE_URL or OPENAI_DEFAULT_BASE_URL, api_key=s.LLM_API_KEY
    )


_provider = build_provider(settings)


def build_model(model_name: str) -> OpenAIChatModel:
    """Build a Chat Completions model bound to the shared provider."""
    return OpenAIChatModel(model_name, provider=_provider)


agent_model: OpenAIChatModel = build_model(settings.MODEL_NAME)

default_model_settings: dict = {"temperature": settings.MODEL_TEMPERATURE}


def build_judge_inference_params(s: Settings) -> dict[str, str]:
    """LiteLLM connection kwargs so MLflow judges hit the same endpoint as the agent."""
    params = {"api_key": s.LLM_API_KEY}
    if s.LLM_BASE_URL:
        params["api_base"] = s.LLM_BASE_URL
    return params


judge_model_uri: str = f"openai:/{settings.JUDGE_MODEL_NAME}"
judge_inference_params: dict[str, str] = build_judge_inference_params(settings)
