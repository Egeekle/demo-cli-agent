from pydantic_ai.models.openai import OpenAIChatModel

from src import providers
from src.settings import Settings, settings

NIM = "https://integrate.api.nvidia.com/v1"


def test_build_model_and_shared_model_exist():
    assert isinstance(providers.agent_model, OpenAIChatModel)
    m = providers.build_model("gpt-4.1-mini")
    assert isinstance(m, OpenAIChatModel)
    assert providers.default_model_settings["temperature"] == 0.0


def test_provider_uses_configured_base_url():
    s = Settings(LLM_API_KEY="k", LLM_BASE_URL=NIM, _env_file=None)
    assert providers.build_provider(s).base_url == NIM + "/"


def test_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    s = Settings(LLM_API_KEY="k", _env_file=None)
    assert providers.build_provider(s).base_url == "https://api.openai.com/v1/"


def test_build_model_accepts_namespaced_id():
    assert providers.build_model("meta/llama-3.3-70b-instruct").model_name == (
        "meta/llama-3.3-70b-instruct"
    )


def test_judge_params_include_base_url_when_set():
    s = Settings(LLM_API_KEY="k", LLM_BASE_URL=NIM, _env_file=None)
    assert providers.build_judge_inference_params(s) == {
        "api_key": "k",
        "api_base": NIM,
    }


def test_judge_params_omit_base_url_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    s = Settings(LLM_API_KEY="k", _env_file=None)
    assert providers.build_judge_inference_params(s) == {"api_key": "k"}


def test_judge_model_uri_uses_judge_model_name():
    assert providers.judge_model_uri == f"openai:/{settings.JUDGE_MODEL_NAME}"


def test_blank_exported_base_url_still_defaults_to_openai(monkeypatch):
    # `uv run --env-file .env` exports `OPENAI_BASE_URL=` as an empty string; the
    # OpenAI client would use it verbatim and fail with "missing protocol".
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    s = Settings(LLM_API_KEY="k", _env_file=None)
    assert providers.build_provider(s).base_url == "https://api.openai.com/v1/"
