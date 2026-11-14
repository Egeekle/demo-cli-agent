import pytest
from pydantic import ValidationError

from src.settings import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")

    settings = Settings(_env_file=None)
    assert settings.LLM_API_KEY == "test-key"
    assert settings.MLFLOW_TRACKING_URI == "http://localhost:9999"
    assert settings.DATABASE_URL == "sqlite+aiosqlite:///test.db"


def test_settings_default_values(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    # Clear other env vars to use defaults
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = Settings(_env_file=None)
    # Just verify they are populated (not empty)
    assert settings.MLFLOW_TRACKING_URI
    assert settings.DATABASE_URL
    assert settings.LLM_BASE_URL is None


def test_judge_and_temperature_defaults(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    s = Settings(LLM_API_KEY="test-key", _env_file=None)
    assert s.JUDGE_MODEL_NAME == "gpt-5.2"
    assert s.MODEL_TEMPERATURE == 0.0


def test_legacy_openai_api_key_still_accepted(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    assert Settings(_env_file=None).LLM_API_KEY == "legacy-key"


def test_llm_api_key_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    assert Settings(_env_file=None).LLM_API_KEY == "new-key"


def test_base_url_read_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    assert (
        Settings(_env_file=None).LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    )


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_base_url_means_default(monkeypatch, blank):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", blank)
    assert Settings(_env_file=None).LLM_BASE_URL is None


def test_missing_key_fails(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_legacy_openai_base_url_still_accepted(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    assert (
        Settings(_env_file=None).LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    )


def test_llm_base_url_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://new.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    assert Settings(_env_file=None).LLM_BASE_URL == "https://new.example/v1"


def _dotenv(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body)
    return path


@pytest.fixture
def clean_llm_env(monkeypatch):
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_dotenv_llm_key_beats_shell_legacy_key(tmp_path, clean_llm_env):
    clean_llm_env.setenv("OPENAI_API_KEY", "sk-shell")
    s = Settings(_env_file=_dotenv(tmp_path, "LLM_API_KEY=nvapi-dotenv\n"))
    assert s.LLM_API_KEY == "nvapi-dotenv"


def test_dotenv_llm_base_url_beats_shell_legacy_url(tmp_path, clean_llm_env):
    clean_llm_env.setenv("OPENAI_BASE_URL", "https://shell.example/v1")
    s = Settings(
        _env_file=_dotenv(tmp_path, "LLM_API_KEY=k\nLLM_BASE_URL=https://nim.example/v1\n")
    )
    assert s.LLM_BASE_URL == "https://nim.example/v1"


def test_blank_llm_base_url_does_not_hide_legacy_url(tmp_path, clean_llm_env):
    body = "LLM_API_KEY=k\nLLM_BASE_URL=\nOPENAI_BASE_URL=https://nim.example/v1\n"
    assert Settings(_env_file=_dotenv(tmp_path, body)).LLM_BASE_URL == (
        "https://nim.example/v1"
    )


def test_blank_llm_key_falls_back_to_legacy_key(tmp_path, clean_llm_env):
    body = "LLM_API_KEY=\nOPENAI_API_KEY=real-key\n"
    assert Settings(_env_file=_dotenv(tmp_path, body)).LLM_API_KEY == "real-key"


def test_blank_key_alone_fails(tmp_path, clean_llm_env):
    with pytest.raises(ValidationError):
        Settings(_env_file=_dotenv(tmp_path, "LLM_API_KEY=\n"))


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        # Pre-async .env files used the sync driver, which create_async_engine rejects.
        ("sqlite:///database.db", "sqlite+aiosqlite:///database.db"),
        ("sqlite://", "sqlite+aiosqlite://"),
        ("sqlite+aiosqlite:///database.db", "sqlite+aiosqlite:///database.db"),
        ("postgresql+asyncpg://u@h/db", "postgresql+asyncpg://u@h/db"),
    ],
)
def test_sync_sqlite_url_uses_async_driver(monkeypatch, given, expected):
    monkeypatch.setenv("DATABASE_URL", given)
    assert Settings(LLM_API_KEY="k", _env_file=None).DATABASE_URL == expected
