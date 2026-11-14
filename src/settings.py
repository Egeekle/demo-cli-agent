from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Any OpenAI-compatible service (OpenAI, NVIDIA NIM, ...). Required: LLM_API_KEY
    # or the legacy OPENAI_API_KEY, so existing .env files keep working.
    LLM_API_KEY: str | None = None
    # None means OpenAI's default endpoint.
    LLM_BASE_URL: str | None = None
    # Legacy names, merged into the LLM_* fields below. They are separate fields
    # (not aliases) so LLM_* wins whichever source each one comes from.
    OPENAI_API_KEY: str | None = Field(default=None, repr=False)
    OPENAI_BASE_URL: str | None = None
    MODEL_NAME: str = "gpt-4.1-mini"
    JUDGE_MODEL_NAME: str = "gpt-5.2"
    MODEL_TEMPERATURE: float = 0.0
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    DATABASE_URL: str = "sqlite+aiosqlite:///database.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `LLM_BASE_URL=` copied from .env.example must not hide a legacy value.
        env_ignore_empty=True,
    )

    @field_validator("LLM_BASE_URL", "OPENAI_BASE_URL", mode="before")
    @classmethod
    def _blank_base_url_is_default(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("DATABASE_URL")
    @classmethod
    def _sqlite_uses_async_driver(cls, value: str) -> str:
        # The engine is async; pre-async .env files still say plain `sqlite://`.
        if value.startswith("sqlite://"):
            return "sqlite+aiosqlite://" + value.removeprefix("sqlite://")
        return value

    @model_validator(mode="after")
    def _merge_legacy_names(self) -> "Settings":
        self.LLM_API_KEY = self.LLM_API_KEY or self.OPENAI_API_KEY
        self.LLM_BASE_URL = self.LLM_BASE_URL or self.OPENAI_BASE_URL
        if not self.LLM_API_KEY:
            raise ValueError("LLM_API_KEY (or legacy OPENAI_API_KEY) is required")
        return self


settings = Settings()
