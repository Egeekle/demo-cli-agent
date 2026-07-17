from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4.1-mini"
    JUDGE_MODEL_NAME: str = "gpt-5.2"
    MODEL_TEMPERATURE: float = 0.0
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    DATABASE_URL: str = "sqlite+aiosqlite:///database.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
