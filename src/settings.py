from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4.1-mini"
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    DATABASE_URL: str = "sqlite:///database.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
