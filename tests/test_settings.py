from src.settings import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")

    settings = Settings()
    assert settings.OPENAI_API_KEY == "test-key"
    assert settings.MLFLOW_TRACKING_URI == "http://localhost:9999"
    assert settings.DATABASE_URL == "sqlite:///test.db"


def test_settings_default_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Clear other env vars to use defaults
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings()
    # Just verify they are populated (not empty)
    assert settings.MLFLOW_TRACKING_URI
    assert settings.DATABASE_URL
