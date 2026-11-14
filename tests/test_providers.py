def test_build_model_and_shared_model_exist():
    from src import providers
    from pydantic_ai.models.openai import OpenAIResponsesModel

    assert isinstance(providers.agent_model, OpenAIResponsesModel)
    m = providers.build_model("gpt-4.1-mini")
    assert isinstance(m, OpenAIResponsesModel)
    assert providers.default_model_settings["temperature"] == 0.0
