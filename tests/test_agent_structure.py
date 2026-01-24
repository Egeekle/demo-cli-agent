from src.agent import appointment_agent
from src.settings import settings
from pydantic_ai.models.openai import OpenAIResponsesModel


def test_agent_initialization():
    # Verify model is correctly initialized from settings
    assert isinstance(appointment_agent.model, OpenAIResponsesModel)
    assert appointment_agent.model.model_name == settings.MODEL_NAME

    # Check that tools are registered
    tool_names = list(appointment_agent._function_toolset.tools.keys())
    assert "list_appointments" in tool_names
    assert "search_appointments" in tool_names
    assert "get_appointment" in tool_names
    assert "create_appointment" in tool_names
    assert "delete_appointment" in tool_names


def test_agent_system_prompt():
    # system_prompt in the constructor is stored in _system_prompts tuple
    combined_prompts = " ".join(appointment_agent._system_prompts)
    assert "IMPORTANT SAFETY RULE" in combined_prompts
    assert "CREATE or DELETE" in combined_prompts
