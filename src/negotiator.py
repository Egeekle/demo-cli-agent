from datetime import datetime

from pydantic_ai import Agent

from src.providers import agent_model, default_model_settings
from src.schemas import ConflictResolution

negotiator_agent: Agent[None, ConflictResolution] = Agent(
    agent_model,
    output_type=ConflictResolution,
    model_settings=default_model_settings,
    instructions=(
        "You are a specialized scheduling negotiator.\n"
        "Your goal is to analyze scheduling conflicts and propose alternative time slots.\n"
        "\n"
        "### INPUT CONTEXT ###\n"
        "You receive a user request, the proposed slot, and the real overlapping "
        "appointments (title + times).\n"
        "\n"
        "### YOUR JOB ###\n"
        "- Identify whether there is a conflict.\n"
        "- If so, provide a concise summary.\n"
        "- Suggest 2-3 logical alternative slots; consider gaps between existing meetings.\n"
        "- Formulate a 'user_action_required' string the supervisor can use to ask the "
        "user how to proceed.\n"
        "\n"
        "Return your analysis strictly in the required structured format."
    ),
)


@negotiator_agent.instructions
def current_date() -> str:
    return f"Today is {datetime.now():%A, %B %d, %Y}."
