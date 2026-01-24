from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.schemas import ConflictResolution
from src.settings import settings
import mlflow.pydantic_ai

# Initialize the provider and model (consistent with main agent)
openai_provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
model = OpenAIResponsesModel(settings.MODEL_NAME, provider=openai_provider)

# Enable MLflow autologging
mlflow.pydantic_ai.autolog()

# Initialize the negotiator agent
negotiator_agent = Agent(
    model,
    output_type=ConflictResolution,
    system_prompt=(
        "You are a specialized scheduling negotiator.\n"
        "Your goal is to analyze scheduling conflicts and propose alternative time slots.\n"
        "\n"
        "### INPUT CONTEXT ###\n"
        "You will receive:\n"
        "1. A user request (what they want to book).\n"
        "2. A list of existing appointments that overlap or are near the requested time.\n"
        "\n"
        "### YOUR JOB ###\n"
        "- Identify if there is a conflict.\n"
        "- If there is a conflict, provide a concise summary.\n"
        "- Suggest 2-3 logical alternative slots. Consider gaps between existing meetings.\n"
        "- Formulate a 'user_action_required' string that the supervisor can use to ask the user how they want to proceed.\n"
        "\n"
        "Return your analysis strictly in the required structured format."
    ),
)

@negotiator_agent.system_prompt
def get_current_date() -> str:
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."