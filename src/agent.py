import mlflow.pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from datetime import datetime

from src.tools import (
    Deps,
    list_appointments,
    search_appointments,
    get_appointment,
    create_appointment,
    delete_appointment,
    resolve_scheduling_conflicts
)
from src.settings import settings

# Configure MLflow tracking
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.pydantic_ai.autolog()

# Initialize the provider and model using settings
openai_provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
model = OpenAIResponsesModel(settings.MODEL_NAME, provider=openai_provider)

# Initialize the agent with Deps
appointment_agent = Agent(
    model,
    deps_type=Deps,
    output_type=str, # Explicitly stating we want string output
    system_prompt=(
        "You are a helpful personal assistant managing appointments.\n"
        "You can list, search, view, create, and delete appointments.\n"
        "\n"
        "### IMPORTANT SAFETY RULE ###\n"
        "1. For any action that modifies data (CREATE or DELETE):\n"
        "   - You MUST first describe exactly what you are going to do to the user.\n"
        "   - You MUST ask for explicit permission (e.g., 'Shall I proceed with this?').\n"
        "   - You MUST wait for the user to say 'yes', 'confirm', or similar before calling the tool.\n"
        "   - If the user says 'no' or expresses doubt, do not call the tool.\n"
        "\n"
        "2. For information requests (LIST, SEARCH, GET):\n"
        "   - You can call these tools immediately to provide information.\n"
        "\n"
        "### CONFLICT RESOLUTION ###\n"
        "If you check for availability (e.g., using `search_appointments`) and find a conflict:\n"
        "1. DO NOT just say 'There is a conflict'.\n"
        "2. Call the `resolve_scheduling_conflicts` tool with the user's request and the list of relevant/overlapping appointments.\n"
        "3. The tool will return structured suggestions. Use them to offer the user specific alternatives.\n"
        "4. PRESENT THE ALTERNATIVES IN A MARKDOWN TABLE with columns: 'Time Slot', 'Reasoning'.\n"
        "\n"
        "Today's date is available to you. Be polite and concise."
    ),
)

# Register tools
appointment_agent.tool(list_appointments)
appointment_agent.tool(search_appointments)
appointment_agent.tool(get_appointment)
appointment_agent.tool(create_appointment)
appointment_agent.tool(delete_appointment)
appointment_agent.tool(resolve_scheduling_conflicts)

@appointment_agent.system_prompt
def get_current_date() -> str:
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."