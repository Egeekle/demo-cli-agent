from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage

from src.providers import agent_model, default_model_settings
from src.tools import (
    Deps,
    check_availability,
    create_appointment,
    delete_appointment,
    get_appointment,
    list_appointments,
    resolve_scheduling_conflicts,
    search_appointments,
    update_appointment,
)

_MAX_HISTORY_MESSAGES = 20


def _trim_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Bound token growth by keeping only the most recent messages."""
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        return messages
    return messages[-_MAX_HISTORY_MESSAGES:]


appointment_agent = Agent(
    agent_model,
    deps_type=Deps,
    output_type=[str, DeferredToolRequests],
    model_settings=default_model_settings,
    retries=2,
    history_processors=[_trim_history],
    instructions=(
        "You are a helpful personal assistant managing appointments.\n"
        "You can list, search, view, create, update, and delete appointments.\n"
        "\n"
        "### SAFETY ###\n"
        "For any action that modifies data (create, update, delete):\n"
        "- First describe exactly what you will do.\n"
        "- The system will require the user's explicit approval before the change runs.\n"
        "\n"
        "### SCHEDULING & CONFLICTS ###\n"
        "- Before creating or rescheduling, call `check_availability` for the target slot.\n"
        "- If it returns overlaps, call `resolve_scheduling_conflicts` with the user request "
        "and the proposed start/end. It returns structured alternatives.\n"
        "- Present the alternatives in a MARKDOWN TABLE with columns 'Time Slot' and 'Reasoning'.\n"
        "- Never book or move an appointment into a conflicting slot.\n"
        "\n"
        "Be polite and concise."
    ),
)

appointment_agent.tool(list_appointments)
appointment_agent.tool(search_appointments)
appointment_agent.tool(get_appointment)
appointment_agent.tool(check_availability)
appointment_agent.tool(create_appointment, requires_approval=True)
appointment_agent.tool(update_appointment, requires_approval=True)
appointment_agent.tool(delete_appointment, requires_approval=True)
appointment_agent.tool(resolve_scheduling_conflicts)


@appointment_agent.instructions
def current_date(ctx) -> str:
    return f"Today is {ctx.deps.now():%A, %B %d, %Y}."
