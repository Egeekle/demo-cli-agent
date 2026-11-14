from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

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


def _is_user_turn_boundary(message: ModelMessage) -> bool:
    """A "user turn" starts at a `ModelRequest` carrying a `UserPromptPart`.

    Trimming must only ever cut the history at one of these boundaries:
    everything else (ToolReturnPart requests, ModelResponse tool calls, plain
    text responses) belongs to the middle of an exchange and can't stand on
    its own as the start of history without orphaning a tool-call/tool-result
    pair, which the OpenAI Responses API rejects with a 400.
    """
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def _trim_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Bound token growth by keeping only the most recent messages, without
    ever splitting a ModelResponse's ToolCallPart from its matching
    ModelRequest ToolReturnPart.

    A naive `messages[-_MAX_HISTORY_MESSAGES:]` slice can land inside an
    exchange (e.g. between a tool call and its return), which produces a
    history that starts with an orphaned tool result / dangling tool call and
    gets rejected by the model API. Instead we only ever cut at a clean
    user-turn boundary (see `_is_user_turn_boundary`):

    1. Compute the naive cut point, then scan forward from it for the
       earliest user-turn boundary and cut there instead. This keeps the
       trimmed history close to `_MAX_HISTORY_MESSAGES` while guaranteeing it
       starts cleanly.
    2. If the tail (from the naive cut point onward) has no boundary at all
       (the whole tail is one long in-flight exchange), fall back to the
       nearest boundary *before* the cut point so we still trim.
    3. If no boundary exists anywhere in the history, return it unchanged —
       we never return a history that begins mid-exchange.
    """
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        return messages

    cut = len(messages) - _MAX_HISTORY_MESSAGES

    for i in range(cut, len(messages)):
        if _is_user_turn_boundary(messages[i]):
            return messages[i:]

    for i in range(cut - 1, -1, -1):
        if _is_user_turn_boundary(messages[i]):
            return messages[i:]

    return messages


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
