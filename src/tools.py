from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional

from pydantic_ai import ModelRetry, RunContext
from sqlmodel import col, select

from src.models import Appointment
from src.schemas import ConflictResolution


@dataclass
class Deps:
    session: Any  # AsyncSession (sqlmodel.ext.asyncio.session)
    negotiator_agent: Any  # Agent[None, ConflictResolution]
    now: Callable[[], datetime] = datetime.now


async def list_appointments(
    ctx: RunContext[Deps], limit: int = 10, offset: int = 0
) -> List[Appointment]:
    """List appointments with pagination."""
    statement = select(Appointment).offset(offset).limit(limit)
    results = await ctx.deps.session.exec(statement)
    return list(results.all())


async def search_appointments(ctx: RunContext[Deps], query: str) -> List[Appointment]:
    """Search appointments by title, location or description."""
    statement = select(Appointment).where(
        (col(Appointment.title).contains(query))
        | (col(Appointment.description).contains(query))
        | (col(Appointment.location).contains(query))
    )
    results = await ctx.deps.session.exec(statement)
    return list(results.all())


async def get_appointment(ctx: RunContext[Deps], id: int) -> Optional[Appointment]:
    """Get a specific appointment by ID."""
    return await ctx.deps.session.get(Appointment, id)

def create_appointment(
    ctx: RunContext[Deps],
    title: str,
    start_time: datetime,
    end_time: datetime,
    location: Optional[str] = None,
    description: Optional[str] = None
) -> Appointment:
    """
    Create a new appointment.
    """
    appointment = Appointment(
        title=title,
        start_time=start_time,
        end_time=end_time,
        location=location,
        description=description
    )
    ctx.deps.session.add(appointment)
    ctx.deps.session.commit()
    ctx.deps.session.refresh(appointment)
    return appointment

def delete_appointment(ctx: RunContext[Deps], appointment_id: int) -> bool:
    """
    Delete an appointment by ID.
    """
    appointment = ctx.deps.session.get(Appointment, appointment_id)
    if not appointment:
        return False
    ctx.deps.session.delete(appointment)
    ctx.deps.session.commit()
    return True

async def check_availability(
    ctx: RunContext[Deps],
    start: datetime,
    end: datetime,
    exclude_id: Optional[int] = None,
) -> List[Appointment]:
    """Return existing appointments that overlap the half-open interval [start, end).

    Use this before creating or rescheduling to detect time conflicts.
    """
    statement = select(Appointment).where(
        col(Appointment.start_time) < end,
        col(Appointment.end_time) > start,
    )
    if exclude_id is not None:
        statement = statement.where(col(Appointment.id) != exclude_id)
    results = await ctx.deps.session.exec(statement)
    return list(results.all())


async def resolve_scheduling_conflicts(
    ctx: RunContext[Deps],
    user_request: str,
    relevant_appointments: List[str]
) -> ConflictResolution:
    """
    Call this tool when you detect a conflict between a user request and existing appointments.
    It will return structured alternative suggestions.

    Args:
        user_request: The original request from the user.
        relevant_appointments: A list of strings describing the conflicting or nearby appointments.
    """
    if not ctx.deps.negotiator_agent:
        raise ValueError("Negotiator agent not provided in dependencies")

    prompt = f"User Request: {user_request}\nExisting Appointments: {relevant_appointments}"

    # Pass usage context to track tokens correctly across agents
    result = await ctx.deps.negotiator_agent.run(prompt, usage=ctx.usage)
    return result.output
