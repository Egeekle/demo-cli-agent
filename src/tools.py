from dataclasses import dataclass
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

def _validate_times(
    ctx: RunContext[Deps], start_time: datetime, end_time: datetime, check_past: bool = True
) -> None:
    if end_time <= start_time:
        raise ModelRetry("end_time must be after start_time.")
    if check_past and start_time < ctx.deps.now():
        raise ModelRetry("start_time is in the past; pick a future time.")


async def create_appointment(
    ctx: RunContext[Deps],
    title: str,
    start_time: datetime,
    end_time: datetime,
    location: Optional[str] = None,
    description: Optional[str] = None,
) -> Appointment:
    """Create a new appointment. Refuses to double-book: if the slot overlaps an
    existing appointment, raises so the model must resolve the conflict first."""
    _validate_times(ctx, start_time, end_time)
    conflicts = await check_availability(ctx, start=start_time, end=end_time)
    if conflicts:
        names = ", ".join(f"{c.title} ({c.start_time:%b %d %H:%M})" for c in conflicts)
        raise ModelRetry(
            f"This slot conflicts with: {names}. Do not create it; resolve the "
            f"conflict (offer the user alternative slots) before creating."
        )
    appointment = Appointment(
        title=title,
        start_time=start_time,
        end_time=end_time,
        location=location,
        description=description,
    )
    ctx.deps.session.add(appointment)
    await ctx.deps.session.commit()
    await ctx.deps.session.refresh(appointment)
    return appointment


async def delete_appointment(ctx: RunContext[Deps], appointment_id: int) -> bool:
    """Delete an appointment by ID. Returns False if it does not exist."""
    appointment = await ctx.deps.session.get(Appointment, appointment_id)
    if not appointment:
        return False
    await ctx.deps.session.delete(appointment)
    await ctx.deps.session.commit()
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
    proposed_start: datetime,
    proposed_end: datetime,
) -> ConflictResolution:
    """Detect and resolve a scheduling conflict for a proposed time window.

    Fetches the real overlapping appointments from the database and asks the
    negotiator subagent to propose alternatives. Returns structured suggestions.
    """
    if ctx.deps.negotiator_agent is None:
        raise ValueError("Negotiator agent not provided in dependencies")

    conflicts = await check_availability(ctx, start=proposed_start, end=proposed_end)
    conflict_lines = "\n".join(
        f"- {c.title}: {c.start_time:%A %b %d %H:%M}-{c.end_time:%H:%M}"
        + (f" @ {c.location}" if c.location else "")
        for c in conflicts
    ) or "- (none)"

    prompt = (
        f"Today is {ctx.deps.now():%A, %B %d, %Y}.\n"
        f"User Request: {user_request}\n"
        f"Proposed Slot: {proposed_start:%A %b %d %H:%M}-{proposed_end:%H:%M}\n"
        f"Overlapping Existing Appointments:\n{conflict_lines}"
    )
    result = await ctx.deps.negotiator_agent.run(prompt, usage=ctx.usage)
    return result.output


async def update_appointment(
    ctx: RunContext[Deps],
    appointment_id: int,
    title: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
) -> Appointment:
    """Partially update an appointment. Only provided fields change. Refuses to
    reschedule into a conflicting slot."""
    appointment = await ctx.deps.session.get(Appointment, appointment_id)
    if not appointment:
        raise ModelRetry(f"No appointment with id {appointment_id} exists.")

    new_start = start_time if start_time is not None else appointment.start_time
    new_end = end_time if end_time is not None else appointment.end_time

    if start_time is not None or end_time is not None:
        # Only re-check the past-start rule when start_time is actually
        # changing; otherwise an unchanged (already-past-relative-to-"now")
        # start_time would wrongly block an end_time-only edit.
        _validate_times(ctx, new_start, new_end, check_past=start_time is not None)
        conflicts = await check_availability(
            ctx, start=new_start, end=new_end, exclude_id=appointment_id
        )
        if conflicts:
            names = ", ".join(f"{c.title} ({c.start_time:%b %d %H:%M})" for c in conflicts)
            raise ModelRetry(
                f"Rescheduling conflicts with: {names}. Resolve the conflict before updating."
            )

    if title is not None:
        appointment.title = title
    if start_time is not None:
        appointment.start_time = start_time
    if end_time is not None:
        appointment.end_time = end_time
    if location is not None:
        appointment.location = location
    if description is not None:
        appointment.description = description

    ctx.deps.session.add(appointment)
    await ctx.deps.session.commit()
    await ctx.deps.session.refresh(appointment)
    return appointment
