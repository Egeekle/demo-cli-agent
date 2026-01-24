from datetime import datetime
from typing import List, Optional
from sqlmodel import select, col, Session
from pydantic_ai import RunContext
from src.models import Appointment


class Deps:
    def __init__(self, session: Session):
        self.session = session


def list_appointments(
    ctx: RunContext[Deps], limit: int = 10, offset: int = 0
) -> List[Appointment]:
    """
    List appointments with pagination.
    """
    statement = select(Appointment).offset(offset).limit(limit)
    results = ctx.deps.session.exec(statement)
    return list(results.all())


def search_appointments(ctx: RunContext[Deps], query: str) -> List[Appointment]:
    """
    Search appointments by title, location or description.
    """
    statement = select(Appointment).where(
        (col(Appointment.title).contains(query))
        | (col(Appointment.description).contains(query))
        | (col(Appointment.location).contains(query))
    )
    results = ctx.deps.session.exec(statement)
    return list(results.all())


def get_appointment(ctx: RunContext[Deps], id: int) -> Optional[Appointment]:
    """
    Get a specific appointment by ID.
    """
    return ctx.deps.session.get(Appointment, id)


def create_appointment(
    ctx: RunContext[Deps],
    title: str,
    start_time: datetime,
    end_time: datetime,
    location: Optional[str] = None,
    description: Optional[str] = None,
) -> Appointment:
    """
    Create a new appointment.
    """
    appointment = Appointment(
        title=title,
        start_time=start_time,
        end_time=end_time,
        location=location,
        description=description,
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
