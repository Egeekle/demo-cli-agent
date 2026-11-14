from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry
from sqlmodel import select

from src import tools
from src.models import Appointment
from src.tools import Deps
from tests.conftest import fixed_clock


def _ctx(session):
    return SimpleNamespace(deps=Deps(session=session, negotiator_agent=None, now=fixed_clock))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_create_free_slot_persists(session):
    appt = await tools.create_appointment(
        _ctx(session),
        title="New Meeting",
        start_time=datetime(2026, 2, 14, 10, 0),
        end_time=datetime(2026, 2, 14, 11, 0),
    )
    assert appt.id is not None
    rows = (await session.exec(select(Appointment).where(Appointment.title == "New Meeting"))).all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_create_overlap_raises_and_does_not_persist(session):
    with pytest.raises(ModelRetry):
        await tools.create_appointment(
            _ctx(session),
            title="Clash",
            start_time=datetime(2026, 2, 12, 14, 30),
            end_time=datetime(2026, 2, 12, 15, 30),
        )
    rows = (await session.exec(select(Appointment).where(Appointment.title == "Clash"))).all()
    assert rows == []


@pytest.mark.asyncio
async def test_create_rejects_inverted_times(session):
    with pytest.raises(ModelRetry):
        await tools.create_appointment(
            _ctx(session),
            title="Bad",
            start_time=datetime(2026, 2, 14, 11, 0),
            end_time=datetime(2026, 2, 14, 10, 0),
        )


@pytest.mark.asyncio
async def test_create_rejects_past_start(session):
    with pytest.raises(ModelRetry):
        await tools.create_appointment(
            _ctx(session),
            title="Past",
            start_time=datetime(2026, 2, 10, 9, 0),  # before fixed_clock (Feb 11)
            end_time=datetime(2026, 2, 10, 10, 0),
        )


@pytest.mark.asyncio
async def test_delete_existing_returns_true(session):
    all_appts = await tools.list_appointments(_ctx(session))
    ok = await tools.delete_appointment(_ctx(session), appointment_id=all_appts[0].id)
    assert ok is True


@pytest.mark.asyncio
async def test_delete_missing_returns_false(session):
    ok = await tools.delete_appointment(_ctx(session), appointment_id=99999)
    assert ok is False
