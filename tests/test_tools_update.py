from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from src import tools
from src.tools import Deps
from tests.conftest import fixed_clock


def _ctx(session):
    return SimpleNamespace(deps=Deps(session=session, negotiator_agent=None, now=fixed_clock))  # type: ignore[arg-type]


async def _doctor(session):
    appts = await tools.list_appointments(_ctx(session))
    return next(a for a in appts if a.title == "Doctor's Appointment")


@pytest.mark.asyncio
async def test_update_title_only(session):
    doctor = await _doctor(session)
    updated = await tools.update_appointment(_ctx(session), appointment_id=doctor.id, title="Dr Visit")
    assert updated.title == "Dr Visit"
    assert updated.start_time == doctor.start_time  # unchanged


@pytest.mark.asyncio
async def test_update_time_to_free_slot(session):
    doctor = await _doctor(session)
    updated = await tools.update_appointment(
        _ctx(session),
        appointment_id=doctor.id,
        start_time=datetime(2026, 2, 12, 16, 0),
        end_time=datetime(2026, 2, 12, 17, 0),
    )
    assert updated.start_time == datetime(2026, 2, 12, 16, 0)


@pytest.mark.asyncio
async def test_update_time_into_conflict_raises(session):
    doctor = await _doctor(session)
    # Move onto Team Sync (09:00-09:30)
    with pytest.raises(ModelRetry):
        await tools.update_appointment(
            _ctx(session),
            appointment_id=doctor.id,
            start_time=datetime(2026, 2, 12, 9, 0),
            end_time=datetime(2026, 2, 12, 9, 30),
        )


@pytest.mark.asyncio
async def test_update_same_time_no_self_conflict(session):
    doctor = await _doctor(session)
    updated = await tools.update_appointment(
        _ctx(session),
        appointment_id=doctor.id,
        location="Room 5",
        start_time=doctor.start_time,
        end_time=doctor.end_time,
    )
    assert updated.location == "Room 5"


@pytest.mark.asyncio
async def test_update_missing_id_raises(session):
    with pytest.raises(ModelRetry):
        await tools.update_appointment(_ctx(session), appointment_id=99999, title="Nope")
