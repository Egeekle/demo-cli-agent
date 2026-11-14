from datetime import datetime
from types import SimpleNamespace

import pytest

from src import tools
from src.tools import Deps
from tests.conftest import fixed_clock


def _ctx(session):
    return SimpleNamespace(deps=Deps(session=session, negotiator_agent=None, now=fixed_clock))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_overlap_detected(session):
    # Doctor's Appointment is 14:00-15:00 on Feb 12
    conflicts = await tools.check_availability(
        _ctx(session), start=datetime(2026, 2, 12, 14, 30), end=datetime(2026, 2, 12, 15, 30)
    )
    assert [a.title for a in conflicts] == ["Doctor's Appointment"]


@pytest.mark.asyncio
async def test_adjacent_slot_is_free(session):
    # 15:00-16:00 starts exactly when Doctor's ends -> no overlap
    conflicts = await tools.check_availability(
        _ctx(session), start=datetime(2026, 2, 12, 15, 0), end=datetime(2026, 2, 12, 16, 0)
    )
    assert conflicts == []


@pytest.mark.asyncio
async def test_exclude_id_ignores_self(session):
    all_appts = await tools.list_appointments(_ctx(session))
    doctor = next(a for a in all_appts if a.title == "Doctor's Appointment")
    conflicts = await tools.check_availability(
        _ctx(session),
        start=datetime(2026, 2, 12, 14, 0),
        end=datetime(2026, 2, 12, 15, 0),
        exclude_id=doctor.id,
    )
    assert conflicts == []
