from types import SimpleNamespace

import pytest

from src import tools
from src.tools import Deps
from tests.conftest import fixed_clock


def _ctx(session):
    deps = Deps(session=session, negotiator_agent=None, now=fixed_clock)  # type: ignore[arg-type]
    return SimpleNamespace(deps=deps)


@pytest.mark.asyncio
async def test_list_appointments_returns_all(session):
    result = await tools.list_appointments(_ctx(session))
    assert len(result) == 3


@pytest.mark.asyncio
async def test_search_appointments_by_title(session):
    result = await tools.search_appointments(_ctx(session), query="Sarah")
    assert len(result) == 1
    assert result[0].title == "Lunch with Sarah"


@pytest.mark.asyncio
async def test_get_appointment_by_id(session):
    all_appts = await tools.list_appointments(_ctx(session))
    target = all_appts[0]
    fetched = await tools.get_appointment(_ctx(session), id=target.id)
    assert fetched is not None
    assert fetched.id == target.id
