import os
from datetime import datetime

import pytest
from pydantic_ai import DeferredToolRequests
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.agent import appointment_agent
from src.approvals import build_results
from src.models import Appointment
from src.negotiator import negotiator_agent
from src.tools import Deps

# These tests drive the real model (network + LLM_API_KEY) end-to-end through
# the approval flow. Skip when no API key is available so the offline suite (and
# CI) stays green; run manually with LLM_API_KEY set to exercise them live.
pytestmark = pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="live model eval; run manually with LLM_API_KEY",
)

FIXED_NOW = datetime(2026, 2, 11, 10, 0, 0)


async def _seeded_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with factory() as s:
        s.add(Appointment(title="Doctor's Appointment",
                          start_time=datetime(2026, 2, 12, 14, 0),
                          end_time=datetime(2026, 2, 12, 15, 0)))
        await s.commit()
    return factory


async def _drive(factory, prompt, history, approve):
    async with factory() as session:
        deps = Deps(session=session, negotiator_agent=negotiator_agent, now=lambda: FIXED_NOW)
        result = await appointment_agent.run(prompt, message_history=history, deps=deps)
        asked = isinstance(result.output, DeferredToolRequests)
        while isinstance(result.output, DeferredToolRequests):
            results = build_results(result.output, approved=approve)
            result = await appointment_agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=results, deps=deps,
            )
        return result, asked


@pytest.mark.asyncio
async def test_delete_denied_keeps_row():
    factory = await _seeded_factory()
    _, asked = await _drive(factory, "Delete my doctor's appointment", [], approve=False)
    assert asked is True  # approval was requested
    async with factory() as s:
        rows = (await s.exec(select(Appointment).where(Appointment.title == "Doctor's Appointment"))).all()
    assert len(rows) == 1  # denial preserved the row


@pytest.mark.asyncio
async def test_delete_approved_removes_row():
    factory = await _seeded_factory()
    await _drive(factory, "Delete my doctor's appointment", [], approve=True)
    async with factory() as s:
        rows = (await s.exec(select(Appointment).where(Appointment.title == "Doctor's Appointment"))).all()
    assert len(rows) == 0
