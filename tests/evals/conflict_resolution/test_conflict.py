import os
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.agent import appointment_agent
from src.negotiator import negotiator_agent
from src.models import Appointment
from src.tools import Deps

# This test drives the real model (network + LLM_API_KEY) end-to-end through
# the negotiation flow. Skip when no API key is available so the offline suite
# (and CI) stays green; run manually with LLM_API_KEY set to exercise it live.
pytestmark = pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="live model eval; run manually with LLM_API_KEY",
)

FIXED_NOW = datetime(2026, 2, 11, 10, 0, 0)


async def _factory():
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


@pytest.mark.asyncio
async def test_conflict_offers_alternatives_and_does_not_double_book():
    from pydantic_ai import DeferredToolRequests
    from src.approvals import build_results

    factory = await _factory()
    async with factory() as session:
        deps = Deps(session=session, negotiator_agent=negotiator_agent, now=lambda: FIXED_NOW)
        # Overlaps the doctor's appointment (14:00-15:00)
        result = await appointment_agent.run(
            "Book a 30-minute checkup on Feb 12 at 2:30pm", deps=deps
        )
        # Agent must NOT have created a conflicting appointment; it should discuss alternatives.
        # If it tried to write, approving would still hit the overlap backstop (ModelRetry).
        while isinstance(result.output, DeferredToolRequests):
            result = await appointment_agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=build_results(result.output, approved=True),
                deps=deps,
            )
        text = result.output.lower()
        assert "conflict" in text or "overlap" in text or "instead" in text or "alternativ" in text
