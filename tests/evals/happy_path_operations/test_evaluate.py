from datetime import datetime

import pytest
from sqlmodel import select

import src.agent
import src.database
import src.negotiator
from src.models import Appointment
from tests.evals.happy_path_operations import evaluate


@pytest.mark.asyncio
async def test_evaluation_context_isolation():
    """Verify evaluation_context provides isolated DB and frozen time."""
    with evaluate.evaluation_context():
        # Check 1: Time is frozen for agent and negotiator
        assert src.agent.datetime.now() == datetime(2026, 2, 11, 10, 0, 0)
        assert src.negotiator.datetime.now() == datetime(2026, 2, 11, 10, 0, 0)

        # Check 2: DB is isolated and seeded with 5 appointments
        with src.database.get_session() as session:
            appts = session.exec(select(Appointment)).all()
            assert len(appts) == 5

            # Verify specific appointments exist
            titles = {a.title for a in appts}
            assert "Doctor's Appointment" in titles
            assert "Team Sync" in titles
            assert "Lunch with Sarah" in titles
            assert "Marketing Team Meeting" in titles
            assert "Dentist Appointment" in titles


@pytest.mark.asyncio
async def test_predict_fn_returns_response():
    """Verify predict_fn returns a string response."""
    with evaluate.evaluation_context():
        response = await evaluate.predict_fn("What appointments do I have tomorrow?")
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention tomorrow's appointments
        lowered = response.lower()
        assert "doctor" in lowered or "team sync" in lowered
