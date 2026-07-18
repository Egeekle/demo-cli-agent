from datetime import datetime

import pytest
from sqlmodel import select

from src.models import Appointment
from tests.evals.happy_path_operations import evaluate


async def test_evaluation_context_isolation():
    """Verify evaluation_context provides an isolated async DB and a fixed clock."""
    async with evaluate.evaluation_context() as factory:
        # Check 1: The injected clock is frozen (no datetime patching involved).
        assert evaluate.FIXED_NOW == datetime(2026, 2, 11, 10, 0, 0)

        # Check 2: DB is isolated and seeded with 5 appointments.
        async with factory() as session:
            appts = (await session.exec(select(Appointment))).all()
            assert len(appts) == 5

            # Verify specific appointments exist
            titles = {a.title for a in appts}
            assert "Doctor's Appointment" in titles
            assert "Team Sync" in titles
            assert "Lunch with Sarah" in titles
            assert "Marketing Team Meeting" in titles
            assert "Dentist Appointment" in titles


@pytest.mark.skip(
    reason="Requires a live OpenAI call (appointment_agent.run against the real "
    "model); no network/API keys in this environment. Run manually with "
    "credentials configured."
)
async def test_predict_fn_returns_response():
    """Verify predict_fn returns a string response."""
    async with evaluate.evaluation_context():
        response = await evaluate.predict_fn("What appointments do I have tomorrow?")
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention tomorrow's appointments
        lowered = response.lower()
        assert "doctor" in lowered or "team sync" in lowered
