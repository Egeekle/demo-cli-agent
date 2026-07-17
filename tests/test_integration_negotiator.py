from datetime import datetime
from types import SimpleNamespace

import pytest

from src import tools
from src.schemas import ConflictResolution, ProposedSlot
from src.tools import Deps
from tests.conftest import fixed_clock


class FakeNegotiator:
    def __init__(self):
        self.received_prompt = None

    async def run(self, prompt, usage=None):
        self.received_prompt = prompt
        output = ConflictResolution(
            has_conflict=True,
            conflict_summary="Overlaps Doctor's Appointment",
            alternative_suggestions=[
                ProposedSlot(
                    start_time=datetime(2026, 2, 12, 16, 0),
                    end_time=datetime(2026, 2, 12, 17, 0),
                    reasoning="Free after the doctor's visit",
                )
            ],
            user_action_required="Pick an alternative slot",
        )
        return SimpleNamespace(output=output)


def _ctx(session, negotiator):
    deps = Deps(session=session, negotiator_agent=negotiator, now=fixed_clock)
    return SimpleNamespace(deps=deps, usage=None)


@pytest.mark.asyncio
async def test_resolve_passes_real_overlap_titles_to_negotiator(session):
    fake = FakeNegotiator()
    result = await tools.resolve_scheduling_conflicts(
        _ctx(session, fake),
        user_request="Book a checkup Feb 12 at 2:30pm",
        proposed_start=datetime(2026, 2, 12, 14, 30),
        proposed_end=datetime(2026, 2, 12, 15, 30),
    )
    assert isinstance(result, ConflictResolution)
    # The real overlapping appointment title must be in the prompt (no telephone game)
    assert "Doctor's Appointment" in fake.received_prompt
    assert result.alternative_suggestions[0].start_time == datetime(2026, 2, 12, 16, 0)
