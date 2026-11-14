import os
from datetime import datetime

import pytest

from src.negotiator import negotiator_agent
from src.schemas import ConflictResolution

# This test drives the real model (network + LLM_API_KEY). Skip when no API
# key is available so the offline suite (and CI) stays green; run manually
# with LLM_API_KEY set to exercise it live.
pytestmark = pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="live model eval; run manually with LLM_API_KEY",
)


@pytest.mark.asyncio
async def test_negotiator_returns_structured_output():
    # Simulate a conflict scenario
    user_request = "Schedule a meeting for tomorrow at 2pm"
    existing_appointments = [
        "Meeting with Bob: Tomorrow 1:30 PM - 2:30 PM"
    ]
    
    prompt = f"""
    Today is {datetime.now():%A, %B %d, %Y}.
    User Request: {user_request}
    Existing Appointments: {existing_appointments}
    """
    
    result = await negotiator_agent.run(prompt)
    
    # Verify the output is structured correctly
    assert isinstance(result.output, ConflictResolution)
    assert result.output.has_conflict is True
    assert len(result.output.alternative_suggestions) > 0
    
    # Check that suggestions are logical (e.g., after the conflict)
    first_suggestion = result.output.alternative_suggestions[0]
    # Remove timezone info for comparison if it exists
    suggestion_time = first_suggestion.start_time.replace(tzinfo=None)
    assert suggestion_time > datetime.now()
