import pytest
from unittest.mock import AsyncMock, MagicMock
from src.tools import resolve_scheduling_conflicts, Deps
from src.schemas import ConflictResolution, ProposedSlot
from datetime import datetime
from pydantic_ai import RunContext

@pytest.mark.asyncio
async def test_resolve_scheduling_conflicts_calls_subagent():
    # Mock the negotiator agent
    mock_negotiator = MagicMock()
    mock_negotiator.run = AsyncMock()
    
    # Mock return value
    expected_resolution = ConflictResolution(
        has_conflict=True,
        conflict_summary="Conflict detected",
        alternative_suggestions=[
            ProposedSlot(start_time=datetime.now(), end_time=datetime.now(), reasoning="Alternative")
        ],
        user_action_required="Choose a slot"
    )
    # The run method returns a RunResult, which has an 'output' attribute
    mock_result = MagicMock()
    mock_result.output = expected_resolution
    mock_negotiator.run.return_value = mock_result
    
    # Mock context
    mock_session = MagicMock()
    deps = Deps(session=mock_session, negotiator_agent=mock_negotiator)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = deps
    mock_ctx.usage = MagicMock()
    
    # Call the tool
    result = await resolve_scheduling_conflicts(
        mock_ctx, 
        user_request="Book meeting", 
        relevant_appointments=["Existing meeting"]
    )
    
    # Verify the subagent was called with correct usage propagation
    mock_negotiator.run.assert_called_once()
    call_args = mock_negotiator.run.call_args
    assert "User Request: Book meeting" in call_args[0][0]
    assert call_args[1]['usage'] == mock_ctx.usage
    
    # Verify output
    assert result == expected_resolution
