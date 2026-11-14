from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai.messages import ToolCallPart

from src.approvals import build_results, summarize_approvals


def _requests():
    part = ToolCallPart(
        tool_name="create_appointment",
        args={"title": "Sync", "start_time": "2026-02-14T10:00:00"},
        tool_call_id="call_1",
    )
    return DeferredToolRequests(approvals=[part])


def test_summarize_lists_tool_and_args():
    text = summarize_approvals(_requests())
    assert "create_appointment" in text
    assert "Sync" in text


def test_build_results_approved():
    results = build_results(_requests(), approved=True)
    assert isinstance(results, DeferredToolResults)
    assert isinstance(results.approvals["call_1"], ToolApproved)


def test_build_results_denied():
    results = build_results(_requests(), approved=False)
    assert isinstance(results.approvals["call_1"], ToolDenied)
