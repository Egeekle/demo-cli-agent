from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied


def summarize_approvals(requests: DeferredToolRequests) -> str:
    """Render pending human-in-the-loop approvals as a readable summary."""
    lines = []
    for part in requests.approvals:
        args = part.args_as_dict()
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        lines.append(f"- {part.tool_name}({arg_str})")
    return "\n".join(lines)


def build_results(
    requests: DeferredToolRequests,
    approved: bool,
    denial_message: str = "The user declined this action.",
) -> DeferredToolResults:
    """Approve or deny every pending approval request."""
    decision = ToolApproved() if approved else ToolDenied(denial_message)
    return DeferredToolResults(
        approvals={part.tool_call_id: decision for part in requests.approvals}
    )
