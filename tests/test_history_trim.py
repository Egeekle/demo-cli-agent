"""Pure-function tests for `_trim_history`'s turn-boundary trimming.

No model calls here: we build synthetic `ModelMessage` histories directly with
the real pydantic_ai.messages dataclasses and assert on what `_trim_history`
returns.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from src.agent import _MAX_HISTORY_MESSAGES, _trim_history


def _user_turn(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tool_call_response(tool_name: str, call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={}, tool_call_id=call_id)])


def _tool_return_request(tool_name: str, call_id: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=tool_name, content="ok", tool_call_id=call_id)])


def _normal_turn(tag: str) -> list[ModelMessage]:
    return [_user_turn(f"user-{tag}"), _text_response(f"assistant-{tag}")]


def _tool_turn(tag: str, call_id: str) -> list[ModelMessage]:
    """A full user turn that includes a tool-call / tool-return pair."""
    return [
        _user_turn(f"user-{tag}"),
        _tool_call_response(f"tool_{tag}", call_id),
        _tool_return_request(f"tool_{tag}", call_id),
        _text_response(f"assistant-{tag}"),
    ]


def _all_tool_return_parts_have_matching_call(messages: list[ModelMessage]) -> bool:
    seen_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    seen_call_ids.add(part.tool_call_id)
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id not in seen_call_ids:
                    return False
    return True


def test_trim_history_advances_past_split_tool_pair_to_next_boundary():
    """A naive `messages[-20:]` slice would cut right on the ToolReturnPart of
    the first tool turn below (orphaning it from its ToolCallPart). The fix
    must instead advance to the next clean user-turn boundary.
    """
    messages: list[ModelMessage] = []

    # 8 padding turns (16 messages) before the tool turn that spans the cut.
    for i in range(8):
        messages.extend(_normal_turn(f"pad-before-{i}"))

    # This tool turn's ToolReturnPart request lands exactly on the naive cut
    # point (len - _MAX_HISTORY_MESSAGES == 18), which is not a user-turn
    # boundary. It must be excluded, not split, by the trim.
    messages.extend(_tool_turn("split", "call-split"))

    # A second tool turn, fully inside the retained tail, so assertion (b)
    # below is a real check (a matching call/return pair present together)
    # rather than a vacuous one.
    messages.extend(_tool_turn("kept", "call-kept"))

    # Padding turns after, to pad the total length out.
    for i in range(7):
        messages.extend(_normal_turn(f"pad-after-{i}"))

    assert len(messages) == 38
    naive_cut = len(messages) - _MAX_HISTORY_MESSAGES
    assert isinstance(messages[naive_cut], ModelRequest)
    assert isinstance(messages[naive_cut].parts[0], ToolReturnPart)

    trimmed = _trim_history(messages)

    # (a) The returned history begins with a ModelRequest containing a
    # UserPromptPart -- never mid-exchange.
    first = trimmed[0]
    assert isinstance(first, ModelRequest)
    assert any(isinstance(part, UserPromptPart) for part in first.parts)

    # (b) No ToolReturnPart appears in the result without its matching
    # ToolCallPart appearing earlier in the result.
    assert _all_tool_return_parts_have_matching_call(trimmed)

    # The split tool turn's orphan-prone ToolReturnPart must not be present.
    for message in trimmed:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    assert part.tool_call_id != "call-split"

    # The fully-retained tool turn's pair should both still be present.
    call_ids = {
        part.tool_call_id
        for message in trimmed
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    return_ids = {
        part.tool_call_id
        for message in trimmed
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert "call-kept" in call_ids
    assert "call-kept" in return_ids


def test_trim_history_short_history_untouched():
    messages: list[ModelMessage] = []
    for i in range(3):
        messages.extend(_normal_turn(str(i)))
    assert len(messages) <= _MAX_HISTORY_MESSAGES
    assert _trim_history(messages) is messages


def test_trim_history_falls_back_to_full_list_when_tail_has_no_boundary():
    """Degenerate case: everything after the single opening user turn is one
    long tool-call/tool-return exchange, longer than the cap, with no
    user-turn boundary anywhere in the tail. There's also none *before* the
    naive cut except index 0, so both the forward scan and the backward
    fallback land on the very first message -- the full list comes back
    unchanged rather than a slice that begins mid-exchange.
    """
    messages: list[ModelMessage] = [_user_turn("start")]
    for i in range(30):
        messages.append(_tool_call_response("t", f"call-{i}"))
        messages.append(_tool_return_request("t", f"call-{i}"))
    assert len(messages) > _MAX_HISTORY_MESSAGES

    trimmed = _trim_history(messages)

    assert trimmed == messages
