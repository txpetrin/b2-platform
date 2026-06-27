from typing import Sequence
from pydantic_ai.messages import (ModelMessage,
                                  ModelRequest,
                                  UserPromptPart,
                                  tool_call_message)

from return_history import serialize_messages

def test_empty_messages() -> None:
    messages: Sequence[ModelMessage] = []

    result: list[dict] = serialize_messages(messages)

    assert result == []

def test_single_text_message() -> None:
    messages: Sequence[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="Hello"
                )
            ]
        )
    ]

    result: list[dict] = serialize_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], dict)
    

def test_tool_call_preserved() -> None:
    messages: Sequence[ModelMessage] = [
        tool_call_message
    ]

    result: list[dict] = serialize_messages(messages)

    assert len(result) == 1

    parts = result[0]["parts"]

    assert len(parts) == 1
    assert parts[0]["part_kind"] == "tool-call"