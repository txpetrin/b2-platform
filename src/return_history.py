from typing import Sequence
from pydantic_ai.messages import ModelMessage

def serialize_messages(
        messages: Sequence[ModelMessage],
    ) -> list[dict]:

        return [
            message.model_dump(mode="json")
            for message in messages
        ]