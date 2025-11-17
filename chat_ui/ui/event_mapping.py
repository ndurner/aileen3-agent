from __future__ import annotations

import json
from typing import List

from gradio import ChatMessage


def decode_event(event: dict) -> ChatMessage | None:
    content = event.get("content", {}) or {}
    parts = content.get("parts") or []
    if not parts:
        return None

    part = parts[0]

    if "function_call" in part:
        fc = part["function_call"]
        tool_name = fc.get("name", "tool")
        args = fc.get("args", {})
        return ChatMessage(
                role="assistant",
                content=f"Calling `{tool_name}` with arguments:",
                metadata={
                    "title": f"Tool call: {tool_name}",
                    "log": json.dumps(args, indent=2),
                    "status": "pending",
                },
            )

    if "function_response" in part:
        fr = part["function_response"]
        tool_name = fr.get("name", "tool")
        resp = fr.get("response", {})
        return ChatMessage(
                role="assistant",
                content=json.dumps(resp, indent=2),
                metadata={
                    "title": f"Tool result: {tool_name}",
                    "status": "done",
                },
            )

    if "text" in part:
        text = part["text"]
        return ChatMessage(role="assistant", content=text)
