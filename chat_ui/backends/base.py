from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AgentBackend(ABC):
    @staticmethod
    def _is_tool_event(event: dict) -> bool:
        """
        Return True when the event represents a tool call or response.

        We keep this lightweight and backend-agnostic so both the API server
        and Vertex Agent Engine backends can surface tool usage to the UI.
        """
        if not isinstance(event, dict):
            return False

        if event.get("author") != "assistant_agent":
            return False

        content = event.get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return False

        part = parts[0] or {}
        # Support both snake_case (ADK python) and camelCase (Vertex / API server)
        return any(
            key in part
            for key in (
                "function_call",
                "function_response",
                "functionCall",
                "functionResponse",
            )
        )

    @staticmethod
    def _is_displayable_event(event: dict) -> bool:
        """
        Return True when the event should be forwarded to the UI.

        This includes:
        - Streaming assistant response chunks (partial text)
        - Tool call / tool result events (so the UI can render tool messages)
        """
        if not isinstance(event, dict):
            return False

        author = event.get("author")
        partial = event.get("partial")

        # Streaming text partials from the assistant.
        if author == "assistant_agent" and bool(partial):
            return True

        # Tool calls and tool responses, which may not be marked as partials.
        return AgentBackend._is_tool_event(event)

    @abstractmethod
    async def ensure_session(
        self,
        user_id: str,
        existing_session_id: str | None,
        session_state: dict[str, str],
    ) -> str:
        """Return a session id, creating one if needed."""

    @abstractmethod
    async def stream_events(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncIterator[dict]:
        """Yield ADK event dicts for the given user, session, and message."""

    async def delete_session(
        self,
        user_id: str,
        session_id: str,
    ) -> None:
        """Delete a session if supported by the backend."""
        return None
