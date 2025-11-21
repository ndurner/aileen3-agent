from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AgentBackend(ABC):
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
