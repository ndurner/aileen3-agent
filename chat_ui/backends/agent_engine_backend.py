from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import vertexai

from .base import AgentBackend
from chat_ui.config import AgentEngineConfig


@dataclass
class AgentEngineBackend(AgentBackend):
    config: AgentEngineConfig

    def __post_init__(self) -> None:
        # Initialize client once; AdkApp provides session and streaming helpers.
        self._client = vertexai.Client(
            project=self.config.project_id,
            location=self.config.location,
        )
        self._adk_app = self._client.agent_engines.get(
            name=self.config.agent_engine_name
        )

    async def ensure_session(
        self,
        user_id: str,
        existing_session_id: str | None,
        session_state: dict[str, str],
    ) -> str:
        if existing_session_id:
            return existing_session_id

        session = await self._adk_app.async_create_session(user_id=user_id, state=session_state)
        return session["id"]

    async def stream_events(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncIterator[dict]:
        async for event in self._adk_app.async_stream_query(
            user_id=user_id,
            session_id=session_id,
            message=message,
        ):
            if self._is_displayable_event(event):
                yield event
