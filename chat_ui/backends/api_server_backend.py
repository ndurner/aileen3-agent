from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from .base import AgentBackend
from chat_ui.config import ApiServerConfig


@dataclass
class ApiServerBackend(AgentBackend):
    config: ApiServerConfig

    async def ensure_session(
        self,
        user_id: str,
        existing_session_id: str | None,
        session_state: dict[str, str],
    ) -> str:
        if existing_session_id:
            return existing_session_id

        base = self.config.base_url.rstrip("/")
        url = f"{base}/apps/{self.config.app_name}/users/{user_id}/sessions"

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(url, json={"state": session_state})
            response.raise_for_status()
            data = response.json()

        # Session responses from the ADK API server include the id field.
        session_id = data.get("id") or data.get("session_id")
        if not session_id:
            raise RuntimeError("API server did not return a session id")

        return session_id

    async def stream_events(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncIterator[dict]:
        payload = {
            "app_name": self.config.app_name,
            "user_id": user_id,
            "session_id": session_id,
            "new_message": {
                "role": "user",
                "parts": [{"text": message}],
            },
            "streaming": True
        }

        url = f"{self.config.base_url.rstrip('/')}/run_sse"

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():

                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    
                    # Gate events so only the streaming assistant partials reach the UI.
                    if self._is_displayable_event(event):
                        yield event

    async def delete_session(
        self,
        user_id: str,
        session_id: str,
    ) -> None:
        if not session_id:
            return

        base = self.config.base_url.rstrip("/")
        url = f"{base}/apps/{self.config.app_name}/users/{user_id}/sessions/{session_id}"

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.delete(url)
            # Treat missing sessions as already-deleted; raise for other errors.
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()
