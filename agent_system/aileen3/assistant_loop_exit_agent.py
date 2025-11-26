from __future__ import annotations

import logging
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

DUMMY_SIGNATURE = "context_engineering_is_the_way_to_go"


class AssistantLoopExitAgent(BaseAgent):
    """Escalates the loop once the assistant has emitted a final response."""

    watched_agent_name: str
    response_state_key: str | None
    continue_prompt: str = "continue"

    def __init__(
        self, *, watched_agent_name: str, response_state_key: str | None = None
    ) -> None:
        super().__init__(
            name="assistant_loop_exit_agent",
            description=(
                "Checks whether the assistant agent produced a final response and "
                "escalates so the LoopAgent can stop iterating."
            ),
            watched_agent_name=watched_agent_name,
            response_state_key=response_state_key,
        )
        self.watched_agent_name = watched_agent_name
        self.response_state_key = response_state_key

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        latest = self._latest_event_from_assistant(ctx)
        if latest is None:
            logger.debug(
                "AssistantLoopExitAgent: no events found for %s; continuing loop.",
                self.watched_agent_name,
            )
            return

        if not latest.is_final_response():
            logger.debug(
                "AssistantLoopExitAgent: latest %s event %s is not final; continuing loop.",
                self.watched_agent_name,
                latest.id,
            )
            return

        has_state_response = bool(
            self.response_state_key
            and self._has_nonempty_response(ctx, self.response_state_key)
        )
        response_text = self._extract_text(latest)
        has_text_response = bool(
            response_text.strip() or self._has_text_response_since_last_user(ctx)
        )

        if not (has_state_response or has_text_response):
            patched = self._patch_missing_signatures(ctx)
            logger.debug(
                (
                    "AssistantLoopExitAgent: final %s event %s has no persisted state"
                    " and no textual response yet; patched %d missing signatures and"
                    " rerunning without escalating."
                ),
                self.watched_agent_name,
                latest.id,
                patched,
            )
            return

        logger.debug(
            "AssistantLoopExitAgent: detected final response from %s (event %s); escalating to stop loop.",
            self.watched_agent_name,
            latest.id,
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(escalate=True),
        )

    def _latest_event_from_assistant(
        self, ctx: InvocationContext
    ) -> Event | None:
        session = ctx.session
        if session is None or not session.events:
            logger.debug(
                "AssistantLoopExitAgent: session has no events; cannot inspect %s output.",
                self.watched_agent_name,
            )
            return None

        for event in reversed(session.events):
            if event.author == self.watched_agent_name:
                logger.debug(
                    "AssistantLoopExitAgent: inspecting event %s from %s.",
                    event.id,
                    self.watched_agent_name,
                )
                return event
        logger.debug(
            "AssistantLoopExitAgent: no events authored by %s found in session history.",
            self.watched_agent_name,
        )
        return None

    def _has_nonempty_response(
        self, ctx: InvocationContext, state_key: str
    ) -> bool:
        session = ctx.session
        if session is None or session.state is None:
            logger.debug(
                "AssistantLoopExitAgent: session/state missing; cannot read '%s'.",
                state_key,
            )
            return False

        value = session.state.get(state_key)
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return bool(value)

    def _extract_text(self, event: Event) -> str:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None)
        if not parts:
            return ""
        texts: list[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        return "".join(texts)

    def _patch_missing_signatures(self, ctx: InvocationContext) -> int:
        session = ctx.session
        if session is None or not session.events:
            return 0

        patched = 0
        for event in session.events:
            if event.author != self.watched_agent_name:
                continue
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                missing_call_sig = not getattr(
                    function_call, "thought_signature", None
                )
                missing_part_sig = not getattr(part, "thought_signature", None)
                if missing_call_sig or missing_part_sig:
                    function_call.thought_signature = DUMMY_SIGNATURE
                    part.thought_signature = DUMMY_SIGNATURE.encode("utf-8")
                    patched += 1
        return patched

    def _has_text_response_since_last_user(
        self, ctx: InvocationContext
    ) -> bool:
        session = ctx.session
        if session is None or not session.events:
            return False

        for event in reversed(session.events):
            if event.author == "user":
                break
            if event.author == self.watched_agent_name:
                if self._extract_text(event).strip():
                    return True
        return False
