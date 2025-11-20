from __future__ import annotations

from typing import Any, AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events import Event


class ConditionalPrepAgent(BaseAgent):
    """Custom agent that conditionally runs prep work for the assistant.

    - On the first invocation for a given session, it delegates to a
      `briefing_agent` (e.g. `briefing_refinement_agent`) to refine the
      structured briefing form and persist it via that agent's output_key.
    - On every invocation, it normalizes the latest user message via a
      `message_agent` and stores it in session state for downstream agents.
    """

    # Let Pydantic hold arbitrary agent instances.
    briefing_agent: LlmAgent
    message_agent: LlmAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, *, briefing_agent: LlmAgent, message_agent: LlmAgent) -> None:
        sub_agents_list = [briefing_agent, message_agent]
        super().__init__(
            name="conditional_prep_agent",
            description=(
                "Refines the initial briefing once per session and "
                "normalizes user messages on every turn."
            ),
            briefing_agent=briefing_agent,
            message_agent=message_agent,
            sub_agents=sub_agents_list,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Session state is the canonical, persisted store across turns.
        state = ctx.session.state
        state_dict: dict[str, Any] = dict(state) if state is not None else {}

        # 1) Run briefing refinement only once per session.
        has_refined_briefing = "briefing_refined" in state_dict
        if not has_refined_briefing:
            # Run the underlying LlmAgent once to refine the briefing.
            # We *must* forward its events so that ADK can apply the
            # output_key-based state update and, if desired, surface
            # the content in the transcript.
            async for event in self.briefing_agent.run_async(ctx):
                yield event

        # 2) On every invocation, normalize the latest user message.
        user_text: str | None = None
        user_content = getattr(ctx, "user_content", None)
        if user_content is not None and getattr(user_content, "parts", None):
            for part in user_content.parts:
                text = getattr(part, "text", None)
                if text:
                    user_text = text
                    break

        if user_text and state is not None:
            # Make the raw text available for the message fix agent
            # via its prompt template.
            state["latest_user_message"] = user_text
            # Forward message-fix events as well so its output_key
            # (`normalized_user_message`) is applied and visible.
            async for event in self.message_agent.run_async(ctx):
                yield event

        # If neither sub-agent produced events, this agent yields nothing.

