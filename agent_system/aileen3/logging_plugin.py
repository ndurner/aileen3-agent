"""
Optional ADK plugin that logs key lifecycle events for the Aileen agent.

The plugin is referenced via the ADK CLI flag:

    adk api_server agent_system --extra_plugins agent_system.aileen3.logging_plugin.LoggingPlugin

It keeps the logging surface minimal so it works even in stripped-down
environments where google-adk may evolve. All attribute access is guarded
to avoid breaking if the callback payloads change shape.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin


def _configure_default_logging(logger: logging.Logger) -> None:
    """Ensure the logger has a sane handler when run standalone."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    root_level = logging.getLogger().level
    logger.setLevel(root_level if root_level != logging.NOTSET else logging.INFO)


class LoggingPlugin(BasePlugin):
    """Lightweight plugin that logs agent + LLM calls."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        name: str | None = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(name=name or "aileen_logging", **base_kwargs)
        self.logger = logger or logging.getLogger("aileen3.agent")
        _configure_default_logging(self.logger)

    async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        agent_name = getattr(agent, "name", agent.__class__.__name__)
        session_id = callback_context.session.id
        state = getattr(callback_context, "state", None)
        state_dict = state.to_dict() if state else None
        self.logger.info(
            "Starting agent '%s' session=%s state=%s",
            agent_name,
            session_id,
            state_dict,
        )

    async def after_agent_callback(
        self, *, agent: Any, callback_context: Any, agent_output: Any | None = None
    ) -> None:
        agent_name = getattr(agent, "name", agent.__class__.__name__)
        session_id = callback_context.session.id
        state = getattr(callback_context, "state", None)
        state_dict = state.to_dict() if state else None
        self.logger.info(
            "Finished agent '%s' session=%s state=%s",
            agent_name,
            session_id,
            state_dict,
        )

    async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
        session_id = callback_context.session.id
        self.logger.info(
            "LLM request session=%s", session_id
        )

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> None:
        session_id = callback_context.session.id
        self.logger.info(
            "LLM response session=%s", session_id
        )
