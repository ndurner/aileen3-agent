"""Vertex AI Memory Bank access tool for the assistant agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field
import vertexai
from vertexai import Client, types as vertex_types

from env_support import get_env_value

_STATE_KEY_PROJECT_ID = "vertex_project_id"
_STATE_KEY_LOCATION = "vertex_location"
_STATE_KEY_ENGINE = "vertex_agent_engine_name"
_STATE_KEY_API_KEY = "vertex_api_key"
_STATE_KEY_APP_NAME = "app_name"
_STATE_KEY_DEFAULT_USER = "default_user_id"

_ENV_KEY_PROJECTS = ("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT")
_ENV_KEY_LOCATIONS = ("VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION")
_ENV_KEY_API = ("VERTEX_API_KEY", "GOOGLE_API_KEY")
_ENV_KEY_ENGINE = ("AGENT_ENGINE_NAME", "AGENT_ENGINE_NAME")

_PAGE_SIZE = 100


@dataclass(slots=True)
class _VertexMemorySettings:
    project: str
    location: str
    agent_engine_name: str
    api_key: str | None
    scope: dict[str, str]


def _read_state_value(state: Any, key: str) -> str:
    if hasattr(state, "get"):
        value = state.get(key)
    else:
        value = None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # Stored as dict already; return JSON for downstream parsing.
        return json.dumps(value)
    return "" if value is None else str(value)


def _env_lookup(options: tuple[str, ...]) -> str:
    for env_key in options:
        value = get_env_value(env_key)
        if value:
            return value
    return ""


def _normalize_engine_name(raw_name: str, project: str, location: str) -> str:
    raw_name = raw_name.strip()
    if not raw_name:
        return ""
    if raw_name.startswith("projects/"):
        return raw_name
    if not project or not location:
        return raw_name
    return f"projects/{project}/locations/{location}/reasoningEngines/{raw_name}"

def _parse_engine_resource(name: str) -> tuple[str | None, str | None]:
    """Return (project, location) when embedded in projects/.../locations/..."""
    if not name or "projects/" not in name or "/locations/" not in name:
        return None, None
    try:
        _, rest = name.split("projects/", 1)
        project, remainder = rest.split("/locations/", 1)
        location = remainder.split("/", 1)[0]
        return project or None, location or None
    except ValueError:
        return None, None

def _extract_topics(memory: Any) -> list[str]:
    topics: list[str] = []
    for topic in getattr(memory, "topics", []) or []:
        label = getattr(topic, "custom_memory_topic_label", None)
        managed = getattr(topic, "managed_memory_topic", None)
        if label:
            topics.append(str(label))
        elif managed:
            topics.append(str(managed))
    return topics


def _resolve_settings(tool_context: ToolContext) -> _VertexMemorySettings:
    project = _read_state_value(tool_context.state, _STATE_KEY_PROJECT_ID) or _env_lookup(
        _ENV_KEY_PROJECTS
    )
    location = _read_state_value(tool_context.state, _STATE_KEY_LOCATION) or _env_lookup(
        _ENV_KEY_LOCATIONS
    )
    agent_engine_name = _read_state_value(tool_context.state, _STATE_KEY_ENGINE) or _env_lookup(
        _ENV_KEY_ENGINE
    )
    api_key = _read_state_value(tool_context.state, _STATE_KEY_API_KEY) or _env_lookup(
        _ENV_KEY_API
    )
    app_name = _read_state_value(tool_context.state, _STATE_KEY_APP_NAME)
    if not app_name:
        app_name = getattr(getattr(tool_context, "session", None), "app_name", "")
    default_user = _read_state_value(tool_context.state, _STATE_KEY_DEFAULT_USER) or tool_context.user_id
    scope = {
        "app_name": app_name or "aileen3",
        "user_id": default_user or tool_context.user_id,
    }

    if not project or not location:
        parsed_project, parsed_location = _parse_engine_resource(agent_engine_name)
        project = project or parsed_project
        location = location or parsed_location

    if not project:
        raise ValueError("Vertex project id is missing. Provide it in the UI or env.")
    if not location:
        raise ValueError("Vertex location is missing. Provide it in the UI or env.")
    if not agent_engine_name:
        raise ValueError("Agent Engine name is missing. Provide it in the UI.")

    normalized_name = _normalize_engine_name(agent_engine_name, project, location)

    return _VertexMemorySettings(
        project=project,
        location=location,
        agent_engine_name=normalized_name,
        api_key=api_key or None,
        scope=scope,
    )


def _retrieve_memories(
    *,
    settings: _VertexMemorySettings,
    query: str,
) -> str:
    if settings.api_key:
        vertexai.init(project=settings.project, location=settings.location)
        client = Client(api_key=settings.api_key)
    else:
        client = Client(project=settings.project, location=settings.location)

    simple_params = vertex_types.RetrieveMemoriesRequestSimpleRetrievalParams(
        page_size=_PAGE_SIZE,
    )

    pager = client.agent_engines.memories.retrieve(
        name=settings.agent_engine_name,
        scope=settings.scope,
        simple_retrieval_params=simple_params,
    )

    facts: str = ""
    for item in pager:
        memory = getattr(item, "memory", None)
        if not memory or not memory.fact:
            continue

        # return flat facts for now. Future improvements could include create_time, update_time, `extract_topics` or (similarity) distance to query
        facts = f"{facts}<fact>{memory.fact}</fact>"

    next_page_token = None
    pager_config = getattr(pager, "config", None)
    if isinstance(pager_config, dict):
        next_page_token = pager_config.get("page_token") or None

    if not facts:
        return '<memory isError="isError">No factual memories were found</memory>'
    else:
        return f'<memory>{facts}</memory>'

async def get_factual_memory(
    query: str | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """Retrieve factual memories from the AI Memory Bank for this app and user.

    Use this tool when:
      - you are preparing to analyze a new talk and want a baseline of what is already known
        in this workspace about the domain, models, products or policies, or
      - the user asks whether something is really new or has been said before.

    The tool returns XML of the form:
        <memory>
          <fact>...</fact>
          <fact>...</fact>
          ...
        </memory>

    How to use the result:
      - Treat these facts as priors and baseline knowledge for later analysis.
      - You may quote or paraphrase relevant facts when answering the user.
      - Absence of a fact does not prove that something is new, it only means
        nothing has been stored in the memory bank.

    Args:
        query: Optional filter text (currently unused and may be left empty).
    """

    if tool_context is None:
        raise ValueError("Tool context is required when calling get_factual_memory.")
    trimmed_query = (query or "").strip()
    settings = _resolve_settings(tool_context)

    try:
        ret = await asyncio.to_thread(
            _retrieve_memories,
            settings=settings,
            query=trimmed_query,
        )
        return ret
    except Exception as exc:  # pragma: no cover - surfaced to the LLM/tool log
        message = str(exc)
        # If the Vertex endpoint/region is misconfigured (common during setup),
        # fail soft and report no memories instead of crashing the whole flow.
        if "location ID doesn't match the endpoint" in message or "INVALID_ARGUMENT" in message:
            return '<memory isError="isError">Vertex AI Memory is not available or misconfigured; continuing without long-term memory.</memory>'
        raise RuntimeError(f"Vertex AI Memory lookup failed: {exc}") from exc


get_factual_memory_tool = FunctionTool(get_factual_memory)
