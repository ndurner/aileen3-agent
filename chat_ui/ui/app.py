from __future__ import annotations

from typing import List, Tuple, Any

import gradio as gr
from gradio import ChatMessage

from chat_ui.config import load_config, AppConfig, BackendKind, ApiServerConfig, AgentEngineConfig
from chat_ui.backends import make_backend
from .event_mapping import (
    init_tool_display_state,
    process_event as process_tool_event,
    get_ordered_tool_messages,
)

_base_config = load_config()


def _build_settings_accordion() -> tuple[
    gr.Dropdown,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.Textbox,
    gr.State,
]:
    """
    Returns controls for overriding backend settings at runtime and a state object for session id.
    """
    with gr.Accordion("Technical settings", open=False):
        backend_kind = gr.Dropdown(
            label="Backend",
            choices=[BackendKind.API_SERVER.value, BackendKind.AGENT_ENGINE.value],
            value=_base_config.backend_kind.value,
        )
        api_url = gr.Textbox(
            label="API Server URL",
            value=_base_config.api_server.base_url if _base_config.api_server else "",
            placeholder="http://localhost:8000",
        )
        api_app = gr.Textbox(
            label="API Server App Name",
            value=_base_config.api_server.app_name if _base_config.api_server else "",
        )
        project_id = gr.Textbox(
            label="Vertex Project ID",
            value=_base_config.agent_engine.project_id if _base_config.agent_engine else "",
        )
        location = gr.Textbox(
            label="Vertex Location",
            value=_base_config.agent_engine.location if _base_config.agent_engine else "",
            placeholder="us-central1",
        )
        ae_name = gr.Textbox(
            label="Agent Engine Resource Name",
            value=_base_config.agent_engine.agent_engine_name if _base_config.agent_engine else "",
            placeholder="projects/../reasoningEngines/..",
        )
        vertex_api_key = gr.Textbox(
            label="Vertex API Key (optional)",
            type="password",
            placeholder="Needed only when ADC/service account auth is unavailable",
        )
        default_user = gr.Textbox(
            label="Default User ID",
            value=_base_config.default_user_id,
        )

    with gr.Accordion("Briefing", open=True):
        media_url = gr.Textbox(
            label="Media URL",
            placeholder="https://youtube.com/...",
        )
        context = gr.Textbox(
            label="Context (scene setting)",
            placeholder="People, venue, goal, spelled names or acronyms you already provided",
            lines=3,
        )
        expectations = gr.Textbox(
            label="Expectations (anticipated insights)",
            placeholder="Facts or takeaways you expect—only what you've already stated",
            lines=3,
        )
        prior_knowledge = gr.Textbox(
            label="Prior Knowledge (what you already know)",
            placeholder="Acronyms, previous meetings, background you mentioned",
            lines=3,
        )
        questions = gr.Textbox(
            label="Questions to Answer",
            placeholder="Specific questions you've supplied for this chat",
            lines=3,
        )

    session_state = gr.State(value=None)
    return (
        backend_kind,
        api_url,
        api_app,
        project_id,
        location,
        ae_name,
        vertex_api_key,
        default_user,
        media_url,
        context,
        expectations,
        prior_knowledge,
        questions,
        session_state,
    )


def _override_config(
    backend_kind: str,
    api_url: str,
    api_app: str,
    project_id: str,
    location: str,
    ae_name: str,
    default_user: str,
) -> AppConfig:
    bk = BackendKind(backend_kind)
    if bk == BackendKind.API_SERVER:
        return AppConfig(
            backend_kind=bk,
            api_server=ApiServerConfig(base_url=api_url or "http://localhost:8000", app_name=api_app),
            default_user_id=default_user or _base_config.default_user_id,
        )
    return AppConfig(
        backend_kind=bk,
        agent_engine=AgentEngineConfig(
            project_id=project_id,
            location=location or (_base_config.agent_engine.location if _base_config.agent_engine else ""),
            agent_engine_name=ae_name,
        ),
        default_user_id=default_user or _base_config.default_user_id,
    )


async def chat_fn(
    message: str | ChatMessage,
    history: List[ChatMessage],
    session_id: str | None,
    backend_kind: str,
    api_url: str,
    api_app: str,
    project_id: str,
    location: str,
    ae_name: str,
    default_user: str,
    vertex_api_key: str,
    media_url: str,
    context: str,
    expectations: str,
    prior_knowledge: str,
    questions: str,
) -> Tuple[Any, str | None]:
    config = _override_config(backend_kind, api_url, api_app, project_id, location, ae_name, default_user)
    active_backend = make_backend(config)

    user_text = str(message.content if isinstance(message, ChatMessage) else message)

    user_id = config.default_user_id

    # Maintain per-turn state for tool thought messages and streaming text.
    tool_state = init_tool_display_state()
    answer_text_chunks: list[str] = []

    session_state_payload = {
        "user_media_url": media_url or "",
        "user_context": context or "",
        "user_expectations": expectations or "",
        "user_prior_knowledge": prior_knowledge or "",
        "user_questions": questions or "",
        "vertex_project_id": project_id or "",
        "vertex_location": location or "",
        "vertex_agent_engine_name": ae_name or "",
        "vertex_api_key": vertex_api_key or "",
        "app_name": api_app or "",
        "default_user_id": default_user or "",
    }

    try:
        did_yield = False
        session_id = await active_backend.ensure_session(
            user_id=user_id,
            existing_session_id=session_id,
            session_state=session_state_payload,
        )
        async for event in active_backend.stream_events(
            user_id=user_id,
            session_id=session_id,
            message=user_text,
        ):
            text_delta = process_tool_event(event, tool_state)
            if isinstance(text_delta, str) and text_delta:
                # Stream assistant text as it arrives.
                answer_text_chunks.append(text_delta)

            # Build the current assistant-side output for this turn:
            # - one Gradio tool message per tool run (with emoji + status)
            # - plus the natural-language answer text so far (if any).
            tool_messages = get_ordered_tool_messages(tool_state)
            answer_text = "".join(answer_text_chunks)

            outputs: list[Any] = []
            outputs.extend(tool_messages)
            if answer_text:
                outputs.append(answer_text)

            if not outputs:
                # Nothing user-visible yet for this event.
                continue

            if len(outputs) == 1:
                yield outputs[0], session_id
            else:
                yield outputs, session_id
            did_yield = True
    except Exception as e:
        # Show the underlying error message so that backend-originated
        # messages (such as token limit errors) surface cleanly in the UI.
        raise gr.Error(str(e))

    # If the backend produced no visible events at all for this turn,
    # return an empty assistant reply to avoid StopAsyncIteration errors.
    if not did_yield:
        yield "", session_id


async def clear_session_fn(
    backend_kind: str,
    api_url: str,
    api_app: str,
    project_id: str,
    location: str,
    ae_name: str,
    default_user: str,
    session_id: str | None,
) -> str | None:
    """
    Clear the backend session (if any) and reset the stored session id.
    """
    if session_id:
        config = _override_config(backend_kind, api_url, api_app, project_id, location, ae_name, default_user)
        active_backend = make_backend(config)

        user_id = config.default_user_id
        try:
            await active_backend.delete_session(user_id=user_id, session_id=session_id)
        except Exception:
            # If deletion fails, we still forget the session id locally so the next
            # turn will use a fresh session.
            pass

    # Forget the session id so the next turn starts fresh.
    return None


def build_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Aileen 3 Agent")

        # Remove undo/redo buttons from chatbot
        gr.HTML("""
                <style>
                    button[aria-label*="undo" i],
                    button[title*="undo" i],
                    button[aria-label*="retry" i],
                    button[title*="retry" i] {
                        display: none !important;
                    }
                </style>
                """)
        chatbot = gr.Chatbot(label="Aileen3 Chat", render_markdown=True)
        (
            backend_kind,
            api_url,
            api_app,
            project_id,
            location,
            ae_name,
            vertex_api_key,
            default_user,
            media_url,
            context,
            expectations,
            prior_knowledge,
            questions,
            session_state,
        ) = _build_settings_accordion()

        # When the built-in trash icon is clicked, delete/reset the backend session id.
        chatbot.clear(
            fn=clear_session_fn,
            inputs=[
                backend_kind,
                api_url,
                api_app,
                project_id,
                location,
                ae_name,
                default_user,
                session_state,
            ],
            outputs=[session_state],
        )

        gr.ChatInterface(
            fn=chat_fn,
            chatbot=chatbot,
            additional_inputs=[
                session_state,
                backend_kind,
                api_url,
                api_app,
                project_id,
                location,
                ae_name,
                default_user,
                vertex_api_key,
                media_url,
                context,
                expectations,
                prior_knowledge,
                questions,
            ],
            additional_outputs=[session_state],
        )

    return demo
