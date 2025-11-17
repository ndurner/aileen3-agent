from __future__ import annotations

from typing import List, Tuple

import gradio as gr
from gradio import ChatMessage

from chat_ui.config import load_config, AppConfig, BackendKind, ApiServerConfig, AgentEngineConfig
from chat_ui.backends import make_backend
from .event_mapping import decode_event

_base_config = load_config()


def _build_settings_accordion() -> tuple[
    gr.Dropdown, gr.Textbox, gr.Textbox, gr.Textbox, gr.Textbox, gr.Textbox, gr.Textbox, gr.State
]:
    """
    Returns controls for overriding backend settings at runtime and a state object for session id.
    """
    with gr.Accordion("Settings", open=False):
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
        default_user = gr.Textbox(
            label="Default User ID",
            value=_base_config.default_user_id,
        )
    session_state = gr.State(value=None)
    return backend_kind, api_url, api_app, project_id, location, ae_name, default_user, session_state


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
) -> Tuple[ChatMessage, str | None]:
    config = _override_config(backend_kind, api_url, api_app, project_id, location, ae_name, default_user)
    active_backend = make_backend(config)

    user_text = str(message.content if isinstance(message, ChatMessage) else message)

    user_id = config.default_user_id
    rets = []

    try:
        session_id = await active_backend.ensure_session(user_id=user_id, existing_session_id=session_id)
        async for event in active_backend.stream_events(user_id=user_id, session_id=session_id, message=user_text):
            ret = decode_event(event)
            if ret:
                rets.append(ret)
            yield rets, session_id
    except Exception as e:
        raise gr.Error(f"There was an error contacting the agent backend: {str(e)}")
    
    yield rets, session_id


def build_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Aileen3 Chat", render_markdown=True)
        (
            backend_kind,
            api_url,
            api_app,
            project_id,
            location,
            ae_name,
            default_user,
            session_state,
        ) = _build_settings_accordion()

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
            ],
            additional_outputs=[session_state],
        )

    return demo
