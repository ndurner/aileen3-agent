from chat_ui.config import AppConfig, BackendKind

from .agent_engine_backend import AgentEngineBackend
from .api_server_backend import ApiServerBackend
from .base import AgentBackend


def make_backend(config: AppConfig) -> AgentBackend:
    if config.backend_kind == BackendKind.API_SERVER:
        assert config.api_server
        return ApiServerBackend(config.api_server)

    if config.backend_kind == BackendKind.AGENT_ENGINE:
        assert config.agent_engine
        return AgentEngineBackend(config.agent_engine)

    raise ValueError(f"Unsupported backend kind: {config.backend_kind}")
