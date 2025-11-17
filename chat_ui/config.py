import os
from dataclasses import dataclass
from enum import Enum


class BackendKind(str, Enum):
    API_SERVER = "api_server"
    AGENT_ENGINE = "aileen3"


@dataclass
class ApiServerConfig:
    base_url: str
    app_name: str


@dataclass
class AgentEngineConfig:
    project_id: str
    location: str
    agent_engine_name: str


@dataclass
class AppConfig:
    backend_kind: BackendKind
    api_server: ApiServerConfig | None = None
    agent_engine: AgentEngineConfig | None = None
    default_user_id: str = "demo-user"


def load_config() -> AppConfig:
    kind = os.getenv("ADK_BACKEND_KIND", "api_server")
    backend = BackendKind(kind)

    if backend == BackendKind.API_SERVER:
        return AppConfig(
            backend_kind=backend,
            api_server=ApiServerConfig(
                base_url=os.getenv("ADK_API_SERVER_URL", "http://localhost:8000"),
                app_name="aileen3",
            ),
        )

    if backend == BackendKind.AGENT_ENGINE:
        return AppConfig(
            backend_kind=backend,
            agent_engine=AgentEngineConfig(
                project_id=os.environ["VERTEX_PROJECT_ID"],
                location=os.environ["VERTEX_LOCATION"],
                agent_engine_name=os.environ["AGENT_ENGINE_NAME"],
            ),
        )

    raise ValueError(f"Unsupported backend kind: {backend}")
