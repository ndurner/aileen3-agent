# Repository Guidelines

## Project Structure & Modules
- `agent_system/` minimal wrapper registering the `root_agent` via Google ADK (`Agent` from `google.adk.agents.llm_agent`).
- `chat_ui/` Gradio-based front end: `main.py` entrypoint, `ui/` layout + event routing, `backends/` adapters for API server vs Vertex Agent Engine, `config.py` env-driven settings.
- `requirements.txt` lists runtime deps (gradio, httpx, vertexai, google-adk). No dedicated tests folder yet.

## Coding Style & Naming
- Python 3 style, PEP 8, 4-space indent; prefer type hints (used across UI/backends).
- Keep modules small: UI work in `chat_ui/ui/*`, transport in `chat_ui/backends/*`, config in `chat_ui/config.py`.
- Use descriptive snake_case for variables and functions; keep Gradio component builders prefixed with `_build_*`, backend factories in `make_backend`.
- No repo formatter pinned; if you use `ruff`/`black`, run locally and avoid unrelated churn.

## Terminology
- "ADK" in this context refers to the Google Agent Development Kit

## Environment
- activate the .venv venv