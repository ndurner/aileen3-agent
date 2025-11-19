# aileen3-agent

## Running
1. Start agent server: `python -m agent_system.run_api_server --log_level debug`
    - this wraps `adk api_server` and adds explicit logging of agent invocations. Accepts the same command line parameters as `api_server`.
2. Start Gradio web chat: `python -m chat_ui.main`