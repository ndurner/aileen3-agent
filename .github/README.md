# aileen3-agent

## Running
1. Start agent server: `python -m agent_system.run_api_server --log_level debug`
    - this wraps `adk api_server` and adds explicit logging of agent invocations. Accepts the same command line parameters as `api_server`.
2. Start Gradio web chat: `python -m chat_ui.main`

## Memory Bank CLI

A helper script in `tools/memory_bank_cli.py` automates Vertex AI Memory Bank
management. Configure `.env` with at least `VERTEX_API_KEY`,
`VERTEX_PROJECT_ID`, and `VERTEX_LOCATION`. Commands that operate on an
existing reasoning engine also expect `AGENT_ENGINE_NAME` (or the `--engine`
flag). Common invocations:

- `python tools/memory_bank_cli.py create-bank --display-name "Aileen Memory"`
  to provision a brand-new Vertex Agent Engine that already contains the
  curated topics and few-shot examples. Copy the printed resource name into
  `AGENT_ENGINE_NAME` afterward.
- `python tools/memory_bank_cli.py configure-bank` to apply the curated topics
  and few-shot examples to the reasoning engine.
- `python tools/memory_bank_cli.py delete-bank` to clear the memory bank
  configuration.
- `python tools/memory_bank_cli.py add-facts --fact "..." --app-name aileen3 --user-id demo`
  to store pre-extracted facts for a scope.
- `python tools/memory_bank_cli.py generate --text-file panel.txt --app-name aileen3 --user-id demo`
  to let Vertex auto-extract memories from raw transcripts.

All commands accept `--scope key=value` for additional scope labels and reuse
the same {app_name, user_id} scheme expected by the assistant’s tool. Similarity
search remains disabled at runtime; the CLI focuses on provisioning and
populating the memory bank.
