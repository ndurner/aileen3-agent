# aileen3-agent

## Setup
1. Obtain a `VERTEX_AI_KEY` from the [Vertex AI Studio](https://console.cloud.google.com/vertex-ai/studio/settings/api-keys)
2. Add `VERTEX_AI_KEY` and `VERTEX_PROJECT_ID` to .env
3. Create memory bank
  1. run `python tools/memory_bank_cli.py create-bank --display-name "Aileen Memory"`
  2. add the "resource name" returned to .env: `AGENT_ENGINE_NAME=projects/...`
4. (Optional) Plant memories: `python tools/memory_bank_cli.py add-facts --user-id neal --fact "The user's name is Neal"`

## Running
1. Start agent server: `python -m agent_system.run_api_server --log_level debug`
    - this wraps `adk api_server` and adds explicit logging of agent invocations. Accepts the same command line parameters as `api_server`.
    - the launcher automatically loads environment variables from `.env` so manual shells behave the same way as the VS Code launch config.
2. Start Gradio web chat: `python -m chat_ui.main`

## Docker

The provided `Dockerfile` runs both the ADK API server and the Gradio UI (via `scripts/start_combined.sh`), making it deployable on Google Cloud Run and Hugging Face Spaces.

To try the container locally:

```
docker build -t aileen3-agent .
docker run --env-file .env -p 7860:7860 -p 8000:8000 aileen3-agent
```

Supplying `.env` (or equivalent `-e KEY=value` flags) is required so the agent can authenticate with Vertex / ADK; the launcher inside the container reads the same variables you provide at runtime. Port `7860` exposes the Gradio UI, while `8000` publishes the embedded API server for debugging.

## Memory Bank CLI

A helper script in `tools/memory_bank_cli.py` automates Vertex AI Memory Bank
management. Configure `.env` with `VERTEX_PROJECT_ID` (always) and, when using
ADC, `VERTEX_LOCATION`. If
`VERTEX_API_KEY` (or `GOOGLE_API_KEY`) is set, the CLI uses express-mode API
keys (always `us-central1`, so `VERTEX_LOCATION` is optional). Otherwise it
falls back to Application Default
Credentials (run `gcloud auth application-default login` or point `GOOGLE_APPLICATION_CREDENTIALS`
to a service account key) so you can target other regions. Commands that operate
on an existing reasoning engine also expect `AGENT_ENGINE_NAME` (or the
`--engine` flag). Common invocations:

- `python tools/memory_bank_cli.py create-bank --display-name "Aileen Memory"`
  to provision a brand-new Vertex Agent Engine that already contains the
  curated topics and few-shot examples. Copy the printed resource name into
  `AGENT_ENGINE_NAME` afterward. (When authenticating with a Vertex API key,
  this always provisions in `us-central1`; use ADC if you need another region.)
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
