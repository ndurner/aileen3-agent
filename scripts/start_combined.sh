#!/usr/bin/env bash
set -euo pipefail

API_SERVER_HOST="${API_SERVER_HOST:-0.0.0.0}"
API_SERVER_PORT="${API_SERVER_PORT:-8000}"
# The Gradio UI should bind to whatever port the platform injects via PORT.
PORT_VALUE="${PORT:-7860}"

# Make the ADK API server discoverable to the UI inside the container unless the
# caller already provided a specific URL.
: "${ADK_API_SERVER_URL:="http://127.0.0.1:${API_SERVER_PORT}"}"

export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-${PORT_VALUE}}"

echo "Starting ADK API server on ${API_SERVER_HOST}:${API_SERVER_PORT}"
python -m agent_system.run_api_server --host "${API_SERVER_HOST}" --port "${API_SERVER_PORT}" &
API_PID=$!

echo "Starting Gradio UI on port ${GRADIO_SERVER_PORT}"
python -m chat_ui.main &
UI_PID=$!

cleanup() {
  trap - TERM INT
  if kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
  if kill -0 "${UI_PID}" 2>/dev/null; then
    kill "${UI_PID}" 2>/dev/null || true
  fi
  wait "${API_PID}" 2>/dev/null || true
  wait "${UI_PID}" 2>/dev/null || true
}

trap cleanup TERM INT

wait -n "${API_PID}" "${UI_PID}"
EXIT_CODE=$?

cleanup
exit "${EXIT_CODE}"
