#!/usr/bin/env bash

set -euo pipefail

if ! git status >/dev/null 2>&1; then
  echo "error: this script must be run from the repository root" >&2
  exit 1
fi

git config core.hooksPath .githooks
echo "Configured git to use .githooks/pre-commit (run this once per clone)."
