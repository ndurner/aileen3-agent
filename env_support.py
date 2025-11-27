"""
Utilities for loading configuration from environment variables with optional .env fallback.

The helpers in this module always prefer existing os.environ values (so Docker or shell
settings win) while providing a shared parser for the repository-level .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"


def _strip_wrapper(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _normalize_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.lower().startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key or key.startswith("#"):
        return None
    return key, _strip_wrapper(value)


@lru_cache(maxsize=8)
def _parse_env_file(env_path: str) -> Dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        normalized = _normalize_line(raw_line)
        if not normalized:
            continue
        key, value = normalized
        values[key] = value
    return values


def get_env_value(key: str, *, env_path: Path | None = None) -> str | None:
    """Return env var from os.environ or .env (without mutating os.environ)."""
    value = os.environ.get(key)
    if value and value.strip():
        return value.strip()

    parsed = _parse_env_file(str(env_path or _DEFAULT_ENV_PATH))
    value = parsed.get(key)
    if value and value.strip():
        return value.strip()
    return None


def ensure_env_loaded(*, env_path: Path | None = None, overwrite: bool = False) -> dict[str, str]:
    """
    Populate os.environ with key/value pairs from .env.

    Returns the values that were read so callers can log/inspect them.
    """
    parsed = _parse_env_file(str(env_path or _DEFAULT_ENV_PATH))
    for key, value in parsed.items():
        if overwrite or key not in os.environ:
            os.environ[key] = value
    return parsed


def get_multiple_env_values(keys: tuple[str, ...], *, env_path: Path | None = None) -> dict[str, str]:
    """Return a mapping of the first available key in keys -> value."""
    resolved: dict[str, str] = {}
    for key in keys:
        value = get_env_value(key, env_path=env_path)
        if value:
            resolved[key] = value
    return resolved
