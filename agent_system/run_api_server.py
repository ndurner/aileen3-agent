"""
Convenience launcher for the ADK API server that always enables the logging plugin.

Usage:
    python -m agent_system.run_api_server [api_server options...]

This calls the ADK CLI entry point directly (no subprocess) and appends the
logging plugin + agents directory automatically so the flag cannot be missed.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # Python 3.12+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

PLUGIN_SPEC = "agent_system.aileen3.logging_plugin.LoggingPlugin"
AGENTS_DIR = Path(__file__).parent


def _load_adk_entrypoint():
    """Return the callable behind the `adk` console script."""
    entry_points = importlib_metadata.entry_points()
    if hasattr(entry_points, "select"):  # python >=3.10 API
        candidates = entry_points.select(group="console_scripts", name="adk")
    else:  # pragma: no cover
        candidates = [
            ep for ep in entry_points if ep.group == "console_scripts" and ep.name == "adk"
        ]
    try:
        entry_point = next(iter(candidates))
    except StopIteration as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not find the 'adk' console entry point. Is google-adk installed?"
        ) from exc
    return entry_point.load()


def _split_plugin_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _consume_extra_plugins(args: list[str]) -> tuple[list[str], list[str]]:
    cleaned: list[str] = []
    collected: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--extra_plugins="):
            collected.extend(_split_plugin_values(arg.split("=", 1)[1]))
        elif arg == "--extra_plugins":
            if i + 1 < len(args):
                collected.extend(_split_plugin_values(args[i + 1]))
                i += 1
        else:
            cleaned.append(arg)
        i += 1
    return cleaned, collected


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_plugin_name(value: str) -> str:
    module, sep, attr = value.partition(":")
    if sep and attr:
        return f"{module}.{attr}"
    return value


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    remaining_args, user_plugins = _consume_extra_plugins(user_args)
    merged_plugins = _dedupe_preserve_order([*user_plugins, PLUGIN_SPEC])
    normalized_plugins = [_normalize_plugin_name(plugin) for plugin in merged_plugins]

    args = [
        "api_server",
        str(AGENTS_DIR),
        *remaining_args,
    ]
    for plugin in normalized_plugins:
        args.extend(["--extra_plugins", plugin])

    adk_cli = _load_adk_entrypoint()

    try:
        if hasattr(adk_cli, "main"):
            adk_cli.main(args=args, prog_name="adk", standalone_mode=False)  # type: ignore[attr-defined]
        else:  # pragma: no cover
            adk_cli(args)  # type: ignore[call-arg]
        return 0
    except SystemExit as exc:  # click uses SystemExit for control flow
        return int(exc.code)


if __name__ == "__main__":
    sys.exit(main())
