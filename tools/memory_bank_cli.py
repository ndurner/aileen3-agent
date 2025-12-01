#!/usr/bin/env python3
"""Utility CLI for managing the Vertex AI Memory Bank used by Aileen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import vertexai

# Ensure project root (where env_support.py lives) is importable when this file is
# invoked as `python tools/memory_bank_cli.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_support import ensure_env_loaded


DOMAIN_CLAIMS_TOPIC = {
    "custom_memory_topic": {
        "label": "domain_claims",
        "description": (
            "Facts and claims about how the world works in this domain, "
            "such as typical challenges, current practices, architectural "
            "patterns, and recurring failure modes. Ignore generic smalltalk."
        ),
    }
}

CAPABILITY_TOPIC = {
    "custom_memory_topic": {
        "label": "capability_and_feature_claims",
        "description": (
            "Statements about what a product, model, system or process can do, "
            "cannot do, or is newly able to do. Include descriptions of inputs, "
            "outputs, and important constraints."
        ),
    }
}

METRICS_TOPIC = {
    "custom_memory_topic": {
        "label": "metrics_and_kpis",
        "description": (
            "Any quantitative metrics, targets, KPIs, benchmark results, or "
            "percentages mentioned in the content."
        ),
    }
}

RISKS_TOPIC = {
    "custom_memory_topic": {
        "label": "risks_obstacles_and_promises",
        "description": (
            "Risks, obstacles, dependencies, timelines, and explicit promises or "
            "commitments made by speakers or authors."
        ),
    }
}

MEMORY_TOPICS = [
    DOMAIN_CLAIMS_TOPIC,
    CAPABILITY_TOPIC,
    METRICS_TOPIC,
    RISKS_TOPIC,
]


EXAMPLE_PANEL = {
    "conversationSource": {
        "events": [
            {
                "content": {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Moderator: Let's talk about digital public services.\n\n"
                                "Speaker A: Most municipalities already have an online front door, "
                                "but fewer than 20 percent manage end-to-end digital processing. "
                                "The rest still print and retype data.\n\n"
                                "Speaker B: The biggest blocker is fragmented registers and "
                                "inconsistent data quality, not the lack of frontends.\n\n"
                                "Moderator: Thank you. Let's move to audience questions."
                            )
                        }
                    ],
                }
            }
        ]
    },
    "generatedMemories": [
        {
            "fact": (
                "[KPI]  Speaker A on unknown date at an unnamed panel: "
                "Fewer than 20 percent of municipalities process cases end-to-end "
                "digitally; most still retype printed data."
            )
        },
        {
            "fact": (
                "[RISK]  Speaker B on unknown date at an unnamed panel: "
                "Fragmented registers and poor data quality are seen as the main "
                "blocker for digital public services, more than frontends."
            )
        },
    ],
}

EXAMPLE_MODEL_LAUNCH = {
    "conversationSource": {
        "events": [
            {
                "content": {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Presenter: Today we are launching PixelGen 3, our new text-and-image-"
                                "to-image model. It can take an existing UI screenshot and translate "
                                "it into a different visual style while preserving layout and text. "
                                "For example, you can keep the same form but restyle it from "
                                "government portal to consumer banking.\n\n"
                                "This was not supported in PixelGen 2, which only accepted text prompts."
                            )
                        }
                    ],
                }
            }
        ]
    },
    "generatedMemories": [
        {
            "fact": (
                "[CAPABILITY]  PixelGen 3 launch keynote: "
                "The model can perform image-to-image translation, preserving layout "
                "and text while changing visual style."
            )
        },
        {
            "fact": (
                "[DOMAIN]  PixelGen 3 launch keynote: "
                "PixelGen 2 only accepted text prompts; image-to-image workflows were "
                "not supported previously."
            )
        },
    ],
}

EXAMPLE_LOW_SIGNAL = {
    "conversationSource": {
        "events": [
            {
                "content": {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Speaker: Digital transformation is important. "
                                "We must collaborate, think about users, and break silos. "
                                "We need to move faster and embrace innovation."
                            )
                        }
                    ],
                }
            }
        ]
    },
    "generatedMemories": [],
}

GENERATE_MEMORIES_EXAMPLES = [
    EXAMPLE_PANEL,
    EXAMPLE_MODEL_LAUNCH,
    EXAMPLE_LOW_SIGNAL,
]


def ensure_setting(value: str | None, *, flag: str, env_name: str) -> str:
    if value:
        return value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    raise SystemExit(f"Missing required setting for {flag}; set {flag} or {env_name} in .env")


def build_client(args: argparse.Namespace) -> tuple[vertexai.Client, str, str]:
    ensure_env_loaded(env_path=Path(args.env_file))
    project = ensure_setting(args.project, flag="--project", env_name="VERTEX_PROJECT_ID")
    requested_location = args.location or os.environ.get("VERTEX_LOCATION")
    api_key = os.environ.get("VERTEX_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        # With API keys, the allowed region is tied to the key.
        # Require an explicit location via flag or env so we don't guess.
        if not requested_location:
            raise SystemExit(
                "Missing required setting for --location; set --location or "
                "VERTEX_LOCATION in .env when using an API key."
            )
        vertexai.init(project=project, location=requested_location)
        return vertexai.Client(api_key=api_key), project, requested_location

    # ADC path: rely on user-provided project/location and default credentials.
    if not requested_location:
        raise SystemExit("Missing required setting for --location; set --location or VERTEX_LOCATION in .env")
    vertexai.init(project=project, location=requested_location)
    return vertexai.Client(project=project, location=requested_location), project, requested_location


def build_memory_bank_config(project: str, location: str) -> dict[str, Any]:
    return {
        "customization_configs": [
            {
                "memory_topics": MEMORY_TOPICS,
                "generate_memories_examples": GENERATE_MEMORIES_EXAMPLES,
            }
        ],
    }


def parse_labels(raw: list[str] | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise SystemExit(f"Invalid label '{item}'. Expected key=value format.")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if not key or not value:
            raise SystemExit(f"Invalid label '{item}'.")
        labels[key] = value
    return labels


def create_memory_bank(args: argparse.Namespace) -> None:
    client, project, location = build_client(args)
    config: dict[str, Any] = {
        "display_name": args.display_name or "Aileen Memory Bank",
        "description": args.description or "Vertex Agent Engine for Aileen factual memory storage",
        "context_spec": {"memory_bank_config": build_memory_bank_config(project, location)},
    }
    labels = parse_labels(args.label)
    if labels:
        config["labels"] = labels

    engine = client.agent_engines.create(config=config)
    api_resource = getattr(engine, "api_resource", None)
    name = getattr(api_resource, "name", None)
    print("Created Vertex Agent Engine with memory bank.")
    print(f"resource name: {name or '(not returned)'}")
    if api_resource and getattr(api_resource, "spec", None):
        context_spec = getattr(api_resource.spec, "context_spec", None)
        if context_spec:
            print("Context spec preview:")
            print(json.dumps(context_spec, indent=2))
    print("Store this resource name as AGENT_ENGINE_NAME in your .env before using the chat UI.")


def configure_memory_bank(args: argparse.Namespace) -> None:
    client, project, location = build_client(args)
    engine_name = ensure_setting(args.engine, flag="--engine", env_name="AGENT_ENGINE_NAME")
    config = {"context_spec": {"memory_bank_config": build_memory_bank_config(project, location)}}
    engine = client.agent_engines.update(name=engine_name, config=config)
    print(f"Memory bank configuration applied to {engine_name}.")
    spec = getattr(getattr(engine, "api_resource", None), "spec", None)
    context_spec = getattr(spec, "context_spec", None)
    if context_spec is not None:
        print("Updated context spec:")
        print(json.dumps(context_spec, indent=2))


def delete_memory_bank(args: argparse.Namespace) -> None:
    client, _, _ = build_client(args)
    engine_name = ensure_setting(args.engine, flag="--engine", env_name="AGENT_ENGINE_NAME")
    engine = client.agent_engines.update(name=engine_name, config={"context_spec": {}})
    print(f"Memory bank configuration removed from {engine_name}.")
    spec = getattr(getattr(engine, "api_resource", None), "spec", None)
    context_spec = getattr(spec, "context_spec", None)
    print(json.dumps(context_spec or {}, indent=2))


def parse_scope(args: argparse.Namespace) -> dict[str, str]:
    scope: dict[str, str] = {}
    if args.app_name:
        scope["app_name"] = args.app_name
    if args.user_id:
        scope["user_id"] = args.user_id
    for item in args.scope or []:
        if "=" not in item:
            raise SystemExit(f"Invalid scope entry '{item}'. Expected key=value format.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Invalid scope entry '{item}'.")
        scope[key] = value
    if not scope:
        raise SystemExit("Provide at least one scope key via --app-name/--user-id or --scope key=value.")
    return scope


def add_facts(args: argparse.Namespace) -> None:
    client, _, _ = build_client(args)
    engine_name = ensure_setting(args.engine, flag="--engine", env_name="AGENT_ENGINE_NAME")
    scope = parse_scope(args)
    facts: list[str] = []
    if args.fact:
        facts.extend([value.strip() for value in args.fact if value and value.strip()])
    if args.facts_file:
        for line in Path(args.facts_file).read_text().splitlines():
            cleaned = line.strip()
            if cleaned:
                facts.append(cleaned)
    if not facts:
        raise SystemExit("No facts provided. Use --fact or --facts-file.")

    for fact in facts:
        operation = client.agent_engines.memories.create(
            name=engine_name,
            fact=fact,
            scope=scope,
            config={"wait_for_completion": True},
        )
        memory_name = operation.response.name if operation.response else "(pending)"
        print(f"Stored fact -> {memory_name}")


def build_events_from_text(text: str) -> list[dict[str, Any]]:
    return [
        {
            "content": {
                "role": "user",
                "parts": [
                    {
                        "text": text,
                    }
                ],
            }
        }
    ]


def generate_memories(args: argparse.Namespace) -> None:
    client, _, _ = build_client(args)
    engine_name = ensure_setting(args.engine, flag="--engine", env_name="AGENT_ENGINE_NAME")
    scope = parse_scope(args)
    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text().strip()
    if not text:
        raise SystemExit("Provide conversation text via --text or --text-file.")

    operation = client.agent_engines.memories.generate(
        name=engine_name,
        direct_contents_source={"events": build_events_from_text(text)},
        scope=scope,
        config={"wait_for_completion": True},
    )
    response = operation.response or {}
    print("Generated memories response:")
    print(json.dumps(response, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to .env containing VERTEX_API_KEY")
    parser.add_argument("--project", help="Vertex project id (falls back to VERTEX_PROJECT_ID)")
    parser.add_argument("--location", help="Vertex region (falls back to VERTEX_LOCATION)")
    parser.add_argument("--engine", help="Reasoning Engine resource name (or AGENT_ENGINE_NAME env)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-bank",
        help="Provision a brand-new reasoning engine configured with the curated memory bank",
    )
    create.add_argument("--display-name", help="Display name for the new Vertex Agent Engine")
    create.add_argument("--description", help="Description for the new Vertex Agent Engine")
    create.add_argument(
        "--label",
        action="append",
        help="Optional label key=value (repeatable). Keys/values are lower-cased for GCP.",
    )
    create.set_defaults(func=create_memory_bank)

    configure = subparsers.add_parser("configure-bank", help="Apply the curated memory topics & examples")
    configure.set_defaults(func=configure_memory_bank)

    delete = subparsers.add_parser("delete-bank", help="Remove memory bank configuration")
    delete.set_defaults(func=delete_memory_bank)

    facts = subparsers.add_parser("add-facts", help="Add pre-extracted factual memories")
    facts.add_argument("--fact", action="append", help="Literal fact string (repeatable)")
    facts.add_argument("--facts-file", help="Path to newline-delimited facts")
    facts.add_argument("--app-name", help="Scope app_name value", default="aileen3")
    facts.add_argument("--user-id", help="Scope user_id value")
    facts.add_argument(
        "--scope",
        action="append",
        help="Extra scope key=value pairs (repeatable)",
    )
    facts.set_defaults(func=add_facts)

    generate = subparsers.add_parser(
        "generate",
        help="Generate memories from raw conversation text via Vertex auto extraction",
    )
    generate.add_argument("--text", help="Inline conversation text")
    generate.add_argument("--text-file", help="File containing conversation text")
    generate.add_argument("--app-name", help="Scope app_name value")
    generate.add_argument("--user-id", help="Scope user_id value")
    generate.add_argument(
        "--scope",
        action="append",
        help="Extra scope key=value pairs (repeatable)",
    )
    generate.set_defaults(func=generate_memories)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
