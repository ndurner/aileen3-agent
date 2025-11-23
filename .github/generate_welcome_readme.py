#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re


FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
MARKDOWN_LINK = re.compile(r"(!?)\[(?P<text>[^\]]*)\]\((?P<target>[^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def strip_frontmatter(raw: str) -> str:
    lines = raw.splitlines()
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return raw
    end_index = None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            end_index = i
            break
    if end_index is None:
        return raw
    body_lines = lines[end_index + 1 :]
    return "\n".join(body_lines).lstrip("\n")


def rewrite_links(raw: str, readme_path: Path, output_path: Path) -> str:
    readme_dir = readme_path.resolve().parent
    output_dir = output_path.resolve().parent

    def replace(match: re.Match) -> str:
        bang = match.group(1)
        text = match.group("text")
        target = match.group("target").strip()

        if target.startswith(SKIP_PREFIXES) or target.startswith("/"):
            return match.group(0)

        source_target = (readme_dir / target).resolve()
        try:
            new_target = os.path.relpath(source_target, start=output_dir)
        except ValueError:
            new_target = target

        return f"{bang}[{text}]({new_target})"

    return MARKDOWN_LINK.sub(replace, raw)


def transform(readme_path: Path, output_path: Path) -> None:
    raw = readme_path.read_text(encoding="utf-8")
    no_frontmatter = strip_frontmatter(raw)
    rewritten = rewrite_links(no_frontmatter, readme_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rewritten, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a welcome-page-friendly README by stripping frontmatter "
            "and rewriting relative links."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("README.md"),
        help="Source README file (default: README.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".github/README.md"),
        help="Output Markdown file (default: .github/README.md)",
    )
    args = parser.parse_args()

    transform(args.input, args.output)


if __name__ == "__main__":
    main()

