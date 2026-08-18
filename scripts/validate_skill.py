#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def fail(message: str) -> None:
    print(f"skill validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    skill_file = root / "SKILL.md"
    try:
        source = skill_file.read_text(encoding="utf-8")
    except OSError as exc:
        fail(str(exc))
    match = re.match(r"^---\n(.*?)\n---(?:\n|$)", source, re.DOTALL)
    if match is None:
        fail("SKILL.md must start with YAML frontmatter")
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        fail(f"invalid YAML frontmatter: {exc}")
    if not isinstance(frontmatter, dict):
        fail("frontmatter must be a mapping")
    if set(frontmatter) != {"name", "description"}:
        fail("frontmatter must contain exactly name and description")
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        fail("name must use lowercase hyphen-case")
    if name != root.name:
        fail(f"name {name!r} must match directory {root.name!r}")
    if not isinstance(description, str) or not description.strip():
        fail("description must be a non-empty string")
    if len(description) > 1024:
        fail("description must not exceed 1024 characters")
    if not (root / "agents" / "openai.yaml").is_file():
        fail("agents/openai.yaml is required")
    print("skill validation passed")


if __name__ == "__main__":
    main()
