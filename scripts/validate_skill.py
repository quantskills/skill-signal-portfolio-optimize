#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REQUIRED_ROOT_FILES = ("SKILL.md", "README.md", "README.en.md", "LICENSE")
REQUIRED_QUANTSKILLS_FIELDS = {
    "schema_version",
    "organization",
    "organization_url",
    "repository",
    "repository_url",
    "project_type",
    "license",
    "maintainer",
    "catalog",
    "workflow",
    "summary_zh",
    "summary_en",
    "status",
    "validation_level",
    "maintainer_type",
    "platforms",
    "interface",
}
EXPECTED_PLATFORMS = ["cursor", "claude-code", "codex", "hermes", "openclaw"]
ALLOWED_TOP_LEVEL_FIELDS = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "user-invocable",
    "disable-model-invocation",
    "supported-runtimes",
    "compatibility",
    "version",
    "author",
    "metadata",
    "quantSkills",
}


def fail(message: str) -> None:
    print(f"skill validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a non-empty string")
    return value.strip()


def validate_frontmatter(root: Path) -> dict:
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

    unknown = set(frontmatter) - ALLOWED_TOP_LEVEL_FIELDS
    if unknown:
        fail(f"unsupported frontmatter fields: {sorted(unknown)}")
    for key in ("name", "description", "quantSkills"):
        if key not in frontmatter:
            fail(f"frontmatter is missing {key}")

    name = frontmatter["name"]
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        fail("name must use lowercase hyphen-case")
    if name != root.name:
        fail(f"name {name!r} must match directory {root.name!r}")

    description = require_non_empty_string(frontmatter["description"], "description")
    if len(description) < 60 or len(description) > 1024:
        fail("description must contain 60 to 1024 characters")
    if "Use when" not in description:
        fail("description must state when the skill should be used")

    return frontmatter


def validate_quantskills(frontmatter: dict) -> None:
    metadata = frontmatter["quantSkills"]
    if not isinstance(metadata, dict):
        fail("quantSkills must be a mapping")

    missing = REQUIRED_QUANTSKILLS_FIELDS - set(metadata)
    if missing:
        fail(f"quantSkills is missing fields: {sorted(missing)}")

    expected_values = {
        "schema_version": "2.1.0",
        "organization": "quantskills",
        "organization_url": "https://github.com/quantskills",
        "repository": "skill-signal-portfolio-optimize",
        "repository_url": (
            "https://github.com/quantskills/skill-signal-portfolio-optimize"
        ),
        "project_type": "skill",
        "license": "GPL-3.0-only",
        "maintainer": "X-Tech-group",
        "status": "active",
        "validation_level": "runnable",
        "maintainer_type": "community",
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            fail(f"quantSkills.{key} must equal {expected!r}")

    catalog = metadata.get("catalog")
    if not isinstance(catalog, dict):
        fail("quantSkills.catalog must be a mapping")
    if catalog.get("category") != "05":
        fail("quantSkills.catalog.category must equal '05'")
    if catalog.get("subcategory") != "05.portfolio-construction":
        fail(
            "quantSkills.catalog.subcategory must equal "
            "'05.portfolio-construction'"
        )

    workflow = metadata.get("workflow")
    if not isinstance(workflow, dict):
        fail("quantSkills.workflow must be a mapping")
    stages = workflow.get("workflow_stages")
    if workflow.get("primary_stage") != "portfolio-construction":
        fail("quantSkills.workflow.primary_stage must be portfolio-construction")
    if not isinstance(stages, list) or "portfolio-construction" not in stages:
        fail("workflow_stages must include portfolio-construction")
    if len(stages) != len(set(stages)):
        fail("workflow_stages must be unique")

    tags = metadata.get("tags")
    if not isinstance(tags, list) or not (5 <= len(tags) <= 8):
        fail("quantSkills.tags must contain 5 to 8 entries")
    if len(tags) != len(set(tags)):
        fail("quantSkills.tags must be unique")
    if any(not isinstance(tag, str) or not NAME_PATTERN.fullmatch(tag) for tag in tags):
        fail("quantSkills.tags must use lowercase hyphen-case")

    if metadata.get("platforms") != EXPECTED_PLATFORMS:
        fail(
            "quantSkills.platforms must list cursor, claude-code, codex, "
            "hermes, and openclaw in canonical order"
        )

    for key in ("summary_zh", "summary_en"):
        summary = require_non_empty_string(metadata.get(key), f"quantSkills.{key}")
        if len(summary) > 240:
            fail(f"quantSkills.{key} must not exceed 240 characters")

    interface = metadata.get("interface")
    if not isinstance(interface, dict) or interface.get("mode") != "natural-language":
        fail("quantSkills.interface.mode must equal 'natural-language'")


def validate_local_links(root: Path) -> None:
    markdown_files = [
        root / "SKILL.md",
        root / "README.md",
        root / "README.en.md",
        *sorted((root / "references").glob("*.md")),
    ]
    for markdown_file in markdown_files:
        source = markdown_file.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(source):
            target = target.strip().strip("<>")
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
            ):
                continue
            local_path = target.split("#", 1)[0]
            if not local_path:
                continue
            resolved = (markdown_file.parent / local_path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                fail(f"{markdown_file.relative_to(root)} links outside the repository")
            if not resolved.exists():
                fail(
                    f"broken local link in {markdown_file.relative_to(root)}: "
                    f"{target}"
                )


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

    for relative_path in REQUIRED_ROOT_FILES:
        if not (root / relative_path).is_file():
            fail(f"{relative_path} is required")
    if not (root / "agents" / "openai.yaml").is_file():
        fail("agents/openai.yaml is required")

    frontmatter = validate_frontmatter(root)
    validate_quantskills(frontmatter)
    validate_local_links(root)

    public_docs = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("SKILL.md", "README.md", "README.en.md")
    )
    if "/home/" in public_docs:
        fail("public documentation must not contain machine-specific /home paths")
    if "investment advice" not in public_docs:
        fail("public documentation must include a research-use risk disclosure")

    print("skill validation passed (QuantSkills Catalog Contract v2.1, runnable)")


if __name__ == "__main__":
    main()
