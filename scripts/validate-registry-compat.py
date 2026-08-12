#!/usr/bin/env python3
"""Bridge legacy Registry validation while preserving essential v2 guarantees."""
import json
import subprocess
import sys
from pathlib import Path

import yaml


def fail(message):
    print(f"compatibility bridge: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    if len(sys.argv) != 3:
        fail("usage: validate-registry-compat.py VALIDATOR TARGET")
    validator, target = map(Path, sys.argv[1:])
    declaration = target / "SKILL.md"
    try:
        frontmatter = yaml.safe_load(declaration.read_text(encoding="utf-8").split("---", 2)[1])
        qs = frontmatter["quantSkills"]
    except (IndexError, KeyError, OSError, yaml.YAMLError) as exc:
        fail(f"invalid declaration: {exc.__class__.__name__}")
    expected = {
        "schema_version": "2.0.0", "organization": "quantskills", "organization_url": "https://github.com/quantskills",
        "repository": target.name, "repository_url": f"https://github.com/quantskills/{target.name}", "project_type": "skill",
        "license": "GPL-3.0-only", "status": "active", "validation_level": "runnable", "maintainer_type": "community",
    }
    if any(qs.get(key) != value for key, value in expected.items()) or not all(str(qs.get(key, "")).strip() for key in ("maintainer", "collection", "summary_zh", "summary_en")):
        fail("essential v2 metadata mismatch")
    if qs.get("catalog") != {"category": "10", "subcategory": "10.skill-template"}:
        fail("catalog metadata mismatch")
    if qs.get("workflow", {}).get("primary_stage") != "orchestration" or "orchestration" not in qs.get("workflow", {}).get("workflow_stages", []):
        fail("workflow metadata mismatch")
    if set(qs.get("platforms", [])) != {"cursor", "claude-code", "codex", "hermes", "openclaw"} or len(qs.get("platforms", [])) != 5:
        fail("platform metadata mismatch")
    if qs.get("interface") != {"mode": "not-applicable", "reason": "orchestration-only"}:
        fail("interface metadata mismatch")
    result = subprocess.run([sys.executable, str(validator), str(target), "--json"], text=True, capture_output=True, check=False)
    try:
        items = json.loads(result.stdout)["items"]
    except (json.JSONDecodeError, KeyError):
        fail("legacy validator emitted invalid JSON")
    allowed = (result.returncode == 1 and len(items) == 2 and all(item.get("level") == "warn" and item.get("check") == "frontmatter" for item in items)
               and any("category" in item.get("detail", "") for item in items)
               and any("hermes" in item.get("detail", "") for item in items))
    if not allowed:
        fail("legacy validator emitted unexpected result")
    print("compatibility bridge: accepted known legacy v2 warnings")


if __name__ == "__main__":
    main()
