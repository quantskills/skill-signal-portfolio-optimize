---
name: skill-template
description: A canonical QuantSkills skill template for authoring portable, catalog-contract v2 skill projects across Cursor, Claude Code, Codex, Hermes, and OpenClaw. Use when creating or migrating a QuantSkills skill repository.
quantSkills:
  schema_version: 2.0.0
  organization: quantskills
  organization_url: https://github.com/quantskills
  repository: skill-template
  repository_url: https://github.com/quantskills/skill-template
  project_type: skill
  license: GPL-3.0-only
  maintainer: abgyjaguo
  collection: infrastructure
  catalog: {category: "10", subcategory: 10.skill-template}
  workflow: {primary_stage: orchestration, workflow_stages: [orchestration]}
  tags: [template, skill-authoring]
  requires: []
  summary_zh: 用于创建可移植 QuantSkills 技能项目的规范模板。
  summary_en: Canonical template for portable QuantSkills skill projects.
  status: active
  validation_level: runnable
  maintainer_type: community
  platforms: [cursor, claude-code, codex, hermes, openclaw]
  interface: {mode: not-applicable, reason: orchestration-only}
---

# QuantSkills Skill Template

Use this repository as the canonical starting point for a portable QuantSkills skill.

## Authoring boundary

- Keep secrets and private datasets out of the repository.
- Cite upstream sources and preserve required licenses when adapting material.
- For quantitative projects, disclose data sources, assumptions, parameters, limitations, risk boundaries, and research-only status. Never promise returns or present output as investment advice.
- The root declaration describes this `skill-template` repository. A copy-ready factor-evaluation declaration is in `references/declaration-example.yml`.

## Optional qsh-form

The `qsh-form` block below is an optional UI enhancement. It is independent of catalog data-interface compatibility; removing it keeps this template contract-valid.

```json qsh-form
{"version":1,"task":{"placeholder":"Describe the research task","required":true},"fields":[],"prompt_template":"Handle this task: {{task}}. Attachments: {{#attachments}}"}
```

## References

Use `references/source_boundary.md` for source boundaries.
