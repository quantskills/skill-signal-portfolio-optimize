# QuantSkills Skill Template

[简体中文](README.md) | English

This is the canonical `skill-template` repository, maintained by `abgyjaguo`, for creating catalog-contract v2 QuantSkills skills. The root `SKILL.md` describes this template repository; the copy-ready factor-evaluation declaration lives only in `references/declaration-example.yml`.

| Runtime | Entry point or thin adapter |
| --- | --- |
| Codex | `SKILL.md` |
| Claude Code | `SKILL.md` |
| Cursor | `agents/cursor-rule.mdc` → `SKILL.md` |
| Hermes | `agents/portable-loader.md` → `SKILL.md` |
| OpenClaw | `agents/openai.yaml` → `SKILL.md` |

Do not include secrets or private data. Cite upstream sources and preserve licenses. Quantitative projects must disclose data sources, assumptions, parameters, limitations, risk boundaries, and research-only scope; they must not promise returns or constitute investment advice. `qsh-form` is optional and independent from data-interface compatibility.

## License

`GPL-3.0-only`. See [LICENSE](LICENSE).
