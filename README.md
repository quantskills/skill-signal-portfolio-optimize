# QuantSkills Skill Template

简体中文 | [English](README.en.md)

这是 `skill-template` 仓库的规范模板，由 `abgyjaguo` 维护，用于创建符合 catalog-contract v2 的 QuantSkills skill 项目。根目录 `SKILL.md` 描述本模板仓库；可复制的因子评估项目声明仅在 `references/declaration-example.yml`。

| 运行时 | 入口或薄适配 |
| --- | --- |
| Codex | `SKILL.md` |
| Claude Code | `SKILL.md` |
| Cursor | `agents/cursor-rule.mdc` → `SKILL.md` |
| Hermes | `agents/portable-loader.md` → `SKILL.md` |
| OpenClaw | `agents/openai.yaml` → `SKILL.md` |

## 编写边界

- 不提交密钥、令牌、私人数据或机密材料。
- 改编上游策略、论文、数据或代码时，注明来源并保留许可证要求。
- 定量项目须披露数据来源、假设、参数、已知限制、风险边界及仅供研究或教育使用的范围；不得承诺收益，也不构成投资建议。
- `qsh-form` 是可选界面增强，和数据接口兼容性相互独立。

## 许可证

`GPL-3.0-only`。见 [LICENSE](LICENSE)。
