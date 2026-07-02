---
description: Idea discovery — themes, supply chain, alt-plays, federal contracts, screens, sector overviews
argument-hint: "[optional: ticker, theme, sector, or skill name]"
---

The user invoked `/discover`. This is the idea-discovery category.

**Step 1 — Routing.**

- If `$ARGUMENTS` clearly matches one of the skills below (e.g., "themes", "supply-chain on NVDA", "screen for value"), go straight to Step 3.
- Otherwise, ask the user which lens they want. Present the menu below verbatim (table or numbered list) and end with the default closer: *"or just describe what you want — I'll route."*

**Step 2 — Menu to present:**

| # | Skill | Use when |
|---|---|---|
| 1 | `themes` | reading what the market is rewarding from the numbers up, no sector bias |
| 2 | `supply-chain` | mapping upstream / downstream from a name or theme to find hidden champions |
| 3 | `alt-plays` | finding a better-priced expression of a thesis you already hold |
| 4 | `gov-contracts` | federal contract awards as a leading revenue indicator |
| 5 | `idea-generation` | systematic screens — value / growth / quality / short / special-situation |
| 6 | `sector-overview` | sector-level state of play |

End with: *"Or just describe what you want — I'll pick the right lens."*

**Step 3 — Load and follow:**

- 1 / themes → `community-skills/discover/themes.md`
- 2 / supply-chain → `community-skills/discover/supply-chain.md`
- 3 / alt-plays → `community-skills/discover/alt-plays.md`
- 4 / gov-contracts → `community-skills/discover/gov-contracts.md`
- 5 / idea-generation → `anthropic-equity-research-skills/idea-generation/SKILL.md`
- 6 / sector-overview → `anthropic-equity-research-skills/sector-overview/SKILL.md`
- Free-text → match to the closest skill; ask one clarifying question only if truly ambiguous.

Read the chosen file and execute its workflow.
