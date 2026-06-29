---
description: Macro & economic research — yield curve, trade flows, labor market
argument-hint: "[optional: skill name or what to look at]"
---

The user invoked `/macro`. This is the macro / economic-research category.

**Step 1 — Routing.**

- If `$ARGUMENTS` clearly matches a skill (e.g., "yield curve", "trade flows for HS 8542", "labor market read"), go straight to Step 3.
- Otherwise, ask the user which lens they want. Present the menu below verbatim and end with the default closer.

**Step 2 — Menu to present:**

| # | Skill | Use when |
|---|---|---|
| 1 | `yield-curve` | rate cycle, inversion / re-steepening, credit spreads, combined recession signal |
| 2 | `trade-flows` | supply-chain relocation at HS-code level; transshipment cross-check; tie shifts to public companies |
| 3 | `labor-market` | leading labor indicators beyond the headline payroll number — claims, quits, temp employment, household vs. establishment |

End with: *"Or just describe what you want — I'll pick the right lens."*

**Step 3 — Load and follow:**

- 1 / yield-curve → `community-skills/economic-research/yield-curve.md`
- 2 / trade-flows → `community-skills/economic-research/trade-flows.md`
- 3 / labor-market → `community-skills/economic-research/labor-market.md`
- Free-text → match to the closest skill; ask one clarifying question only if truly ambiguous.

Read the chosen file and execute its workflow.
