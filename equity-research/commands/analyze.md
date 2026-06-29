---
description: Single-company analysis — business model, earnings tone, forensics, reporting drift, management, initiation, model update
argument-hint: "[ticker] [optional: skill name or what to analyze]"
---

The user invoked `/analyze`. This is the single-company deep-work category.

**Step 1 — Routing.**

- If `$ARGUMENTS` clearly names a skill (e.g., "forensics on NKE", "score NVDA's last call", "initiate on PWR"), go straight to Step 3.
- Otherwise, ask the user which lens they want. Present the menu below verbatim and end with the default closer.

**Step 2 — Menu to present:**

| # | Skill | Use when |
|---|---|---|
| 1 | `business-model` | how the company makes money, scored across 8 economist's dimensions (pricing power, moat, cyclicality, AI exposure, etc.) |
| 2 | `earnings-scorecard` | 8-dimension tone scorecard + 6 content-integrity checks on a recent call |
| 3 | `financial-forensics` | FCF vs net-income gap, SBC dilution, channel-stuffing signals, non-GAAP widening, capitalization shifts |
| 4 | `reporting-quality` | metric-definition drift, above/below-the-line moves, segment restructuring, restatements |
| 5 | `management` | capital-allocation track record, comp alignment, insider patterns, exec turnover |
| 6 | `initiating-coverage` (Anthropic) | full initiation note: thesis, model, valuation, risks |
| 7 | `earnings-preview` (Anthropic) | pre-print: what to watch on the upcoming call |
| 8 | `earnings-analysis` (Anthropic) | post-print: review and writeup |
| 9 | `model-update` (Anthropic) | update a financial model after new data lands |

End with: *"Or just describe what you want — I'll pick the right lens."*

If the user did not name a ticker, ask for one before loading any skill (this category always needs a name).

**Step 3 — Load and follow:**

- 1 / business-model → `community-skills/analyze/business-model.md`
- 2 / earnings-scorecard → `community-skills/analyze/earnings-scorecard.md`
- 3 / financial-forensics → `community-skills/analyze/financial-forensics.md`
- 4 / reporting-quality → `community-skills/analyze/reporting-quality.md`
- 5 / management → `community-skills/analyze/management.md`
- 6 / initiating-coverage → `anthropic-equity-research-skills/initiating-coverage/SKILL.md`
- 7 / earnings-preview → `anthropic-equity-research-skills/earnings-preview/SKILL.md`
- 8 / earnings-analysis → `anthropic-equity-research-skills/earnings-analysis/SKILL.md`
- 9 / model-update → `anthropic-equity-research-skills/model-update/SKILL.md`
- Free-text → match to the closest skill; ask one clarifying question only if truly ambiguous.

Read the chosen file and execute its workflow.
