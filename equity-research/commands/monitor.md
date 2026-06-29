---
description: Position monitoring — watchlist, thesis check, event radar, thesis tracker, catalyst calendar, morning note
argument-hint: "[optional: ticker, watchlist, or skill name]"
---

The user invoked `/monitor`. This is the position-tracking and forward-looking category.

**Step 1 — Routing.**

- If `$ARGUMENTS` clearly matches a skill (e.g., "thesis check on NKE", "event radar on watchlist", "morning note"), go straight to Step 3.
- Otherwise, ask the user which lens they want. Present the menu below verbatim and end with the default closer.

**Step 2 — Menu to present:**

| # | Skill | Use when |
|---|---|---|
| 1 | `watchlist` | view, add to, or remove from the tracked list of tickers and themes |
| 2 | `thesis-check` | quarterly review: are the original reasons for owning still intact? Verdict: Intact / Improved / Weakening / Broken |
| 3 | `event-radar` | material events since last review — 8-Ks, deals, exec changes, insider flow, ratings shifts (default 90-day window) |
| 4 | `thesis-tracker` (Anthropic) | create or update an investment thesis with new data points |
| 5 | `catalyst-calendar` (Anthropic) | forward-looking calendar of upcoming catalysts for a name, watchlist, or sector |
| 6 | `morning-note` (Anthropic) | desk-style morning note for the day's flow across the watchlist |

End with: *"Or just describe what you want — I'll pick the right lens."*

**Step 3 — Load and follow:**

- 1 / watchlist → `community-skills/monitor/watchlist.md`
- 2 / thesis-check → `community-skills/monitor/thesis-check.md`
- 3 / event-radar → `community-skills/monitor/event-radar.md`
- 4 / thesis-tracker → `anthropic-equity-research-skills/thesis-tracker/SKILL.md`
- 5 / catalyst-calendar → `anthropic-equity-research-skills/catalyst-calendar/SKILL.md`
- 6 / morning-note → `anthropic-equity-research-skills/morning-note/SKILL.md`
- Free-text → match to the closest skill; ask one clarifying question only if truly ambiguous.

Read the chosen file and execute its workflow. If the skill needs a universe and the user didn't supply one, default to `community-skills/monitor/watchlist.md` contents.
