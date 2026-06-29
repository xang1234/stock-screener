# Orientation

An open-source project for fundamental equity analysis in Claude Code.

Anthropic recently shipped an excellent [equity-research skills bundle](https://github.com/anthropics/financial-services/tree/main/plugins/vertical-plugins/equity-research/skills) — Apache-licensed, nine institutional workflow templates, written abstractly so they don't hardcode any data provider. The catch: to actually run them, you need a data connector. Anthropic's reference [`.mcp.json`](https://github.com/anthropics/financial-services/blob/main/plugins/vertical-plugins/financial-analysis/.mcp.json) wires through **eleven separate institutional MCPs** — FactSet, LSEG, S&P Global, Morningstar, and others. No single provider covers fundamentals + filings + transcripts + news in one place, so the full bundle stacks multiple subscriptions, each typically five figures a year per seat. Out of reach for independent analysts, academic economists, and anyone who wants institutional-grade workflows without an institutional budget.

This project closes that gap. **Both Anthropic's Apache-licensed equity-research skill bundle and a community-maintained library of fundamental analysis skills**, running on a single data connector — the drillr MCP — that consolidates what would otherwise require multiple institutional subscriptions. drillr sources its data directly from primary sources (SEC EDGAR, company IR pages, government databases, customs filings, public market venues) via agentic AI, rather than reselling proprietary feeds. Free quota for individual users; no FactSet, LSEG, S&P Global, or Morningstar subscription required.

---

## What the drillr MCP provides

Institutional-grade access to:

- **Structured fundamentals** — financial statements, ratios, multi-year history
- **SEC filing search** — 10-K, 10-Q, 8-K, proxy, S-1, full-text and structured
- **Ontology-based company discovery** — find companies by product offerings, competitive positioning, supply-chain edges, or founder backgrounds. Peers, suppliers, and customers too.
- **Alternative data** — government contracts, hiring trends, geospatial signals
- **News, deals, and events** — breaking news, M&A and deal flow, corporate events, regulatory and legislative actions, earnings calls

---

## Two skill libraries, one data layer

### Anthropic equity-research skill bundle (Apache-licensed)

Nine institutional workflow templates from `anthropics/financial-services`. Abstract methodology files, no hardcoded data provider — drillr slots in as the data layer.

- **`initiating-coverage`** — Full initiation note: thesis, model, valuation, risks.
- **`catalyst-calendar`** — Forward-looking catalyst tracker for a name or sector.
- **`earnings-analysis`** — Post-print review and writeup.
- **`morning-note`** — Desk-style morning note for the day's flow.
- **`thesis-tracker`** — Track an active thesis as confirms and breaks accumulate.
- *Plus `earnings-preview`, `idea-generation`, `model-update`, `sector-overview`.*

### Community-maintained fundamental analysis skills

Analyst-contributed lenses — opinionated methodologies, not buttons. How to read what the market is rewarding from the numbers up, how to walk a supply chain to find hidden champions, how to score an earnings call when management is dodging questions, how to detect channel stuffing or metric definition drift.

- **`discover/themes`** — Read what the market is actually rewarding, from the numbers up. The sharpest entry point: it tells you which lenses to pull next.
- **`discover/supply-chain`** — Walk upstream and downstream from a theme to find the picks-and-shovels names everyone else missed.
- **`analyze/earnings-scorecard`** — Quantitative + qualitative scoring of calls. Tone, hedging, what got dropped from the prepared remarks.
- **`analyze/financial-forensics`** — FCF gap, SBC dilution, non-GAAP creep, channel stuffing. Catches what the press release won't tell you.
- **`analyze/reporting-quality`** — Metric definition drift across quarters, selective omission, language patterns that signal management is hiding something.
- **`discover/alt-plays`** — When you like a thesis but hate the valuation, find a better-priced expression of the same idea.
- **`discover/gov-contracts`** — Federal contract awards as a leading revenue indicator. See the contract before it shows up in the income statement.

In our experience, this repo also works well for academic economists doing exploratory company-level analysis.

---

## How to invoke a skill

Two ways, same skills underneath — pick whichever feels natural.

**Plain language.** Just say what you want — *"run forensics on NKE"*, *"what's coming up for PLTR"*, *"build a theme around AI capex"*. I'll match your request to the right skill.

**Four slash commands.** Prefer a menu? Four category dispatchers cover everything:

- **`/discover`** — idea generation (themes, supply chain, alt-plays, federal contracts, screens, sector overviews)
- **`/analyze`** — single-company deep work (business model, earnings tone, forensics, reporting drift, management, initiation, model updates)
- **`/monitor`** — position tracking (watchlist, thesis check, event radar, thesis tracker, catalyst calendar, morning note)
- **`/macro`** — economic research (yield curve, trade flows, labor market)

Each command opens a short menu of the available lenses; pick one, or just describe what you want and I'll route from there. The slash-command surface stays at four commands no matter how big the skill library grows — new skills are added inside the dispatchers, not as new commands.

---

## Contribute a skill — please

Contributions go to the **community library**. If you have a lens you use — a sector framework, a forensic check, a screen, a macro signal — contribute it. Skills are short markdown files; adding one is a single PR. See `CONTRIBUTING.md`. The community library only gets sharp if working analysts share what they actually do. (The Anthropic equity-research bundle is upstream-mirrored, not directly contributed to here.)

---

## Heads up: you need MCP connected

Without the MCP data connector live, this toolkit can't do real work — only talk about it. Run `/mcp` in Claude Code to check status, and authenticate if prompted.

---

## Where to start

Two natural entry points if you don't already have something specific in mind:

1. **Bring a ticker.** Anything from "run forensics on NKE" to "score PLTR's last call" to "is PWR the best expression of the data-center power thesis" — name a ticker and I'll route to the right skill.

2. **Read recent market themes.** Don't know what's working right now? I can run the `themes` skill — clusters the top-performing names bottom-up over the last 1m / 3m / 6m windows, ignores GICS labels, and tells you which lenses to pull next.

**Which would you like — a specific ticker, or a read of recent market themes?**
