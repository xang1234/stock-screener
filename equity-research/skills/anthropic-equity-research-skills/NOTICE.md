# NOTICE

This directory contains skills vendored from Anthropic's open-source [`anthropics/financial-services`](https://github.com/anthropics/financial-services) repository.

**Upstream:** https://github.com/anthropics/financial-services
**Source path:** `plugins/vertical-plugins/equity-research/skills/`
**Vendored at commit:** `853f755a61f7bbb045c681327f46b354419030a1`
**Vendored on:** 2026-05-10
**License:** Apache License 2.0 (see `LICENSE` in this directory)

---

## What's vendored

Nine equity-research skills, copied verbatim from upstream:

- `catalyst-calendar`
- `earnings-analysis`
- `earnings-preview`
- `idea-generation`
- `initiating-coverage`
- `model-update`
- `morning-note`
- `sector-overview`
- `thesis-tracker`

## Modifications

**None.** These files are reproduced verbatim from upstream. If any file is later modified in this repository, the modification will be marked at the top of that file per Apache 2.0 §4(b).

## Why these were vendored

These skills are abstract methodology files — report frameworks and workflow guides that do not hardcode any data provider. They run cleanly on top of this project's MCP data connector, so we ship them alongside our own community-contributed skills as a curated bundle.

## Trademark

This project is not affiliated with or endorsed by Anthropic. The Anthropic name and trademarks remain the property of Anthropic, PBC, and are referenced here solely for descriptive attribution as permitted under Apache 2.0 §6.

## Updating

To pull updates from upstream:

```bash
git clone --depth 1 https://github.com/anthropics/financial-services.git /tmp/fs
diff -r /tmp/fs/plugins/vertical-plugins/equity-research/skills/ anthropic-equity-research-skills/ | head -50
# Review and merge changes manually; update commit SHA above.
```
