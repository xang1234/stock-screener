# XUI Reader Operator Handoff Checklist

Use this checklist when handing XUI Reader operations to another maintainer/operator.

## 1. Environment and Access

- [ ] Confirm Python version (`>=3.11`) and local virtualenv workflow
- [ ] Confirm Playwright Chromium installation on target machine
- [ ] Confirm profile/config paths used in production-like runs

## 2. Core Operational Commands

- [ ] `xui auth status --profile <name>` reviewed
- [ ] `xui read --limit <n>` dry run completed
- [ ] `xui watch --max-cycles <n>` dry run completed
- [ ] `xui doctor --json` reviewed for actionable, redacted output

## 3. Diagnostics and Artifacts

- [ ] Operator knows default artifact location under profile `logs/`
- [ ] Operator understands raw HTML capture is opt-in only
- [ ] Operator can locate selector report and watch counters outputs

## 4. Rollback Notes

If release behavior regresses:

1. Stop active watch processes.
2. Revert to the last known-good tag/commit.
3. Re-run:
   - `xui --help`
   - `xui auth status --profile <name>`
   - `xui read --limit 5`
4. Validate that diagnostics outputs are still redacted.
5. File a blocker issue with:
   - failing command
   - run id / artifact paths
   - exact error output

## 5. Runbook Pointers

- `xui-reader/README.md` for install, local workflow, and debug safety defaults
- `xui-reader/docs/release-readiness-checklist.md` for release gates
- `xui-reader/docs/release-notes-template.md` for release communication structure
