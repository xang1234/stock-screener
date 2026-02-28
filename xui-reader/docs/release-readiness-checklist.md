# XUI Reader v2 Release Readiness Checklist

Use this checklist before tagging or publishing a `xui-reader` release.

## 1. Packaging and Install Smoke

- [ ] `python -m pip install -e ".[dev]"` succeeds from `xui-reader/`
- [ ] `python -m playwright install chromium` succeeds
- [ ] `xui --help` succeeds in a fresh shell
- [ ] `python -c "import xui_reader; print(xui_reader.__version__)"` succeeds

## 2. Quality Gates

- [ ] `pytest -q` passes
- [ ] `ruff check src tests` passes
- [ ] No skipped failures hidden behind `xfail(strict=False)` for WS7 scope

## 3. Security and Secret Hygiene

- [ ] Debug artifacts redact storage/cookie/token material by default
- [ ] HTML artifact full-capture remains opt-in (`XUI_DEBUG_RAW_HTML=1`)
- [ ] Fixture sanitizer is used for newly committed extractor snapshots

## 4. Milestone Gates

### v2.0 (first user-visible value)

- [ ] Auth login flow works and storage state persists to profile path
- [ ] One list read succeeds with bounded runtime
- [ ] `--new` emits only unseen items with deterministic ordering

### v2.1 (watch/diagnostics hardening)

- [ ] Watch loop respects interval/jitter/shutdown window controls
- [ ] Budget stop behavior is deterministic and surfaced in exit codes
- [ ] `xui doctor` outputs actionable and redacted diagnostics sections

### v2.2 (release hardening and operator handoff)

- [ ] Release docs are updated (this checklist + handoff + notes template)
- [ ] Operator handoff checklist is reviewed with next maintainer
- [ ] Rollback notes are validated against current CLI behavior

## 5. Release Output

- [ ] Release notes drafted from `docs/release-notes-template.md`
- [ ] Tag/version matches release notes heading and changelog summary
- [ ] Final sign-off captured by maintainer in release PR
