# XUI Reader Clean-Machine Onboarding Drill

This runbook validates first-run onboarding without tribal knowledge.

## Goal

Confirm a new operator can:

1. initialize config/profile state,
2. understand auth prerequisites,
3. run doctor/read/watch with actionable guidance when auth is missing.

## Preconditions

- Python `>=3.11`
- Playwright Chromium installed
- `xui` CLI available in the active virtualenv

## Drill Steps

Use an isolated temp config path:

```bash
TMP_DIR="$(mktemp -d)"
CONFIG_PATH="$TMP_DIR/config.toml"
```

1. Initialize config:

```bash
xui config init --path "$CONFIG_PATH"
xui config show --path "$CONFIG_PATH"
```

Expected:
- Config path resolves correctly.
- Default profile is `default`.

2. Bootstrap profile directory:

```bash
xui profiles create default --path "$CONFIG_PATH"
xui profiles list --path "$CONFIG_PATH"
```

Expected:
- `default` profile exists and is active.

3. Verify auth guidance before login:

```bash
xui auth status --path "$CONFIG_PATH" --profile default
```

Expected:
- Exit code `2`
- Status `missing_storage_state`
- Next-step command includes `xui auth login --profile default --path "$CONFIG_PATH"`

4. Run doctor preflight:

```bash
xui doctor --path "$CONFIG_PATH"
```

Expected:
- Structured sections (`config`, `auth`, `source_selection`, `smoke`)
- Clear failure summary for missing auth
- Actionable guidance with login command

5. Validate read/watch auth-failure UX:

```bash
xui read --path "$CONFIG_PATH" --limit 5
xui watch --path "$CONFIG_PATH" --max-cycles 1 --limit 5
```

Expected:
- `read` prints per-source failures plus explicit next-step login command
- `watch` prints `auth_fail` exit state plus explicit next-step login command

## Friction Points Captured and Resolved

1. Friction: unclear first command order on fresh setup.
Resolution: this runbook defines an explicit command sequence from `config init` to `doctor/read/watch`.

2. Friction: auth-missing errors in `read/watch` required inference.
Resolution: CLI now prints explicit next-step login commands in those auth-failure paths.

3. Friction: optional module invocation (`python -m xui_reader.cli`) may fail without installed deps.
Resolution: runbook standardizes operator entrypoint as `xui` from the prepared environment.
