# xui-reader

`xui-reader` is the XUI Reader v2 scaffold package for read-only timeline/list collection.

## Bootstrap

1. Enter package directory:
   - `cd xui-reader`
2. Create and activate a virtual environment.
3. Install package and dev tools:
   - `pip install -e ".[dev]"`
4. Install Playwright Chromium:
   - `python -m playwright install chromium`

## Local Workflow

- CLI help: `xui --help`
- Run tests: `pytest`
- Lint: `ruff check .`

This baseline intentionally keeps behavior minimal while stabilizing module contracts.

## Watch Exit Codes

`xui watch` uses a stable exit-code matrix for automation wrappers:

| State | Exit code | Meaning |
| --- | ---: | --- |
| `success` | `0` | Run completed without auth-fail or budget-stop terminal state. |
| `budget_stop` | `4` | Run stopped because configured cycle budget (`--max-cycles > 1`) was exhausted. |
| `auth_fail` | `5` | All source failures in the run were auth-related (missing/invalid session). |
| `interrupted` | `130` | Process interrupted by operator signal/keyboard interrupt. |

## Debug Artifact Safety Defaults

- Diagnostic artifact values are redacted by default (`sessionid`, `auth_token`, `authorization`, `storage_state`, etc.).
- Full HTML capture is **opt-in only**. By default, HTML artifacts are redacted snippets.
- To opt into full redacted HTML capture, set `XUI_DEBUG_RAW_HTML=1`.

## Debug Event Schema Compatibility

Structured debug events use schema version `v1` and always include:

- `schema_version`
- `event_type`
- `occurred_at`
- `run_id`
- `source_id`
- `payload`

Compatibility policy:

- Top-level fields are append-only within a schema major version.
- Consumers should ignore unknown top-level fields.
- Breaking top-level changes require a schema major bump.
