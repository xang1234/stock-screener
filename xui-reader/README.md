# xui-reader

`xui-reader` is the XUI Reader v2 scaffold package for read-only timeline/list collection.

## Bootstrap

1. Create and activate a virtual environment.
2. Install package and dev tools:
   - `pip install -e ".[dev]"`
3. Install Playwright Chromium:
   - `python -m playwright install chromium`

## Local Workflow

- CLI help: `xui --help`
- Run tests: `pytest`
- Lint: `ruff check .`

This baseline intentionally keeps behavior minimal while stabilizing module contracts.
