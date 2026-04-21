from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_static_site_market_build_failures_are_not_marked_continue_on_error() -> None:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    build_market_job = content.split("  build-market:\n", 1)[1].split(
        "\n  combine-and-build:",
        1,
    )[0]

    assert "continue-on-error: true" not in build_market_job
