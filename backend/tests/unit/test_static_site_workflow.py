from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _build_market_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  build-market:\n", 1)[1].split(
        "\n  combine-and-build:",
        1,
    )[0]


def test_static_site_market_build_failures_are_not_marked_continue_on_error() -> None:
    build_market_job = _build_market_job()

    assert "continue-on-error: true" not in build_market_job


def test_static_site_daily_price_seed_allows_stale_bootstrap() -> None:
    build_market_job = _build_market_job()
    seed_step = build_market_job.split("      - name: Seed daily price bundle from GitHub\n", 1)[1].split(
        "\n      - name: Export market static data bundle",
        1,
    )[0]

    assert "--allow-stale" in seed_step
