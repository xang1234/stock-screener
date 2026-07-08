"""Static-site workflow market matrix coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.domain.markets import market_registry


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _weekly_reference_matrix_markets(path: str) -> list[str]:
    """Parse the weekly-reference matrix from its ``|| '[...]'`` dispatch form."""
    content = (_PROJECT_ROOT / path).read_text(encoding="utf-8")
    match = re.search(r"\|\|\s*'(\[[^']+\])'", content)
    assert match is not None, f"{path} does not declare a market matrix"
    return list(json.loads(match.group(1)))


def _static_site_market_group(name: str) -> list[str]:
    """Parse a market-group JSON array assigned in the select-markets job.

    The static-site workflow no longer uses a literal matrix; build-market
    consumes ``fromJSON(needs.select-markets.outputs.markets)`` and the
    select-markets job derives the group from shell vars like ``ASIA='[...]'``.
    """
    content = (_PROJECT_ROOT / ".github/workflows/static-site.yml").read_text(encoding="utf-8")
    match = re.search(rf"^\s*{name}='(\[[^']*\])'", content, re.MULTILINE)
    assert match is not None, f"static-site.yml does not define a {name} market group"
    return list(json.loads(match.group(1)))


def test_static_site_all_group_and_weekly_reference_cover_supported_markets():
    expected = list(market_registry.supported_market_codes())

    # The dispatch "all" group must stay in lockstep with the registry so a
    # manual full rebuild covers every supported market in the canonical order.
    assert _static_site_market_group("ALL") == expected
    assert _weekly_reference_matrix_markets(".github/workflows/weekly-reference-data.yml") == expected


def test_static_site_schedule_groups_partition_supported_markets():
    asia = _static_site_market_group("ASIA")
    us = _static_site_market_group("US")
    all_markets = _static_site_market_group("ALL")

    # The two scheduled groups must contain the intended exact markets...
    assert asia == ["HK", "IN", "JP", "KR", "TW", "CN", "SG", "MY", "AU"]
    assert us == ["US", "CA", "DE"]

    # ...and together partition the full market set (disjoint + exhaustive), so
    # every supported market is published by exactly one scheduled run.
    assert set(asia).isdisjoint(us)
    assert set(asia) | set(us) == set(all_markets)
    assert set(all_markets) == set(market_registry.supported_market_codes())


def test_static_site_manual_dispatch_can_run_china_only():
    content = (_PROJECT_ROOT / ".github/workflows/static-site.yml").read_text(encoding="utf-8")
    select_markets_job = content.split("  select-markets:\n", 1)[1].split(
        "\n  ensure_daily_price_release:",
        1,
    )[0]

    assert "          - china" in content
    assert _static_site_market_group("CN_ONLY") == ["CN"]
    assert 'china) markets="$CN_ONLY" ;;' in select_markets_job


def test_static_workflow_legacy_weekly_reference_manifest_is_us_only():
    content = (_PROJECT_ROOT / ".github/workflows/static-site.yml").read_text(encoding="utf-8")

    assert '[ "${{ matrix.market }}" = "US" ] &&' in content
    assert "No market-scoped weekly reference manifest found" in content


def test_static_workflow_fails_fast_when_weekly_reference_assets_cannot_be_listed():
    content = (_PROJECT_ROOT / ".github/workflows/static-site.yml").read_text(encoding="utf-8")

    assert 'if ! ASSET_NAMES="$(retry_list_assets)"; then' in content
    assert "Failed to list weekly-reference release assets" in content


def test_weekly_reference_defaults_to_partial_publish_for_transient_tw_source_failures():
    content = (_PROJECT_ROOT / ".github/workflows/weekly-reference-data.yml").read_text(
        encoding="utf-8"
    )

    assert '[ "$MATRIX_MARKET" = "CN" ] || [ "$MATRIX_MARKET" = "TW" ]' in content


def test_local_celery_startup_derives_market_workers_from_backend_topology():
    content = (_PROJECT_ROOT / "backend" / "start_celery.sh").read_text(encoding="utf-8")

    assert "from app.tasks.market_queues import SUPPORTED_MARKETS" in content
    assert "from app.tasks.market_queues import all_data_fetch_queues" in content
    assert 'ENABLED_MARKETS="${ENABLED_MARKETS:-$SUPPORTED_MARKETS}"' in content
    assert '-Q "$DATA_FETCH_QUEUES"' in content
    assert "US|HK|IN|JP|KR|TW|CN|CA|DE|SG|MY" not in content
