"""Deployment worker queue coverage for supported markets."""

from __future__ import annotations

from pathlib import Path

from app.tasks.market_queues import SUPPORTED_MARKETS

ROOT = Path(__file__).resolve().parents[3]


def test_docker_compose_profiles_every_market_worker():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for market in SUPPORTED_MARKETS:
        suffix = market.lower()
        assert f"market_jobs_{suffix}" in compose
        assert f"user_scans_{suffix}" in compose
        assert f'profiles: ["market-{suffix}"]' in compose


def test_docker_compose_datafetch_queues_are_derived_from_enabled_markets():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "compose_enabled_markets.py queues" in compose
    assert "data_fetch_shared,data_fetch_us,data_fetch_hk" not in compose
    assert "-Q \"$$QUEUES\"" in compose


def test_docker_compose_forwards_opendart_api_key_to_app_env():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "OPENDART_API_KEY: ${OPENDART_API_KEY:-}" in compose


def test_local_celery_script_consumes_every_supported_market_queue():
    script = (ROOT / "backend" / "start_celery.sh").read_text(encoding="utf-8")
    supported_csv = ",".join(SUPPORTED_MARKETS)
    supported_case = "|".join(SUPPORTED_MARKETS)

    assert f'ENABLED_MARKETS="${{ENABLED_MARKETS:-{supported_csv}}}"' in script
    assert f"data_fetch_shared,{','.join(f'data_fetch_{market.lower()}' for market in SUPPORTED_MARKETS)}" in script
    assert f"{supported_case}) ;;" in script


def test_enabled_market_compose_wrapper_reads_env_files_and_preserves_profiles():
    script = (ROOT / "scripts" / "docker-compose-enabled-markets.sh").read_text(encoding="utf-8")

    assert "read_env_value" in script
    assert "env_file_from_args" in script
    assert "read_env_value \"$ROOT_DIR/.env\" ENABLED_MARKETS" in script
    assert "read_env_value \"$ROOT_DIR/.env.docker\" ENABLED_MARKETS" in script
    assert "PROFILES=\"$COMPOSE_PROFILES,$MARKET_PROFILES\"" in script
