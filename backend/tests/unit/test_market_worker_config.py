"""Deployment worker queue coverage for supported markets."""

from __future__ import annotations

from pathlib import Path

from app.tasks.market_queues import SUPPORTED_MARKETS

ROOT = Path(__file__).resolve().parents[3]


def test_docker_compose_consumes_every_supported_market_queue():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for market in SUPPORTED_MARKETS:
        suffix = market.lower()
        assert f"data_fetch_{suffix}" in compose
        assert f"market_jobs_{suffix}" in compose
        assert f"user_scans_{suffix}" in compose


def test_local_celery_script_consumes_every_supported_market_queue():
    script = (ROOT / "backend" / "start_celery.sh").read_text(encoding="utf-8")
    supported_csv = ",".join(SUPPORTED_MARKETS)
    supported_case = "|".join(SUPPORTED_MARKETS)

    assert f'ENABLED_MARKETS="${{ENABLED_MARKETS:-{supported_csv}}}"' in script
    assert f"data_fetch_shared,{','.join(f'data_fetch_{market.lower()}' for market in SUPPORTED_MARKETS)}" in script
    assert f"{supported_case}) ;;" in script
