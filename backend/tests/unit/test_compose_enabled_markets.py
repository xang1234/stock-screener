from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compose_enabled_markets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compose_enabled_markets", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_markets_uppercases_deduplicates_and_preserves_order():
    module = load_module()

    assert module.normalize_markets(" us, hk,US,cn ") == ["US", "HK", "CN"]


def test_supported_markets_match_app_queue_topology():
    from app.tasks.market_queues import SUPPORTED_MARKETS

    module = load_module()

    assert module.SUPPORTED_MARKETS == SUPPORTED_MARKETS


def test_normalize_markets_defaults_to_us_when_empty():
    module = load_module()

    assert module.normalize_markets("") == ["US"]
    assert module.normalize_markets(None) == ["US"]


def test_normalize_markets_rejects_unsupported_market():
    module = load_module()

    with pytest.raises(ValueError, match="Unsupported market"):
        module.normalize_markets("US,ZZ")


def test_compose_profiles_for_markets():
    module = load_module()

    assert module.compose_profiles_for_markets(["US", "HK", "CN"]) == [
        "market-us",
        "market-hk",
        "market-cn",
    ]


def test_datafetch_queues_for_markets_include_shared_first():
    module = load_module()

    assert module.datafetch_queues_for_markets(["US", "HK", "CN"]) == [
        "data_fetch_shared",
        "data_fetch_us",
        "data_fetch_hk",
        "data_fetch_cn",
    ]


def test_cli_profiles_output(capsys):
    module = load_module()

    exit_code = module.main(["profiles", "--markets", "US,HK,CN"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "market-us,market-hk,market-cn"


def test_cli_queues_output(capsys):
    module = load_module()

    exit_code = module.main(["queues", "--markets", "US,HK,CN"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "data_fetch_shared,data_fetch_us,data_fetch_hk,data_fetch_cn"


def test_user_scans_queues_for_markets_include_shared_first():
    module = load_module()

    assert module.user_scans_queues_for_markets(["US", "HK", "CN"]) == [
        "user_scans_shared",
        "user_scans_us",
        "user_scans_hk",
        "user_scans_cn",
    ]


def test_market_jobs_queues_for_markets_have_no_shared_fallback():
    module = load_module()

    assert module.market_jobs_queues_for_markets(["US", "HK", "CN"]) == [
        "market_jobs_us",
        "market_jobs_hk",
        "market_jobs_cn",
    ]


def test_cli_queues_output_for_userscans_queue_set(capsys):
    module = load_module()

    exit_code = module.main(["queues", "--queue-set", "userscans", "--markets", "US,HK,CN"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "user_scans_shared,user_scans_us,user_scans_hk,user_scans_cn"


def test_cli_queues_output_for_marketjobs_queue_set(capsys):
    module = load_module()

    exit_code = module.main(["queues", "--queue-set", "marketjobs", "--markets", "US,HK,CN"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "market_jobs_us,market_jobs_hk,market_jobs_cn"


def test_cli_env_output(capsys):
    module = load_module()

    exit_code = module.main(["env", "--markets", "US,HK,CN"])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "COMPOSE_PROFILES=market-us,market-hk,market-cn",
        "DATAFETCH_QUEUES=data_fetch_shared,data_fetch_us,data_fetch_hk,data_fetch_cn",
        "USERSCANS_QUEUES=user_scans_shared,user_scans_us,user_scans_hk,user_scans_cn",
        "MARKETJOBS_QUEUES=market_jobs_us,market_jobs_hk,market_jobs_cn",
    ]
