"""Tests for backfill_theme_aliases script behavior."""

import argparse
from unittest.mock import MagicMock, patch

from scripts import backfill_theme_aliases as script


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "dry_run": False,
        "mention_limit": 0,
        "report_file": "/tmp/theme_alias_backfill_report.json",
        "yes": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@patch("scripts.backfill_theme_aliases._save_json")
@patch("scripts.backfill_theme_aliases.ThemeAliasBackfillService")
@patch("scripts.backfill_theme_aliases.SessionLocal")
@patch("scripts.backfill_theme_aliases._build_parser")
def test_main_writes_report_for_dry_run(
    mock_build_parser,
    mock_session_local,
    mock_service_cls,
    mock_save_json,
):
    parser = MagicMock()
    parser.parse_args.return_value = _args(dry_run=True)
    mock_build_parser.return_value = parser

    db = MagicMock()
    mock_session_local.return_value = db

    service = MagicMock()
    service.run.return_value = {
        "dry_run": True,
        "totals": {
            "clusters_scanned": 1,
            "mentions_scanned": 2,
            "candidate_groups": 2,
            "planned_inserts": 2,
            "inserted": 0,
            "collisions_total": 1,
        },
        "collisions": {"by_bucket": {"existing_alias_conflict": 1}, "inventory": []},
    }
    mock_service_cls.return_value = service

    script.main()

    mock_save_json.assert_called_once()
    saved_path = mock_save_json.call_args.args[0]
    assert str(saved_path) == "/tmp/theme_alias_backfill_report.json"
    db.close.assert_called_once()


@patch("scripts.backfill_theme_aliases._save_json")
@patch("scripts.backfill_theme_aliases.ThemeAliasBackfillService")
@patch("scripts.backfill_theme_aliases.SessionLocal")
@patch("scripts.backfill_theme_aliases._build_parser")
def test_main_calls_service_with_mention_limit(
    mock_build_parser,
    mock_session_local,
    mock_service_cls,
    mock_save_json,
):
    parser = MagicMock()
    parser.parse_args.return_value = _args(dry_run=False, mention_limit=100)
    mock_build_parser.return_value = parser

    db = MagicMock()
    mock_session_local.return_value = db

    service = MagicMock()
    service.run.return_value = {
        "dry_run": False,
        "totals": {
            "clusters_scanned": 1,
            "mentions_scanned": 100,
            "candidate_groups": 2,
            "planned_inserts": 2,
            "inserted": 2,
            "collisions_total": 0,
        },
        "collisions": {"by_bucket": {}, "inventory": []},
    }
    mock_service_cls.return_value = service

    script.main()

    service.run.assert_called_once_with(dry_run=False, mention_limit=100)
    mock_save_json.assert_called_once()
