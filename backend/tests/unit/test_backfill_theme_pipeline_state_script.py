"""Tests for backfill_theme_pipeline_state script behavior."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import backfill_theme_pipeline_state as script


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "chunk_size": 500,
        "max_chunks": 0,
        "no_resume": False,
        "reset_checkpoint": False,
        "checkpoint_file": "/tmp/backfill_checkpoint.json",
        "report_file": "/tmp/backfill_report.json",
        "max_age_days": 30,
        "threshold_processed_without_mentions_ratio": 0.1,
        "threshold_parse_failure_rate": 0.3,
        "threshold_retryable_growth_ratio": 2.0,
        "threshold_retryable_growth_delta": 25,
        "dry_run": False,
        "yes": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _chunk(rows_read: int, rows_written: int, conflicts: int, next_cursor: int):
    return SimpleNamespace(
        rows_read=rows_read,
        rows_written=rows_written,
        conflicts=conflicts,
        next_cursor=next_cursor,
        writes_by_pipeline_status={},
    )


def _summary_payload():
    return {
        "by_pipeline_status": {},
        "by_pipeline_status_scope": {
            "type": "all_time_table_counts",
            "window_days": None,
        },
        "drift": {
            "scope": {
                "type": "published_at_window",
                "window_days": 30,
            },
            "window_days": 30,
            "thresholds": {},
            "pipelines": [],
        },
    }


@patch("scripts.backfill_theme_pipeline_state._save_checkpoint")
@patch("scripts.backfill_theme_pipeline_state.ThemePipelineStateBackfillService")
@patch("scripts.backfill_theme_pipeline_state.SessionLocal")
@patch("scripts.backfill_theme_pipeline_state._build_parser")
def test_main_dry_run_skips_checkpoint_and_report_writes(
    mock_build_parser,
    mock_session_local,
    mock_service_cls,
    mock_save_checkpoint,
):
    parser = MagicMock()
    parser.parse_args.return_value = _args(dry_run=True)
    mock_build_parser.return_value = parser

    db = MagicMock()
    mock_session_local.return_value = db

    service = MagicMock()
    service.process_chunk.side_effect = [
        _chunk(rows_read=1, rows_written=2, conflicts=0, next_cursor=10),
        _chunk(rows_read=0, rows_written=0, conflicts=0, next_cursor=10),
    ]
    service.summary_counts.return_value = _summary_payload()
    mock_service_cls.return_value = service

    script.main()

    mock_save_checkpoint.assert_not_called()
    db.close.assert_called_once()


@patch("scripts.backfill_theme_pipeline_state._save_checkpoint")
@patch("scripts.backfill_theme_pipeline_state.ThemePipelineStateBackfillService")
@patch("scripts.backfill_theme_pipeline_state.SessionLocal")
@patch("scripts.backfill_theme_pipeline_state._build_parser")
def test_main_non_dry_run_writes_checkpoint_and_report(
    mock_build_parser,
    mock_session_local,
    mock_service_cls,
    mock_save_checkpoint,
):
    parser = MagicMock()
    parser.parse_args.return_value = _args(dry_run=False)
    mock_build_parser.return_value = parser

    db = MagicMock()
    mock_session_local.return_value = db

    service = MagicMock()
    service.process_chunk.side_effect = [
        _chunk(rows_read=1, rows_written=2, conflicts=0, next_cursor=10),
        _chunk(rows_read=0, rows_written=0, conflicts=0, next_cursor=10),
    ]
    service.summary_counts.return_value = _summary_payload()
    mock_service_cls.return_value = service

    script.main()

    # One write for checkpoint progress, one for final report.
    assert mock_save_checkpoint.call_count == 2
    first_path = mock_save_checkpoint.call_args_list[0].args[0]
    second_path = mock_save_checkpoint.call_args_list[1].args[0]
    assert str(first_path) == "/tmp/backfill_checkpoint.json"
    assert str(second_path) == "/tmp/backfill_report.json"
