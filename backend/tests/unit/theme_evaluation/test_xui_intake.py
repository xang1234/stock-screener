from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.theme_evaluation.xui_intake import import_xui, read_required_lists


NOW = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)


def ingest(payloads, limit=100):
    return import_xui(payloads, captured_at=NOW, max_posts_per_source=limit,
                      requested_limit=5, mode='controlled')


def test_shared_post_keeps_both_sources(xui_payloads):
    result = ingest(xui_payloads)
    assert len(result.documents) == 1
    assert {m.source_id for m in result.documents[0].memberships} == {
        'x-list:1986290701492232693', 'x-list:1522014550211457024'}


def test_export_count_is_not_observed_count_and_selection_is_stable(xui_payloads):
    first = xui_payloads['1986290701492232693']
    row = first['items'][0]
    first['items'] = [dict(row, tweet_id=str(i), created_at=f'2026-09-01T0{i}:00:00Z')
                      for i in range(1, 9)]
    first['outcomes'][0].update(item_count=8, observed_ids=5)
    result = ingest(xui_payloads, 2)
    assert result.source_outcomes[0].returned_count == 8
    assert result.source_outcomes[0].observed_ids == 5
    assert result.selection['selected_ids']['x-list:1986290701492232693'] == ['8', '7']
    first['items'].reverse()
    reordered = ingest(xui_payloads, 2)
    assert reordered.selection == result.selection
    assert reordered.documents == result.documents


def test_conflicting_bodies_are_excluded(xui_payloads):
    xui_payloads['1522014550211457024']['items'][0]['text'] = 'Different body.'
    result = ingest(xui_payloads)
    assert result.documents == []
    assert result.selection['conflicting_post_ids'] == ['1']


def test_missing_required_source_is_explicit(xui_payloads):
    del xui_payloads['1522014550211457024']
    result = ingest(xui_payloads)
    assert result.source_outcomes[1].status == 'failed'
    assert result.source_outcomes[1].error_code == 'source_not_supplied'
    assert len(result.documents) == 1


def test_fallback_observation_does_not_become_publication(xui_payloads):
    for payload in xui_payloads.values():
        payload['items'][0].pop('observed_at')
        payload['items'][0]['created_at'] = '2026-01-01T00:00:00Z'
    doc = ingest(xui_payloads).documents[0]
    assert doc.retrieved_at == NOW
    assert doc.published_at.month == 1
    assert doc.source_metadata.observed_at_fallback


def test_unknown_row_source_rejected(xui_payloads):
    xui_payloads['1986290701492232693']['items'][0]['source_id'] = 'list:wrong'
    with pytest.raises(ValueError, match='source_mismatch'):
        ingest(xui_payloads)


def test_auth_failure_stops_second_read_and_preserves_safe_outcome(tmp_path):
    calls = []

    def run_command(args, **kwargs):
        calls.append(args)
        assert kwargs['timeout'] == 180
        assert '--login-policy' in args and args[args.index('--login-policy') + 1] == 'prompt'
        assert '--config-path' in args
        assert 'shell' not in kwargs
        return SimpleNamespace(returncode=2, stdout='{"error_code":"reauth_required"}')

    values = read_required_lists(wrapper=Path('/reader.py'), python=Path('/python'),
                                 xui_bin=Path('/bin/xui'), config=Path('/config.toml'),
                                 profile='default', limit=5, run_command=run_command)
    assert len(calls) == 1
    result = ingest(values)
    assert result.source_outcomes[0].status == 'reauth_required'
    assert result.source_outcomes[1].status == 'failed'


def test_failed_source_cannot_admit_stale_exported_rows(xui_payloads):
    xui_payloads['1986290701492232693']['outcomes'][0].update(ok=False, error='unavailable')
    result = ingest(xui_payloads)
    assert len(result.documents[0].memberships) == 1
    assert result.source_outcomes[0].selected_count == 0


def test_reader_quality_full_does_not_prove_complete_long_post(xui_payloads):
    for payload in xui_payloads.values():
        payload['items'][0]['text'] = 'A long post that stops mid sentence ' * 8
    assert ingest(xui_payloads).documents[0].capture_status == 'partial'


def test_auth_error_inside_outcome_also_stops_live_reads():
    calls = []
    def run_command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='{"items":[],"outcomes":[{"ok":false,"error":{"error_code":"reauth_required"}}]}')
    read_required_lists(wrapper=Path('/reader.py'), python=Path('/python'), xui_bin=Path('/bin/xui'),
                        config=Path('/config'), profile='default', limit=5, run_command=run_command)
    assert len(calls) == 1
