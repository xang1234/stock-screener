import csv

from app.services.theme_evaluation.records import Bundle
from app.services.theme_evaluation.review import coverage_summary, render_review


def test_unreviewed_empty_bundle_does_not_claim_success(bundle):
    summary = coverage_summary(Bundle.model_validate(bundle(documents=[])))
    assert summary['required_sources'] == 'incomplete'
    assert summary['reference_screening'] == 'pending'
    assert summary['evidence_review'] == 'pending'
    assert summary['extraction_generation'] == 'awaiting_evidence_approval'
    assert summary['ranking_evaluation'] == 'unavailable'


def test_report_preserves_original_text_and_escapes_spreadsheet_formulas(tmp_path, bundle, document):
    text = '  =HYPERLINK("bad")\n# Forged heading\n<script>alert(1)</script>'
    result = Bundle.model_validate(bundle(documents=[document(text=text)]))
    files = render_review(result, tmp_path / 'report')
    with (tmp_path / 'report' / 'documents.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert row['text'] == "'" + text
    report = (tmp_path / 'report' / 'evidence.md').read_text()
    assert '<script>' not in report
    assert '\n# Forged heading' not in report
    assert 'No extractions have been generated' in report
    assert len(files) >= 4


def test_duplicate_membership_counts_one_post(xui_payloads):
    from datetime import datetime, timezone
    from app.services.theme_evaluation.xui_intake import import_xui
    result = import_xui(xui_payloads, captured_at=datetime.now(timezone.utc), max_posts_per_source=5)
    summary = coverage_summary(result)
    assert summary['unique_posts'] == 1
    assert summary['shared_posts'] == 1
    assert summary['required_sources'] == 'complete'
    assert summary['evidence_review'] == 'pending'
