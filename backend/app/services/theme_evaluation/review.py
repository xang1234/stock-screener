"""Source-only review packets. No theme judgments or extraction calls."""

import csv
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from .bundle import canonical_bytes, sha256, validate_bundle
from .records import Bundle, REQUIRED_SOURCE_IDS


def coverage_summary(bundle: Bundle) -> dict:
    sources = {s.source_id: s for s in bundle.source_outcomes}
    complete = all(key in sources and sources[key].status == 'success'
                   and sources[key].selected_count > 0 for key in REQUIRED_SOURCE_IDS)
    posts = [d for d in bundle.documents if d.kind == 'post']
    screening = bundle.selection.get('reference_screening', {}).get('status', 'pending')
    gaps = sum(r.status not in {'resolved', 'not_article', 'skipped_noninvestment'}
               for r in bundle.followups)
    return dict(
        bundle_id=sha256(canonical_bytes(bundle.model_dump(mode='json'))), mode=bundle.mode,
        required_sources='complete' if complete else 'incomplete',
        unique_posts=len(posts), shared_posts=sum(len(d.memberships) > 1 for d in posts),
        articles=sum(d.kind == 'article' for d in bundle.documents),
        partial_documents=sum(d.capture_status == 'partial' for d in bundle.documents),
        unknown_language=sum(d.original_language is None for d in bundle.documents),
        reference_screening=screening, followup_gaps=gaps,
        article_followup='complete' if screening == 'complete' and not gaps else 'incomplete',
        evidence_review='pending', extraction_generation='awaiting_evidence_approval',
        ranking_evaluation='unavailable',
    )


def _text(value) -> str:
    return re.sub(r'([\\`*_{}\[\]()#+.!|>~-])', r'\\\1', html.escape(str(value)))


def _csv_value(value):
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, '')) for k in fields})


def _anchor(document_id: str) -> str:
    return 'evidence-' + sha256(document_id.encode())[:16]


def render_review(bundle: Bundle, output: Path) -> list[Path]:
    validate_bundle(bundle)
    summary = coverage_summary(bundle)
    output.mkdir(parents=True, exist_ok=False)
    dates = [d.published_at for d in bundle.documents if d.kind == 'post' and d.published_at]
    report = [
        '# Collected evidence — review before extraction', '',
        '**No extractions have been generated. User approval of this evidence version is pending.**', '',
        'Review the source coverage, captured text, article lookup outcomes, and missing context below.',
        'These posts are source claims, not verified facts or proposed theme labels.', '',
        f'Bundle: `{summary["bundle_id"]}`', '',
        f'{summary["unique_posts"]} unique posts; {summary["shared_posts"]} shared across lists; '
        f'{summary["articles"]} captured articles; {summary["followup_gaps"]} follow-up gaps.', '',
        f'Published range (UTC): {min(dates).isoformat()} to {max(dates).isoformat()}.' if dates else
        'Published range unavailable.', '',
        '## Source coverage', '',
        '| Source | Read | Returned | Selected | Reader observed IDs |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for source in bundle.source_outcomes:
        report.append(f'| {_text(source.source_id)} | {source.status} | {source.returned_count} | '
                      f'{source.selected_count} | {source.observed_ids if source.observed_ids is not None else "unknown"} |')
    report.extend(['', '## Review status', '',
                   f'- Required source access: {summary["required_sources"]}.',
                   f'- Article follow-up: {summary["article_followup"]}; {summary["followup_gaps"]} gaps.',
                   f'- Partial or potentially truncated documents: {summary["partial_documents"]}.',
                   f'- Documents without supplied language metadata: {summary["unknown_language"]}.',
                   '- User evidence approval: pending.', '- Extraction: waiting for evidence approval.',
                   '- Ranking comparison: unavailable at this stage.', '', '## Collection limitations', ''])
    report.extend('- ' + _text(value) for value in bundle.limitations)
    report.extend(['', 'Selection: ' + _text(bundle.selection.get('rule', 'controlled fixture')), '',
                   '## Article follow-ups', '', '| Post | Outcome | Reason |', '| --- | --- | --- |'])
    authors = {d.document_id: d.author or d.title for d in bundle.documents}
    for ref in bundle.followups:
        if ref.status in {'not_article', 'skipped_noninvestment'}:
            continue
        report.append(f'| [{_text(authors[ref.post_id])}](#{_anchor(ref.post_id)}) | {ref.status} | {_text(ref.screen_reason)} |')
    report.extend(['', 'All reference decisions, including attached media and subscription links, are in followups.csv.', ''])
    report.extend(['', '## Original evidence', ''])
    documents = sorted(bundle.documents, key=lambda d: (d.retrieved_at, d.published_at or d.retrieved_at,
                                                         d.document_id))
    for number, doc in enumerate(documents, 1):
        report.extend([
            f'<a id="{_anchor(doc.document_id)}"></a>', '', f'### {number}. {_text(doc.title)}', '',
            f'[Original source]({quote(doc.url, safe=":/?=&%#")}) · `{_text(doc.document_id)}`', '',
            f'Published: {_text(doc.published_at or "unknown")} · Observed/retrieved: {_text(doc.retrieved_at)}',
            f'Language: {_text(doc.original_language or "unknown")} · Capture: {doc.capture_status}',
            'Sources: ' + _text(', '.join(m.source_id for m in doc.memberships) or doc.publisher or 'unknown'), '',
        ])
        report.extend('> ' + _text(line) for line in doc.text.splitlines())
        for ref in (r for r in bundle.followups if r.post_id == doc.document_id):
            report.extend(['', f'Article/reference: **{ref.status}** — {_text(ref.reference_text)}',
                           'Lookup: ' + _text(ref.lookup_method or 'pending'),
                           'Outcome: ' + _text(ref.screen_reason)])
            if ref.article_id:
                report.append(f'[Captured article](#{_anchor(ref.article_id)})')
        for derivative in (d for d in bundle.derivatives if d.document_id == doc.document_id):
            report.extend(['', f'Translation ({_text(derivative.target_language)}): {derivative.status}', ''])
            report.extend('> ' + _text(line) for line in derivative.text.splitlines())
        report.append('')
    (output / 'evidence.md').write_text('\n'.join(report), encoding='utf-8')
    (output / 'coverage.json').write_bytes(canonical_bytes(summary))
    rows = []
    for doc in documents:
        row = doc.model_dump(mode='json')
        row['memberships'] = json.dumps(row['memberships'], ensure_ascii=False)
        rows.append(row)
    _write_csv(output / 'documents.csv', ['document_id', 'kind', 'author', 'title', 'url',
               'published_at', 'retrieved_at', 'original_language', 'capture_status', 'memberships',
               'text_sha256', 'text'], rows)
    fields = list(bundle.followups[0].model_fields) if bundle.followups else [
        'reference_id', 'post_id', 'candidate_url', 'investment_related', 'status', 'screen_reason',
        'article_id', 'attempted_at', 'lookup_method', 'error_code']
    rows = [r.model_dump(mode='json') for r in bundle.followups]
    for row in rows:
        row['evidence_urls'] = json.dumps(row['evidence_urls'], ensure_ascii=False)
    _write_csv(output / 'followups.csv', fields, rows)
    _write_csv(output / 'coverage.csv', ['metric', 'value'], [dict(metric=k, value=v) for k, v in summary.items()])
    return sorted(output.iterdir())
