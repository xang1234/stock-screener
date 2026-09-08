"""Read-only X capture and deterministic local selection."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, TypeAdapter

from .bundle import canonical_bytes, sha256, validate_bundle
from .records import (
    REQUIRED_LIST_IDS,
    Bundle,
    Document,
    Membership,
    SourceMetadata,
    SourceOutcome,
)

_DATE = TypeAdapter(AwareDatetime)


def _date(value):
    return _DATE.validate_python(value) if value else None


def read_required_lists(*, wrapper: Path, python: Path, xui_bin: Path,
                        config: Path, profile: str, limit: int,
                        run_command=subprocess.run) -> dict[str, dict]:
    if limit <= 0:
        raise ValueError('limit_must_be_positive')
    env = dict(os.environ, PATH=str(xui_bin.parent) + os.pathsep + os.environ.get('PATH', ''))
    results = {}
    for list_id in REQUIRED_LIST_IDS:
        started = datetime.now(timezone.utc)
        args = [str(python), str(wrapper), 'list', list_id, '--config-path', str(config),
                '--profile', profile, '--login-policy', 'prompt', '--limit', str(limit)]
        try:
            response = run_command(args, env=env, capture_output=True, text=True, timeout=180)
            payload = json.loads(response.stdout)
            if not isinstance(payload, dict):
                raise ValueError('reader_object_required')  # noqa: TRY004 - handled as invalid reader data
            for outcome in payload.get('outcomes', []):
                error = outcome.get('error')
                if error == 'reauth_required' or (isinstance(error, dict)
                                                 and error.get('error_code') == 'reauth_required'):
                    payload['error_code'] = 'reauth_required'
            if response.returncode and not payload.get('error_code'):
                payload['error_code'] = 'reader_failed'
            raw_hash = sha256(response.stdout.encode())
        except subprocess.TimeoutExpired:
            payload, raw_hash = {'error_code': 'reader_timeout'}, None
        except (OSError, ValueError):
            payload, raw_hash = {'error_code': 'reader_response_unavailable'}, None
        payload['_capture'] = {
            'started_at': started.isoformat(),
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'raw_sha256': raw_hash,
            'raw_json': response.stdout if raw_hash is not None else None,
        }
        results[list_id] = payload
        if payload.get('error_code') == 'reauth_required':
            break
    return results


def _sort_row(row):
    published = _date(row.get('created_at'))
    return (-(published.timestamp()) if published else float('inf'), str(row['tweet_id']))


def _capture_status(row):
    # A complete post preview is not the full body of a native X Article.
    if row.get('is_article') or row.get('incomplete_text_reasons'):
        return 'partial'
    if 'text_complete' in row:
        return 'full' if row['text_complete'] is True else 'partial'
    # Legacy exports have no explicit completeness signal.
    return ('full' if row.get('quality_tier') == 'full' and len(row['text']) < 270
            and not row['text'].rstrip().endswith(('…', '...')) else 'partial')


def import_xui(payloads: dict[str, dict], *, captured_at: datetime,
               max_posts_per_source: int, requested_limit: int | None = None,
               mode: str = 'observed_capture', raw_hashes: dict[str, str] | None = None) -> Bundle:
    captured_at = _DATE.validate_python(captured_at)
    if max_posts_per_source <= 0 or set(payloads) - set(REQUIRED_LIST_IDS):
        raise ValueError('invalid_source_selection')
    outcomes, candidates, all_rows = [], {}, {}
    for list_id in REQUIRED_LIST_IDS:
        source_id = 'x-list:' + list_id
        payload = payloads.get(list_id)
        capture = (payload or {}).get('_capture', {})
        matching = [o for o in (payload or {}).get('outcomes', [])
                    if o.get('source_id') == 'list:' + list_id]
        if len(matching) > 1:
            raise ValueError('duplicate_reader_outcome')
        outcome = matching[0] if matching else {}
        error = (payload or {}).get('error_code')
        ok = outcome.get('ok') is True and not error
        rows = (payload or {}).get('items', [])
        for row in rows:
            if row.get('source_id') != 'list:' + list_id:
                raise ValueError('source_mismatch')
            if not row.get('tweet_id') or not isinstance(row.get('text'), str):
                raise ValueError('invalid_reader_post')
        if not ok and not error:
            error = 'source_not_supplied' if payload is None else 'reader_failed'
        raw_hash = (raw_hashes or {}).get(list_id) or capture.get('raw_sha256')
        if raw_hash is None and payload is not None:
            raw_hash = sha256(canonical_bytes({k: v for k, v in payload.items() if k != '_capture'}))
        outcomes.append(SourceOutcome(
            source_id=source_id, required=True,
            status='success' if ok else ('reauth_required' if error == 'reauth_required' else 'failed'),
            requested_limit=requested_limit, returned_count=len(rows), selected_count=0,
            observed_ids=outcome.get('observed_ids'),
            captured_at=_date(capture.get('finished_at')) or captured_at,
            error_code=error, raw_sha256=raw_hash, source_config_revision=None,
        ))
        candidates[source_id] = rows if ok else []
        for row in candidates[source_id]:
            all_rows.setdefault(str(row['tweet_id']), []).append(row)
    conflicts = sorted(key for key, rows in all_rows.items()
                       if len({row['text'] for row in rows}) > 1)
    selected, documents = {}, {}
    for source in outcomes:
        unique = {}
        for row in sorted(candidates[source.source_id], key=lambda r: (
            *_sort_row(r), canonical_bytes(r)
        )):
            unique.setdefault(str(row['tweet_id']), row)
        rows = [row for key, row in unique.items() if key not in conflicts][:max_posts_per_source]
        selected[source.source_id] = [str(row['tweet_id']) for row in rows]
        source.selected_count = len(rows)
        for row in rows:
            key = 'post:' + str(row['tweet_id'])
            observed = _date(row.get('observed_at')) or source.captured_at
            member = Membership(source_id=source.source_id, observed_at=observed)
            if key in documents:
                documents[key].memberships.append(member)
                documents[key].retrieved_at = min(documents[key].retrieved_at, observed)
                continue
            meta_fields = ('extraction_method', 'extraction_version', 'quality_tier',
                           'quality_score', 'quote_tweet_id', 'is_reply', 'is_article', 'image_urls',
                           'image_captions', 'article_urls', 'reply_tweet_id', 'text_source',
                           'text_complete', 'incomplete_text_reasons')
            metadata = {k: row[k] for k in meta_fields if row.get(k) is not None}
            metadata['observed_at_fallback'] = not bool(row.get('observed_at'))
            author = row.get('author_handle') or row.get('author')
            documents[key] = Document(
                document_id=key, kind='post', title=f'Post by {author or "unknown author"}',
                text=row['text'], url=row['tweet_url'], author=author, publisher='X',
                published_at=_date(row.get('created_at')), updated_at=None,
                retrieved_at=observed, original_language=row.get('lang') or row.get('language'),
                memberships=[member], capture_status=_capture_status(row),
                text_sha256=sha256(row['text'].encode()), reference_only=False,
                source_metadata=SourceMetadata.model_validate(metadata),
            )
    result = Bundle(
        schema_version=1, mode=mode, availability_rule=None, source_outcomes=outcomes,
        documents=sorted(documents.values(), key=lambda d: (d.retrieved_at, d.document_id)),
        derivatives=[], followups=[], extractions=[], labels=[],
        selection={
            'rule': 'recent-per-source-v1: publication descending, tweet ID ascending; unknown last',
            'max_posts_per_source': max_posts_per_source, 'selected_ids': selected,
            'conflicting_post_ids': conflicts, 'original_unique_post_count': len(all_rows),
        },
        limitations=['Recent captured sample, not a complete historical archive.',
                     'Older publication dates do not establish historical observation.',
                     'Language metadata is unknown where the reader does not supply it.',
                     ('Explicit reader completeness is preserved; unknown or incomplete text and native '
                      'Article previews are partial. Legacy exports use conservative length/ellipsis checks.'),
                     'Evidence requires user review before extraction.'],
    )
    validate_bundle(result)
    return result
