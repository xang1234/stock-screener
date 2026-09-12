"""Explicit preparation stages; no extraction or production database access."""

from pathlib import Path
from time import monotonic

from pydantic import field_validator

from .article_recovery import ArticleRecovery
from .bundle import IntegrityError, sha256
from .image_stage import ImageStage
from .multilingual_preparation import prepare_text
from .multilingual_v2 import preparation_cache_policy, prepare_text_v2
from .preparation_article_stage import prepare_articles
from .preparation_records import (
    Handoff,
    PreparationBinding,
)
from .preparation_results import (
    ArticleRequest,
    ArticleResult,
    ImageRequest,
    TextRequest,
    TextResult,
)
from .preparation_run import CountedTranslator, save_run
from .preparation_state import PreparationState
from .preparation_store import PreparationStore, validate_handoff
from .public_fetch import fetch_public
from .records import URL, Record
from .translation_selection import select_translation
from .translation_selection_store import (
    save_translation_selection,
    selection_for_preparation,
    selection_record,
)
from .xui_translation import captured_translation_result


def _request(stage, digest, client=None, **options):
    request_type = {
        "article": ArticleRequest,
        "text": TextRequest,
        "image": ImageRequest,
    }[stage]
    return request_type(
        input_sha256=digest,
        provider=client.provider if client else "local",
        model=client.model if client else None,
        policy_version=client.policy_version if client else stage + "-v1",
        **options,
    )


def _binding(stage, kind, doc, result_id, source_id=None, parent=None, locator=None):
    return PreparationBinding(
        stage=stage,
        source_kind=kind,
        source_id=source_id or doc.document_id,
        source_text_sha256=doc.text_sha256,
        result_id=result_id,
        parent_result_id=parent,
        input_locator=locator,
    )


def _text_result(store, text, language, translator, text_policy="v1"):
    request = _request(
        "text",
        sha256(text.encode()),
        translator,
        language=language,
        target_language="en",
        max_chars=4000,
    )
    if text_policy == "v2":
        request = request.model_copy(
            update={"policy_version": preparation_cache_policy(translator)}
        )
    cached = store.cached(request)
    if cached:
        return cached[0]
    text_preparer = prepare_text_v2 if text_policy == "v2" else prepare_text
    prepared = text_preparer(text, language=language, translator=translator)
    return store.save_result(TextResult(request=request, payload=prepared))


def _document_translation(state, store, doc, language, translator, text_policy="v1"):
    captured = captured_translation_result(doc)
    x_id = store.save_result(captured) if captured else None
    initial = select_translation(doc.text, language, captured, None)
    kimi_id = None
    kimi = None
    if (
        initial.selected_candidate is None
        and initial.assessment.disposition == "fallback"
    ):
        kimi_id = _text_result(store, doc.text, language, translator, text_policy)
        kimi = store.load_result(kimi_id)
    decision = select_translation(doc.text, language, captured, kimi)
    for result_id in (x_id, kimi_id):
        if result_id is not None:
            state.record(_binding("text", "document", doc, result_id))
    return selection_record(doc.document_id, x_id, kimi_id, decision)


def prepare(
    base: Path,
    store: PreparationStore,
    handoff: Handoff,
    *,
    stages,
    vision=None,
    translator=None,
    fetcher=fetch_public,
    allow_network=False,
    prior_id=None,
    max_images=100,
    max_documents=500,
    text_policy="v1",
    run_summary=None,
) -> str:
    bundle = validate_handoff(base, handoff)
    if text_policy not in {"v1", "v2"}:
        raise ValueError("invalid_text_policy")
    started = monotonic()
    stats = {}
    translator = CountedTranslator(translator) if translator is not None else None
    if not stages or not set(stages).issubset({"article", "text", "image"}):
        raise ValueError("invalid_preparation_stages")
    if len(bundle.documents) > max_documents or max_images < 0 or max_documents < 1:
        raise ValueError("preparation_limit_exceeded")
    state = PreparationState(base, store, handoff, prior_id)
    documents = {d.document_id: d for d in bundle.documents}
    images = []
    for doc in bundle.documents:
        override = handoff.documents.get(doc.document_id)
        urls = list(
            dict.fromkeys(
                doc.source_metadata.image_urls
                + (override.image_urls if override else [])
            )
        )
        files = list(dict.fromkeys(override.local_images)) if override else []
        images.extend((doc, url, False) for url in urls)
        images.extend((doc, file, True) for file in files)
    if "image" in stages and len(images) > max_images:
        raise ValueError("image_limit_exceeded")
    if "article" in stages:
        stats.update(
            prepare_articles(
                bundle,
                store,
                state,
                handoff,
                fetcher=fetcher,
                allow_network=allow_network,
            )
        )
    if "image" in stages:
        image_stage = ImageStage(
            store, vision=vision, fetcher=fetcher, allow_network=allow_network
        )
        for doc, location, is_local in images:
            outcome = image_stage.process(location, is_local=is_local)
            for rid in outcome.attempt_ids:
                state.record(_binding("image", "document", doc, rid, locator=location))
        stats.update(
            image_model_request_count=image_stage.model_request_count,
            image_download_request_count=image_stage.download_request_count,
        )
    if "text" in stages:
        translation_decisions = []
        for doc in bundle.documents:
            override = handoff.documents.get(doc.document_id)
            language = (
                override.language
                if override and override.language
                else doc.original_language
            )
            translation_decisions.append(
                _document_translation(
                    state, store, doc, language, translator, text_policy
                )
            )
        references = {r.reference_id: r for r in bundle.followups}
        for binding in state.manifest.current_bindings:
            result = store.load_result(binding.result_id)
            if (
                result.stage not in {"article", "image"}
                or not result.source_text.strip()
            ):
                continue
            text = result.source_text
            doc = (
                documents[binding.source_id]
                if binding.source_kind == "document"
                else documents[references[binding.source_id].post_id]
            )
            rid = _text_result(store, text, None, translator, text_policy)
            state.record(
                _binding(
                    "text",
                    binding.source_kind,
                    doc,
                    rid,
                    binding.source_id,
                    binding.result_id,
                    locator=binding.input_locator,
                )
            )
    preparation_id = store.seal(base, state.manifest)
    if "text" not in stages:
        translation_decisions = [] if not prior_id else None
        if prior_id:
            try:
                translation_decisions = list(
                    selection_for_preparation(base, store, prior_id)[1].decisions
                )
            except IntegrityError as exc:
                if str(exc) != "missing_translation_selection":
                    raise
    if translation_decisions is not None:
        stats["selection_id"] = save_translation_selection(
            base, store, preparation_id, translation_decisions
        )
    stats["translation_model_request_count"] = translator.calls if translator else 0
    stats["elapsed_seconds"] = round(monotonic() - started, 6)
    stats["text_policy"] = text_policy
    stats["run_id"] = save_run(store, base.name, preparation_id, stats)
    if run_summary is not None:
        run_summary.update(stats)
    return preparation_id


class ArticleImport(Record):
    reference_id: str
    destination_url: URL
    article: ArticleRecovery

    @field_validator("article")
    @classmethod
    def actual_capture_time_required(cls, article):
        if "retrieved_at" not in article.model_fields_set:
            raise ValueError("article_import_capture_time_required")
        return article


def import_articles(base, store, handoff, records, *, prior_id=None):
    bundle = validate_handoff(base, handoff)
    refs = {r.reference_id: r for r in bundle.followups}
    docs = {d.document_id: d for d in bundle.documents}
    imports = [ArticleImport.model_validate(row) for row in records]
    # Validate every entry before writing anything.
    for row in imports:
        ref = refs.get(row.reference_id)
        if ref is None or row.destination_url != handoff.references.get(
            row.reference_id, ref.candidate_url
        ):
            raise ValueError("article_import_reference_mismatch")
        if (
            row.article.method != "browser_import"
            or not row.article.match_basis
            or not row.article.text.strip()
        ):
            raise ValueError("article_import_provenance_required")
        if (
            row.article.capture_status == "full"
            and not (row.article.completeness_basis or "").strip()
        ):
            raise ValueError("article_import_completeness_basis_required")
        if row.article.body_sha256 is not None and row.article.body_sha256 != sha256(
            row.article.text.encode()
        ):
            raise ValueError("article_import_body_hash_mismatch")
        if sha256(row.article.text.encode()) != row.article.response_sha256:
            raise ValueError("article_import_text_hash_mismatch")
    state = PreparationState(base, store, handoff, prior_id)
    for row in imports:
        asset = store.save_asset(row.article.text.encode())
        request = _request("article", asset, destination_url=row.destination_url)
        request = request.model_copy(update={"policy_version": "article-v2"})
        rid = store.save_result(
            ArticleResult(
                request=request,
                payload=row.article,
                assets=[asset],
                source_url=row.article.final_url,
                created_at=row.article.retrieved_at,
            )
        )
        state.record(
            _binding(
                "article",
                "reference",
                docs[refs[row.reference_id].post_id],
                rid,
                row.reference_id,
            )
        )
    return store.seal(base, state.manifest)
