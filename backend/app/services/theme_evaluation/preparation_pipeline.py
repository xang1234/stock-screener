"""Explicit preparation stages; no extraction or production database access."""

from pathlib import Path

from pydantic import field_validator

from .article_recovery import ArticleRecovery, parse_article
from .bundle import sha256
from .image_preparation import prepare_image, validate_image
from .multilingual_preparation import prepare_text
from .preparation_records import (
    Handoff,
    PreparationBinding,
    PreparationManifest,
    PreparationRequest,
    PreparationResult,
)
from .preparation_store import PreparationStore, deduplicate_bindings, validate_handoff
from .public_fetch import fetch_public
from .records import URL, Record


def _request(stage, digest, client=None, **options):
    return PreparationRequest(
        stage=stage,
        input_sha256=digest,
        provider=client.provider if client else "local",
        model=client.model if client else None,
        policy_version=client.policy_version if client else stage + "-v1",
        options=options,
    )


def _binding(kind, doc, result_id, source_id=None, parent=None, locator=None):
    return PreparationBinding(
        source_kind=kind,
        source_id=source_id or doc.document_id,
        source_text_sha256=doc.text_sha256,
        result_id=result_id,
        parent_result_id=parent,
        input_locator=locator,
    )


def _text_result(store, text, language, translator):
    request = _request(
        "text",
        sha256(text.encode()),
        translator,
        language=language,
        target_language="en",
        max_chars=4000,
    )
    cached = store.cached(request)
    if cached:
        return cached[0]
    prepared = prepare_text(text, language=language, translator=translator)
    return store.save_result(
        PreparationResult(
            request=request,
            status=prepared.status,
            payload=prepared.model_dump(mode="json"),
            warnings=prepared.warnings,
        )
    )


def _image_result(store, location, is_local, vision, fetcher, allow_network):
    assets = []
    request = _request("image", sha256(location.encode()), vision)
    try:
        if is_local:
            with Path(location).open("rb") as handle:
                data = handle.read(10 * 1024 * 1024 + 1)
        elif allow_network:
            data = fetcher(location, max_bytes=10 * 1024 * 1024).body
        else:
            return store.save_result(
                PreparationResult(
                    request=request,
                    status="unavailable",
                    warnings=["network_disabled"],
                    source_url=location,
                )
            )
        metadata = validate_image(data)
        assets = [store.save_asset(data)]
        request = _request("image", metadata["sha256"], vision)
        cached = store.cached(request)
        if cached:
            return cached[0]
        if vision is None:
            return store.save_result(
                PreparationResult(
                    request=request,
                    status="unavailable",
                    assets=assets,
                    warnings=["vision_provider_unavailable"],
                    source_url=None if is_local else location,
                )
            )
        prepared = prepare_image(data, model_client=vision)
        result = PreparationResult(
            request=request,
            status="needs_review" if prepared.uncertainties else "success",
            payload=prepared.model_dump(mode="json"),
            warnings=prepared.uncertainties,
            assets=assets,
            source_url=None if is_local else location,
        )
    except (ValueError, OSError, RuntimeError):
        result = PreparationResult(
            request=request,
            status="unavailable",
            assets=assets,
            warnings=["image_validation_or_processing_failed"],
            source_url=None if is_local else location,
        )
    return store.save_result(result)


def _article_result(store, url, fetcher, allow_network):
    request = _request("article", sha256(url.encode()), destination_url=url)
    if not allow_network:
        return store.save_result(
            PreparationResult(
                request=request,
                status="unavailable",
                warnings=["network_disabled", "browser_followup_required"],
                source_url=url,
            )
        )
    try:
        response = fetcher(url)
        if response.content_type and not any(
            kind in response.content_type.lower() for kind in ("html", "xhtml")
        ):
            raise ValueError("unsupported_article_type")
        raw_id = store.save_asset(response.body)
        article = parse_article(response.body, response.final_url)
        request = _request("article", raw_id, destination_url=url)
        return store.save_result(
            PreparationResult(
                request=request,
                status="partial" if article.text else "unavailable",
                payload=article.model_dump(mode="json"),
                assets=[raw_id],
                warnings=article.warnings + ["browser_followup_required"],
                source_url=response.final_url,
            )
        )
    except (ValueError, OSError):
        return store.save_result(
            PreparationResult(
                request=request,
                status="unavailable",
                warnings=["article_fetch_or_parse_failed", "browser_followup_required"],
                source_url=url,
            )
        )


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
) -> str:
    bundle = validate_handoff(base, handoff)
    if not stages or not set(stages).issubset({"article", "text", "image"}):
        raise ValueError("invalid_preparation_stages")
    if len(bundle.documents) > max_documents or max_images < 0 or max_documents < 1:
        raise ValueError("preparation_limit_exceeded")
    bindings = []
    if prior_id:
        prior = store.load(base, prior_id)
        if prior.handoff != handoff:
            raise ValueError("prior_handoff_mismatch")
        bindings.extend(prior.bindings)
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
        for ref in bundle.followups:
            if ref.investment_related == "no" or ref.status in {
                "not_article",
                "skipped_noninvestment",
            }:
                continue
            url = handoff.references.get(ref.reference_id, ref.candidate_url)
            if url:
                rid = _article_result(store, url, fetcher, allow_network)
            else:
                rid = store.save_result(
                    PreparationResult(
                        request=_request(
                            "article", sha256(ref.reference_text.encode())
                        ),
                        status="unavailable",
                        warnings=[
                            "article_destination_unknown",
                            "browser_followup_required",
                        ],
                    )
                )
            bindings.append(
                _binding("reference", documents[ref.post_id], rid, ref.reference_id)
            )
    if "image" in stages:
        for doc, location, is_local in images:
            rid = _image_result(
                store, location, is_local, vision, fetcher, allow_network
            )
            bindings.append(_binding("document", doc, rid, locator=location))
    if "text" in stages:
        for doc in bundle.documents:
            override = handoff.documents.get(doc.document_id)
            language = (
                override.language
                if override and override.language
                else doc.original_language
            )
            rid = _text_result(store, doc.text, language, translator)
            bindings.append(_binding("document", doc, rid))
        references = {r.reference_id: r for r in bundle.followups}
        for binding in list(bindings):
            result = store.load_result(binding.result_id)
            text = (
                result.payload.get("text")
                if result.request.stage == "article"
                else result.payload.get("transcription")
            )
            if not isinstance(text, str) or not text.strip():
                continue
            doc = (
                documents[binding.source_id]
                if binding.source_kind == "document"
                else documents[references[binding.source_id].post_id]
            )
            rid = _text_result(store, text, None, translator)
            bindings.append(
                _binding(
                    binding.source_kind, doc, rid, binding.source_id, binding.result_id
                )
            )
    return store.seal(
        base,
        PreparationManifest(
            bundle_id=base.name,
            handoff=handoff,
            bindings=deduplicate_bindings(bindings),
        ),
    )


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
        if sha256(row.article.text.encode()) != row.article.response_sha256:
            raise ValueError("article_import_text_hash_mismatch")
    bindings = []
    if prior_id:
        prior = store.load(base, prior_id)
        if prior.handoff != handoff:
            raise ValueError("prior_handoff_mismatch")
        bindings.extend(prior.bindings)
    for row in imports:
        asset = store.save_asset(row.article.text.encode())
        request = _request("article", asset, destination_url=row.destination_url)
        rid = store.save_result(
            PreparationResult(
                request=request,
                status="success" if row.article.capture_status == "full" else "partial",
                payload=row.article.model_dump(mode="json"),
                warnings=row.article.warnings,
                assets=[asset],
                source_url=row.article.final_url,
                created_at=row.article.retrieved_at,
            )
        )
        bindings.append(
            _binding(
                "reference", docs[refs[row.reference_id].post_id], rid, row.reference_id
            )
        )
    return store.seal(
        base,
        PreparationManifest(
            bundle_id=base.name,
            handoff=handoff,
            bindings=deduplicate_bindings(bindings),
        ),
    )
