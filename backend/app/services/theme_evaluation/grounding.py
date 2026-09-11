"""Freeze and validate bounded, ID-linked grounding for extraction inputs."""

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, TypeAdapter, ValidationError

from app.services.theme_grounding_context import (
    MAX_RELATED_CHARACTERS,
    POLICY_VERSION,
    GroundingContext,
    GroundingEvidence,
)

from .bundle import canonical_bytes
from .extraction_intake import ExtractionApproval, build_extraction_inputs
from .extraction_records import ExtractionInput
from .preparation_store import PreparationStore

_DATETIME = TypeAdapter(AwareDatetime)
_PACKET_KEYS = {
    "policy_version",
    "run_id",
    "bundle_id",
    "preparation_id",
    "assessment_id",
    "approval",
    "as_of",
    "contexts",
    "digest",
}
_INHERITED_PARENT_COMPANY_CONTEXT = "company_context_inherited_from_parent:"


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _packet_digest(value: dict[str, Any]) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _aware(value: datetime) -> datetime:
    try:
        return _DATETIME.validate_python(value)
    except ValidationError as exc:
        raise ValueError("grounding_as_of_invalid") from exc


def _inputs(value: list[ExtractionInput]) -> list[ExtractionInput]:
    try:
        inputs = [ExtractionInput.model_validate(item) for item in value]
    except (TypeError, ValidationError, ValueError) as exc:
        raise ValueError("grounding_inputs_invalid") from exc
    if len(inputs) != len({item.input_id for item in inputs}):
        raise ValueError("grounding_duplicate_input")
    return inputs


def _bound_run(run: dict[str, Any]) -> tuple[dict[str, Any], list[ExtractionInput]]:
    if not isinstance(run, dict) or not _is_digest(run.get("run_id")):
        raise ValueError("grounding_run_invalid")
    manifest = run.get("manifest")
    if not isinstance(manifest, dict):
        raise TypeError("grounding_run_manifest_invalid")
    bindings = ("bundle_id", "preparation_id", "assessment_id")
    if any(not _is_digest(manifest.get(key)) for key in bindings):
        raise ValueError("grounding_run_manifest_invalid")
    if not isinstance(manifest.get("approval"), dict):
        raise TypeError("grounding_run_approval_missing")
    return manifest, _inputs(run.get("inputs"))


def _rederive_admission(
    run: dict[str, Any], base: Path, store: PreparationStore
) -> list[ExtractionInput]:
    manifest, bound_inputs = _bound_run(run)
    try:
        approval = ExtractionApproval.model_validate(manifest["approval"])
        derived, derived_manifest = build_extraction_inputs(
            base,
            store,
            manifest["preparation_id"],
            manifest["assessment_id"],
            approval,
        )
    except (TypeError, ValidationError, ValueError) as exc:
        raise ValueError("grounding_admission_unverifiable") from exc
    if (
        manifest["bundle_id"] != base.name
        or approval.bundle_id != manifest["bundle_id"]
        or approval.preparation_id != manifest["preparation_id"]
        or approval.assessment_id != manifest["assessment_id"]
        or manifest.get("approval") != derived_manifest["approval"]
        or manifest.get("admitted_inputs") != derived_manifest["admitted_inputs"]
        or manifest.get("exclusions") != derived_manifest["exclusions"]
        or [item.model_dump(mode="json") for item in bound_inputs]
        != [item.model_dump(mode="json") for item in derived]
    ):
        raise ValueError("grounding_admission_mismatch")
    return bound_inputs


def _evidence(
    item: ExtractionInput, relation: str, remaining: int
) -> tuple[GroundingEvidence | None, str | None]:
    if remaining <= 0:
        return None, "grounding_related_evidence_omitted_budget:" + item.input_id
    full_text = item.text
    text = full_text[:remaining]
    truncated = len(text) != len(full_text)
    warnings = list(item.warnings)
    if truncated:
        warnings.append("grounding_related_text_truncated")
    return (
        GroundingEvidence(
            input_id=item.input_id,
            source_id=item.source_id,
            source_url=item.source_url,
            input_kind=item.input_kind,
            relation=relation,
            text=text,
            original_text_sha256=sha256(full_text.encode()).hexdigest(),
            text_sha256=sha256(text.encode()).hexdigest(),
            available_at=item.available_at,
            warnings=list(dict.fromkeys(warnings)),
            truncated=truncated,
        ),
        None,
    )


def _article_edges(bundle, post_id: str) -> set[tuple[str, str]]:
    return {
        (ref.reference_id, ref.article_id)
        for ref in bundle.followups
        if ref.status in {"resolved", "partial"}
        and ref.post_id == post_id
        and ref.article_id
    }


def _related_inputs(
    primary: ExtractionInput,
    inputs: list[ExtractionInput],
    *,
    base: Path,
) -> list[tuple[ExtractionInput, str]]:
    """Return only relationships persisted by the reviewed bundle/preparation path."""
    if primary.input_kind == "image_transcription":
        parents = [
            item
            for item in inputs
            if item.source_id == primary.source_id
            and item.input_kind in {"original", "translation"}
        ]
        return [(item, "parent_post") for item in parents]
    if primary.input_kind not in {"original", "translation"}:
        return []

    images = [
        item
        for item in inputs
        if item.source_id == primary.source_id
        and item.input_kind == "image_transcription"
    ]
    # Article inputs are keyed by followup reference ID.  The bundle is the only
    # place where that ID is explicitly connected to this post and article.
    from .bundle import load_bundle

    bundle = load_bundle(base)
    article_edges = _article_edges(bundle, primary.source_id)
    articles = [
        item
        for item in inputs
        if (
            any(item.source_id == article_id for _, article_id in article_edges)
            and item.source_kind == "article"
            and item.input_kind in {"original", "translation"}
        )
        or (
            item.input_kind == "article"
            and any(item.source_id == reference_id for reference_id, _ in article_edges)
        )
    ]
    return [
        *[(item, "attached_image") for item in images],
        *[(item, "linked_article") for item in articles],
    ]


def _validated_company_context(value: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) - {"companies", "warnings"}
        or not isinstance(value.get("companies", []), list)
        or not isinstance(value.get("warnings", []), list)
    ):
        raise ValueError("grounding_company_context_invalid")
    return {
        "companies": value.get("companies", []),
        "warnings": value.get("warnings", []),
    }


def _company_context_for(
    primary: ExtractionInput,
    inputs: list[ExtractionInput],
    company_contexts: dict[str, dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    """Use direct profile facts, or the admitted post directly behind an image.

    An image transcription is a derivative of its post, so sparse OCR may use
    that post's cached company facts.  This deliberately reads only the flat
    input-keyed contexts supplied for an admitted original/translation parent:
    it never follows a related-evidence edge or looks through other posts.
    """
    direct = _validated_company_context(company_contexts.get(primary.input_id, {}))
    if primary.input_kind != "image_transcription" or direct["companies"]:
        return direct

    parents = sorted(
        (
            item
            for item in inputs
            if item.source_id == primary.source_id
            and item.input_kind in {"original", "translation"}
            and item.available_at <= as_of
        ),
        key=lambda item: item.input_kind != "original",
    )
    for parent in parents:
        parent_context = _validated_company_context(
            company_contexts.get(parent.input_id, {})
        )
        if not parent_context["companies"]:
            continue
        return {
            "companies": parent_context["companies"],
            "warnings": list(
                dict.fromkeys(
                    [
                        *direct["warnings"],
                        *parent_context["warnings"],
                        _INHERITED_PARENT_COMPANY_CONTEXT + parent.input_id,
                    ]
                )
            ),
        }
    return direct


def _context_for(
    primary: ExtractionInput,
    inputs: list[ExtractionInput],
    company_context: dict[str, Any],
    as_of: datetime,
    *,
    base: Path,
) -> GroundingContext:
    if primary.available_at > as_of:
        raise ValueError("grounding_primary_not_yet_available")
    company_context = _validated_company_context(company_context)
    related = _related_inputs(primary, inputs, base=base)
    evidence, warnings = [], list(company_context.get("warnings", []))
    remaining = MAX_RELATED_CHARACTERS
    for item, relation in related:
        if item.available_at > as_of:
            raise ValueError("grounding_evidence_not_yet_available")
        record, warning = _evidence(item, relation, remaining)
        if record is None:
            warnings.append(warning)
            continue
        evidence.append(record)
        remaining -= len(record.text)
    try:
        return GroundingContext(
            primary_input_id=primary.input_id,
            # This pilot deliberately exposes the current, frozen context time.
            context_available_at=as_of,
            companies=company_context.get("companies", []),
            evidence=evidence,
            warnings=list(dict.fromkeys(warnings)),
        )
    except (TypeError, ValidationError, ValueError) as exc:
        raise ValueError("grounding_context_invalid") from exc


def prepare_grounding(
    run: dict,
    base: Path,
    store: PreparationStore,
    company_contexts: dict,
    as_of: datetime,
) -> dict:
    """Build an immutable, JSON-safe grounding packet for one frozen input run."""
    manifest, _ = _bound_run(run)
    inputs = _rederive_admission(run, base, store)
    as_of = _aware(as_of)
    if not isinstance(company_contexts, dict):
        raise TypeError("grounding_company_contexts_invalid")
    unknown = set(company_contexts) - {item.input_id for item in inputs}
    if unknown:
        raise ValueError("grounding_unknown_company_context")
    contexts = {
        item.input_id: _context_for(
            item,
            inputs,
            _company_context_for(item, inputs, company_contexts, as_of),
            as_of,
            base=base,
        ).model_dump(mode="json")
        for item in inputs
    }
    packet = {
        "policy_version": POLICY_VERSION,
        "run_id": run["run_id"],
        "bundle_id": manifest["bundle_id"],
        "preparation_id": manifest["preparation_id"],
        "assessment_id": manifest["assessment_id"],
        "approval": manifest["approval"],
        "as_of": as_of.isoformat(),
        "contexts": contexts,
    }
    packet = {**packet, "digest": _packet_digest(packet)}
    validate_grounding(
        packet, inputs, run_id=run["run_id"], manifest=manifest, base=base
    )
    return packet


def validate_grounding(
    packet: dict,
    inputs: list[ExtractionInput],
    *,
    run_id: str | None = None,
    manifest: dict,
    base: Path | None = None,
) -> dict[str, GroundingContext]:
    """Verify a packet before it enters a provider prompt or persistence layer."""
    if not isinstance(packet, dict) or set(packet) != _PACKET_KEYS:
        raise ValueError("grounding_packet_invalid")
    unsigned = {key: value for key, value in packet.items() if key != "digest"}
    if not isinstance(packet.get("digest"), str) or packet["digest"] != _packet_digest(
        unsigned
    ):
        raise ValueError("grounding_packet_digest_mismatch")
    if packet.get("policy_version") != POLICY_VERSION:
        raise ValueError("grounding_packet_policy_invalid")
    if run_id is not None and packet.get("run_id") != run_id:
        raise ValueError("grounding_packet_run_mismatch")
    if not isinstance(manifest, dict):
        raise TypeError("grounding_manifest_invalid")
    if any(
        packet[key] != manifest.get(key)
        for key in ("bundle_id", "preparation_id", "assessment_id")
    ) or packet["approval"] != manifest.get("approval"):
        raise ValueError("grounding_packet_manifest_mismatch")
    if (
        not _is_digest(packet.get("run_id"))
        or any(
            not _is_digest(packet.get(key))
            for key in ("bundle_id", "preparation_id", "assessment_id")
        )
        or not isinstance(packet.get("approval"), dict)
    ):
        raise ValueError("grounding_packet_invalid")
    as_of = _aware(packet.get("as_of"))
    bound_inputs = _inputs(inputs)
    known = {item.input_id: item for item in bound_inputs}
    contexts = packet.get("contexts")
    if not isinstance(contexts, dict) or set(contexts) != set(known):
        raise ValueError("grounding_packet_coverage_mismatch")
    parsed: dict[str, GroundingContext] = {}
    bundle = None
    for input_id, raw_context in contexts.items():
        try:
            context = GroundingContext.model_validate(raw_context)
        except (TypeError, ValidationError, ValueError) as exc:
            raise ValueError("grounding_context_invalid") from exc
        primary = known[input_id]
        if context.primary_input_id != input_id:
            raise ValueError("grounding_primary_mismatch")
        if primary.available_at > as_of:
            raise ValueError("grounding_primary_not_yet_available")
        if context.context_available_at != as_of:
            raise ValueError("grounding_context_time_mismatch")
        for evidence in context.evidence:
            actual = known.get(evidence.input_id)
            if actual is None:
                raise ValueError("grounding_unknown_evidence")
            if evidence.input_id == input_id:
                raise ValueError("grounding_self_evidence")
            if evidence.relation == "linked_article":
                if base is None:
                    raise ValueError("grounding_article_base_required")
                if base.name != packet["bundle_id"]:
                    raise ValueError("grounding_article_base_mismatch")
                if bundle is None:
                    from .bundle import load_bundle

                    bundle = load_bundle(base)
                article_edges = _article_edges(bundle, primary.source_id)
                if not (
                    any(
                        actual.source_id == article_id
                        for _, article_id in article_edges
                    )
                    or any(
                        actual.input_kind == "article"
                        and actual.source_id == reference_id
                        for reference_id, _ in article_edges
                    )
                ):
                    raise ValueError("grounding_article_relation_mismatch")
            relation_valid = (
                (
                    evidence.relation == "attached_image"
                    and primary.input_kind in {"original", "translation"}
                    and actual.input_kind == "image_transcription"
                    and actual.source_id == primary.source_id
                )
                or (
                    evidence.relation == "parent_post"
                    and primary.input_kind == "image_transcription"
                    and actual.input_kind in {"original", "translation"}
                    and actual.source_id == primary.source_id
                )
                or (
                    evidence.relation == "linked_article"
                    and primary.input_kind in {"original", "translation"}
                    and (
                        actual.input_kind == "article"
                        or (
                            actual.source_kind == "article"
                            and actual.input_kind in {"original", "translation"}
                        )
                    )
                )
            )
            expected_warnings = [*actual.warnings]
            if evidence.truncated:
                expected_warnings.append("grounding_related_text_truncated")
            if (
                not relation_valid
                or evidence.source_id != actual.source_id
                or evidence.source_url != actual.source_url
                or evidence.input_kind != actual.input_kind
                or evidence.available_at != actual.available_at
                or evidence.available_at > as_of
                or evidence.original_text_sha256
                != sha256(actual.text.encode()).hexdigest()
                or evidence.text != actual.text[: len(evidence.text)]
                or (not evidence.truncated and evidence.text != actual.text)
                or evidence.warnings != list(dict.fromkeys(expected_warnings))
            ):
                raise ValueError("grounding_evidence_mismatch")
        parsed[input_id] = context
    for input_id, context in parsed.items():
        inherited = [
            warning.removeprefix(_INHERITED_PARENT_COMPANY_CONTEXT)
            for warning in context.warnings
            if warning.startswith(_INHERITED_PARENT_COMPANY_CONTEXT)
        ]
        if not inherited:
            continue
        if len(inherited) != 1:
            raise ValueError("grounding_inherited_company_context_mismatch")
        parent = known.get(inherited[0])
        if (
            parent is None
            or known[input_id].input_kind != "image_transcription"
            or parent.input_kind not in {"original", "translation"}
            or parent.source_id != known[input_id].source_id
            or parent.available_at > as_of
            or context.companies != parsed[parent.input_id].companies
        ):
            raise ValueError("grounding_inherited_company_context_mismatch")
    return parsed
