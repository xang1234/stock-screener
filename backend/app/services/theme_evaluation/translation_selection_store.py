"""Immutable, content-addressed persistence for translation selections."""

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, model_validator

from .bundle import IntegrityError, canonical_bytes, load_bundle, sha256
from .preparation_store import PreparationStore, _atomic
from .records import SHA, Record
from .translation_quality import (
    QUALITY_POLICY_V1,
    QUALITY_POLICY_V2,
    QualityIssue,
    QualityPolicy,
)
from .translation_selection import TranslationSelection, select_translation
from .xui_translation import captured_translation_result


class TranslationSelectionRecord(Record):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    document_id: str
    source_text_sha256: SHA
    x_result_id: SHA | None = None
    kimi_result_id: SHA | None = None
    selected_result_id: SHA | None = None
    eligible: bool
    disposition: Literal["use", "fallback", "review"]
    issues: tuple[QualityIssue, ...]

    @model_validator(mode="after")
    def consistent_decision(self):
        candidates = {
            value for value in (self.x_result_id, self.kimi_result_id) if value
        }
        if (
            self.selected_result_id is not None
            and self.selected_result_id not in candidates
        ):
            raise ValueError("selection_decision_inconsistent")
        if self.disposition == "use":
            valid = self.eligible and self.selected_result_id is not None
        elif self.disposition == "review":
            valid = not self.eligible and self.selected_result_id is not None
        else:
            valid = not self.eligible and self.selected_result_id is None
        if not valid:
            raise ValueError("selection_decision_inconsistent")
        return self


class TranslationSelectionSidecar(Record):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal[1] = 1
    policy_version: Literal["translation-quality-v1", "translation-quality-v2"] = (
        QUALITY_POLICY_V2
    )
    bundle_id: SHA
    preparation_id: SHA
    decisions: tuple[TranslationSelectionRecord, ...]

    @model_validator(mode="after")
    def unique_documents(self):
        if len({decision.document_id for decision in self.decisions}) != len(
            self.decisions
        ):
            raise ValueError("duplicate_translation_selection")
        return self


def selection_record(
    document_id: str,
    x_result_id: str | None,
    kimi_result_id: str | None,
    selection: TranslationSelection,
) -> TranslationSelectionRecord:
    selected_result_id = {
        "x": x_result_id,
        "kimi": kimi_result_id,
        None: None,
    }[selection.selected_candidate]
    if selection.selected_candidate is not None and selected_result_id is None:
        raise ValueError("selection_candidate_id_missing")
    return TranslationSelectionRecord(
        document_id=document_id,
        source_text_sha256=selection.source_text_sha256,
        x_result_id=x_result_id,
        kimi_result_id=kimi_result_id,
        selected_result_id=selected_result_id,
        eligible=selection.eligible,
        disposition=selection.assessment.disposition,
        issues=selection.issues,
    )


def _validate_sidecar(
    base: Path,
    store: PreparationStore,
    sidecar: TranslationSelectionSidecar,
) -> None:
    manifest = store.load(base, sidecar.preparation_id)
    if sidecar.bundle_id != base.name or manifest.bundle_id != sidecar.bundle_id:
        raise ValueError("selection_bundle_mismatch")
    documents = {
        document.document_id: document for document in load_bundle(base).documents
    }
    bound = {
        (binding.source_id, binding.source_text_sha256, binding.result_id)
        for binding in manifest.bindings
        if binding.stage == "text"
        and binding.source_kind == "document"
        and binding.parent_result_id is None
    }
    expected_documents = {document_id for document_id, _, _ in bound}
    if {record.document_id for record in sidecar.decisions} != expected_documents:
        raise ValueError("selection_decision_coverage_mismatch")
    for record in sidecar.decisions:
        document = documents.get(record.document_id)
        if document is None or document.text_sha256 != record.source_text_sha256:
            raise ValueError("selection_source_mismatch")
        results = []
        for result_id in (record.x_result_id, record.kimi_result_id):
            if result_id is None:
                results.append(None)
                continue
            if (record.document_id, record.source_text_sha256, result_id) not in bound:
                raise ValueError("selection_candidate_not_bound")
            result = store.load_result(result_id)
            if result.stage != "text":
                raise ValueError("selection_candidate_stage_mismatch")
            results.append(result)
        x_result, kimi_result = results
        if x_result is not None and (
            x_result.request.provider != "X translation"
            or x_result.request.method != "translation_import"
        ):
            raise ValueError("selection_x_candidate_mismatch")
        expected_x = captured_translation_result(document)
        if x_result is not None and (
            expected_x is None
            or x_result.model_dump(mode="json") != expected_x.model_dump(mode="json")
        ):
            raise ValueError("selection_x_capture_mismatch")
        if x_result is None and expected_x is not None:
            raise ValueError("selection_x_capture_missing")
        if kimi_result is not None and (
            kimi_result.request.provider == "X translation"
            or kimi_result.request.method != "prepare"
        ):
            raise ValueError("selection_kimi_candidate_mismatch")
        language = (
            manifest.handoff.documents.get(record.document_id).language
            if record.document_id in manifest.handoff.documents
            and manifest.handoff.documents[record.document_id].language
            else document.original_language
        )
        expected = selection_record(
            record.document_id,
            record.x_result_id,
            record.kimi_result_id,
            select_translation(
                document.text,
                language,
                x_result,
                kimi_result,
                policy_version=sidecar.policy_version,
            ),
        )
        if record != expected:
            raise ValueError("selection_decision_inconsistent")


def save_translation_selection(
    base: Path,
    store: PreparationStore,
    preparation_id: str,
    decisions,
    *,
    policy_version: QualityPolicy = QUALITY_POLICY_V2,
) -> str:
    sidecar = TranslationSelectionSidecar(
        policy_version=policy_version,
        bundle_id=base.name,
        preparation_id=preparation_id,
        decisions=tuple(
            TranslationSelectionRecord.model_validate(decision)
            for decision in decisions
        ),
    )
    _validate_sidecar(base, store, sidecar)
    raw = canonical_bytes(sidecar.model_dump(mode="json"))
    digest = sha256(raw)
    _atomic(store._path("selection-decisions", digest, ".json"), raw)
    return digest


def load_translation_selection(
    base: Path,
    store: PreparationStore,
    preparation_id: str,
    sidecar_id: str,
) -> TranslationSelectionSidecar:
    sidecar = TranslationSelectionSidecar.model_validate_json(
        store._read("selection-decisions", sidecar_id, ".json")
    )
    if sidecar.preparation_id != preparation_id:
        raise ValueError("selection_preparation_mismatch")
    _validate_sidecar(base, store, sidecar)
    return sidecar


def selection_for_preparation(
    base: Path,
    store: PreparationStore,
    preparation_id: str,
    *,
    policy_version: QualityPolicy | None = None,
) -> tuple[str, TranslationSelectionSidecar]:
    matches = []
    for path in sorted((store.root / "selection-decisions").glob("*.json")):
        raw = path.read_bytes()
        if sha256(raw) != path.stem:
            raise IntegrityError("preparation_hash_mismatch")
        candidate = TranslationSelectionSidecar.model_validate_json(raw)
        if candidate.preparation_id == preparation_id and (
            policy_version is None or candidate.policy_version == policy_version
        ):
            matches.append(
                (
                    path.stem,
                    load_translation_selection(base, store, preparation_id, path.stem),
                )
            )
    if not matches:
        raise IntegrityError("missing_translation_selection")
    by_policy = {
        quality_policy: [
            match for match in matches if match[1].policy_version == quality_policy
        ]
        for quality_policy in (QUALITY_POLICY_V1, QUALITY_POLICY_V2)
    }
    if any(len(policy_matches) > 1 for policy_matches in by_policy.values()):
        raise IntegrityError("ambiguous_translation_selection")
    if policy_version is not None:
        return by_policy[policy_version][0]
    return (by_policy[QUALITY_POLICY_V2] or by_policy[QUALITY_POLICY_V1])[0]
