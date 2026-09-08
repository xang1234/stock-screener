"""Import attributed translations without selecting or calling an unapproved service."""

from pydantic import AwareDatetime, Field

from .multilingual_preparation import TextPreparation, prepare_text
from .preparation_records import PreparationManifest, PreparationResult
from .preparation_store import deduplicate_bindings
from .records import SHA, Record


class TranslationImport(Record):
    result_id: SHA
    source_text_sha256: SHA
    translations: list[str | None]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    generated_at: AwareDatetime


def import_translations(base, store, preparation_id, records):
    manifest = store.load(base, preparation_id)
    rows = [TranslationImport.model_validate(value) for value in records]
    if len({row.result_id for row in rows}) != len(rows):
        raise ValueError("duplicate_translation_import")
    known = {binding.result_id for binding in manifest.bindings}
    pending = []
    for row in rows:
        if row.result_id not in known:
            raise ValueError("unknown_translation_source")
        original = store.load_result(row.result_id)
        if (
            original.request.stage != "text"
            or row.source_text_sha256 != original.request.input_sha256
        ):
            raise ValueError("translation_import_source_mismatch")
        previous = TextPreparation.model_validate(original.payload)
        if len(row.translations) != len(previous.segments):
            raise ValueError("translation_import_segment_count")
        # Preserve source segmentation including identity/blank segments. prepare_text
        # intentionally skips those, so validate them and omit them from the iterator.
        translating = []
        for segment, translated in zip(previous.segments, row.translations):
            if segment.status == "identity":
                if translated != segment.original:
                    raise ValueError("identity_translation_changed")
            else:
                translating.append(translated)

        translation_values = iter(translating)

        def translate(text, source, target, values=translation_values):
            return next(values)

        text = "".join(segment.original for segment in previous.segments)
        prepared = prepare_text(
            text,
            language=previous.supplied_language,
            target_language=previous.target_language,
            translator=translate,
            max_chars=original.request.options.get("max_chars", 4000),
        )
        request = original.request.model_copy(
            update={
                "provider": row.provider,
                "model": row.model,
                "policy_version": row.policy_version,
                "options": {**original.request.options, "method": "translation_import"},
            }
        )
        result = PreparationResult(
            request=request,
            status=prepared.status,
            payload=prepared.model_dump(mode="json"),
            warnings=prepared.warnings,
            created_at=row.generated_at,
        )
        pending.append((row.result_id, result))
    bindings = list(manifest.bindings)
    for old_id, result in pending:
        rid = store.save_result(result)
        for binding in manifest.bindings:
            if binding.result_id == old_id:
                bindings.append(binding.model_copy(update={"result_id": rid}))
    return store.seal(
        base,
        PreparationManifest(
            bundle_id=manifest.bundle_id,
            handoff=manifest.handoff,
            bindings=deduplicate_bindings(bindings),
        ),
    )
