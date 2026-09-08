"""Import attributed translations without selecting or calling an unapproved service."""

from pydantic import AwareDatetime, Field

from .multilingual_preparation import finalize_translation
from .preparation_results import TextResult
from .preparation_state import PreparationState
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
            not isinstance(original, TextResult)
            or row.source_text_sha256 != original.request.input_sha256
        ):
            raise ValueError("translation_import_source_mismatch")
        prepared = finalize_translation(original.payload, row.translations)
        request = original.request.model_copy(
            update={
                "provider": row.provider,
                "model": row.model,
                "policy_version": row.policy_version,
                "method": "translation_import",
            }
        )
        result = TextResult(
            request=request,
            payload=prepared,
            created_at=row.generated_at,
        )
        pending.append((row.result_id, result))
    state = PreparationState(base, store, manifest.handoff, preparation_id)
    for old_id, result in pending:
        rid = store.save_result(result)
        for binding in manifest.bindings:
            if binding.result_id == old_id:
                state.record(binding.model_copy(update={"result_id": rid}))
    return store.seal(base, state.manifest)
