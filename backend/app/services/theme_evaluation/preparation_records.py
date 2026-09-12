"""Preparation state is separate from frozen version-1 evidence bundles."""

from typing import Literal

from pydantic import Field, model_validator

from .bundle import canonical_bytes, sha256
from .records import SHA, URL, Record


class DocumentHandoff(Record):
    source_text_sha256: SHA
    language: str | None = None
    image_urls: list[URL] = Field(default_factory=list)
    local_images: list[str] = Field(default_factory=list)


class Handoff(Record):
    bundle_id: SHA
    documents: dict[str, DocumentHandoff] = Field(default_factory=dict)
    references: dict[str, URL] = Field(default_factory=dict)


class PreparationBinding(Record):
    stage: Literal["article", "text", "image"]
    source_kind: Literal["document", "reference"]
    source_id: str
    source_text_sha256: SHA
    result_id: SHA
    parent_result_id: SHA | None = None
    input_locator: str | None = None

    @property
    def binding_id(self):
        return sha256(canonical_bytes(self.model_dump(mode="json")))

    @property
    def slot_id(self):
        return sha256(
            canonical_bytes(
                self.model_dump(
                    mode="json",
                    exclude={"source_text_sha256", "result_id"},
                )
            )
        )


class PreparationManifest(Record):
    schema_version: Literal[2] = 2
    bundle_id: SHA
    handoff: Handoff
    bindings: list[PreparationBinding] = Field(default_factory=list)
    current: dict[SHA, SHA] = Field(default_factory=dict)
    evidence_review: Literal["pending"] = "pending"
    extraction: Literal["awaiting_evidence_approval"] = "awaiting_evidence_approval"

    @property
    def current_bindings(self):
        selected = set(self.current.values())
        return [b for b in self.bindings if b.binding_id in selected]

    @model_validator(mode="after")
    def valid_selection(self):
        history = {b.binding_id: b for b in self.bindings}
        if len(history) != len(self.bindings):
            raise ValueError("duplicate_preparation_binding")
        for slot, selected in self.current.items():
            if selected not in history or history[selected].slot_id != slot:
                raise ValueError("invalid_current_preparation")
        roots = {
            (b.source_kind, b.source_id, b.result_id, b.input_locator)
            for b in self.current_bindings
            if b.parent_result_id is None
        }
        for binding in self.bindings:
            if binding.parent_result_id is None and binding.slot_id not in self.current:
                raise ValueError("missing_current_preparation")
        for binding in self.current_bindings:
            if (
                binding.parent_result_id
                and (
                    binding.source_kind,
                    binding.source_id,
                    binding.parent_result_id,
                    binding.input_locator,
                )
                not in roots
            ):
                raise ValueError("current_preparation_parent_superseded")
        return self
