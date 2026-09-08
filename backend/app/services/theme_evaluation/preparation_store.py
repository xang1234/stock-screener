"""Atomic content-addressed assets, results, and review manifests."""

import os
import re
import tempfile
from pathlib import Path

from .bundle import IntegrityError, canonical_bytes, load_bundle, sha256
from .preparation_records import (
    Handoff,
    PreparationManifest,
    PreparationRequest,
    PreparationResult,
)
from .preparation_validation import validate_result


def validate_handoff(base: Path, handoff: Handoff):
    bundle = load_bundle(base)
    if handoff.bundle_id != base.name:
        raise ValueError("handoff_base_mismatch")
    documents = {d.document_id: d for d in bundle.documents}
    for key, value in handoff.documents.items():
        if (
            key not in documents
            or value.source_text_sha256 != documents[key].text_sha256
        ):
            raise ValueError("handoff_source_mismatch")
    references = {r.reference_id for r in bundle.followups}
    if not set(handoff.references).issubset(references):
        raise ValueError("handoff_unknown_reference")
    return bundle


def _digest(value):
    return sha256(canonical_bytes(value.model_dump(mode="json")))


def _atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise IntegrityError("existing_preparation_file_mismatch")
    finally:
        os.unlink(temporary)


def deduplicate_bindings(bindings):
    # Last occurrence is the latest selection, including a return to a cached result.
    unique = {}
    for binding in bindings:
        key = canonical_bytes(binding.model_dump(mode="json"))
        unique.pop(key, None)
        unique[key] = binding
    return list(unique.values())


class PreparationStore:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, folder: str, digest: str, suffix="") -> Path:
        if not re.fullmatch("[a-f0-9]{64}", digest):
            raise IntegrityError("invalid_preparation_id")
        return self.root / folder / (digest + suffix)

    def _read(self, folder: str, digest: str, suffix="") -> bytes:
        try:
            raw = self._path(folder, digest, suffix).read_bytes()
        except OSError as exc:
            raise IntegrityError("missing_preparation_file") from exc
        if sha256(raw) != digest:
            raise IntegrityError("preparation_hash_mismatch")
        return raw

    def save_asset(self, raw: bytes) -> str:
        digest = sha256(raw)
        _atomic(self._path("assets", digest), raw)
        return digest

    def load_asset(self, digest: str) -> bytes:
        return self._read("assets", digest)

    def save_result(self, result: PreparationResult) -> str:
        result = PreparationResult.model_validate(result.model_dump())
        validate_result(result)
        for asset in result.assets:
            self.load_asset(asset)
        raw = canonical_bytes(result.model_dump(mode="json"))
        digest = sha256(raw)
        _atomic(self._path("results", digest, ".json"), raw)
        return digest

    def load_result(self, digest: str) -> PreparationResult:
        result = PreparationResult.model_validate_json(
            self._read("results", digest, ".json")
        )
        validate_result(result)
        for asset in result.assets:
            self.load_asset(asset)
        return result

    def cached(self, request: PreparationRequest):
        # Small offline pilots need no database or mutable cache index.
        signature = _digest(request)
        for path in sorted((self.root / "results").glob("*.json")):
            result = self.load_result(path.stem)
            if result.status == "success" and _digest(result.request) == signature:
                return path.stem, result
        return None

    def validate(self, base: Path, manifest: PreparationManifest):
        bundle = validate_handoff(base, manifest.handoff)
        if manifest.bundle_id != base.name:
            raise ValueError("preparation_base_mismatch")
        documents = {d.document_id: d for d in bundle.documents}
        references = {r.reference_id: r for r in bundle.followups}
        keys = set()
        for binding in manifest.bindings:
            if binding.source_kind == "reference":
                ref = references.get(binding.source_id)
                doc = documents.get(ref.post_id) if ref else None
            else:
                doc = documents.get(binding.source_id)
            if doc is None or doc.text_sha256 != binding.source_text_sha256:
                raise ValueError("preparation_source_mismatch")
            key = (
                binding.source_kind,
                binding.source_id,
                binding.result_id,
                binding.parent_result_id,
                binding.input_locator,
            )
            if key in keys:
                raise ValueError("duplicate_preparation_binding")
            keys.add(key)
            result = self.load_result(binding.result_id)
            if result.request.stage == "image":
                override = manifest.handoff.documents.get(doc.document_id)
                locators = doc.source_metadata.image_urls + (
                    override.image_urls + override.local_images if override else []
                )
                if (
                    binding.source_kind != "document"
                    or binding.input_locator not in locators
                ):
                    raise ValueError("image_source_locator_mismatch")
            if result.request.stage == "article":
                if binding.source_kind != "reference":
                    raise ValueError("article_reference_required")
                destination = manifest.handoff.references.get(
                    binding.source_id, references[binding.source_id].candidate_url
                )
                if result.request.options.get("destination_url") != destination:
                    raise ValueError("article_destination_mismatch")
            if binding.parent_result_id:
                parents = [
                    b
                    for b in manifest.bindings
                    if b.result_id == binding.parent_result_id
                    and b.source_id == binding.source_id
                    and b.source_kind == binding.source_kind
                    and b.parent_result_id is None
                ]
                if not parents:
                    raise ValueError("missing_preparation_parent")
                parent = self.load_result(binding.parent_result_id)
                parent_text = parent.payload.get(
                    "text", parent.payload.get("transcription")
                )
                if (
                    result.request.stage != "text"
                    or not isinstance(parent_text, str)
                    or sha256(parent_text.encode()) != result.request.input_sha256
                ):
                    raise ValueError("preparation_parent_input_mismatch")
            elif (
                result.request.stage == "text"
                and result.request.input_sha256 != doc.text_sha256
            ):
                raise ValueError("preparation_text_input_mismatch")
        return bundle

    def seal(self, base: Path, manifest: PreparationManifest) -> str:
        self.validate(base, manifest)
        raw = canonical_bytes(manifest.model_dump(mode="json"))
        digest = sha256(raw)
        _atomic(self._path("preparations", digest, ".json"), raw)
        return digest

    def load(self, base: Path, digest: str) -> PreparationManifest:
        manifest = PreparationManifest.model_validate_json(
            self._read("preparations", digest, ".json")
        )
        self.validate(base, manifest)
        return manifest
