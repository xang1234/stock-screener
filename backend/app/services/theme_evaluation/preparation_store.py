"""Atomic content-addressed assets, results, and review manifests."""

import os
import re
import tempfile
from pathlib import Path

from .bundle import IntegrityError, canonical_bytes, load_bundle, sha256
from .preparation_records import (
    Handoff,
    PreparationManifest,
)
from .preparation_results import RESULT_ADAPTER, PreparationRequest, PreparationResult


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
        result = RESULT_ADAPTER.validate_python(result.model_dump())
        for asset in result.assets:
            self.load_asset(asset)
        raw = canonical_bytes(result.model_dump(mode="json"))
        digest = sha256(raw)
        _atomic(self._path("results", digest, ".json"), raw)
        if result.status == "success":
            self._index_result(digest, result)
        return digest

    def load_result(self, digest: str) -> PreparationResult:
        result = RESULT_ADAPTER.validate_json(self._read("results", digest, ".json"))
        for asset in result.assets:
            self.load_asset(asset)
        return result

    def _index_result(self, digest, result):
        # Immutable entries avoid a mutable pointer and concurrent-writer races.
        _atomic(self._path("cache", _digest(result.request)) / digest, b"")

    def cached(self, request: PreparationRequest):
        signature = _digest(request)
        for path in sorted(self._path("cache", signature).glob("*")):
            if path.name.startswith(".pending-"):
                continue
            result = self.load_result(path.name)
            if (
                path.read_bytes()
                or result.status != "success"
                or _digest(result.request) != signature
            ):
                raise IntegrityError("preparation_cache_mismatch")
            return path.name, result
        return None

    def verify_all(self):
        """Audit every asset/result, including unbound history, and repair missing indexes.

        A crash after a result write but before indexing is a harmless cache miss.
        Reindex only after all content and existing entries pass verification.
        """
        for path in sorted((self.root / "assets").glob("*")):
            if not path.name.startswith(".pending-"):
                self.load_asset(path.name)
        successful = []
        for path in sorted((self.root / "results").glob("*.json")):
            result = self.load_result(path.stem)
            if result.status == "success":
                successful.append((path.stem, result))
        for directory in sorted((self.root / "cache").glob("*")):
            for entry in sorted(directory.glob("*")):
                if entry.name.startswith(".pending-"):
                    continue
                result = self.load_result(entry.name)
                if (
                    entry.read_bytes()
                    or result.status != "success"
                    or _digest(result.request) != directory.name
                ):
                    raise IntegrityError("preparation_cache_mismatch")
        for digest, result in successful:
            self._index_result(digest, result)

    def validate(self, base: Path, manifest: PreparationManifest):
        manifest = PreparationManifest.model_validate(manifest.model_dump())
        bundle = validate_handoff(base, manifest.handoff)
        if manifest.bundle_id != base.name:
            raise ValueError("preparation_base_mismatch")
        documents = {d.document_id: d for d in bundle.documents}
        references = {r.reference_id: r for r in bundle.followups}
        for binding in manifest.bindings:
            if binding.source_kind == "reference":
                ref = references.get(binding.source_id)
                doc = documents.get(ref.post_id) if ref else None
            else:
                doc = documents.get(binding.source_id)
            if doc is None or doc.text_sha256 != binding.source_text_sha256:
                raise ValueError("preparation_source_mismatch")
            result = self.load_result(binding.result_id)
            if result.stage != binding.stage:
                raise ValueError("preparation_stage_mismatch")
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
                if result.request.destination_url != destination:
                    raise ValueError("article_destination_mismatch")
            if binding.parent_result_id:
                parents = [
                    b
                    for b in manifest.bindings
                    if b.result_id == binding.parent_result_id
                    and b.source_id == binding.source_id
                    and b.source_kind == binding.source_kind
                    and b.parent_result_id is None
                    and b.input_locator == binding.input_locator
                ]
                if not parents:
                    raise ValueError("missing_preparation_parent")
                parent = self.load_result(binding.parent_result_id)
                parent_text = parent.source_text
                if (
                    result.request.stage != "text"
                    or parent.stage not in {"article", "image"}
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
