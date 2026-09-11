"""Provider-call provenance and immutable storage for frozen extraction runs."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle import IntegrityError, canonical_bytes, sha256
from .extraction_records import ExtractionInput, ExtractionRecord

_MANIFEST_BINDINGS = ("bundle_id", "preparation_id", "assessment_id")
_REQUEST_PARAMETERS = (
    "temperature",
    "max_tokens",
    "allow_fallbacks",
    "thinking",
    "num_retries",
    "metered",
)


def _public_dump(value: Any) -> Any:
    """Use public Pydantic serialization only; never inspect provider internals."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return None


def _requested_model(wrapped: Any, kwargs: dict[str, Any]) -> str | None:
    model = kwargs.get("model")
    if isinstance(model, str):
        return model
    preset = getattr(wrapped, "preset", None)
    primary = getattr(preset, "primary", None)
    configured = getattr(primary, "model_id", None)
    return configured if isinstance(configured, str) else None


class RecordingLLM:
    """Transparent completion proxy that retains only safe public provenance."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def _request_record(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        parameters = {
            name: kwargs[name] for name in _REQUEST_PARAMETERS if name in kwargs
        }
        return {
            "requested_model": _requested_model(self.wrapped, kwargs),
            "messages_sha256": sha256(
                canonical_bytes({"messages": kwargs.get("messages", [])})
            ),
            "parameters_sha256": sha256(canonical_bytes(parameters)),
            "parameters": parameters,
        }

    def _record(self, kwargs: dict[str, Any], response) -> dict[str, Any]:
        call = self._request_record(kwargs)
        usage = _public_dump(getattr(response, "usage", None))
        choices = [
            _public_dump(choice)
            for choice in (getattr(response, "choices", None) or [])
        ]
        call.update(
            status="success",
            actual_model=getattr(response, "model", None)
            if isinstance(getattr(response, "model", None), str)
            else None,
            provider=getattr(response, "provider", None)
            if isinstance(getattr(response, "provider", None), str)
            else None,
            usage=usage or {},
            choices=choices,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        canonical_bytes(call)  # reject non-JSON provider values before storing.
        return call

    def _failure_record(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        call = self._request_record(kwargs)
        call.update(
            status="failed",
            error_code="provider_completion_failed",
            actual_model=None,
            provider=None,
            usage={},
            choices=[],
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        return call

    async def completion(self, **kwargs):
        try:
            response = await self.wrapped.completion(**kwargs)
        except Exception:
            self.calls.append(self._failure_record(kwargs))
            raise
        self.calls.append(self._record(kwargs, response))
        return response


def _manifest(value: dict) -> dict:
    if not isinstance(value, dict):
        raise TypeError("extraction_manifest_invalid")
    missing = [key for key in _MANIFEST_BINDINGS if not isinstance(value.get(key), str)]
    if missing:
        raise ValueError("extraction_manifest_missing_provenance")
    # Existing preparation and assessment artifacts are content-addressed IDs.
    for key in _MANIFEST_BINDINGS:
        identifier = value[key]
        if len(identifier) != 64 or any(
            char not in "0123456789abcdef" for char in identifier
        ):
            raise ValueError("extraction_manifest_invalid_provenance")
    canonical_bytes(value)
    return value


def _validate(
    manifest: dict, inputs: list[ExtractionInput], records: list[ExtractionRecord]
) -> None:
    _manifest(manifest)
    if len(inputs) != len({item.input_id for item in inputs}):
        raise ValueError("duplicate_extraction_input")
    known = {item.input_id for item in inputs}
    keys = set()
    for record in records:
        if record.input_id not in known:
            raise ValueError("unknown_extraction_input")
        key = (record.input_id, record.pipeline)
        if key in keys:
            raise ValueError("duplicate_extraction")
        keys.add(key)


def _run_payload(
    manifest: dict, inputs: list[ExtractionInput], records: list[ExtractionRecord]
) -> tuple[str, bytes, bytes]:
    manifest_raw = canonical_bytes(manifest)
    run = {
        "schema_version": 1,
        "manifest_sha256": sha256(manifest_raw),
        "input_ids": [item.input_id for item in inputs],
        "input_count": len(inputs),
        "record_count": len(records),
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "records": [item.model_dump(mode="json") for item in records],
    }
    run_id = sha256(canonical_bytes({"manifest": manifest, "run": run}))
    run["run_id"] = run_id
    return run_id, canonical_bytes(run), manifest_raw


def save_extraction_run(
    root: Path,
    *,
    manifest: dict,
    inputs: list[ExtractionInput],
    records: list[ExtractionRecord],
) -> Path:
    """Atomically persist an independently verifiable, content-addressed run."""
    manifest = _manifest(manifest)
    inputs = [ExtractionInput.model_validate(item) for item in inputs]
    records = [ExtractionRecord.model_validate(item) for item in records]
    _validate(manifest, inputs, records)
    run_id, run_raw, manifest_raw = _run_payload(manifest, inputs, records)
    parent = root / "extraction-runs"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / run_id
    if target.exists():
        load_extraction_run(target)
        return target
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        for name, payload in (("run.json", run_raw), ("manifest.json", manifest_raw)):
            with (temporary / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            load_extraction_run(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_extraction_run(path: Path) -> dict:
    try:
        run_raw = (path / "run.json").read_bytes()
        manifest_raw = (path / "manifest.json").read_bytes()
        run = json.loads(run_raw)
        manifest = json.loads(manifest_raw)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise IntegrityError("extraction_run_invalid") from exc
    if canonical_bytes(run) != run_raw or canonical_bytes(manifest) != manifest_raw:
        raise IntegrityError("extraction_run_noncanonical")
    _manifest(manifest)
    if run.get("schema_version") != 1 or run.get("manifest_sha256") != sha256(
        manifest_raw
    ):
        raise IntegrityError("extraction_run_manifest_mismatch")
    try:
        inputs = [ExtractionInput.model_validate(item) for item in run["inputs"]]
        records = [ExtractionRecord.model_validate(item) for item in run["records"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError("extraction_run_records_invalid") from exc
    _validate(manifest, inputs, records)
    if run.get("input_ids") != [item.input_id for item in inputs]:
        raise IntegrityError("extraction_run_input_ids_mismatch")
    if run.get("input_count") != len(inputs) or run.get("record_count") != len(records):
        raise IntegrityError("extraction_run_counts_mismatch")
    payload = {key: value for key, value in run.items() if key != "run_id"}
    run_id = sha256(canonical_bytes({"manifest": manifest, "run": payload}))
    if run.get("run_id") != run_id or path.name != run_id:
        raise IntegrityError("extraction_run_hash_mismatch")
    return {
        "run_id": run_id,
        "manifest": manifest,
        "inputs": inputs,
        "records": records,
    }


def verify_extraction_run(path: Path) -> dict:
    value = load_extraction_run(path)
    return {
        "run_id": value["run_id"],
        "input_count": len(value["inputs"]),
        "record_count": len(value["records"]),
        "integrity": "verified",
    }
