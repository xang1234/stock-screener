"""Immutable operational run counts, separate from evidence eligibility."""

import json

from .bundle import canonical_bytes, sha256
from .preparation_store import _atomic


def save_run(store, bundle_id, preparation_id, summary):
    raw = canonical_bytes(
        dict(
            schema_version=1,
            bundle_id=bundle_id,
            preparation_id=preparation_id,
            **summary,
        )
    )
    digest = sha256(raw)
    _atomic(store._path("preparation-runs", digest, ".json"), raw)
    return digest


def runs_for_preparation(store, bundle_id, preparation_id):
    reports = []
    for path in sorted((store.root / "preparation-runs").glob("*.json")):
        report = json.loads(store._read("preparation-runs", path.stem, ".json"))
        if (
            report["bundle_id"] == bundle_id
            and report["preparation_id"] == preparation_id
        ):
            reports.append(dict(run_id=path.stem, **report))
    return reports


class CountedTranslator:
    """Count actual adapter invocations, including failed segment attempts."""

    def __init__(self, client):
        self.client = client
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.client(*args, **kwargs)
