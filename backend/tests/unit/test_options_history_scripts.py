from __future__ import annotations

from app.scripts.export_options_history import write_history_bundle
from app.scripts.import_options_history import read_history_bundle

from .test_options_history_transfer import _bundle


def test_history_scripts_round_trip_deterministic_gzip(tmp_path):
    path = tmp_path / "options-history-us-v1.json.gz"
    bundle = _bundle()

    write_history_bundle(path, bundle)

    assert read_history_bundle(path) == bundle


def test_missing_history_input_visibly_starts_fresh(tmp_path):
    result = read_history_bundle(tmp_path / "missing.json.gz", allow_missing=True)
    assert result == {
        "status": "fresh_history",
        "reason": "history_bundle_missing",
    }
