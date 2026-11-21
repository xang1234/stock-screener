import logging
from types import SimpleNamespace
from uuid import uuid4

from app.models.economic_taxonomy_runtime import (
    ClassificationAttempt,
    GenerationInputManifest,
    ServingGeneration,
    TaxonomyAuthority,
)
from app.services.economic_taxonomy_interpretations import (
    EconomicTaxonomyInterpretationService,
)


class _Session:
    def __init__(self, rows):
        self._rows = rows

    def get(self, model, key):
        return self._rows.get((model, str(key)))


def test_rejected_override_serves_default_and_logs_reason(monkeypatch, caplog):
    lineage_id, override_id = uuid4(), uuid4()
    selected = SimpleNamespace(id=uuid4())
    default = SimpleNamespace(id=uuid4())
    session = _Session(
        {
            (TaxonomyAuthority, "1"): SimpleNamespace(serving_generation_id="g"),
            (ServingGeneration, "g"): SimpleNamespace(generation_input_manifest_id="m"),
            (GenerationInputManifest, "m"): SimpleNamespace(
                selections=[
                    {
                        "lineage": str(lineage_id),
                        "selected_attempt_id": str(selected.id),
                        "interpretation_override_revision_id": str(override_id),
                    }
                ]
            ),
            (ClassificationAttempt, str(selected.id)): selected,
            # No InterpretationOverrideRevision row: the override is not pinned.
        }
    )
    service = EconomicTaxonomyInterpretationService(session_factory=None)
    monkeypatch.setattr(service, "_attempt_completed", lambda _s, _a: True)
    monkeypatch.setattr(service, "_default_attempt", lambda _s, _l: default)

    with caplog.at_level(logging.WARNING):
        assert service.serving_attempt(session, lineage_id) is default

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert str(override_id) in message
    assert str(lineage_id) in message
    assert "override_not_pinned" in message
