from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.company_exposure.factory import FixedClock

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "company_exposure"


@pytest.fixture
def contract_manifest() -> dict[str, dict]:
    payload = json.loads((FIXTURE_DIR / "contract_cases.json").read_text("utf-8"))
    return payload["cases"]


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


class DossierEnv:
    """Issuer, theme, retained passages and a real assessment service."""

    def __init__(self, db, clock):
        from app.services.company_exposure.assessments import ExposureAssessmentService
        from app.services.company_exposure.claims import AssessmentScope
        from tests.fixtures.company_exposure.factory import (
            make_document,
            make_issuer,
            make_passage,
            make_revision,
            make_theme,
        )

        self.db = db
        self.clock = clock
        self.issuer = make_issuer(db, "dossier-issuer")
        self.theme = make_theme(db, "dossier-theme")
        document = make_document(db, "sec:10-K:dossier", issuer=self.issuer)
        self.revision = make_revision(db, document, b"annual report")
        self.passages = {
            name: make_passage(db, self.revision, text, index=i)
            for i, (name, text) in enumerate(
                {
                    "role": "We sell HBM test equipment to memory makers.",
                    "materiality": "Memory test was 20% of revenue in fiscal 2025.",
                    "exit": "We exited the HBM test equipment business in June 2025.",
                    "other": "Our probe cards support HBM wafer sort.",
                }.items()
            )
        }
        self.scope = AssessmentScope(
            issuer_id=self.issuer.id,
            economic_theme_id=self.theme.id,
            theme_fingerprint="f" * 64,
            theme_label="AI Memory",
            theme_terms=("HBM",),
        )
        self.service = ExposureAssessmentService(db, clock=clock.now)

    def attempt(self, *claims, coverage=(), scope=None, **kwargs):
        from app.services.company_exposure.assessments import AssessmentAttemptInput

        return AssessmentAttemptInput(
            scope=scope or self.scope,
            claims=tuple(claims),
            document_revision_ids=(self.revision.id,),
            coverage=tuple(coverage),
            **kwargs,
        )

    def persist(self, attempt):
        result = self.service.assess(attempt)
        ref = self.service.persist_assessment(
            result, expected_prior_revision_id=result.prior_revision_id
        )
        return result, ref


@pytest.fixture
def dossier(db_session, clock):
    return DossierEnv(db_session, clock)
