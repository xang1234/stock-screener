"""Public facade for company-exposure research persistence.

Importing this module registers every company-exposure model together with
its ORM and PostgreSQL immutability protections.
"""

# ruff: noqa: F401

from app.models.company_exposure_common import (
    APPEND_ONLY_EXPOSURE_MODELS,
    SEALABLE_EXPOSURE_MODELS,
)
from app.models.company_exposure_documents import (
    DocumentCaptureEvent,
    DocumentRelationRevision,
    EvidenceTombstoneEvent,
    ExposureDocument,
    ExposureDocumentRevision,
    ExposurePassage,
    PassageDerivative,
)
from app.models.company_exposure_identity import (
    ExposureIssuer,
    IssuerIdentifierRevision,
    IssuerSecurityLinkRevision,
    LegacyIssuerAttestationBridge,
)
