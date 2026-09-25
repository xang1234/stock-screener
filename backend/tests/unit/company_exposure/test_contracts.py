from __future__ import annotations

import hashlib
import re
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from app.domain.company_exposure.contracts import (
    ClaimKind,
    CoverageOutcome,
    LLMBillingMode,
    ResearchLimits,
    ResearchMode,
    SearchResult,
    TaskOutcome,
    canonical_json,
    content_hash,
)
from app.domain.company_exposure.manifest import CASE_IDS
from app.domain.company_exposure.policy import (
    freshness_deadline,
    is_fresh,
    may_dispatch_paid_search,
    may_dispatch_research,
    within_synthesis_bound,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC = REPO_ROOT / "docs/superpowers/specs/2026-09-25-company-exposure-map-design.md"
PLAN = REPO_ROOT / "docs/superpowers/plans/2026-09-25-company-exposure-map.md"
ACCEPTED_SPEC_SHA256 = "84784bdea4d789c2f7bb0f1e5529054259ab2fa8c66c504f30678efcc8fae2df"


@pytest.mark.case("R01")
@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_default_configuration_does_not_spend():
    limits = ResearchLimits()
    assert limits.research_mode is ResearchMode.DISABLED
    assert limits.paid_search_enabled is False
    assert limits.llm_billing_mode is LLMBillingMode.SUBSCRIPTION
    assert limits.provider_dispatch_configured is False
    assert not may_dispatch_research(limits.research_mode)
    assert not may_dispatch_paid_search(enabled=False, cap=None, key_present=True)


@pytest.mark.case("R01")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    ("enabled", "cap", "key_present", "allowed"),
    [
        (False, "5", True, False),
        (True, None, True, False),
        (True, "0", True, False),
        (True, "-1", True, False),
        (True, "NaN", True, False),
        (True, "not-a-number", True, False),
        (True, "5", False, False),
        (True, "5", True, True),
    ],
)
def test_paid_search_needs_enablement_cap_and_key(enabled, cap, key_present, allowed):
    assert (
        may_dispatch_paid_search(enabled=enabled, cap=cap, key_present=key_present)
        is allowed
    )


def test_limits_match_approved_defaults():
    limits = ResearchLimits()
    assert (
        limits.concurrent_investigations,
        limits.concurrent_provider_requests,
        limits.new_issuers_per_discovery_job,
        limits.search_queries_per_root_job,
        limits.documents_per_issuer,
        limits.download_bytes_per_document,
        limits.text_pages_per_document,
        limits.passages_per_issuer,
        limits.image_pages_per_issuer,
        limits.browser_navigations_per_issuer,
        limits.provider_attempts_per_issuer,
        limits.provider_attempts_per_root_job,
    ) == (1, 1, 10, 6, 12, 25 * 1024**2, 300, 24, 8, 4, 24, 24)
    assert limits.storage_max_bytes == 5 * 1024**3
    assert limits.storage_min_free_bytes == 1024**3
    assert limits.scratch_bytes == 512 * 1024**2
    assert limits.daily_request_allocation is None


def test_limits_are_frozen_and_validated():
    limits = ResearchLimits()
    with pytest.raises(FrozenInstanceError):
        limits.research_mode = ResearchMode.LIVE
    with pytest.raises(ValueError):
        ResearchLimits(documents_per_issuer=-1)
    with pytest.raises(ValueError):
        ResearchLimits(research_mode="enabled")
    with pytest.raises(ValueError):
        ResearchLimits(paid_search_enabled="true")


def test_manifest_preserves_all_approved_case_ids(contract_manifest):
    expected = (
        {f"E{i:02}" for i in range(1, 16)}
        | {f"I{i:02}" for i in range(1, 12)}
        | {f"R{i:02}" for i in range(1, 16)}
    )
    assert set(contract_manifest) == expected == CASE_IDS


def test_manifest_descriptions_are_verbatim_from_accepted_spec(contract_manifest):
    spec = SPEC.read_text("utf-8")
    table = {
        match.group(1): match.group(2).strip()
        for match in re.finditer(r"^\| ([EIR]\d\d) \| ([^|]+) \|\s*$", spec, re.M)
    }
    assert {key: row["description"] for key, row in contract_manifest.items()} == table


def test_accepted_spec_hash_is_pinned_in_plan():
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert digest == ACCEPTED_SPEC_SHA256
    assert ACCEPTED_SPEC_SHA256 in PLAN.read_text("utf-8")


def test_canonical_json_is_stable_and_rejects_ambiguous_values():
    when = datetime(2026, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=8)))
    payload = {
        "b": Decimal("0.20"),
        "a": UUID("00000000-0000-0000-0000-000000000001"),
        "t": when,
        "k": ClaimKind.ROLE,
    }
    assert canonical_json(payload) == (
        '{"a":"00000000-0000-0000-0000-000000000001","b":"0.20",'
        '"k":"role","t":"2026-01-01T19:04:00+00:00"}'
    )
    assert content_hash(payload) == content_hash(dict(reversed(payload.items())))
    with pytest.raises(ValueError):
        canonical_json({"x": 0.2})
    with pytest.raises(ValueError):
        canonical_json({"x": datetime(2026, 1, 1)})
    with pytest.raises(ValueError):
        canonical_json({"x": Decimal("NaN")})


@pytest.mark.case("E12")
@pytest.mark.exposure_layer("unit")
def test_freshness_windows_follow_claim_kind():
    anchor = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert freshness_deadline(anchor, ClaimKind.ROLE) == anchor + timedelta(days=450)
    assert freshness_deadline(
        anchor, ClaimKind.CUSTOMER_RELATIONSHIP
    ) == anchor + timedelta(days=180)
    assert freshness_deadline(
        anchor, ClaimKind.COMMERCIAL_STATUS
    ) == anchor + timedelta(days=180)
    assert freshness_deadline(anchor, ClaimKind.MATERIALITY) is None
    assert freshness_deadline(None, ClaimKind.ROLE) is None
    deadline = freshness_deadline(anchor, ClaimKind.ROLE)
    assert is_fresh(deadline, deadline - timedelta(microseconds=1))
    assert not is_fresh(deadline, deadline)
    assert not is_fresh(None, anchor)
    with pytest.raises(ValueError):
        freshness_deadline(datetime(2025, 1, 1), ClaimKind.ROLE)


def test_synthesis_bound():
    assert within_synthesis_bound(["a", "b", "c"], ["j1", "j2"])
    assert within_synthesis_bound(["a", "a"], ["j1"])
    assert not within_synthesis_bound(["a", "b", "c", "d"], [])
    assert not within_synthesis_bound(["a"], ["j1", "j2", "j3"])
    assert not within_synthesis_bound([], [])


def test_named_empty_outcomes_are_explicit():
    disabled = SearchResult.disabled("paid_search_disabled_or_unbudgeted")
    assert disabled.items == ()
    assert disabled.coverage[0].outcome is CoverageOutcome.DISABLED
    skipped = TaskOutcome.skipped("research_disabled")
    assert (skipped.status, skipped.reason) == ("skipped", "research_disabled")
