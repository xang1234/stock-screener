"""US official-document route: SEC EDGAR filings plus CIK resolution.

* CIK resolution (spec §4.2): SEC's ticker-to-CIK file
  (``company_tickers_exchange.json``, falling back to ``company_tickers.json``)
  gives candidates; each candidate's submissions record must list the ticker.
  Both documents are retained as captured revisions. The result is a
  ``RegistryMatch`` for ``IssuerIdentityAdapter.accept_registry_match``.
* Filing discovery reads ``filings.recent`` from the submissions record. Its
  parallel arrays must have equal lengths; a mismatch is reported as a
  partial route rather than silently shifting rows.
* Every SEC request uses the operator's identifying User-Agent and the
  existing ``sec_edgar`` pacing key. Without a User-Agent the route is a
  typed capability gap.

SEC metadata and XBRL facts are identity/discovery inputs, never proof of a
theme-specific business exposure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    CaptureResult,
    CoverageItem,
    CoverageOutcome,
    DocumentTarget,
    RegistryMatch,
)
from app.models.company_exposure import ExposureDocumentRevision
from app.models.stock_universe import StockUniverse
from app.services.company_exposure.acquisition import (
    DocumentAcquisitionRegistry,
    JobBudgetRef,
)
from app.services.company_exposure.markets.base import (
    AcquisitionLimits,
    DiscoveryResult,
    DocumentQuery,
)

SEC_HOSTS = ("www.sec.gov", "data.sec.gov")
TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
RESOLVER_POLICY = "sec-ticker-cik-v1"
MAX_CIK_CANDIDATES = 3
ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
FORMS_BY_KIND = {
    "annual_report": ANNUAL_FORMS,
    "quarterly_report": ("10-Q", "10-Q/A"),
    "current_report": ("8-K", "8-K/A", "6-K", "6-K/A"),
}
_RECENT_FIELDS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "form",
    "primaryDocument",
)


def sec_submission_url(cik: str) -> str:
    if not cik.isdigit() or len(cik) > 10:
        raise ValueError("invalid_cik")
    return f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"


def sec_ticker(symbol: str) -> str:
    """SEC tickers use '-' for share classes (``BRK-B``); listings may use '.'."""

    return symbol.strip().upper().replace(".", "-").replace("/", "-")


def parse_ticker_file(payload: dict) -> list[dict]:
    """Rows of ``{cik, name, ticker, exchange}`` from either SEC file shape."""

    rows = []
    if isinstance(payload, dict) and "fields" in payload and "data" in payload:
        fields = [str(name) for name in payload["fields"]]
        for values in payload["data"]:
            record = dict(zip(fields, values, strict=False))
            rows.append(
                {
                    "cik": str(record.get("cik", "")),
                    "name": record.get("name"),
                    "ticker": str(record.get("ticker", "")).upper(),
                    "exchange": record.get("exchange"),
                }
            )
        return rows
    if isinstance(payload, dict):
        for record in payload.values():
            if isinstance(record, dict) and "cik_str" in record:
                rows.append(
                    {
                        "cik": str(record["cik_str"]),
                        "name": record.get("title"),
                        "ticker": str(record.get("ticker", "")).upper(),
                        "exchange": None,
                    }
                )
    return rows


@dataclass(frozen=True, slots=True)
class FilingRows:
    rows: tuple[dict, ...]
    complete: bool
    reason: str | None = None


def recent_filings(submissions: dict) -> FilingRows:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    columns = {name: recent.get(name) for name in _RECENT_FIELDS}
    if any(not isinstance(values, list) for values in columns.values()):
        return FilingRows((), False, "filing_arrays_missing")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        # Never realign mismatched arrays; the whole block is untrusted.
        return FilingRows((), False, "mismatched_filing_arrays")
    count = lengths.pop()
    rows = tuple(
        {name: columns[name][index] for name in _RECENT_FIELDS} for index in range(count)
    )
    return FilingRows(rows, True)


class USDocumentAdapter:
    market = "US"

    def __init__(
        self,
        session: Session,
        acquisition: DocumentAcquisitionRegistry,
        *,
        user_agent: str,
    ):
        self.session = session
        self.acquisition = acquisition
        self.user_agent = (user_agent or "").strip()
        self._json_cache: dict[tuple[UUID | None, str], tuple[CaptureResult, dict | None]] = {}

    # -- plumbing -------------------------------------------------------------

    def _unconfigured(self) -> CoverageItem | None:
        if not self.user_agent:
            return CoverageItem(
                route="us_sec",
                outcome=CoverageOutcome.NOT_CONFIGURED,
                reason="sec_user_agent_not_configured",
            )
        return None

    def _json_target(self, identity_key: str, url: str, source_kind: str) -> DocumentTarget:
        return DocumentTarget(
            adapter="us_sec",
            provider="sec",
            identity_key=identity_key,
            url=url,
            source_kind=source_kind,
            allowed_hosts=SEC_HOSTS,
            rate_provider="sec_edgar",
            market="US",
            publisher="U.S. Securities and Exchange Commission",
            verified_origin="sec.gov",
            accept="application/json",
            max_bytes=25 * 1024 * 1024,
        )

    def _fetch_json(
        self, target: DocumentTarget, budget: JobBudgetRef
    ) -> tuple[CaptureResult, dict | None]:
        key = (budget.root_request_id, target.identity_key)
        if key in self._json_cache:
            return self._json_cache[key]
        capture = self.acquisition.fetch(target, budget)
        payload = None
        if capture.revision_id is not None:
            revision = self.session.get(ExposureDocumentRevision, capture.revision_id)
            try:
                payload = json.loads(self.acquisition.store.read(revision.blob_key))
            except (ValueError, UnicodeDecodeError):
                # e.g. an HTML blocking page served with HTTP 200: retry it
                # as a failed fetch, not a successful retrieval.
                payload = None
                capture = replace(
                    capture,
                    coverage=CoverageItem(
                        route=target.adapter,
                        outcome=CoverageOutcome.FETCH_FAILED,
                        reason="invalid_json_payload",
                        detail={"identity_key": target.identity_key},
                    ),
                )
        self._json_cache[key] = (capture, payload)
        return capture, payload

    # -- discovery ------------------------------------------------------------

    def resolve_target(self, raw_metadata: dict) -> DocumentTarget:
        cik = str(raw_metadata["cik"]).lstrip("0") or "0"
        accession = str(raw_metadata["accessionNumber"])
        primary = str(raw_metadata["primaryDocument"])
        form = str(raw_metadata["form"])
        filed = raw_metadata.get("filingDate")
        published = (
            datetime.fromisoformat(filed).replace(tzinfo=timezone.utc) if filed else None
        )
        is_amendment = form.endswith("/A")
        return DocumentTarget(
            adapter="us_sec",
            provider="sec",
            identity_key=f"sec:accession:{accession}:{primary}",
            url=(
                "https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{accession.replace('-', '')}/{primary}"
            ),
            source_kind="annual_report" if form in ANNUAL_FORMS else "filing",
            allowed_hosts=SEC_HOSTS,
            rate_provider="sec_edgar",
            market="US",
            publisher="Issuer filing via SEC EDGAR",
            verified_origin="sec.gov",
            provider_document_id=accession,
            published_at=published,
            reporting_period=raw_metadata.get("reportDate") or None,
            correction_identity={
                "accession": accession,
                "form": form,
                "is_amendment": is_amendment,
            },
            metadata={
                "cik": cik.zfill(10),
                "form": form,
                "is_amendment": is_amendment,
                "filing_date": filed,
                "report_date": raw_metadata.get("reportDate"),
                "primary_document": primary,
                "language": "en",
            },
        )

    def discover(
        self,
        issuer,
        query: DocumentQuery,
        limits: AcquisitionLimits,
        budget: JobBudgetRef | None = None,
    ) -> DiscoveryResult:
        gap = self._unconfigured()
        if gap is not None:
            return DiscoveryResult(coverage=(gap,))
        cik = (getattr(issuer, "identifiers", {}) or {}).get(("US", "cik"))
        if not cik:
            return DiscoveryResult(
                coverage=(
                    CoverageItem(
                        route="us_sec",
                        outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                        reason="issuer_cik_unresolved",
                    ),
                )
            )
        budget = budget or JobBudgetRef(root_request_id=None)
        capture, submissions = self._fetch_json(
            self._json_target(f"sec:submissions:{cik}", sec_submission_url(cik), "filing_index"),
            budget,
        )
        if submissions is None:
            return DiscoveryResult(coverage=(capture.coverage,))
        filings = recent_filings(submissions)
        coverage = [capture.coverage]
        if not filings.complete:
            coverage.append(
                CoverageItem(
                    route="us_sec_filings",
                    outcome=CoverageOutcome.PARTIAL,
                    reason=filings.reason,
                )
            )
            return DiscoveryResult(coverage=tuple(coverage))
        wanted = {form for kind in query.document_kinds for form in FORMS_BY_KIND.get(kind, ())}
        matching = [
            row
            for row in filings.rows
            if row["form"] in wanted
            and (
                query.since_year is None
                or str(row.get("filingDate", ""))[:4] >= str(query.since_year)
            )
        ]
        limit = min(query.max_documents, limits.max_documents)
        chosen = sorted(matching, key=lambda r: r["filingDate"], reverse=True)[:limit]
        targets = tuple(self.resolve_target({**row, "cik": cik}) for row in chosen)
        if not targets:
            coverage.append(
                CoverageItem(
                    route="us_sec_filings",
                    outcome=CoverageOutcome.NO_MATCHING_DOCUMENT,
                    reason="no_matching_filing",
                )
            )
        elif len(matching) > limit:
            coverage.append(
                CoverageItem(
                    route="us_sec_filings",
                    outcome=CoverageOutcome.PARTIAL,
                    reason="document_limit",
                    detail={"matching": len(matching), "selected": limit},
                )
            )
        return DiscoveryResult(targets=targets, coverage=tuple(coverage))

    def fetch(self, target: DocumentTarget, budget: JobBudgetRef) -> CaptureResult:
        gap = self._unconfigured()
        if gap is not None:
            return CaptureResult(None, None, None, None, False, gap)
        return self.acquisition.fetch(target, budget)


class USIssuerResolver:
    """Ticker → CIK from SEC's official files, confirmed by submissions."""

    def __init__(self, session: Session, adapter: USDocumentAdapter):
        self.session = session
        self.adapter = adapter

    def resolve_cik(self, security_id: int, budget: JobBudgetRef) -> RegistryMatch | CoverageItem:
        gap = self.adapter._unconfigured()
        if gap is not None:
            return gap
        security = self.session.get(StockUniverse, security_id)
        if security is None or security.market != "US":
            return CoverageItem(
                route="us_sec_registry",
                outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                reason="not_a_us_listing",
            )
        ticker = sec_ticker(security.symbol)
        registry_capture, payload = self.adapter._fetch_json(
            self.adapter._json_target(
                "sec:company_tickers_exchange", TICKERS_EXCHANGE_URL, "identifier_registry"
            ),
            budget,
        )
        if payload is None:
            registry_capture, payload = self.adapter._fetch_json(
                self.adapter._json_target(
                    "sec:company_tickers", TICKERS_URL, "identifier_registry"
                ),
                budget,
            )
        if payload is None:
            return registry_capture.coverage
        rows = [row for row in parse_ticker_file(payload) if row["ticker"] == ticker]
        candidates = sorted({row["cik"].zfill(10) for row in rows if row["cik"].isdigit()})
        confirmed: list[tuple[str, CaptureResult]] = []
        official_capture = None
        for cik in candidates[:MAX_CIK_CANDIDATES]:
            capture, submissions = self.adapter._fetch_json(
                self.adapter._json_target(
                    f"sec:submissions:{cik}", sec_submission_url(cik), "filing_index"
                ),
                budget,
            )
            tickers = {str(t).upper() for t in (submissions or {}).get("tickers", [])}
            if ticker in tickers:
                confirmed.append((cik, capture))
        exchange = next((row["exchange"] for row in rows if row.get("exchange")), None)
        title = next((row["name"] for row in rows if row.get("name")), None)
        chosen = None
        if len(candidates) == 1 and len(confirmed) == 1:
            chosen, official_capture = confirmed[0]
        elif confirmed:
            official_capture = confirmed[0][1]
        return RegistryMatch(
            security_id=security.id,
            market="US",
            scheme="cik",
            value=chosen if chosen is not None else (candidates[0] if len(candidates) == 1 else None),
            candidate_count=len(candidates),
            ticker_confirmed=bool(confirmed) and len(candidates) == 1,
            matched_ticker=security.symbol,
            registry_capture_revision_id=registry_capture.revision_id,
            official_record_capture_revision_id=(
                None if official_capture is None else official_capture.revision_id
            ),
            entity_title=title,
            matched_exchange=exchange,
            resolver_policy_version=RESOLVER_POLICY,
            candidates=tuple(candidates),
        )

