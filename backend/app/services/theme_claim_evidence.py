"""Strict citation provenance and profile grounding for theme claim review."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.theme_grounding_context import GroundingContext


class ClaimReviewError(ValueError):
    """Stable failure with the audit retained even when no candidates pass."""

    def __init__(self, code, *, audit=None):
        super().__init__(code)
        self.audit = audit


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=2000)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["supported", "inferred", "unsupported", "absent"]
    reason: str = Field(min_length=1, max_length=1000)
    evidence: list[Citation] = Field(max_length=8)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, strict=True)
    theme: Verdict
    development: Verdict


def evidence_sources(primary_text: str, context: GroundingContext) -> dict[str, str]:
    # Extraction already bounds the body; keep its title and exact source bytes.
    sources = {"primary": primary_text}
    sources.update({f"related:{e.input_id}": e.text for e in context.evidence})
    for company in context.companies:
        for field in ("name", "sector", "industry", "business_description"):
            value = company.get(field)
            if isinstance(value, str) and value:
                sources[f"company:{company['symbol']}:{field}"] = value
    return sources


def _normalize_verdict(
    verdict: Verdict, sources: dict[str, str], *, theme=False, theme_name=""
) -> list[str]:
    """Correct provenance only when exact supplied bytes prove the correction."""
    adjustments = []
    for citation in verdict.evidence:
        if (
            citation.source_id in sources
            and citation.quote not in sources[citation.source_id]
        ):
            words = citation.quote.split()
            if not words:
                raise ClaimReviewError("claim_review_citation_invalid")

            def quote_pattern(word):
                return "".join(
                    "['‘’]"
                    if char in "'‘’"
                    else '["“”]'
                    if char in '"“”'
                    else re.escape(char)
                    for char in word
                )

            pattern = re.compile(r"\s+".join(quote_pattern(word) for word in words))
            matches = {
                key: match.group()
                for key, text in sources.items()
                if (match := pattern.search(text)) is not None
            }
            selected = (
                citation.source_id
                if citation.source_id in matches
                else (next(iter(matches)) if len(matches) == 1 else None)
            )
            if selected is not None:
                if len(matches[selected]) > 2000:
                    raise ClaimReviewError("claim_review_citation_invalid")
                if citation.source_id != selected:
                    adjustments.append("citation_source_corrected_by_exact_match")
                if citation.quote != matches[selected]:
                    typography = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
                    if (
                        citation.quote.translate(typography) != citation.quote
                        or matches[selected].translate(typography) != matches[selected]
                    ):
                        adjustments.append("citation_typography_normalized_to_source")
                    else:
                        adjustments.append("citation_whitespace_normalized_to_source")
                citation.source_id = selected
                citation.quote = matches[selected]
    # An optional identity citation must not turn a directly named theme into
    # a company-business inference. It still has to be a real source quotation.
    explicit_theme = theme_name and any(
        (c.source_id == "primary" or c.source_id.startswith("related:"))
        and c.quote in sources.get(c.source_id, "")
        and re.search(
            r"(?<!\w)" + re.escape(theme_name) + r"(?!\w)", c.quote, re.IGNORECASE
        )
        for c in verdict.evidence
    )
    if theme and explicit_theme:
        identity_refs = [
            c
            for c in verdict.evidence
            if c.source_id.startswith("company:") and c.source_id.endswith(":name")
        ]
        for citation in identity_refs:
            if citation.quote not in sources.get(citation.source_id, ""):
                raise ClaimReviewError("claim_review_citation_invalid")
            verdict.evidence.remove(citation)
            adjustments.append("ancillary_identity_citation_removed")
    profile_refs = [c for c in verdict.evidence if c.source_id.startswith("company:")]
    if verdict.status == "supported" and profile_refs:
        verdict.status = "inferred"
        adjustments.append("profile_support_labelled_inference")
    if theme and verdict.status == "inferred" and profile_refs:
        if not any(
            c.source_id.rsplit(":", 1)[-1]
            in {"sector", "industry", "business_description"}
            for c in profile_refs
        ):
            raise ClaimReviewError("claim_review_profile_exposure_required")
        # Every cited issuer must occur in event evidence, even when another
        # unrelated event quote is already present. Company names are not proof.
        symbols = {c.source_id.split(":")[1] for c in profile_refs}
        for symbol in sorted(symbols):
            pattern = (
                r"(?<![A-Za-z0-9_$])\$"
                + re.escape(symbol)
                + r"(?![A-Za-z0-9]|\.[A-Za-z0-9])"
            )
            event_sources = {
                k: v
                for k, v in sources.items()
                if k == "primary" or k.startswith("related:")
            }
            anchored = False
            for source_id, text in event_sources.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if not match:
                    continue
                anchored = any(
                    c.source_id == source_id
                    and re.search(pattern, c.quote, re.IGNORECASE)
                    for c in verdict.evidence
                )
                if not anchored and len(verdict.evidence) < 8:
                    verdict.evidence.append(
                        Citation(source_id=source_id, quote=match.group())
                    )
                    adjustments.append("profile_issuer_anchored_to_explicit_cashtag")
                    anchored = True
                if anchored:
                    break
            if not anchored:
                raise ClaimReviewError("claim_review_profile_issuer_anchor_required")
    return adjustments


def _validate_verdict(verdict: Verdict, sources: dict[str, str]):
    for citation in verdict.evidence:
        if (
            citation.source_id not in sources
            or citation.quote not in sources[citation.source_id]
        ):
            raise ClaimReviewError("claim_review_citation_invalid")
    if verdict.status not in {"supported", "inferred"}:
        return
    if not verdict.evidence or not any(
        c.source_id == "primary" or c.source_id.startswith("related:")
        for c in verdict.evidence
    ):
        raise ClaimReviewError("claim_review_event_evidence_required")
    if verdict.status == "supported" and any(
        c.source_id.startswith("company:") for c in verdict.evidence
    ):
        raise ClaimReviewError("claim_review_profile_inference_required")
