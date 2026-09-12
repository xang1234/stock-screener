"""Resolve, route, and recover references while reusing destination work."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from .article_recovery import ArticleRecovery, parse_article
from .public_fetch import PublicResponse, fetch_public
from .reference_routing import (
    ReferenceRoute,
    destination_identity,
    normalize_destination,
    route_reference,
)


@dataclass(frozen=True)
class ReferenceInput:
    reference_id: str
    source_post_id: str
    url: str
    source_depth: int = 0


@dataclass(frozen=True)
class ArticleCapture:
    identity_url: str
    response: PublicResponse
    recovery: ArticleRecovery


@dataclass(frozen=True)
class ReferenceAssessment:
    reference_id: str
    code: str


@dataclass(frozen=True)
class ArticleStageOutcome:
    routes: tuple[ReferenceRoute, ...]
    captures: dict[str, ArticleCapture]
    assessments: tuple[ReferenceAssessment, ...]
    reference_captures: dict[str, str]

    def capture_for(self, reference_id):
        identity = self.reference_captures.get(reference_id)
        return self.captures.get(identity) if identity else None

    def assessment_for(self, reference_id):
        return next(
            item.code for item in self.assessments if item.reference_id == reference_id
        )


def assess_recovery(recovery: ArticleRecovery) -> str:
    for code in ("access_restricted", "body_ambiguous", "render_required", "body_missing"):
        if code in recovery.warnings:
            return code
    return "captured_partial" if recovery.text else "body_missing"


def _fetch_error(exc):
    value = str(exc)
    if value == "http_status_401":
        return "http_401"
    if value == "http_status_403":
        return "http_403"
    return "fetch_failed"


def recover_references(
    references: Iterable[ReferenceInput],
    *,
    fetcher: Callable[[str], PublicResponse] = fetch_public,
    parser: Callable[[bytes, str], ArticleRecovery] = parse_article,
) -> ArticleStageOutcome:
    """Fetch each unresolved URL and reuse parsing for known final destinations."""
    request_cache = {}
    aliases = {}
    captures = {}
    reference_captures = {}
    routes = []
    assessments = []
    for reference in references:
        initial = route_reference(
            reference_id=reference.reference_id,
            source_post_id=reference.source_post_id,
            original_url=reference.url,
            final_url=reference.url,
            source_depth=reference.source_depth,
        )
        if initial.classification != "article_candidate":
            routes.append(initial)
            assessments.append(
                ReferenceAssessment(reference.reference_id, initial.classification)
            )
            continue

        request_key = normalize_destination(reference.url)
        try:
            response = request_cache.get(request_key)
            if response is None:
                response = fetcher(reference.url)
                request_cache[request_key] = response
        except (ValueError, OSError) as exc:
            routes.append(initial)
            assessments.append(ReferenceAssessment(reference.reference_id, _fetch_error(exc)))
            continue

        route = route_reference(
            reference_id=reference.reference_id,
            source_post_id=reference.source_post_id,
            original_url=reference.url,
            final_url=response.final_url,
            source_depth=reference.source_depth,
        )
        routes.append(route)
        if route.classification != "article_candidate":
            assessments.append(
                ReferenceAssessment(reference.reference_id, route.classification)
            )
            continue
        if response.content_type and not any(
            value in response.content_type.lower() for value in ("html", "xhtml")
        ):
            assessments.append(ReferenceAssessment(reference.reference_id, "not_article"))
            continue

        final_key = normalize_destination(response.final_url)
        identity = aliases.get(final_key)
        if identity is None:
            recovery = parser(response.body, response.final_url)
            identity = destination_identity(response.final_url, recovery.canonical_url)
            existing = captures.get(identity)
            if existing is None:
                captures[identity] = ArticleCapture(identity, response, recovery)
            aliases[final_key] = identity
            aliases[identity] = identity
        capture = captures[identity]
        route = replace(route, canonical_url=capture.recovery.canonical_url)
        routes[-1] = route
        reference_captures[reference.reference_id] = identity
        assessments.append(
            ReferenceAssessment(reference.reference_id, assess_recovery(capture.recovery))
        )
    return ArticleStageOutcome(
        routes=tuple(routes),
        captures=captures,
        assessments=tuple(assessments),
        reference_captures=reference_captures,
    )
