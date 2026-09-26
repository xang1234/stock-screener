"""Bounded primary synthesis and evidence-dependency checks (spec §5.4, §12.3).

A synthesized claim may join at most three original primary premises with
at most two explicit links. Each link must be stated in a premise quote that
names both ends, and only these relationships can carry a product toward a
theme application:

* ``issuer_offers_product`` — the issuer sells/ships/offers the product;
* ``product_supports_application`` — the product is designed for / supports
  / tests the application;
* ``segment_of_issuer`` — a segment/subsidiary belongs to the issuer.

Customer/supplier relationships (``supplies_to``, ``customer_of``,
``manufactures``) never carry an application across companies: "A supplies
B and B makes HBM" does not establish that A's product serves HBM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.company_exposure.policy import (
    MAX_SYNTHESIS_LINKS,
    MAX_SYNTHESIS_PRIMARY_PREMISES,
    within_synthesis_bound,
)

APPLICATION_LINKS = frozenset(
    {"issuer_offers_product", "product_supports_application", "segment_of_issuer"}
)
CROSS_COMPANY_LINKS = frozenset({"supplies_to", "customer_of", "manufactures"})


class EvidenceCycle(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Premise:
    ref: str
    quote: str
    primary: bool


@dataclass(frozen=True, slots=True)
class Link:
    source: str
    target: str
    relationship: str
    premise_ref: str


@dataclass(frozen=True, slots=True)
class SynthesisDecision:
    permitted: bool
    reasons: tuple[str, ...] = ()
    premises: tuple[Premise, ...] = ()
    links: tuple[Link, ...] = ()
    forbidden_extrapolations: tuple[str, ...] = field(
        default=("theme_specific_sales", "named_customers", "revenue_share")
    )


def _mentions(quote: str, entity: str) -> bool:
    return entity.casefold() in quote.casefold()


def validate_synthesis(
    premises: list[Premise],
    links: list[Link],
    *,
    subject: str,
    application: str,
) -> SynthesisDecision:
    """Permit only a fully evidenced chain from the issuer's subject to the
    application; anything missing or cross-company is held."""

    reasons: list[str] = []
    by_ref = {premise.ref: premise for premise in premises}
    if not within_synthesis_bound([p.ref for p in premises], links):
        reasons.append(
            f"exceeds_bound_{MAX_SYNTHESIS_PRIMARY_PREMISES}_premises_"
            f"{MAX_SYNTHESIS_LINKS}_links"
        )
    if any(not premise.primary for premise in premises):
        reasons.append("non_primary_premise")
    for link in links:
        premise = by_ref.get(link.premise_ref)
        if premise is None:
            reasons.append("link_premise_missing")
            continue
        if not (_mentions(premise.quote, link.source) and _mentions(premise.quote, link.target)):
            reasons.append("link_not_stated_in_premise")
        if link.relationship in CROSS_COMPANY_LINKS:
            reasons.append("cross_company_link_cannot_carry_application")
        elif link.relationship not in APPLICATION_LINKS:
            reasons.append("unsupported_link_relationship")
    # The chain must actually connect subject -> application.
    edges = {
        link.source.casefold(): link.target.casefold()
        for link in links
        if link.relationship in APPLICATION_LINKS
    }
    node = subject.casefold()
    seen = set()
    while node in edges and node not in seen:
        seen.add(node)
        node = edges[node]
    if node != application.casefold():
        reasons.append("application_link_missing")
    reasons = list(dict.fromkeys(reasons))
    return SynthesisDecision(
        permitted=not reasons,
        reasons=tuple(reasons),
        premises=tuple(premises),
        links=tuple(links),
    )


def verify_evidence_dag(edges: dict[str, list[str]]) -> None:
    """Reject any dependency cycle (e.g. research → classification → research).

    ``edges`` maps a node to the nodes it depends on. Raises
    ``EvidenceCycle`` naming the cycle.
    """

    white, grey, black = 0, 1, 2
    colour: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        colour[node] = grey
        for dependency in edges.get(node, []):
            state = colour.get(dependency, white)
            if state == grey:
                cycle = path[path.index(dependency) :] + [dependency] if dependency in path else [node, dependency]
                raise EvidenceCycle("evidence_cycle:" + "->".join(cycle))
            if state == white:
                visit(dependency, [*path, dependency])
        colour[node] = black

    for start in list(edges):
        if colour.get(start, white) == white:
            visit(start, [start])


def primary_leaves(edges: dict[str, list[str]], leaf_roles: dict[str, str], root: str) -> set[str]:
    """Original-primary leaves reachable from ``root``; derivatives and
    generated artifacts are traversed through, never counted as leaves."""

    verify_evidence_dag(edges)
    leaves, stack, seen = set(), [root], set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        dependencies = edges.get(node, [])
        if not dependencies and leaf_roles.get(node) == "original_primary":
            leaves.add(node)
        stack.extend(dependencies)
    return leaves
