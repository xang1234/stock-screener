"""Pure, replayable social-signal-v1 scoring over persisted evidence judgments."""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from math import log1p
from typing import Iterable

from .records import (
    ComponentScore, ConfirmationInput, SignalStateDecision, SocialEvidenceInput,
    SocialPostRecord, SocialScoreResult, SocialSnapshotRecord, validate_utc_timestamp,
)

SOCIAL_WEIGHTS = (("acceleration", 20), ("authors", 15), ("engagement", 15), ("recency", 5), ("cross_list", 5))
CONFIRMATION_WEIGHTS = (("setup", 20), ("rs", 10), ("group", 5), ("theme", 5))
PRECISION = Decimal("0.0001")


def _decimal(value) -> Decimal:
    return Decimal(str(value))


def _round(value) -> Decimal:
    return _decimal(value).quantize(PRECISION, rounding=ROUND_HALF_UP)


def engagement_value(*, likes, reposts, replies, quotes=None, bookmarks=None, views=None) -> float | None:
    if any(v is None for v in (likes, reposts, replies)):
        return None
    total = sum((_decimal(v) * _decimal(w) for v, w in (
        (likes, 1), (reposts, 2), (replies, 1.5), (quotes, 1.5), (bookmarks, 2), (views, .001)
    ) if v is not None), Decimal(0))
    return log1p(float(total))


def recency_score(*, age_hours) -> Decimal:
    if age_hours < 0:
        raise ValueError("future_mention")
    return _round(Decimal(100) * (Decimal(2) ** (-_decimal(age_hours) / 48)))


def queue_score(*, social, confirmation) -> Decimal | None:
    if social is None or confirmation is None:
        return None
    return _round(_decimal(social) * Decimal("0.6") + _decimal(confirmation) * Decimal("0.4"))


def midrank_percentile(value, cohort: Iterable) -> Decimal | None:
    values = tuple(_decimal(v) for v in cohort if v is not None)
    if value is None or not values:
        return None
    number = _decimal(value)
    return _round(100 * (sum(v < number for v in values) + Decimal("0.5") * sum(v == number for v in values)) / len(values))


def percentile_threshold(values: Iterable, percentile=Decimal("0.95")) -> Decimal | None:
    ordered = sorted(_decimal(v) for v in values if v is not None)
    if not ordered:
        return None
    offset = (len(ordered) - 1) * _decimal(percentile)
    lower = int(offset)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (offset - lower)


def _component(value, weight, *, observed=1, count=1, reasons=(), selected_key=None) -> ComponentScore:
    return ComponentScore(None if value is None else _round(value), Decimal(weight if value is not None else 0),
                          Decimal(weight), tuple(sorted(set(reasons))), observed if value is not None else 0,
                          count, selected_key=selected_key)


def _aggregate(components: tuple[tuple[str, ComponentScore], ...]) -> ComponentScore:
    available = sum((c.available_weight for _, c in components), Decimal(0))
    total = sum((c.total_weight for _, c in components), Decimal(0))
    value = None if not available else _round(sum((c.value * c.available_weight for _, c in components if c.value is not None), Decimal(0)) / available)
    return ComponentScore(value, available, total,
        tuple(sorted({reason for _, c in components for reason in c.reasons})),
        sum(c.value is not None for _, c in components), len(components), components)


def _post_key(post: SocialPostRecord) -> str:
    return f"{post.provider}:{post.provider_post_id}"


@dataclass
class _Candidate:
    evidence: SocialEvidenceInput
    posts: tuple[SocialPostRecord, ...]
    memberships: tuple[tuple[str, tuple[str, ...]], ...]
    exclusions: tuple[tuple[str, str], ...]
    acceleration: Decimal | None
    authors: int | None
    engagement: tuple[float, ...]
    lists: frozenset[str]
    cross_list: int | None


def _prepare(evidence: SocialEvidenceInput, window_days: int, now: datetime) -> _Candidate:
    enabled = set(evidence.enabled_source_ids)
    grouped = defaultdict(list)
    memberships = defaultdict(set)
    exclusions = set()
    for post in evidence.posts:
        key = _post_key(post)
        try:
            validate_utc_timestamp(post.created_at, "created_at")
            validate_utc_timestamp(post.observed_at, "observed_at")
        except ValueError:
            exclusions.add((key, "invalid_timestamp"))
            continue
        if post.created_at > now or post.observed_at > now:
            exclusions.add((key, "future_timestamp"))
            continue
        memberships[key].add(post.source_id)
        if post.source_id not in enabled:
            exclusions.add((key, "disabled_source"))
        else:
            grouped[key].append(post)
    # Latest saved observation wins; repr gives a stable tie-break for conflicting copies.
    unique = [max(rows, key=lambda p: (p.observed_at, repr(p))) for rows in grouped.values()]
    unique.sort(key=lambda p: (p.created_at, _post_key(p)))
    accepted = []
    authors = defaultdict(deque)
    urls, claims = set(), set()
    for post in unique:
        key = _post_key(post)
        reason = None
        if post.is_repost:
            reason = "repost"
        elif post.quoted_text is not None and post.has_new_thesis is not True:
            reason = "quote_without_new_thesis"
        elif post.canonical_url and post.canonical_url in urls:
            reason = "repeated_canonical_url"
        elif post.canonical_claim_key and post.canonical_claim_key in claims:
            reason = "copied_claim"
        recent = authors[post.author_handle.casefold().lstrip("@")]
        while recent and recent[0] <= post.created_at - timedelta(hours=24):
            recent.popleft()
        if reason is None and len(recent) >= 3:
            reason = "author_24h_cap"
        if reason:
            exclusions.add((key, reason))
            continue
        recent.append(post.created_at)
        if post.canonical_url:
            urls.add(post.canonical_url)
        if post.canonical_claim_key:
            claims.add(post.canonical_claim_key)
        # Carry saved earlier author activity into the first scoring-day cap.
        if post.created_at < now - timedelta(days=14):
            exclusions.add((key, "outside_window"))
            continue
        accepted.append(post)
    current = tuple(p for p in accepted if p.created_at >= now - timedelta(days=window_days))
    recent_count = sum(p.created_at >= now - timedelta(days=1) for p in accepted)
    previous_count = len(accepted) - recent_count
    acceleration = (Decimal(recent_count + 1) / (Decimal(previous_count) / 13 + 1)
                    if evidence.history_complete and current else None)
    values = tuple(value for p in current if (value := engagement_value(
        likes=p.likes, reposts=p.reposts, replies=p.replies, quotes=p.quotes, bookmarks=p.bookmarks, views=p.views)) is not None)
    current_lists = frozenset(s for p in current for s in memberships[_post_key(p)] if s in enabled)
    cross_list = int(len(current) >= 2 and len(current_lists) >= 2) * 100 if current else None
    return _Candidate(evidence, current, tuple(sorted((k, tuple(sorted(v))) for k, v in memberships.items())),
                      tuple(sorted(exclusions)), acceleration,
                      len({p.author_handle.casefold().lstrip("@") for p in current}) if current else None,
                      values, current_lists, cross_list)


def score_social_candidates(evidence: Iterable[SocialEvidenceInput], window_days: int, now: datetime) -> tuple[SocialScoreResult, ...]:
    validate_utc_timestamp(now, "now")
    if window_days not in {1, 7, 14}:
        raise ValueError("invalid_window")
    inputs = sorted((e for e in evidence if e.resolved and e.security_kind in {"stock", "thematic_etf"}), key=lambda e: e.candidate_key)
    if len({e.candidate_key for e in inputs}) != len(inputs):
        raise ValueError("duplicate_candidate_key")
    candidates = [_prepare(e, window_days, now) for e in inputs]
    window_candidates = [c for c in candidates if c.posts]
    markets = defaultdict(list)
    for c in window_candidates:
        markets[c.evidence.market].append(c)
    results = []
    for c in candidates:
        local = markets[c.evidence.market]
        scope = "market" if len(local) >= 20 else "global_fallback"
        cohort = local if scope == "market" else window_candidates
        # A provider post occurring for several tickers is still one per-post observation.
        engagement_posts = {}
        for peer in cohort:
            for p in peer.posts:
                key = _post_key(p)
                if key not in engagement_posts or (p.observed_at, repr(p)) > (engagement_posts[key].observed_at, repr(engagement_posts[key])):
                    engagement_posts[key] = p
        threshold = percentile_threshold(engagement_value(likes=p.likes, reposts=p.reposts, replies=p.replies,
            quotes=p.quotes, bookmarks=p.bookmarks, views=p.views) for p in engagement_posts.values())
        def engagement_sum(peer):
            if not peer.engagement or threshold is None:
                return None
            return sum((min(_decimal(v), threshold) for v in peer.engagement), Decimal(0))
        summed = engagement_sum(c)
        latest = max((p.created_at for p in c.posts), default=None)
        missing = c.evidence.coverage_reasons
        values = (
            ("acceleration", _component(midrank_percentile(c.acceleration, (p.acceleration for p in cohort)), 20,
                reasons=missing + (() if c.acceleration is not None else ("missing_history" if not c.evidence.history_complete else "no_qualifying_posts",)))),
            ("authors", _component(midrank_percentile(c.authors, (p.authors for p in cohort)), 15,
                observed=len(c.posts), count=len(c.posts), reasons=missing + (() if c.posts else ("no_qualifying_posts",)))),
            ("engagement", _component(midrank_percentile(summed, (engagement_sum(p) for p in cohort)), 15,
                observed=len(c.engagement), count=len(c.posts), reasons=missing + (("missing_engagement_metrics",) if len(c.engagement) < len(c.posts) else ()) + (() if c.posts else ("no_qualifying_posts",)))),
            ("recency", _component(recency_score(age_hours=(now-latest).total_seconds()/3600) if latest else None, 5,
                reasons=missing + (() if latest else ("no_qualifying_posts",)))),
            ("cross_list", _component(c.cross_list, 5, observed=len(c.lists), count=len(c.evidence.enabled_source_ids),
                reasons=missing + (() if c.posts else ("no_qualifying_posts",)))),
        )
        results.append(SocialScoreResult(_aggregate(values).value, values,
            SignalStateDecision("watch", ("confirmation_not_evaluated",)), c.evidence.candidate_key,
            c.evidence.canonical_symbol, c.evidence.market, scope, latest, len(c.posts), len(c.lists),
            len(set(c.evidence.enabled_source_ids)), c.memberships, c.exclusions, c.acceleration,
            float(summed) if summed is not None else None))
    return tuple(results)


def _rating(value) -> Decimal | None:
    if value is None:
        return None
    value = _decimal(value)
    return value if value.is_finite() and 0 <= value <= 100 else None


def _theme_component(input: ConfirmationInput) -> ComponentScore:
    choices, reasons = [], []
    for theme in sorted(input.theme_confirmations, key=lambda t: t.theme_key):
        prefix = f"theme:{theme.theme_key}"
        reasons.extend(f"{prefix}:{k}:{v}" for k, v in theme.reasons)
        if theme.market != input.market or not input.market_benchmark or theme.benchmark_symbol != input.market_benchmark:
            reasons.append(f"{prefix}:market_or_benchmark_mismatch")
            continue
        if theme.session_date > input.observed_at.date():
            reasons.append(f"{prefix}:future_session")
            continue
        metrics, counts = dict(theme.components), dict(theme.measured_company_counts)
        valid = []
        for canonical in ("basket_rs_vs_benchmark", "avg_rs_rating", "pct_above_50ma"):
            name = "basket_rs_vs_spy" if canonical == "basket_rs_vs_benchmark" and canonical not in metrics and input.market == "US" else canonical
            value = _rating(metrics.get(name))
            count = counts.get(name, 0)
            if count < 3 or count * 10 < theme.accepted_company_count * 7:
                reasons.append(f"{prefix}:{name}:insufficient_coverage")
            elif value is None:
                reasons.append(f"{prefix}:{name}:missing_value")
            else:
                valid.append(value)
        if valid:
            choices.append((sum(valid) / len(valid), theme.theme_key, len(valid)))
    if not choices:
        return _component(None, 5, count=3, reasons=tuple(reasons) + ("missing_theme",))
    value, key, count = sorted(choices, key=lambda c: (-c[0], c[1]))[0]
    return _component(value, 5, observed=count, count=3, reasons=reasons, selected_key=key)


def score_confirmation(input: ConfirmationInput) -> ComponentScore:
    setup = _rating(input.setup_score)
    rs = []
    reasons = []
    for name, value, weight in (("rs_rating_1m", input.rs_rating_1m, 4), ("rs_rating_3m", input.rs_rating_3m, 6)):
        value = _rating(value)
        if value is None:
            reasons.append(f"missing_{name}")
        else:
            rs.append((value, weight))
    relative = sum(v * w for v, w in rs) / sum(w for _, w in rs) if rs else None
    group = None
    if input.group_rank is not None and input.market_group_count is not None and input.market_group_count >= 2:
        group = max(Decimal(0), min(Decimal(100), 100 * (1 - Decimal(input.group_rank - 1) / (input.market_group_count - 1))))
    return _aggregate((
        ("setup", _component(setup, 20, reasons=() if setup is not None else ("missing_setup",))),
        ("rs", _component(relative, 10, observed=len(rs), count=2, reasons=reasons)),
        ("group", _component(group, 5, reasons=() if group is not None else ("missing_group",))),
        ("theme", _theme_component(input)),
    ))


def snapshot_rank_order(mode: str) -> tuple[tuple[str, str, str], ...]:
    """Shared field/direction/null policy for pure and persisted ranked rows."""
    if mode not in {"pure_social", "blended"}:
        raise ValueError("invalid_rank_mode")
    social = (("social_score", "desc", "last"), ("latest_mention", "desc", "last"),
              ("canonical_symbol", "asc", "last"), ("candidate_key", "asc", "last"))
    return (("queue_score", "desc", "last"),) + social if mode == "blended" else social


def rank_snapshots(rows: Iterable[SocialSnapshotRecord], mode: str) -> tuple[SocialSnapshotRecord, ...]:
    ordering = snapshot_rank_order(mode)
    def key(row):
        values = []
        for field, direction, _ in ordering:
            value = getattr(row, field)
            if field == "latest_mention" and value is not None:
                validate_utc_timestamp(value, "latest_mention")
                value = value.timestamp()
            values.append((value is None, 0 if value is None else -value if direction == "desc" else value))
        return tuple(values)
    return tuple(sorted((r for r in rows if r.candidate_state not in {"context", "unresolved"}), key=key))
