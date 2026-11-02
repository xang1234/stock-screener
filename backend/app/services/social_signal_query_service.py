"""Pointer-scoped, frozen Social Signal read projections."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from html import unescape
from html.parser import HTMLParser
from urllib.parse import quote

from sqlalchemy import select

from app.config import settings
from app.domain.social_signals.states import ACTIONABLE_EXPOSURE_FLOOR
from app.infra.db.models.social_signals import (
    SocialPostTicker, SocialSignalRun, SocialSignalRunPointer,
    SocialSignalSnapshot, SocialSourceRegistry,
)
from app.infra.db.repositories.published_social_signal_reader import (
    PublishedSocialSignalReader,
)


WINDOW_DAYS = {"1d": 1, "7d": 7, "14d": 14}
_PUBLIC_ATTEMPT_REASONS = {
    "bounded_provider_read", "daily_limit_exhausted", "incremental_gap_suspected",
    "provider_error", "provider_network_error",
    "provider_timeout", "provider_unavailable", "rate_limited",
    "reauthentication_required",
}


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _excerpt(value):
    parser = _TextOnly()
    parser.feed(unescape(str(value or "")))
    return " ".join("".join(parser.parts).split())[:280]


class SocialSignalQueries:
    def __init__(self, db, *, clock=None):
        self.db = db
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.reader = PublishedSocialSignalReader(lambda: nullcontext(db))

    def _supported(self):
        registry = self.db.get(SocialSourceRegistry, 1)
        return bool(registry and registry.mode == "live" and registry.provider != "disabled")

    def _publication(self):
        pointer = self.db.get(SocialSignalRunPointer, "latest_published")
        return self.db.get(SocialSignalRun, pointer.run_id) if pointer else None

    @staticmethod
    def _oldest_successful_observation_at(run):
        source_ids = set(run.application_progress_json.get("sources", {}))
        observations = run.application_progress_json.get("observations", {})
        outcomes = run.source_outcomes_json or {}
        observed_at = []
        for source_id in source_ids:
            outcome = (
                outcomes.get(source_id) or outcomes.get(str(source_id)) or {}
            )
            observation = (
                observations.get(source_id)
                or observations.get(str(source_id))
                or {}
            )
            if outcome.get("read_status") == "failed":
                continue  # a failed list is a coverage gap, not stale data
            if (
                outcome.get("read_status") != "success"
                or not observation.get("observed_at")
            ):
                return None
            value = observation["observed_at"]
            try:
                parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
                    str(value).replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                return None
            observed_at.append(_utc(parsed))
        return min(observed_at) if observed_at else None

    def latest_live_attempt(self):
        run = self.db.scalar(select(SocialSignalRun).where(
            SocialSignalRun.mode == "live"
        ).order_by(
            SocialSignalRun.created_at.desc(), SocialSignalRun.id.desc()
        ).limit(1))
        if run is None:
            return None
        configured = run.application_progress_json.get("sources", {})
        outcomes = run.source_outcomes_json or {}
        sources = []
        for source_id, source in sorted(
            configured.items(),
            key=lambda item: (
                0, int(item[0])
            ) if str(item[0]).isdigit() else (1, str(item[0])),
        ):
            outcome = outcomes.get(source_id) or {}
            read_status = outcome.get("read_status")
            if read_status not in {"success", "failed"}:
                read_status = "pending"
            received_count = outcome.get("received_count")
            if not isinstance(received_count, int) or isinstance(received_count, bool):
                received_count = None
            sources.append({
                "name": str(source.get("name") or f"Source {source_id}"),
                "read_status": read_status,
                "received_count": received_count,
                "history_status": outcome.get("history_status"),
                "reason_codes": sorted({
                    reason for reason in outcome.get("coverage_reason_codes", [])
                    if reason in _PUBLIC_ATTEMPT_REASONS
                }),
            })
        statuses = {source["read_status"] for source in sources}
        if statuses == {"failed"}:
            status = "collection_failed"
        elif "pending" in statuses or not sources:
            status = "collecting"
        elif run.status == "running":
            status = "processing"
        elif run.status == "published":
            status = "published"
        else:
            status = "failed"
        return {
            "run_id": run.id,
            "status": status,
            "started_at": _utc(run.created_at),
            "completed_at": _utc(run.completed_at),
            "sources": sources,
        }

    @staticmethod
    def _candidate_explanation(run, candidate_key, window_days):
        prepared = (run.prepared_json or {}) if run else {}
        context = prepared.get("context") or {}
        candidate = next((value for value in context.get("candidates", ())
                          if value.get("candidate_key") == candidate_key
                          and value.get("window_days") == window_days), None)
        if candidate is None:
            return {}
        state_input = candidate.get("state_input") or {}
        social = candidate.get("social_result") or {}
        confirmation = candidate.get("confirmation") or {}
        confirmation_components = dict(confirmation.get("components", ()))
        social_components = dict(social.get("components", ()))
        market_input, facts = None, {}
        for batch in context.get("market_batches", ()):
            market_input = next((value for value in batch.get("inputs", ())
                                 if value.get("candidate_key") == candidate_key), None)
            if market_input:
                facts = dict(batch.get("facts", ())).get(candidate_key.partition(":")[2]) or {}
                break
        feature_freshness = facts.get("feature_freshness") or {}
        theme_component = confirmation_components.get("theme") or {}
        memberships = social.get("post_memberships", [])
        source_ids = set()
        for membership in memberships:
            if isinstance(membership, str):
                source_ids.add(membership)
            elif isinstance(membership, (list, tuple)) and len(membership) == 2:
                values = membership[1]
                if isinstance(values, str):
                    source_ids.add(values)
                elif isinstance(values, (list, tuple)):
                    source_ids.update(str(value) for value in values)
        pins = run.application_progress_json.get("sources", {})
        source_names = sorted({
            str(pin.get("name") or source_id)
            for source_id, pin in pins.items()
            if str(source_id) in source_ids
            or str(pin.get("list_id")) in source_ids
        })
        return {
            "state_reasons": list((candidate.get("state_decision") or {}).get("reasons", ())),
            "security_kind": state_input.get("security_kind"),
            "setup_score": state_input.get("setup_score"),
            "readiness": ("ready" if state_input.get("setup_ready") is True else
                          "not ready" if state_input.get("setup_ready") is False else None),
            "market_exposure": state_input.get("market_exposure"),
            "rs_rating_1m": (market_input or {}).get("rs_rating_1m"),
            "rs_rating_3m": (market_input or {}).get("rs_rating_3m"),
            "group_rank": (market_input or {}).get("group_rank"),
            "feature_sessions_behind": (feature_freshness.get("sessions_behind")
                                        if feature_freshness.get("fresh") else None),
            "feature_required_session": feature_freshness.get("required_session"),
            "theme": theme_component.get("selected_key"),
            "confirmation_components": confirmation_components,
            "social_components": social_components,
            "acceleration": social.get("acceleration"),
            "post_memberships": social.get("post_memberships", []),
            "source_names": source_names,
        }

    @classmethod
    def _item(cls, record, run=None):
        pinned = dict(record.pinned_inputs)
        reasons = pinned.get("state_reasons")
        if isinstance(reasons, str):
            pinned["state_reasons"] = [value for value in reasons.split(",") if value]
        pinned.update(cls._candidate_explanation(
            run, record.candidate_key, record.window_days
        ))
        return {
            "candidate_key": record.candidate_key,
            "canonical_symbol": record.canonical_symbol,
            "market": record.market,
            "state": record.candidate_state,
            "social_score": record.social_score,
            "confirmation_score": record.confirmation_score,
            "queue_score": record.queue_score,
            "latest_mention": record.latest_mention,
            "mention_count": record.mention_count,
            "observed_list_count": record.observed_list_count,
            "enabled_list_count": record.enabled_list_count,
            "normalization_scope": record.normalization_scope,
            "formula_version": record.formula_version,
            "coverage": list(record.coverage),
            "explanation": pinned,
        }

    def queue(
        self, *, market, window, view, rank_mode, page, page_size, filters=None,
        watch_fallback=True,
    ):
        base = {
            "supported": self._supported(), "available": False,
            "reason_code": "social_signals_disabled", "market": market,
            "window": window, "view": view, "rank_mode": rank_mode,
            "page": page, "page_size": page_size, "total": 0, "items": [],
        }
        if not base["supported"]:
            return base
        filters = {
            key: str(value).strip().casefold()
            for key, value in (filters or {}).items()
            if value is not None and str(value).strip()
        }
        run = self._publication()
        if run is None:
            attempt = self.latest_live_attempt()
            return {
                **base, "reason_code": "no_published_run",
                "latest_attempt": attempt and {"status": attempt["status"]},
            }

        def matches(record):
            item = self._item(record, run)
            explanation = item["explanation"]
            haystacks = {
                "source": " ".join(explanation.get("source_names") or ()).casefold(),
                "theme": str(explanation.get("theme") or "").casefold(),
                "instrument": str(explanation.get("security_kind") or "stock").casefold(),
                "state": str(item.get("state") or "").casefold(),
                "ticker": str(item.get("canonical_symbol") or "").casefold(),
            }
            return all(value in haystacks.get(key, "") for key, value in filters.items())

        days, scope = WINDOW_DAYS[window], {"item_filter": matches if filters else None, "run_id": run.id}
        result = self.reader.queue(market, days, view, rank_mode, page, page_size, **scope)
        total, items = result.total, [self._item(item, run) for item in result.items]
        fallback = None
        if watch_fallback and view == "actionable" and not total and not filters:
            watch = self.reader.rows(market, days, "watch", rank_mode, run_id=run.id)
            risk_off = self.reader.queue(market, days, "risk_off", rank_mode, 1, 1, run_id=run.id)
            fallback = self._why_not_actionable(watch, risk_off, run)
            if fallback:
                total = len(watch)
                items = [self._item(record, run)
                         for record in watch[(page - 1) * page_size:page * page_size]]
        generated_at = _utc(run.created_at)
        published_at = _utc(run.published_at)
        oldest_observation_at = self._oldest_successful_observation_at(run)
        return {
            **base,
            "available": True,
            "reason_code": None,
            "total": total,
            "items": items,
            "run_id": run.id,
            "generated_at": generated_at,
            "published_at": published_at,
            "stale": bool(
                oldest_observation_at is None
                or (self.clock() - oldest_observation_at).total_seconds()
                > settings.social_stale_after_hours * 3600
            ),
            "fallback": fallback,
        }

    def _why_not_actionable(self, watch, risk_off, run):
        if not watch and not risk_off.total:
            return None
        reasons = [
            [value for value in dict(record.pinned_inputs).get("state_reasons", "").split(",") if value]
            for record in watch
        ]
        stale = [record for record, values in zip(watch, reasons) if "stale_features" in values]
        counts = Counter(reason for values in reasons for reason in values)
        if risk_off.items:
            code = "market_exposure_below_50"
            exposure = self._candidate_explanation(
                run, risk_off.items[0].candidate_key, risk_off.items[0].window_days
            ).get("market_exposure")
            params = {"market_exposure": exposure, "floor": ACTIONABLE_EXPOSURE_FLOOR}
        elif stale:
            code = "stale_features"
            params = {"count": len(stale), "required_session": self._candidate_explanation(
                run, stale[0].candidate_key, stale[0].window_days
            ).get("feature_required_session")}
        elif counts:
            code, count = counts.most_common(1)[0]
            params = {"count": count}
        else:
            code, params = None, {}
        return {
            "from": "actionable", "to": "watch", "reason_code": code,
            "reason_params": params, "watch_count": len(watch),
            "risk_off_count": risk_off.total,
        }

    def section(self, *, market, window, section, page, page_size):
        result = self.queue(
            market=market, window=window, view=section,
            rank_mode="blended", page=page, page_size=page_size,
        )
        return {
            "supported": result["supported"],
            "available": result["available"],
            "reason_code": result["reason_code"],
            "market": market,
            "window": window,
            "section": section,
            "page": page,
            "page_size": page_size,
            "total": result["total"],
            "items": result["items"],
            "run_id": result.get("run_id"),
        }

    @staticmethod
    def _canonical_post_url(post):
        handle = str(post.get("author_handle") or "unknown").lstrip("@")
        return "https://x.com/{}/status/{}".format(
            quote(handle, safe="_"), quote(str(post["provider_post_id"]), safe="")
        )

    def evidence(self, *, candidate_key, window):
        base = {
            "supported": self._supported(), "available": False,
            "reason_code": "social_signals_disabled", "candidate_key": candidate_key,
            "window": window, "item": None, "posts": [], "related_listings": [],
        }
        if not base["supported"]:
            return base
        run = self._publication()
        if run is None:
            return {**base, "reason_code": "no_published_run"}
        snapshot = self.db.scalar(select(SocialSignalSnapshot).where(
            SocialSignalSnapshot.run_id == run.id,
            SocialSignalSnapshot.window_days == WINDOW_DAYS[window],
            SocialSignalSnapshot.candidate_key == candidate_key,
        ))
        if snapshot is None:
            return {**base, "reason_code": "candidate_not_found"}
        data = dict(snapshot.explanation_json["record"])
        data["pinned_inputs"] = tuple(tuple(value) for value in data["pinned_inputs"])
        data["coverage"] = tuple(data["coverage"])
        data["latest_mention"] = (
            datetime.fromisoformat(data["latest_mention"])
            if data["latest_mention"] else None
        )
        from app.domain.social_signals.records import SocialSnapshotRecord
        for field in ("social_score", "confirmation_score", "queue_score"):
            data[field] = Decimal(data[field]) if data[field] is not None else None
        item = self._item(SocialSnapshotRecord(**data), run)
        pins = run.application_progress_json.get("sources", {})
        collected = {}
        cutoff = _utc(run.created_at) - timedelta(days=WINDOW_DAYS[window])
        from app.infra.db.repositories.social_refresh_support import (
            SocialScoringEvidenceReader,
        )
        rolling = SocialScoringEvidenceReader.read_in_session(
            self.db, run.id, _utc(run.created_at)
        )
        candidate = next(
            (value for value in rolling if value.candidate_key == candidate_key),
            None,
        )
        for record in candidate.posts if candidate else ():
            if record.created_at < cutoff or record.created_at > _utc(run.created_at):
                continue
            post = {
                "provider": record.provider,
                "provider_post_id": record.provider_post_id,
                "author_handle": record.author_handle,
                "created_at": record.created_at.isoformat(),
                "text": record.text,
                **{field: getattr(record, field) for field in (
                    "likes", "reposts", "replies", "quotes", "bookmarks", "views"
                )},
            }
            key = (record.provider, record.provider_post_id)
            value = collected.setdefault(key, {"post": post, "sources": set()})
            value["sources"].add(
                pins.get(record.source_id, {}).get("name", record.source_id)
            )
        posts = []
        for value in collected.values():
            post = value["post"]
            excerpt = _excerpt(post.get("text"))
            posts.append({
                "post_id": str(post["provider_post_id"]),
                "author_handle": str(post.get("author_handle") or ""),
                "created_at": datetime.fromisoformat(post["created_at"]),
                "excerpt": excerpt,
                "url": self._canonical_post_url(post),
                "source_names": sorted(value["sources"]),
                "engagement": {field: post.get(field) for field in (
                    "likes", "reposts", "replies", "quotes", "bookmarks", "views"
                )},
            })
        posts.sort(key=lambda value: (value["created_at"], value["post_id"]), reverse=True)
        from app.models.stock_universe import StockUniverse
        from app.services.social_company_identity_service import (
            SocialCompanyIdentityService,
        )
        from app.services.social_ticker_resolver import SocialTickerResolver
        related, resolver, listing = [], None, None
        if item["canonical_symbol"] and item["market"]:
            identities = SocialCompanyIdentityService(self.db).read()
            resolver = SocialTickerResolver(
                self.db, verified_company_ids=identities.verified_company_ids,
            )
            listing = (item["market"], item["canonical_symbol"])
        # Post hints need no listing, so unresolved candidates get them too.
        claims, hints = self._extractions(run, {post["post_id"] for post in posts}, listing, resolver)
        for post in posts:
            post["post_kind"], post["stance"] = hints.get(post["post_id"], (None, None))
        if resolver is not None:
            resolution = resolver.resolve(item["canonical_symbol"], item["market"])
            if resolution.related_symbols:
                related = [
                    (row.market, row.symbol)
                    for row in self.db.scalars(select(StockUniverse).where(
                        StockUniverse.symbol.in_(resolution.related_symbols),
                        StockUniverse.active_filter(),
                    ).order_by(StockUniverse.market, StockUniverse.symbol))
                ]
        return {
            **base, "available": True, "reason_code": None, "item": item,
            "posts": posts[:3],
            "claims": claims,
            "related_listings": [
                {"market": market, "canonical_symbol": symbol}
                for market, symbol in related
            ],
        }

    _SUPPORT_ORDER = {"supported": 0, "uncertain": 1, "unsupported": 2}

    def _extractions(self, run, post_ids, listing, resolver, limit=5):
        """From the run's pinned work: this listing's claims, supported first, and
        each post's (post_kind, stance) hints."""
        from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
        from app.services.social_theme_projection_service import _decode

        # ponytail: loads every succeeded work pinned to the run, as the scoring
        # reader does; filter by content item ids in SQL if drawer opens get slow.
        # (Not by ticker rows: posts also join a candidate through claim tokens.)
        works = self.db.scalars(select(SocialExtractionWork).join(
            SocialRunWork, SocialRunWork.work_id == SocialExtractionWork.id,
        ).where(
            SocialRunWork.run_id == run.id, SocialExtractionWork.state == "succeeded",
        ).order_by(SocialExtractionWork.id.desc()))
        found, hints, resolved = {}, {}, {}
        for work in works:
            post_id = (work.input_snapshot_json or {}).get("provider_post_id")
            # Newest work per post only: an edited post's older revision is superseded.
            if post_id not in post_ids or post_id in hints:
                continue
            hints[post_id] = (None, None)
            try:
                post, result = _decode(work)
            except ValueError:
                # A saved result that no longer validates is not evidence to show.
                continue
            judgment = result.judgments[0]
            hints[post_id] = (judgment.post_kind, judgment.stance)
            # Claims are only shown for a resolved listing.
            for claim in result.claims if listing is not None else ():
                if claim.company_token not in resolved:
                    resolved[claim.company_token] = resolver.resolve(claim.company_token)
                resolution = resolved[claim.company_token]
                if resolution.status != "resolved" or (resolution.market, resolution.symbol) != listing:
                    continue
                excerpt = _excerpt(claim.excerpt)
                # The parser verified a cited excerpt against this attachment's text.
                attachment = next((item for item in post.prepared_evidence
                                   if claim.evidence_id and item.id == claim.evidence_id), None)
                found.setdefault((post.provider_post_id, claim.raw_theme, excerpt), {
                    "post_id": post.provider_post_id,
                    "author_handle": post.author_handle,
                    "created_at": post.created_at,
                    "url": self._canonical_post_url({
                        "author_handle": post.author_handle,
                        "provider_post_id": post.provider_post_id,
                    }),
                    "theme": claim.raw_theme,
                    "relationship": claim.relationship,
                    "excerpt": excerpt,
                    "support": claim.support,
                    "source": attachment.kind if attachment else "post",
                    "source_url": attachment.url if attachment else None,
                })
        claims = sorted(found.values(), key=lambda value: (
            self._SUPPORT_ORDER[value["support"]], -value["created_at"].timestamp(), value["post_id"],
        ))[:limit]
        return claims, hints

    def summary(self, *, market):
        if not self._supported():
            return {
                "supported": False, "available": False,
                "reason_code": "social_signals_disabled", "market": market,
            }
        queue = self.queue(
            market=market, window="7d", view="actionable",
            rank_mode="blended", page=1, page_size=5, watch_fallback=False,
        )
        if not queue["available"]:
            return {
                "supported": True, "available": False,
                "reason_code": queue["reason_code"], "market": market,
            }
        run = self.db.get(SocialSignalRun, queue["run_id"])
        pins = run.application_progress_json.get("sources", {})
        observations = run.application_progress_json.get("observations", {})
        themes = [value for value in (run.prepared_json or {}).get("theme_evidence", [])
                  if value.get("market") == market]
        themes.sort(key=lambda value: (
            -int(value.get("accepted_company_count", 0)), value.get("theme_key", "")
        ))
        return {
            "supported": True, "available": True, "reason_code": None,
            "market": market, "generated_at": queue["generated_at"],
            "published_at": queue["published_at"], "stale": queue["stale"],
            "top_signals": queue["items"], "dominant_themes": [{
                "theme_key": value.get("theme_key"),
                "accepted_company_count": value.get("accepted_company_count", 0),
                "components": dict(value.get("components", ())),
                "reasons": dict(value.get("reasons", ())),
            } for value in themes[:5]],
            "enabled_source_count": len(pins),
            "participating_source_count": len(observations),
        }


__all__ = ["SocialSignalQueries", "WINDOW_DAYS"]
