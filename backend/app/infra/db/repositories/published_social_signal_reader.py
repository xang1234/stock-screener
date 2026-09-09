"""Read one immutable publication; never join current engagement or work."""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import case, func, select

from app.domain.social_signals.records import QueuePage, SocialSnapshotRecord, SUPPORTED_MARKETS
from app.domain.social_signals.scoring import snapshot_rank_order
from app.infra.db.models.social_signals import SocialSignalRunPointer, SocialSignalSnapshot


class SocialPublicationUnavailable(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class PublishedSocialSignalReader:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    @staticmethod
    def _record(row):
        data = dict(row.explanation_json["record"])
        data["pinned_inputs"] = tuple(tuple(value) for value in data["pinned_inputs"])
        data["coverage"] = tuple(data["coverage"])
        data["latest_mention"] = datetime.fromisoformat(data["latest_mention"]) if data["latest_mention"] else None
        for field in ("social_score", "confirmation_score", "queue_score"):
            data[field] = Decimal(data[field]) if data[field] is not None else None
        return SocialSnapshotRecord(**data)

    def queue(
        self, market, window_days, view, rank_mode, page, page_size,
        *, item_filter=None,
    ):
        if market not in SUPPORTED_MARKETS or window_days not in {1, 7, 14}:
            raise ValueError("invalid_market_or_window")
        if view not in {"all", "actionable", "watch", "risk_off", "context", "unresolved"}:
            raise ValueError("invalid_queue_selection")
        ordering = snapshot_rank_order(rank_mode)
        if page < 1 or not 1 <= page_size <= 200:
            raise ValueError("invalid_pagination")
        with self.session_factory() as db:
            pointer = db.get(SocialSignalRunPointer, "latest_published")
            if pointer is None:
                raise SocialPublicationUnavailable("no_published_run")
            run_id = pointer.run_id
            predicates = [SocialSignalSnapshot.run_id == run_id,
                SocialSignalSnapshot.window_days == window_days,
                (SocialSignalSnapshot.market == market) | SocialSignalSnapshot.market.is_(None)]
            ranked = SocialSignalSnapshot.state.not_in(("context", "unresolved")) & SocialSignalSnapshot.market.is_not(None)
            if view == "all":
                predicates.append(ranked)
            else:
                predicates.append(SocialSignalSnapshot.state == view)
            # Unranked sections ignore scores and dates: context, then resolution,
            # each alphabetically stable. They never join a candidate cohort.
            order_by = [case((ranked, 0), (SocialSignalSnapshot.state == "context", 1), else_=2)]
            for field, direction, null_policy in ordering:
                column = getattr(SocialSignalSnapshot, field)
                if field in {"canonical_symbol", "candidate_key"}:
                    column = column.collate("C" if db.get_bind().dialect.name == "postgresql" else "BINARY")
                else:
                    column = case((ranked, column), else_=None)
                ordered = column.desc() if direction == "desc" else column.asc()
                order_by.append(ordered.nulls_last() if null_policy == "last" else ordered.nulls_first())
            query = select(SocialSignalSnapshot).where(*predicates).order_by(*order_by)
            if item_filter is None:
                total = db.scalar(select(func.count()).select_from(SocialSignalSnapshot).where(*predicates))
                rows = db.scalars(query.offset((page-1)*page_size).limit(page_size)).all()
                items = tuple(self._record(row) for row in rows)
            else:
                records = tuple(self._record(row) for row in db.scalars(query).all())
                filtered = tuple(record for record in records if item_filter(record))
                total = len(filtered)
                start = (page - 1) * page_size
                items = filtered[start:start + page_size]
            return QueuePage(items, page, page_size, total)
