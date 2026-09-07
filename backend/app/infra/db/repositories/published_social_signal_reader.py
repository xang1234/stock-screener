"""Read one immutable publication; never join current engagement or work."""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import select

from app.domain.social_signals.records import QueuePage, SocialSnapshotRecord, SUPPORTED_MARKETS
from app.infra.db.models.social_signals import SocialSignalRunPointer, SocialSignalSnapshot


class SocialPublicationUnavailable(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class PublishedSocialSignalReader:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def queue(self, market, window_days, view, rank_mode, page, page_size):
        if market not in SUPPORTED_MARKETS or window_days not in {1, 7, 14}:
            raise ValueError("invalid_market_or_window")
        if view not in {"all", "actionable", "watch", "risk_off", "context", "unresolved"} or rank_mode not in {"blended", "social"}:
            raise ValueError("invalid_queue_selection")
        if page < 1 or not 1 <= page_size <= 200:
            raise ValueError("invalid_pagination")
        with self.session_factory() as db:
            pointer = db.get(SocialSignalRunPointer, "latest_published")
            if pointer is None:
                raise SocialPublicationUnavailable("no_published_run")
            run_id = pointer.run_id
            rows = db.scalars(select(SocialSignalSnapshot).where(SocialSignalSnapshot.run_id == run_id,
                SocialSignalSnapshot.window_days == window_days,
                (SocialSignalSnapshot.market == market) | SocialSignalSnapshot.market.is_(None))).all()
            if view != "all":
                rows = [row for row in rows if row.state == view]
            def ordering(row):
                score = row.queue_score if rank_mode == "blended" else row.social_score
                return (row.state in {"context", "unresolved"} or row.market is None,
                    score is None, -(score or Decimal(0)), -(row.social_score or Decimal(0)), row.candidate_key)
            rows.sort(key=ordering)
            items = []
            for row in rows[(page-1)*page_size:page*page_size]:
                data = dict(row.explanation_json["record"])
                data["pinned_inputs"] = tuple(tuple(value) for value in data["pinned_inputs"])
                data["coverage"] = tuple(data["coverage"])
                data["latest_mention"] = datetime.fromisoformat(data["latest_mention"]) if data["latest_mention"] else None
                for field in ("social_score", "confirmation_score", "queue_score"):
                    data[field] = Decimal(data[field]) if data[field] is not None else None
                items.append(SocialSnapshotRecord(**data))
            return QueuePage(tuple(items), page, page_size, len(rows))
