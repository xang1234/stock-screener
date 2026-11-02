"""Short serialized dollar reservations. No provider I/O or process-local policy."""
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_CEILING, Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, update

from app.infra.db.models.social_analysis import SocialLLMAttempt, SocialLLMBudgetDay
from app.infra.db.models.social_signals import SocialSourceRegistry
from app.models.app_settings import AppSetting

ZERO = Decimal("0")
MONEY_QUANTUM = Decimal("0.000000000001")


def money(value):
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("invalid_money")
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_CEILING)


def social_budget_date(now: datetime, timezone_name: str) -> date:
    if now.tzinfo is None:
        raise ValueError("naive_timestamp")
    return now.astimezone(ZoneInfo(timezone_name)).date()


def social_budget_period(now: datetime, timezone_name: str) -> tuple[date, datetime, datetime]:
    day = social_budget_date(now, timezone_name)
    zone = ZoneInfo(timezone_name)
    start = datetime.combine(day, time.min, zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time.min, zone).astimezone(timezone.utc)
    return day, start, end


@contextmanager
def social_analysis_transaction(factory):
    """Reuse the migration-seeded registry lock without changing its version."""
    with factory() as db, db.begin():
        dialect = db.get_bind().dialect.name
        if dialect == "sqlite":
            db.execute(update(SocialSourceRegistry).where(SocialSourceRegistry.id == 1)
                .values(version=SocialSourceRegistry.version))
        elif dialect != "postgresql":
            raise ValueError("unsupported_database")
        if db.scalar(select(SocialSourceRegistry).where(SocialSourceRegistry.id == 1).with_for_update()) is None:
            raise ValueError("registry_not_initialized")
        yield db


@dataclass(frozen=True)
class BudgetStatus:
    remaining_usd: Decimal
    next_reset_at: datetime
    budget_day_id: int


@dataclass(frozen=True)
class SocialModelPrice:
    version: str
    provider: str
    actual_models: tuple[str, ...]
    input_rate: Decimal
    output_rate: Decimal

    def cost(self, inputs, outputs):
        return money((self.input_rate * inputs + self.output_rate * outputs) / Decimal("1000000"))


class SocialLLMBudgetService:
    def __init__(
        self,
        session_factory,
        *,
        daily_limit_usd: Decimal | None = None,
        budget_timezone: str | None = None,
    ):
        self.session_factory = session_factory
        self.daily_limit_usd = (
            money(daily_limit_usd) if daily_limit_usd is not None else None
        )
        if budget_timezone is not None:
            ZoneInfo(budget_timezone)
        self.budget_timezone = budget_timezone

    @staticmethod
    def _setting(db, key, default, *, override=None):
        setting = db.scalar(select(AppSetting).where(AppSetting.key == key))
        if setting is None:
            setting = AppSetting(
                key=key,
                value=override if override is not None else default,
                category="social",
            )
            db.add(setting)
            db.flush()
        elif override is not None and setting.value != override:
            setting.value = override
        return setting.value

    def _day(self, db, now):
        zone = self._setting(
            db,
            "social_llm_budget_timezone",
            "Asia/Singapore",
            override=self.budget_timezone,
        )
        limit_override = (
            str(self.daily_limit_usd) if self.daily_limit_usd is not None else None
        )
        limit = money(Decimal(self._setting(
            db,
            "social_llm_daily_limit_usd",
            "2",
            override=limit_override,
        )))
        day, start, end = social_budget_period(now, zone)
        row = db.scalar(select(SocialLLMBudgetDay).where(
            SocialLLMBudgetDay.period_start_utc == start, SocialLLMBudgetDay.period_end_utc == end))
        if row is None:
            row = SocialLLMBudgetDay(budget_date=day, timezone=zone, period_start_utc=start,
                period_end_utc=end, limit_usd=limit, reserved_usd=ZERO, actual_usd=ZERO)
            db.add(row)
            db.flush()
        row.limit_usd = limit
        # Sum native counters once per overlapping dispatch bucket, never copy
        # counters into another bucket: repeated timezone switches cannot mint cash.
        overlapping = db.scalars(select(SocialLLMBudgetDay).where(
            SocialLLMBudgetDay.period_start_utc < end, SocialLLMBudgetDay.period_end_utc > start)).all()
        used = sum((r.actual_usd + r.reserved_usd for r in overlapping), ZERO)
        return row, BudgetStatus(max(ZERO, limit - used), end, row.id)

    def status(self, now):
        with social_analysis_transaction(self.session_factory) as db:
            return self._day(db, now)[1]

    def price(self, model):
        with self.session_factory() as db:
            return self.price_in_transaction(db, model)

    @staticmethod
    def price_in_transaction(db, model):
        setting = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing"))
        try:
            config = json.loads(setting.value)
            blocked = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing_blocks"))
            if blocked:
                model_blocks = json.loads(blocked.value).get(model, {})
                if (model_blocks.get("version") == config["version"]
                        or config["version"] in model_blocks.get("versions", {})):
                    return None
            rate = config["models"][model]
            if not config["version"] or not rate["provider"] or not rate["actual_models"]:
                return None
            if not all(isinstance(v, str) for v in rate["actual_models"]):
                return None
            if not all(isinstance(rate[k], str) for k in ("input_usd_per_million", "output_usd_per_million")):
                return None
            return SocialModelPrice(config["version"], rate["provider"], tuple(rate["actual_models"]),
                money(Decimal(rate["input_usd_per_million"])), money(Decimal(rate["output_usd_per_million"])))
        except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
            return None

    def block_price(self, model, version, reason):
        """A new explicit pricing version is required after known billing mismatch."""
        with social_analysis_transaction(self.session_factory) as db:
            value = self._setting(db, "social_llm_pricing_blocks", "{}")
            blocks = json.loads(value)
            model_blocks = blocks.get(model, {})
            versions = dict(model_blocks.get("versions", {}))
            # Retain the original single-version format during an in-place
            # upgrade, and never let an old completion erase a newer block.
            if "version" in model_blocks:
                versions[model_blocks["version"]] = model_blocks["reason"]
            versions[version] = reason
            blocks[model] = {"versions": versions}
            db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing_blocks")).value = json.dumps(blocks)

    def reserve(
        self,
        attempt_key: str,
        work_ids: tuple[int, ...],
        maximum_usd: Decimal,
        now: datetime,
        *,
        pricing_version="manual-v1",
        input_token_limit=0,
        output_token_limit=0,
        logical_operation_key: str | None = None,
        operation_kind: str = "social_extraction",
    ):
        maximum_usd = money(maximum_usd)
        if not attempt_key or not work_ids or any(type(i) is not int or i <= 0 for i in work_ids):
            raise ValueError("invalid_reservation")
        logical_operation_key = logical_operation_key or attempt_key
        if not logical_operation_key.strip() or not operation_kind.strip():
            raise ValueError("invalid_logical_operation")
        with social_analysis_transaction(self.session_factory) as db:
            old = db.scalar(select(SocialLLMAttempt).where(SocialLLMAttempt.idempotency_key == attempt_key))
            if old:
                if (
                    old.work_ids != list(work_ids)
                    or old.estimated_usd != maximum_usd
                    or old.pricing_version != pricing_version
                    or old.logical_operation_key != logical_operation_key
                    or old.operation_kind != operation_kind
                ):
                    raise ValueError("idempotency_conflict")
                return old.id
            latest = db.scalar(
                select(SocialLLMAttempt)
                .where(
                    SocialLLMAttempt.logical_operation_key == logical_operation_key,
                    SocialLLMAttempt.operation_kind == operation_kind,
                )
                .order_by(SocialLLMAttempt.attempt_number.desc())
                .limit(1)
            )
            if latest is not None and latest.state in {"reserved", "dispatched", "uncertain"}:
                return None
            day, status = self._day(db, now)
            if maximum_usd > status.remaining_usd:
                return None
            attempt_number = int(
                db.scalar(
                    select(func.max(SocialLLMAttempt.attempt_number)).where(
                        SocialLLMAttempt.logical_operation_key
                        == logical_operation_key,
                        SocialLLMAttempt.operation_kind == operation_kind,
                    )
                )
                or 0
            ) + 1
            attempt = SocialLLMAttempt(
                idempotency_key=attempt_key,
                logical_operation_key=logical_operation_key,
                operation_kind=operation_kind,
                attempt_number=attempt_number,
                work_ids=list(work_ids),
                budget_day_id=day.id,
                estimated_usd=maximum_usd,
                pricing_version=pricing_version,
                input_token_limit=input_token_limit,
                output_token_limit=output_token_limit,
                state="reserved",
                created_at=now,
            )
            db.add(attempt)
            day.reserved_usd += maximum_usd
            day.version += 1
            db.flush()
            return attempt.id

    def reservation_is_open(self, attempt_id: int) -> bool:
        with self.session_factory() as db:
            attempt = db.get(SocialLLMAttempt, attempt_id)
            return bool(
                attempt is not None
                and attempt.state in {"reserved", "dispatched", "uncertain"}
            )

    def mark_dispatched(self, attempt_id):
        with social_analysis_transaction(self.session_factory) as db:
            attempt = db.get(SocialLLMAttempt, attempt_id)
            if attempt.state != "reserved":
                return False
            attempt.state = "dispatched"
            return True

    def release(self, attempt_id):
        with social_analysis_transaction(self.session_factory) as db:
            attempt = db.get(SocialLLMAttempt, attempt_id)
            if attempt.state != "reserved":
                return False
            day = db.get(SocialLLMBudgetDay, attempt.budget_day_id)
            day.reserved_usd -= attempt.estimated_usd
            day.version += 1
            attempt.state = "released"
            attempt.completed_at = datetime.now(timezone.utc)
            return True

    def release_unbilled(self, attempt_id):
        """Refund an attempt only when the caller proves it was not billed: never sent, or refused.

        Returns whether it ends released.
        """
        with social_analysis_transaction(self.session_factory) as db:
            attempt = db.get(SocialLLMAttempt, attempt_id)
            if attempt.state == "released":
                return True
            if attempt.state not in {"dispatched", "uncertain"}:
                return False
            day = db.get(SocialLLMBudgetDay, attempt.budget_day_id)
            day.reserved_usd -= attempt.estimated_usd
            day.version += 1
            attempt.state = "released"
            attempt.completed_at = datetime.now(timezone.utc)
            return True

    def reconcile(self, attempt_id: int, actual_usd: Decimal | None,
                  provider_request_id: str | None, *, actual_input_tokens=None, actual_output_tokens=None):
        if actual_usd is not None:
            actual_usd = money(actual_usd)
        with social_analysis_transaction(self.session_factory) as db:
            attempt = db.get(SocialLLMAttempt, attempt_id)
            if attempt.state == "reconciled":
                if actual_usd != attempt.actual_usd:
                    raise ValueError("reconciliation_conflict")
                return
            if attempt.state not in {"dispatched", "uncertain"}:
                raise ValueError("attempt_not_dispatched")
            attempt.provider_request_id = provider_request_id
            attempt.actual_input_tokens = actual_input_tokens
            attempt.actual_output_tokens = actual_output_tokens
            if actual_usd is None:
                attempt.state = "uncertain"
                return
            day = db.get(SocialLLMBudgetDay, attempt.budget_day_id)
            day.reserved_usd -= attempt.estimated_usd
            day.actual_usd += actual_usd
            day.version += 1
            attempt.actual_usd = actual_usd
            attempt.state = "reconciled"
            attempt.completed_at = datetime.now(timezone.utc)


class SocialBudgetReservation:
    """Provider-reservation protocol backed by one durable Social attempt."""

    def __init__(self, budget: SocialLLMBudgetService, attempt_id: int):
        self.budget = budget
        self.attempt_id = attempt_id

    @property
    def state(self) -> str:
        with self.budget.session_factory() as db:
            attempt = db.get(SocialLLMAttempt, self.attempt_id)
            if attempt is None:
                raise KeyError(f"social LLM attempt {self.attempt_id} not found")
            return attempt.state

    def mark_dispatched(self) -> None:
        if not self.budget.mark_dispatched(self.attempt_id):
            raise ValueError("social_attempt_not_reserved")

    def release_pre_dispatch(self) -> None:
        state = self.state
        released = (
            self.budget.release(self.attempt_id)
            if state == "reserved"
            else self.budget.release_unbilled(self.attempt_id)
        )
        if not released:
            raise ValueError("social_attempt_not_releasable")

    def reconcile(
        self, *, actual_cost: Decimal | None, provider_request_id: str | None
    ) -> None:
        self.budget.reconcile(
            self.attempt_id,
            actual_cost,
            provider_request_id,
        )


class SocialEconomicReservationManager:
    """Adapt Social's dollar ledger to Economic Taxonomy provider calls."""

    def __init__(
        self,
        budget: SocialLLMBudgetService,
        *,
        work_ids: tuple[int, ...],
        maximum_usd: Decimal,
        now: datetime,
        pricing_version: str,
        input_token_limit: int,
        output_token_limit: int,
    ):
        self.budget = budget
        self.work_ids = work_ids
        self.maximum_usd = maximum_usd
        self.now = now
        self.pricing_version = pricing_version
        self.input_token_limit = input_token_limit
        self.output_token_limit = output_token_limit

    def reserve(self, *, attempt_key, logical_request_id, operation_kind):
        attempt_id = self.budget.reserve(
            str(attempt_key),
            self.work_ids,
            self.maximum_usd,
            self.now,
            pricing_version=self.pricing_version,
            input_token_limit=self.input_token_limit,
            output_token_limit=self.output_token_limit,
            logical_operation_key=str(logical_request_id),
            operation_kind=str(operation_kind),
        )
        if attempt_id is None:
            return None
        return SocialBudgetReservation(self.budget, attempt_id)
