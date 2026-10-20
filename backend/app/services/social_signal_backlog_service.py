"""Resume saved posts under the shared budget; collection/publication live elsewhere."""
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import re
from time import monotonic
from uuid import uuid4

from sqlalchemy import func, select

from app.domain.social_signals.records import (
    SKIPPED_NO_SIGNAL, BacklogResult, ExtractionPostJudgment, ExtractionResult, SocialPostRecord,
)
from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt, SocialLLMBudgetDay, SocialRunWork
from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
from app.models.stock_universe import StockUniverse
from app.services.llm.llm_service import LLMPreDispatchError, LLMService
from app.services.social_extraction_service import SocialExtractionError, SocialExtractionService, VERSION
from app.services.social_llm_budget_service import SocialLLMBudgetService, social_analysis_transaction
from app.services.social_ticker_resolver import SocialTickerResolver, cashtags


logger = logging.getLogger(__name__)

NO_SIGNAL_GATE = "no_signal_gate"
# Configuration failures repeat for any subset of posts, so splitting the batch cannot help.
_UNSPLITTABLE_ERRORS = frozenset({"llm_configuration_error", "extraction_model_not_configured"})
_WORD = re.compile(r"\w+")
_SYMBOL = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z]{1,3}\b)?")  # MU, BRK.B, 0700.HK, 3008.TWO
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_SYMBOL_BASE = re.compile(r"[.\-]")


def _alias_phrases():
    """Latin company aliases ("Foxconn", "Hon Hai") as space-joined casefolded words,
    each with the canonical listing it names."""
    from app.services.cjk_alias_data import ASIA_ALIAS_CORPUS
    return {
        " ".join(_WORD.findall(alias.casefold())): canonical
        for canonical, entry in ASIA_ALIAS_CORPUS.items() for alias in entry["aliases"]
        if alias.isascii() and _WORD.search(alias)
    }


_ALIAS_PHRASES = _alias_phrases()


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _post_snapshot(post: SocialPostRecord) -> dict:
    """Keep text-only work snapshots byte-compatible with the earlier schema."""
    snapshot = asdict(post)
    for field in ("attachments", "prepared_evidence", "evidence_digest"):
        if not snapshot.get(field):
            snapshot.pop(field, None)
    for item in snapshot.get("prepared_evidence", ()):
        item["available_at"] = item["available_at"].isoformat()
    snapshot["created_at"] = post.created_at.isoformat()
    snapshot["observed_at"] = post.observed_at.isoformat()
    return snapshot


class _Deferred(Exception):
    def __init__(self, count=1, outside=0):
        self.count = count
        self.outside = outside


class _CompletionUnknown(Exception):
    """The dispatched model request itself failed, so whether it billed is unknown."""


class _NoopRequestGate:
    def acquire(self, owner, ttl):
        return True

    def wait_seconds(self):
        return 0

    def mark_started(self, interval):
        pass

    def release(self, owner):
        pass


class _MeteredCompletion:
    """Intercept usage BEFORE parsing, including malformed successful responses."""
    def __init__(self, owner, claims, now, *, retry=0):
        self.owner, self.claims, self.now, self.retry = owner, claims, now, retry
        self.dispatched = self.refunded = False

    async def completion(self, **kwargs):
        budget = self.owner.budget
        price = budget.price(kwargs["model"])
        if price is None:
            self.owner._finish_many(self.claims, "waiting_budget", "pricing_not_configured")
            raise _Deferred(len(self.claims))
        # UTF-8 bytes upper-bound text tokens; fixed margin covers chat envelope.
        inputs = len(json.dumps(kwargs["messages"], ensure_ascii=False).encode("utf-8")) + 1024
        outputs = kwargs["max_tokens"]
        work_ids = tuple(work_id for work_id, _ in self.claims)
        identity = ":".join(f"{work_id}:{token}" for work_id, token in self.claims)
        key = f"social-batch:{hashlib.sha256(identity.encode()).hexdigest()}"
        # A retry is a new logical request; the uncertain first attempt keeps its own charge.
        attempt = budget.reserve(key + (f":retry-{self.retry}" if self.retry else ""), work_ids,
            price.cost(inputs, outputs), self.now(), pricing_version=price.version,
            input_token_limit=inputs, output_token_limit=outputs)
        if attempt is None:
            self.owner._finish_many(self.claims, "waiting_budget", "daily_budget_exhausted")
            raise _Deferred(len(self.claims))
        lease_owner = f"social-llm:{attempt}:{uuid4().hex}"
        try:
            acquired = self.owner.request_gate.acquire(
                lease_owner, self.owner.request_lease_seconds
            )
        except Exception:
            budget.release(attempt)
            self.owner._finish_many(
                self.claims, "pending", "llm_request_gate_unavailable"
            )
            raise _Deferred(len(self.claims)) from None
        if not acquired:
            budget.release(attempt)
            self.owner._finish_many(self.claims, "pending", "llm_request_in_flight")
            raise _Deferred(len(self.claims))
        try:
            try:
                wait = self.owner.request_gate.wait_seconds()
                if wait > 0:
                    await self.owner.sleep(wait)
                self.owner.request_gate.mark_started(self.owner.min_interval_seconds)
            except asyncio.CancelledError:
                # Nothing was sent: refund the reservation and leave the posts claimable.
                # Claim expiry recovers both if this cleanup fails; the cancellation must
                # still propagate so no further batch is dispatched.
                try:
                    budget.release(attempt)
                    self.owner._finish_many(self.claims, "pending", "cancelled_before_dispatch")
                    self.refunded = True
                except Exception:
                    logger.warning("Could not refund a cancelled social LLM reservation", exc_info=True)
                raise
            except Exception:
                budget.release(attempt)
                self.owner._finish_many(
                    self.claims, "pending", "llm_request_gate_unavailable"
                )
                raise _Deferred(len(self.claims)) from None
            # Fence stale owners, daily calls, and reserved cancellation in one transaction.
            with social_analysis_transaction(self.owner.session_factory) as db:
                works = [db.get(SocialExtractionWork, work_id) for work_id in work_ids]
                row = db.get(SocialLLMAttempt, attempt)
                day = db.get(SocialLLMBudgetDay, row.budget_day_id)
                registry = db.get(SocialSourceRegistry, 1, populate_existing=True)
                dispatch_now = self.now()
                owned = all(work is not None and work.claim_token == token and work.state == "running"
                            for work, (_, token) in zip(works, self.claims))
                price_valid = budget.price_in_transaction(db, kwargs["model"]) == price
                runtime_available = (
                    registry is not None
                    and registry.mode in {"validation", "live"}
                    and registry.provider != "disabled"
                )
                outside_ids = {
                    work.id for work in works if work is not None and not work.requested_by_admin
                    and datetime.fromisoformat(work.input_snapshot_json["created_at"])
                    < dispatch_now - timedelta(days=14)
                }
                calls_today = db.scalar(
                    select(func.count(SocialLLMAttempt.id))
                    .join(
                        SocialLLMBudgetDay,
                        SocialLLMBudgetDay.id == SocialLLMAttempt.budget_day_id,
                    )
                    .where(
                        SocialLLMBudgetDay.period_start_utc < day.period_end_utc,
                        SocialLLMBudgetDay.period_end_utc > day.period_start_utc,
                        SocialLLMAttempt.state.in_({
                            "dispatched", "reconciled", "uncertain",
                        }),
                    )
                ) or 0
                call_available = calls_today < self.owner.max_calls_per_day
                valid = (owned and row.state == "reserved"
                    and all(_utc(work.claim_expires_at) > dispatch_now for work in works)
                    and _utc(day.period_start_utc) <= dispatch_now < _utc(day.period_end_utc)
                    and price_valid and runtime_available and not outside_ids and call_available)
                if valid:
                    row.state = "dispatched"
                else:
                    if row.state == "reserved":
                        row.state, row.completed_at = "released", dispatch_now
                        day.reserved_usd -= row.estimated_usd
                        day.version += 1
                    if owned:
                        for work in works:
                            if work.id in outside_ids:
                                work.state, work.error_code = "outside_window", "outside_signal_window"
                            else:
                                work.state = "pending" if price_valid and call_available else "waiting_budget"
                                work.error_code = (
                                    "social_runtime_unavailable" if not runtime_available
                                    else "daily_call_limit_exhausted" if not call_available
                                    else "claim_or_budget_period_expired" if price_valid
                                    else "pricing_changed"
                                )
                            work.claim_token = work.claim_expires_at = None
            if not valid:
                raise _Deferred(len(self.claims) - len(outside_ids), len(outside_ids))
            self.dispatched = True
            try:
                llm = self.owner.llm if self.owner.llm is not None else LLMService(use_case="extraction")
                response = await llm.completion(**kwargs, metered=True)
            except LLMPreDispatchError:
                budget.release_pre_dispatch(attempt)
                raise SocialExtractionError("llm_configuration_error") from None
            except Exception as exc:
                budget.reconcile(attempt, None, None)
                raise _CompletionUnknown() from exc
            except BaseException:
                budget.reconcile(attempt, None, None)
                raise
            usage = getattr(response, "usage", None)
            actual_inputs = getattr(usage, "prompt_tokens", None)
            actual_outputs = getattr(usage, "completion_tokens", None)
            actual_model = getattr(response, "model", None)
            provider = (getattr(response, "_hidden_params", None) or {}).get("custom_llm_provider")
            known = (provider == price.provider and actual_model in price.actual_models
                and type(actual_inputs) is int and actual_inputs >= 0
                and type(actual_outputs) is int and actual_outputs >= 0)
            if provider != price.provider or actual_model not in price.actual_models:
                budget.block_price(kwargs["model"], price.version, "billing_model_mismatch")
            elif known and (actual_inputs > inputs or actual_outputs > outputs):
                budget.block_price(kwargs["model"], price.version, "billing_token_limit_exceeded")
            budget.reconcile(attempt, price.cost(actual_inputs, actual_outputs) if known else None,
                getattr(response, "id", None), actual_input_tokens=actual_inputs, actual_output_tokens=actual_outputs)
            with social_analysis_transaction(self.owner.session_factory) as db:
                for work_id, token in self.claims:
                    work = db.get(SocialExtractionWork, work_id)
                    if work.claim_token == token:
                        work.actual_provider, work.actual_model = provider, actual_model
            return response
        finally:
            try:
                self.owner.request_gate.release(lease_owner)
            except Exception:
                logger.warning("Social LLM request lease release failed", exc_info=True)


class ProcessSocialBacklog:
    def __init__(self, session_factory, *, llm=None, batch_size=1,
                 max_calls_per_run=20, max_calls_per_day=80, request_gate=None,
                 min_interval_seconds=0, request_lease_seconds=600, sleep=asyncio.sleep,
                 daily_limit_usd=None, budget_timezone=None):
        if (not 1 <= batch_size <= 50 or max_calls_per_run <= 0 or max_calls_per_day <= 0
                or min_interval_seconds < 0 or request_lease_seconds <= 0):
            raise ValueError("invalid_social_llm_request_policy")
        self.session_factory, self.llm = session_factory, llm
        self.batch_size = batch_size
        self.max_calls_per_run = max_calls_per_run
        self.max_calls_per_day = max_calls_per_day
        self.request_gate = request_gate or _NoopRequestGate()
        self.min_interval_seconds = min_interval_seconds
        self.request_lease_seconds = request_lease_seconds
        self.sleep = sleep
        self.budget = SocialLLMBudgetService(
            session_factory,
            daily_limit_usd=daily_limit_usd,
            budget_timezone=budget_timezone,
        )
        self._universe_hints = None  # (active symbols, first name words, active aliases), read once

    def enqueue(self, content_item_id, post: SocialPostRecord, *, selected_model,
                now, prompt_version=VERSION, schema_version=VERSION, run_id=None):
        """Pin a per-post content revision; engagement-only updates reuse success."""
        if not selected_model or not prompt_version or not schema_version:
            raise ValueError("extraction_configuration_missing")
        input_hash = SocialExtractionService.input_hash((post,))
        snapshot = _post_snapshot(post)
        with social_analysis_transaction(self.session_factory) as db:
            if run_id:
                run = db.get(SocialSignalRun, run_id)
                if run is None or run.status != "running":
                    raise ValueError("terminal_or_missing_social_run")
            def saved(model):
                return db.scalar(select(SocialExtractionWork).where(
                    SocialExtractionWork.content_item_id == content_item_id,
                    SocialExtractionWork.input_hash == input_hash,
                    SocialExtractionWork.prompt_version == prompt_version,
                    SocialExtractionWork.schema_version == schema_version,
                    SocialExtractionWork.selected_model == model))
            work = saved(selected_model)
            if work is None:
                # Gate skips carry their own model identity and are never rewritten: runs
                # that saved one keep it, and a post that gains signal (the universe now
                # lists what it names) gets separate LLM work.
                skipped = saved(NO_SIGNAL_GATE)
                if not self._has_signal(db, post):
                    work = skipped
                    if work is None:
                        # Saved as an empty judgment, so readiness and replays treat it like any result.
                        result = ExtractionResult(input_hash, NO_SIGNAL_GATE, NO_SIGNAL_GATE, prompt_version,
                            schema_version, (), None, None, (ExtractionPostJudgment(post.provider_post_id, False, None),))
                        work = SocialExtractionWork(content_item_id=content_item_id, input_hash=input_hash,
                            prompt_version=prompt_version, schema_version=schema_version,
                            selected_model=NO_SIGNAL_GATE, input_snapshot_json=snapshot,
                            actual_provider=NO_SIGNAL_GATE, actual_model=NO_SIGNAL_GATE,
                            result_json=asdict(result), error_code=SKIPPED_NO_SIGNAL,
                            state="succeeded", created_at=now, updated_at=now)
            if work is None:
                work = SocialExtractionWork(content_item_id=content_item_id, input_hash=input_hash,
                    prompt_version=prompt_version, schema_version=schema_version,
                    selected_model=selected_model, input_snapshot_json=snapshot,
                    state="pending", created_at=now, updated_at=now)
            if work.id is None:
                db.add(work)
                db.flush()
            if run_id and db.get(SocialRunWork, (run_id, work.id)) is None:
                db.add(SocialRunWork(run_id=run_id, work_id=work.id, input_hash=input_hash, included_at=now))
            return work.id

    def _has_signal(self, db, post: SocialPostRecord) -> bool:
        """Deterministic pre-LLM gate: a repost, or a post naming no listing or company, carries nothing to extract.

        Scoring and theme projection both drop reposts, and every claim needs a
        company token found verbatim in the source.
        """
        if post.is_repost:
            return False
        text = f"{post.text}\n{post.quoted_text or ''}"
        if post.attachments or post.prepared_evidence or cashtags(text) or _CJK.search(text):
            return True
        if self._universe_hints is None:
            symbols, names = set(), set()
            for symbol, name in db.execute(select(StockUniverse.symbol, StockUniverse.name)
                                           .where(StockUniverse.active_filter())):
                symbols.add(symbol)
                first = SocialTickerResolver._issuer_key(name).split()[:1]
                if first and len(first[0]) >= 4:
                    names.add(first[0])
            # An alias of an inactive listing can't yield a usable claim (#619).
            aliases = frozenset(phrase for phrase, canonical in _ALIAS_PHRASES.items()
                                if canonical in symbols)
            self._universe_hints = (symbols, names, aliases)
        symbols, names, aliases = self._universe_hints
        # ponytail: first-word name match lets generic names ("General", "First") through;
        # it errs toward an LLM call. CJK text has no word breaks, so it always passes.
        # Exact case: lowercase "it", "on", "now" are tickers too, and would pass every post.
        # A suffix that isn't a listing ("MU.It" with no space) falls back to its base.
        if any(token in symbols or _SYMBOL_BASE.split(token)[0] in symbols
               for token in _SYMBOL.findall(text)):
            return True
        words = _WORD.findall(text.casefold())
        if any(word in names for word in words):
            return True
        # Aliases that aren't a universe name's first word ("Foxconn", "Hon Hai").
        joined = f" {' '.join(words)} "
        return any(f" {alias} " in joined for alias in aliases)

    def _claim(self, now, limit, admin_work_ids, eligible_work_ids=()):
        claimed, outside, terminal = [], 0, 0
        with social_analysis_transaction(self.session_factory) as db:
            query = select(SocialExtractionWork).where(SocialExtractionWork.state != "succeeded")
            if eligible_work_ids:
                query = query.where(SocialExtractionWork.id.in_(eligible_work_ids))
            rows = db.scalars(query).all()
            for row in rows:
                if row.id in admin_work_ids:
                    row.requested_by_admin = True
                if row.state == "running" and row.claim_expires_at and _utc(row.claim_expires_at) <= now:
                    # A dispatched request may have billed, including after a crash.
                    # Never silently retry it. Reserved-only claims can be released.
                    attempts = [a for a in db.scalars(select(SocialLLMAttempt)).all() if row.id in a.work_ids]
                    ambiguous = any(a.state in {"dispatched", "uncertain", "reconciled"} for a in attempts)
                    for attempt in attempts:
                        if attempt.state == "reserved":
                            day = db.get(SocialLLMBudgetDay, attempt.budget_day_id)
                            day.reserved_usd -= attempt.estimated_usd
                            day.version += 1
                            attempt.state, attempt.completed_at = "released", now
                        elif attempt.state == "dispatched":
                            attempt.state = "uncertain"
                    row.state = "failed_terminal" if ambiguous else "pending"
                    row.error_code = "completion_unknown" if ambiguous else None
                    if not ambiguous:
                        row.claim_token = None
                    row.claim_expires_at = None
                if row.state == "failed_terminal":
                    if eligible_work_ids:
                        terminal += 1
                    continue
                if row.state == "running":
                    continue
                published = datetime.fromisoformat(row.input_snapshot_json["created_at"])
                if published < now - timedelta(days=14) and not row.requested_by_admin:
                    if row.state != "outside_window":
                        outside += 1
                    row.state, row.error_code = "outside_window", "outside_signal_window"
            eligible = [r for r in rows if r.state in {"pending", "waiting_budget", "failed_retryable"}
                or (r.state == "outside_window" and r.requested_by_admin)]
            eligible.sort(key=lambda r: (datetime.fromisoformat(r.input_snapshot_json["created_at"]), r.id))
            for row in eligible[:limit]:
                row.state, row.error_code = "running", None
                row.claim_token = uuid4().hex
                row.claim_expires_at = now + timedelta(minutes=10)
                row.updated_at = now
                claimed.append((row.id, row.claim_token, row.selected_model, row.prompt_version,
                    row.schema_version, dict(row.input_snapshot_json)))
        return claimed, outside, terminal

    def _finish(self, work_id, token, state, error=None, result=None):
        with social_analysis_transaction(self.session_factory) as db:
            row = db.get(SocialExtractionWork, work_id)
            late_completion = row.state == "failed_terminal" and row.error_code == "completion_unknown"
            if (row.state != "running" and not late_completion) or row.claim_token != token:
                return False
            row.state, row.error_code = state, error
            row.claim_token = row.claim_expires_at = None
            row.updated_at = datetime.now(timezone.utc)
            if result is not None:
                row.result_json = asdict(result)
                row.actual_provider, row.actual_model = result.provider, result.model
            return True

    def _finish_many(self, claims, state, error=None):
        return sum(self._finish(work_id, token, state, error) for work_id, token in claims)

    def _settle_unknown(self, claims):
        """Close work a deferral released, unless another claim has taken it since."""
        settled = 0
        with social_analysis_transaction(self.session_factory) as db:
            for work_id, _ in claims:
                row = db.get(SocialExtractionWork, work_id)
                if row.state in {"pending", "waiting_budget"} and row.claim_token is None:
                    row.state, row.error_code = "failed_terminal", "completion_unknown"
                    row.updated_at = datetime.now(timezone.utc)
                    settled += 1
        return settled

    @staticmethod
    def _groups(claimed, batch_size):
        current = []
        current_config = None
        for item in claimed:
            config = item[2:5]
            if current and (config != current_config or len(current) >= batch_size):
                yield current
                current = []
            current.append(item)
            current_config = config
        if current:
            yield current

    @staticmethod
    def _single_result(post, result):
        post_id = post.provider_post_id
        return ExtractionResult(
            SocialExtractionService.input_hash((post,)), result.provider, result.model,
            result.prompt_version, result.schema_version,
            tuple(claim for claim in result.claims if claim.post_id == post_id),
            None, None,
            tuple(value for value in result.judgments if value.post_id == post_id),
        )

    async def execute(self, now: datetime, limit: int, admin_work_ids: tuple[int, ...] = (),
                      work_ids: tuple[int, ...] = ()) -> BacklogResult:
        if now.tzinfo is None or limit <= 0:
            raise ValueError("invalid_backlog_request")
        started = monotonic()
        current_time = lambda: now + timedelta(seconds=monotonic() - started)
        claim_limit = min(limit, self.batch_size * self.max_calls_per_run)
        claimed, outside, failed = self._claim(
            now, claim_limit, admin_work_ids, work_ids
        )
        succeeded = deferred = 0
        batches = list(self._groups(claimed, self.batch_size))
        spare_calls = self.max_calls_per_run - len(batches)
        for batch in batches:
            items = []
            for work_id, token, _, _, _, snapshot in batch:
                snapshot["created_at"] = datetime.fromisoformat(snapshot["created_at"])
                snapshot["observed_at"] = datetime.fromisoformat(snapshot["observed_at"])
                items.append(((work_id, token), SocialPostRecord(**snapshot)))
            outcome, spare_calls = await self._extract(tuple(items), batch[0][2:5], current_time, spare_calls)
            if isinstance(outcome, str):
                outcome, spare_calls = await self._isolate(
                    tuple(items), batch[0][2:5], current_time, outcome, spare_calls
                )
            succeeded, deferred, failed, outside = (
                total + part for total, part in zip((succeeded, deferred, failed, outside), outcome)
            )
        return BacklogResult(succeeded, deferred, failed, outside, self.budget.status(current_time()).next_reset_at)

    async def _extract(self, items, config, current_time, spare_calls):
        """Return (succeeded, deferred, failed, outside) or an unusable response's code, and the calls still spare.

        A request whose outcome is unknown, such as a provider timeout, may have
        billed, so it is retried at most once and only with a call the run's cap left unused.
        """
        model, prompt, schema = config
        claims = tuple(claim for claim, _ in items)
        work_ids = [work_id for work_id, _ in claims]
        for retry in range(2):
            metered = _MeteredCompletion(self, claims, current_time, retry=retry)
            try:
                with self.session_factory() as db:
                    extraction = SocialExtractionService(db, model=model, prompt_version=prompt,
                        schema_version=schema, llm=metered)
                    result = await extraction.extract(tuple(post for _, post in items))
                break
            except _Deferred as exc:
                if retry:
                    # The retry never went out, and deferred work would be claimed again
                    # later, so the first request's unknown outcome stands.
                    return (0, 0, self._settle_unknown(claims), exc.outside), spare_calls
                return (0, exc.count, 0, exc.outside), spare_calls
            except asyncio.CancelledError:
                # A request that never went out leaves its posts pending or, if the refund
                # failed, running for claim expiry to recover. A sent request, or a retry of
                # one, may have billed, so its posts must not be sent again. A retry whose
                # refund failed stays running too: claim expiry releases its reservation and,
                # seeing the first attempt uncertain, settles the posts as completion_unknown.
                if metered.dispatched or (retry and metered.refunded):
                    self._finish_many(claims, "failed_terminal", "completion_unknown")
                    self._settle_unknown(claims)
                raise
            except SocialExtractionError as exc:
                return str(exc), spare_calls
            except _CompletionUnknown:
                logger.warning("Social LLM request outcome unknown (attempt %d) work_ids=%s",
                               retry + 1, work_ids, exc_info=True)
                if retry or spare_calls < 1:
                    return (0, 0, self._finish_many(claims, "failed_terminal", "completion_unknown"), 0), spare_calls
                spare_calls -= 1
            except Exception:
                logger.exception("Social LLM extraction failed work_ids=%s", work_ids)
                return (0, 0, self._finish_many(claims, "failed_terminal", "completion_unknown"), 0), spare_calls
        try:
            succeeded = sum(self._finish(work_id, token, "succeeded", result=self._single_result(post, result))
                            for (work_id, token), post in items)
        except Exception:
            logger.exception("Saving social LLM results failed work_ids=%s", work_ids)
            return (0, 0, self._finish_many(claims, "failed_terminal", "completion_unknown"), 0), spare_calls
        return (succeeded, 0, 0, 0), spare_calls

    async def _isolate(self, items, config, current_time, code, spare_calls):
        """Narrow an unusable batch response to the posts that provoke it.

        Splitting spends only calls the run's cap left unused, so a model that
        fails every request costs no more than one call per claimed batch did.
        Returns the outcome totals and the calls still spare.
        """
        claims = tuple(claim for claim, _ in items)
        if len(items) == 1 or code in _UNSPLITTABLE_ERRORS or spare_calls < 2:
            return (0, 0, self._finish_many(claims, "failed_terminal", code), 0), spare_calls
        spare_calls -= 2
        middle = len(items) // 2
        totals = (0, 0, 0, 0)
        for half in (items[:middle], items[middle:]):
            outcome, spare_calls = await self._extract(half, config, current_time, spare_calls)
            if isinstance(outcome, str):
                outcome, spare_calls = await self._isolate(half, config, current_time, outcome, spare_calls)
            totals = tuple(total + part for total, part in zip(totals, outcome))
        return totals, spare_calls
