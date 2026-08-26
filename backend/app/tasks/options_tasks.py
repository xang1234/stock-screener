from __future__ import annotations

import logging

from celery import shared_task

from .market_queues import data_fetch_queue_for_market
from .options_analysis_tasks import batch_analyze_options_exposure

logger = logging.getLogger(__name__)


@shared_task(name='app.tasks.options_tasks.schedule_daily_update')
def schedule_daily_update(limit: int = 2000):
    """Trigger the consolidated options analysis batch for active US symbols.

    This used to run its own separate yfinance sweep over the whole US
    universe just to populate the `options_metrics:{symbol}` Redis cache
    (used by the Options Analytics Dashboard SummaryCards). That duplicated
    the fetch already done by options_analysis_tasks.analyze_options_exposure
    (Daily Batch Options Analysis), doubling external API calls and general-
    queue worker time. analyze_options_exposure now derives and caches
    options_metrics:{symbol} itself from the same fetched chain, so this task
    just delegates to the shared batch instead of fetching again.
    """
    logger.info("Triggering consolidated options analysis batch (options metrics is now derived from it)")
    # Explicit queue routing -- batch_analyze_options_exposure isn't in the
    # default task_routes map, so a plain .delay() would dispatch it (and
    # the per-batch pacing loop it runs for its full duration) on the
    # general compute queue instead of data_fetch_us, same bug class as
    # max_pain_tasks.schedule_daily_update.
    return batch_analyze_options_exposure.apply_async(
        kwargs={"market": "US", "limit": limit},
        queue=data_fetch_queue_for_market("US"),
    )

