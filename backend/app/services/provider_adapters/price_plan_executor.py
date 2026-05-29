"""Execute price provider plans through provider-specific adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from app.domain.providers.data_plan import (
    DATASET_PRICES,
    PROVIDER_AKSHARE,
    PROVIDER_BAOSTOCK,
    PROVIDER_KRX,
    PROVIDER_YFINANCE,
    ProviderDataPlan,
    ProviderPlanStep,
    provider_data_plan_registry,
)
from app.services.security_master_service import security_master_resolver

logger = logging.getLogger(__name__)

PricePlanResolver = Callable[[str | None, str | None], ProviderDataPlan]


class PriceFetcher(Protocol):
    def _build_error_result(self, symbol: str, error: str) -> dict[str, Any]:
        ...

    def _fetch_yfinance_prices_in_batches(
        self,
        symbols: list[str],
        *,
        period: str,
        start_batch_size: int | None = None,
        market: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        ...

    def _fetch_cn_price_batch(
        self,
        symbols: list[str],
        *,
        period: str,
    ) -> dict[str, dict[str, Any]]:
        ...

    def _fetch_kr_price_batch(
        self,
        symbols: list[str],
        *,
        period: str,
    ) -> dict[str, dict[str, Any]]:
        ...


class PriceProviderPlanExecutor:
    """Run price fetches according to ``ProviderDataPlanRegistry``."""

    def __init__(
        self,
        fetcher: PriceFetcher,
        *,
        plan_resolver: PricePlanResolver | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._plan_resolver = plan_resolver or self._default_plan_resolver

    @staticmethod
    def _default_plan_resolver(
        market: str | None,
        mic: str | None = None,
    ) -> ProviderDataPlan:
        return provider_data_plan_registry.plan_for(market, DATASET_PRICES, mic=mic)

    def fetch(
        self,
        symbols: list[str],
        *,
        period: str = "2y",
        start_batch_size: int | None = None,
        market: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not symbols:
            return {}

        plan = self._resolve_plan(market)
        if not plan.providers:
            return self._with_plan_metadata(
                {
                    symbol: self._fetcher._build_error_result(
                        symbol,
                        f"No price provider plan for market={plan.market!r}",
                    )
                    for symbol in symbols
                },
                plan,
            )

        for step in plan.steps:
            if step.provider in {PROVIDER_AKSHARE, PROVIDER_BAOSTOCK}:
                return self._fetch_cn_native(
                    symbols,
                    period=period,
                    start_batch_size=start_batch_size,
                    market=market,
                    plan=plan,
                )
            if step.provider == PROVIDER_KRX:
                return self._fetch_krx(
                    symbols,
                    period=period,
                    start_batch_size=start_batch_size,
                    market=market,
                    plan=plan,
                )
            if step.provider == PROVIDER_YFINANCE:
                return self._fetch_yfinance(
                    symbols,
                    period=period,
                    start_batch_size=start_batch_size,
                    market=market,
                    plan=plan,
                    step=step,
                )

            logger.warning(
                "Unsupported price provider %r in plan %s/%s version %s",
                step.provider,
                plan.market,
                plan.dataset,
                plan.version,
            )

        return self._with_plan_metadata(
            {
                symbol: self._fetcher._build_error_result(
                    symbol,
                    f"No executable price provider in plan for market={plan.market!r}",
                )
                for symbol in symbols
            },
            plan,
        )

    def _fetch_yfinance(
        self,
        symbols: list[str],
        *,
        period: str,
        start_batch_size: int | None,
        market: str | None,
        plan: ProviderDataPlan,
        step: ProviderPlanStep,
    ) -> dict[str, dict[str, Any]]:
        results = self._fetcher._fetch_yfinance_prices_in_batches(
            symbols,
            period=period,
            start_batch_size=start_batch_size if start_batch_size is not None else step.batch_size,
            market=market,
        )
        return self._with_plan_metadata(results, plan)

    def _fetch_cn_native(
        self,
        symbols: list[str],
        *,
        period: str,
        start_batch_size: int | None,
        market: str | None,
        plan: ProviderDataPlan,
    ) -> dict[str, dict[str, Any]]:
        native_results = self._fetcher._fetch_cn_price_batch(symbols, period=period)
        fallback_symbols = [
            symbol
            for symbol, payload in native_results.items()
            if self._missing_price_data(payload)
            and self._resolve_symbol_plan(symbol, market, plan).allows(PROVIDER_YFINANCE)
        ]
        if not fallback_symbols:
            return self._with_symbol_plan_metadata(native_results, market=market, base_plan=plan)

        fallback_results = self._fetcher._fetch_yfinance_prices_in_batches(
            fallback_symbols,
            period=period,
            start_batch_size=start_batch_size if start_batch_size is not None else plan.step_for(PROVIDER_YFINANCE).batch_size,
            market=market,
        )
        merged = dict(native_results)
        for symbol, fallback_payload in fallback_results.items():
            primary_payload = native_results.get(symbol, {})
            enriched_payload = dict(fallback_payload)
            enriched_payload.setdefault("provider", PROVIDER_YFINANCE)
            enriched_payload["fallback_from"] = "akshare_baostock"
            enriched_payload["primary_provider_failed"] = self._missing_price_data(primary_payload)
            primary_error = primary_payload.get("error")
            if primary_error:
                enriched_payload["primary_provider_error"] = primary_error
            merged[symbol] = enriched_payload
        return self._with_symbol_plan_metadata(merged, market=market, base_plan=plan)

    def _fetch_krx(
        self,
        symbols: list[str],
        *,
        period: str,
        start_batch_size: int | None,
        market: str | None,
        plan: ProviderDataPlan,
    ) -> dict[str, dict[str, Any]]:
        krx_results = self._fetcher._fetch_kr_price_batch(symbols, period=period)
        fallback_symbols = [
            symbol
            for symbol, payload in krx_results.items()
            if self._missing_price_data(payload)
        ]
        if not fallback_symbols or not plan.allows(PROVIDER_YFINANCE):
            return self._with_plan_metadata(krx_results, plan)

        fallback_results = self._fetcher._fetch_yfinance_prices_in_batches(
            fallback_symbols,
            period=period,
            start_batch_size=start_batch_size if start_batch_size is not None else plan.step_for(PROVIDER_YFINANCE).batch_size,
            market=market,
        )
        merged = dict(krx_results)
        for symbol, fallback_payload in fallback_results.items():
            primary_payload = krx_results.get(symbol, {})
            enriched_payload = dict(fallback_payload)
            enriched_payload.setdefault("provider", PROVIDER_YFINANCE)
            enriched_payload["fallback_from"] = PROVIDER_KRX
            enriched_payload["primary_provider_failed"] = self._missing_price_data(primary_payload)
            primary_error = primary_payload.get("error")
            if primary_error:
                enriched_payload["primary_provider_error"] = primary_error
            merged[symbol] = enriched_payload
        return self._with_plan_metadata(merged, plan)

    def _resolve_plan(self, market: str | None, mic: str | None = None) -> ProviderDataPlan:
        return self._plan_resolver(market, mic)

    def _resolve_symbol_plan(
        self,
        symbol: str,
        market: str | None,
        base_plan: ProviderDataPlan,
    ) -> ProviderDataPlan:
        resolved_market = market or base_plan.market
        try:
            identity = security_master_resolver.resolve_identity(
                symbol=symbol,
                market=resolved_market,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Could not resolve price plan identity for %s: %s", symbol, exc)
            return base_plan
        if identity.mic:
            return self._resolve_plan(identity.market, identity.mic)
        return base_plan

    @staticmethod
    def _missing_price_data(payload: dict[str, Any]) -> bool:
        return bool(payload.get("has_error") or payload.get("price_data") is None)

    @staticmethod
    def _with_plan_metadata(
        results: dict[str, dict[str, Any]],
        plan: ProviderDataPlan,
    ) -> dict[str, dict[str, Any]]:
        metadata = plan.provenance_metadata()
        for payload in results.values():
            if isinstance(payload, dict):
                payload["provider_data_plan"] = metadata
        return results

    def _with_symbol_plan_metadata(
        self,
        results: dict[str, dict[str, Any]],
        *,
        market: str | None,
        base_plan: ProviderDataPlan,
    ) -> dict[str, dict[str, Any]]:
        for symbol, payload in results.items():
            if isinstance(payload, dict):
                plan = self._resolve_symbol_plan(symbol, market, base_plan)
                payload["provider_data_plan"] = plan.provenance_metadata()
        return results
