"""Read-only listing resolution; company equivalence requires a trusted mapping."""
from collections.abc import Mapping
from types import MappingProxyType

from app.domain.social_signals.records import SUPPORTED_MARKETS, TickerResolution
from app.models.stock_universe import StockUniverse
from app.services.cjk_alias_resolver_service import (
    METHOD_SYMBOL_NORMALIZED, METHOD_SYMBOL_PASSTHROUGH, resolve_alias,
)
from app.services.multi_market_ticker_validator import TICKER_SHAPE_RE
from app.services.security_master_service import security_master_resolver


class SocialTickerResolver:
    def __init__(self, db, *, verified_company_ids: Mapping[str, str] | None = None,
                 security_kinds: Mapping[str, str] | None = None):
        self.db = db
        # Caller supplies independently verified identities, never model/name guesses.
        self.company_ids = MappingProxyType(dict(verified_company_ids or {}))
        if any(not isinstance(value, str) or not value.strip() for value in self.company_ids.values()):
            raise ValueError("invalid_verified_company_id")
        self.security_kinds = MappingProxyType({"SPY": "broad_etf", "QQQ": "broad_etf",
            "SMH": "thematic_etf", "REMX": "thematic_etf", **(security_kinds or {})})

    def resolve(self, raw_token: str, explicit_market: str | None = None) -> TickerResolution:
        def unresolved(reason):
            return TickerResolution(raw_token, None, None, None, "unresolved", (reason,))

        if explicit_market is not None and explicit_market not in SUPPORTED_MARKETS:
            return unresolved("unsupported_market")
        token = security_master_resolver.normalize_symbol(raw_token)
        raw = raw_token.strip().lstrip("$")
        explicit = raw_token.strip().startswith("$") or (raw == raw.upper() and bool(TICKER_SHAPE_RE.fullmatch(raw)))
        with self.db.no_autoflush:
            # An explicit ticker is never redirected to a company alias (HSBC).
            if explicit:
                canonical = token
                if token.isdigit():
                    if explicit_market is None:
                        return unresolved("ambiguous_numeric_listing")
                    numeric = token.zfill(4) if explicit_market == "HK" else token
                    alias = resolve_alias(numeric, hint_market=explicit_market)
                    canonical = alias.canonical_symbol
                elif "." in token:
                    alias = resolve_alias(token, hint_market=explicit_market)
                    if alias.method in {METHOD_SYMBOL_NORMALIZED, METHOD_SYMBOL_PASSTHROUGH}:
                        canonical = alias.canonical_symbol
            else:
                alias = resolve_alias(raw, hint_market=explicit_market)
                canonical = alias.canonical_symbol
            if not canonical or not TICKER_SHAPE_RE.fullmatch(canonical):
                return unresolved("unknown_listing")
            security = self.db.query(StockUniverse).filter(
                StockUniverse.symbol == canonical, StockUniverse.active_filter()).first()
            if security is None:
                return unresolved("not_in_active_universe")
            if security.market not in SUPPORTED_MARKETS:
                return unresolved("unsupported_market")
            if explicit_market is not None and security.market != explicit_market:
                return unresolved("conflicting_market")
            company_id = self.company_ids.get(canonical)
            related = ()
            if company_id:
                candidates = [symbol for symbol, identity in self.company_ids.items()
                              if identity == company_id and symbol != canonical]
                if candidates:
                    related = tuple(row.symbol for row in self.db.query(StockUniverse).filter(
                        StockUniverse.symbol.in_(candidates), StockUniverse.active_filter(),
                        StockUniverse.market.in_(SUPPORTED_MARKETS)).order_by(StockUniverse.symbol).all())
            kind = self.security_kinds.get(canonical, "stock" if security.is_common_stock else "macro")
            reasons = () if company_id else ("company_identity_unknown",)
            return TickerResolution(raw_token, canonical, security.market, str(security.id), "resolved",
                reasons, company_id, related, kind, kind in {"stock", "thematic_etf"},
                bool(company_id) and kind == "stock", explicit)
