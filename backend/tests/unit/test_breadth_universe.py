from datetime import date

import pandas as pd
from app.services.breadth.market_policy import get_breadth_market_policy
from app.services.breadth.types import BreadthFormulaPolicy, BreadthUniverseMember
from app.services.breadth.universe import (
    build_breadth_universe_snapshots,
    classify_metric_eligibility,
    stockbee_eligibility_signature,
)
from app.services.point_in_time_universe_service import (
    PointInTimeUniverse,
    PointInTimeUniverseMember,
    hash_point_in_time_universe_symbols,
)


class _UniverseResolver:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def resolve(self, _db, *, market, as_of_date):
        assert market == self.snapshot.market
        assert as_of_date == self.snapshot.as_of_date
        return self.snapshot


def test_breadth_snapshot_keeps_only_point_in_time_common_stocks():
    calculation_date = date(2026, 8, 21)
    point_in_time = PointInTimeUniverse(
        market="US",
        as_of_date=calculation_date,
        symbols=("COMMON", "ETF"),
        universe_hash="source-hash",
        members=(
            PointInTimeUniverseMember("COMMON", "USD", True),
            PointInTimeUniverseMember("ETF", "USD", False),
        ),
    )

    snapshots = build_breadth_universe_snapshots(
        object(),
        "US",
        (calculation_date,),
        universe_service=_UniverseResolver(point_in_time),
    )

    result = snapshots[calculation_date]
    assert result.members == (BreadthUniverseMember("COMMON", "USD", True),)
    assert result.broad_signature == hash_point_in_time_universe_symbols(("COMMON",))


def test_metric_eligibility_is_independent_and_stockbee_requires_common_stock():
    calculation_date = pd.Timestamp("2026-08-21")
    features = pd.DataFrame(
        [
            {
                "adjusted_close": 100.0,
                "prior_adjusted_close": 99.0,
                "daily_return": 0.01,
                "volume": 100_000.0,
                "prior_volume": 90_000.0,
                "adtv20_local": 250_000.0,
                "adjusted_close_20": float("nan"),
                "raw_close_local_20": float("nan"),
                "month_return": float("nan"),
                "low_34": float("nan"),
                "high_34": float("nan"),
                "low_65": float("nan"),
                "high_65": float("nan"),
                "sma40": float("nan"),
                "sma50": float("nan"),
                "atr14": float("nan"),
                "adjusted_high": 101.0,
                "adjusted_low": 98.0,
                "previous_251_high": float("nan"),
                "previous_251_low": float("nan"),
                "sma200": float("nan"),
                "high_252": float("nan"),
                "sma20": float("nan"),
                "trend_template": float("nan"),
                "prior_trend_template": float("nan"),
            }
        ],
        index=[calculation_date],
    )

    common = classify_metric_eligibility(
        BreadthUniverseMember("IPO", "USD", True),
        features,
        BreadthFormulaPolicy(),
        get_breadth_market_policy("US"),
        calculation_date=calculation_date.date(),
    )
    excluded = classify_metric_eligibility(
        BreadthUniverseMember("ETF", "USD", False),
        features,
        BreadthFormulaPolicy(),
        get_breadth_market_policy("US"),
        calculation_date=calculation_date.date(),
    )

    assert common.advance_decline is True
    assert common.stockbee_daily is True
    assert common.stockbee_month is False
    assert common.stockbee_quarter is False
    assert excluded.stockbee_daily is False


def test_stockbee_signature_contains_only_liquid_broad_members():
    calculation_date = date(2026, 8, 21)
    common = BreadthUniverseMember("COMMON", "USD", True)
    illiquid = BreadthUniverseMember("ILLIQUID", "USD", True)
    snapshot = build_breadth_universe_snapshots(
        object(),
        "US",
        (calculation_date,),
        universe_service=_UniverseResolver(
            PointInTimeUniverse(
                market="US",
                as_of_date=calculation_date,
                symbols=("COMMON", "ILLIQUID"),
                universe_hash="source-hash",
                members=(
                    PointInTimeUniverseMember("COMMON", "USD", True),
                    PointInTimeUniverseMember("ILLIQUID", "USD", True),
                ),
            )
        ),
    )[calculation_date]
    liquid_features = pd.DataFrame(
        [
            {
                "adjusted_close": 100.0,
                "prior_adjusted_close": 99.0,
                "daily_return": 0.01,
                "volume": 100_000.0,
                "prior_volume": 90_000.0,
                "adtv20_local": 250_000.0,
                "adjusted_close_20": float("nan"),
                "raw_close_local_20": float("nan"),
                "month_return": float("nan"),
                "low_34": float("nan"),
                "high_34": float("nan"),
                "low_65": float("nan"),
                "high_65": float("nan"),
                "sma40": float("nan"),
                "sma50": float("nan"),
                "atr14": float("nan"),
                "adjusted_high": 101.0,
                "adjusted_low": 98.0,
                "previous_251_high": float("nan"),
                "previous_251_low": float("nan"),
                "sma200": float("nan"),
                "high_252": float("nan"),
                "sma20": float("nan"),
                "trend_template": float("nan"),
                "prior_trend_template": float("nan"),
            }
        ],
        index=[pd.Timestamp(calculation_date)],
    )
    illiquid_features = liquid_features.copy()
    illiquid_features.loc[:, "adtv20_local"] = 249_999.0

    signature = stockbee_eligibility_signature(
        snapshot,
        {
            common.symbol: liquid_features,
            illiquid.symbol: illiquid_features,
        },
        BreadthFormulaPolicy(),
        get_breadth_market_policy("US"),
    )

    assert signature == hash_point_in_time_universe_symbols(("COMMON",))
