from app.tasks.cache_tasks import _active_benchmark_markets, _benchmark_markets_for_symbols


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_args, **_kwargs):
        return _FakeQuery(self._rows)


def test_active_benchmark_markets_infers_non_us_when_market_blank():
    db = _FakeDB(
        [
            ("", "XHKG", "0700"),
            (None, "XTKS", "9984"),
            ("US", "NASDAQ", "AAPL"),
        ]
    )

    markets = _active_benchmark_markets(db)

    assert markets == ["HK", "JP", "US"]


def test_benchmark_markets_for_symbols_infers_market_from_symbol_suffix():
    db = _FakeDB(
        [
            ("", None, "3008.TWO"),
            ("", None, "2330.TW"),
        ]
    )

    markets = _benchmark_markets_for_symbols(db, symbols=["3008.TWO", "2330.TW"])

    assert markets == ["TW"]

