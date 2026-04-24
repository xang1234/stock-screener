from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

def _postgres_order_sql(expr) -> str:
    return str(expr.compile(dialect=postgresql.dialect()))


class _RecordingQuery:
    def __init__(self, *, rows=None, first_row=None):
        self.rows = rows or []
        self.first_row = first_row
        self.order_by_args = []

    def filter(self, *args):
        return self

    def group_by(self, *args):
        return self

    def order_by(self, *args):
        self.order_by_args.extend(args)
        return self

    def limit(self, *_args):
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.first_row


class _RecordingSession:
    def __init__(self, queries):
        self._queries = list(queries)
        self.added = []
        self.committed = False
        self.closed = False

    def query(self, *args):
        if not self._queries:
            raise AssertionError(f"unexpected query call for {args!r}")
        return self._queries.pop(0)

    def add(self, item):
        self.added.append(item)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def _group_row():
    return SimpleNamespace(
        ibd_industry_group="Software",
        total_stocks=2,
        avg_rs_1m=70.0,
        avg_rs_3m=72.0,
        avg_rs_12m=74.0,
        avg_minervini_score=68.0,
        avg_composite_score=80.0,
    )


def test_scan_result_repository_peer_queries_sort_null_scores_last(monkeypatch):
    from app.infra.db.repositories import scan_result_repo as module

    repo = module.SqlScanResultRepository(session=object())

    industry_query = _RecordingQuery()
    monkeypatch.setattr(
        module,
        "_scan_results_query",
        lambda _session, _scan_id: industry_query,
    )
    assert repo.get_peers_by_industry("scan-1", "Software") == ()
    assert "NULLS LAST" in _postgres_order_sql(industry_query.order_by_args[0])

    sector_query = _RecordingQuery()
    monkeypatch.setattr(
        module,
        "_scan_results_query",
        lambda _session, _scan_id: sector_query,
    )
    assert repo.get_peers_by_sector("scan-1", "Technology") == ()
    assert "NULLS LAST" in _postgres_order_sql(sector_query.order_by_args[0])


def test_scan_execution_peer_cache_top_symbol_sorts_null_scores_last():
    from app.services.scan_execution import compute_industry_peer_metrics

    group_query = _RecordingQuery(rows=[_group_row()])
    top_query = _RecordingQuery(
        first_row=SimpleNamespace(symbol="SCORED", composite_score=91.0)
    )
    session = _RecordingSession([group_query, top_query])

    compute_industry_peer_metrics(session, "scan-1")

    assert "NULLS LAST" in _postgres_order_sql(top_query.order_by_args[0])


def test_legacy_scan_task_peer_cache_top_symbol_sorts_null_scores_last():
    from app.tasks.scan_tasks import compute_industry_peer_metrics

    group_query = _RecordingQuery(rows=[_group_row()])
    top_query = _RecordingQuery(
        first_row=SimpleNamespace(symbol="SCORED", composite_score=91.0)
    )
    session = _RecordingSession([group_query, top_query])

    compute_industry_peer_metrics(session, "scan-1")

    assert "NULLS LAST" in _postgres_order_sql(top_query.order_by_args[0])


def test_chart_cache_prewarm_selects_top_results_with_null_scores_last(monkeypatch):
    import app.tasks.cache_tasks as module

    top_query = _RecordingQuery(rows=[])
    session = _RecordingSession([top_query])
    monkeypatch.setattr(module, "SessionLocal", lambda: session)

    result = module.prewarm_chart_cache_for_scan.run("scan-1", top_n=50)

    assert result["warmed"] == 0
    assert "NULLS LAST" in _postgres_order_sql(top_query.order_by_args[0])
