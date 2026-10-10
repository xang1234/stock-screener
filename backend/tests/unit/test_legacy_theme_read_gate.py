"""#474: CI gate for legacy Theme reads that bypass authority routing.

Retirement criterion 4 of the economic taxonomy audit
(docs/runbooks/artifacts/economic-taxonomy-readiness-audit-2026-10-01.md).
Every API route, Celery task and MCP tool that reaches a legacy authority model
without routing through ``EconomicThemeReader`` must be listed in ``ALLOWLIST``
with its reason. Each entry is a tracked item to route, replace or remove;
retirement needs the list to be empty.
"""

from __future__ import annotations

import textwrap

import pytest

from tests.helpers.legacy_theme_read_gate import Index, api_entry_points, unrouted_legacy_reads

_REVIEW = "Audit 'Readers with no routing': review/merge GETs serve legacy clusters and suggestions."
_INTELLIGENCE = "Audit 'Readers with no routing': equivalence and development GETs serve legacy identities."
_TELEMETRY = "Audit 'Readers with no routing': matching telemetry serves legacy ThemeMention stats."
_CONTENT = "Audit 'Readers with no routing': content listing annotates items with ThemeMention."
_DEVELOPMENT_BACKFILL = (
    "Audit 'Rollback machinery (keep until retirement)': maps legacy "
    "ThemeDevelopmentTheme links to economic rows for the snapshot builder (#513)."
)
_PIPELINE_DIAGNOSTICS = "Audit 'Readers with no routing': pipeline diagnostics read legacy tables."
_ROLLBACK = (
    "Audit 'Rollback machinery (keep until retirement)': compatibility delivery "
    "maintains legacy projections."
)
_SOCIAL_BRIDGE = (
    "Audit 'Rollback machinery (keep until retirement)': the economic Social taxonomy "
    "adapter keeps Social associations bridged, and economic-mode decisions revise them."
)

# entry point -> (reason, the legacy models it is expected to read)
ALLOWLIST: dict[str, tuple[str, set[str]]] = {
    "GET /api/v1/themes/merge-suggestions": (
        _REVIEW,
        {"ThemeCluster", "ThemeMergeSuggestion"},
    ),
    "GET /api/v1/themes/merge-history": (
        _REVIEW,
        {"ThemeMergeHistory"},
    ),
    "GET /api/v1/themes/merge-plan/dry-run": (
        _REVIEW,
        {"ThemeCluster", "ThemeEmbedding"},
    ),
    "GET /api/v1/themes/candidates/queue": (
        _REVIEW,
        {"ThemeCluster", "ThemeEquivalenceOperation", "ThemeMention", "ThemeMetrics"},
    ),
    "GET /api/v1/themes/relationship-graph": (
        _REVIEW,
        {"ThemeCluster", "ThemeEquivalenceOperation", "ThemeRelationship"},
    ),
    "GET /api/v1/themes/equivalence/preview": (
        _INTELLIGENCE,
        {"ThemeCluster", "ThemeEquivalenceOperation", "ThemeMention"},
    ),
    "GET /api/v1/themes/equivalence/history": (
        _INTELLIGENCE,
        {"ThemeEquivalenceOperation"},
    ),
    "GET /api/v1/themes/equivalence/search": (
        _INTELLIGENCE,
        {"ThemeCluster", "ThemeEquivalenceOperation"},
    ),
    "GET /api/v1/themes/{theme_id}/developments": (
        _INTELLIGENCE,
        {"ThemeCluster", "ThemeDevelopmentTheme", "ThemeEquivalenceOperation", "ThemeMention"},
    ),
    "GET /api/v1/themes/matching/telemetry": (
        _TELEMETRY,
        {"ThemeMention"},
    ),
    "GET /api/v1/themes/content": (
        _CONTENT,
        {"ThemeMention", "table:theme_mentions"},
    ),
    "GET /api/v1/themes/content/export": (
        _CONTENT,
        {"ThemeMention", "table:theme_mentions"},
    ),
    "task app.tasks.economic_taxonomy_tasks.backfill_legacy_developments": (
        _DEVELOPMENT_BACKFILL,
        {"ThemeDevelopmentTheme"},
    ),
    "GET /api/v1/themes/pipeline/state-health": (
        _PIPELINE_DIAGNOSTICS,
        {"ThemeMention"},
    ),
    "GET /api/v1/themes/pipeline/observability": (
        _PIPELINE_DIAGNOSTICS,
        {"ThemeCluster", "ThemeMention", "ThemeMergeSuggestion"},
    ),
    "task app.tasks.economic_taxonomy_tasks.deliver_taxonomy_outbox": (
        _ROLLBACK,
        {"SocialThemeAssociation", "SocialThemeDecision", "ThemeCluster", "ThemeConstituent"},
    ),
    "task app.interfaces.tasks.social_signal_tasks.refresh_social_signals": (
        _SOCIAL_BRIDGE,
        {"SocialThemeAssociation"},
    ),
    "task app.interfaces.tasks.social_signal_tasks.resume_social_analysis": (
        _SOCIAL_BRIDGE,
        {"SocialThemeAssociation"},
    ),
    "task app.tasks.economic_taxonomy_tasks.process_economic_taxonomy_work": (
        _SOCIAL_BRIDGE,
        {"SocialThemeAssociation"},
    ),
    "POST /api/v1/social-signals/admin/economic-associations/{association_id}/decision": (
        _SOCIAL_BRIDGE,
        {"SocialThemeAssociation"},
    ),
}


@pytest.fixture(scope="module")
def findings():
    from app.celery_app import celery_app
    from app.main import app

    return unrouted_legacy_reads(app, celery_app)


def _short(qualname):
    return ".".join(qualname.rsplit(".", 2)[-2:])


def test_no_new_unrouted_legacy_theme_reads(findings):
    # Keyed by entry point and model, so new debt under an allowlisted entry
    # point fails too. Not by call path: refactors would churn it.
    new = [
        finding
        for entry, found in sorted(findings.items())
        for finding in found
        if finding.model not in ALLOWLIST.get(entry, ("", set()))[1]
    ]
    report = "\n".join(
        f"  {f.entry}: {f.model} via {' > '.join(_short(p) for p in f.path)}" for f in new
    )
    assert not new, (
        "These entry points read legacy Theme tables without routing through "
        "EconomicThemeReader (or a #472 guard):\n"
        f"{report}\n"
        "Route the read by authority mode, or add it to ALLOWLIST with a reason."
    )


def test_allowlist_has_no_stale_entries(findings):
    from app.config.settings import settings

    stale = {
        entry: sorted(models - {f.model for f in findings.get(entry, [])})
        for entry, (_, models) in ALLOWLIST.items()
        # POST /mcp/ exists only when the MCP HTTP transport is enabled.
        if entry != "POST /mcp/" or settings.mcp_http_enabled
    }
    stale = {entry: models for entry, models in stale.items() if models}
    assert not stale, (
        f"No longer read unrouted (routed, removed or renamed); shrink ALLOWLIST: {stale}"
    )


# -- analyzer behaviour, on a small fixture package ----------------------------

_FIXTURE = {
    "models/theme.py": """
        class ThemeCluster: ...
    """,
    "services/economic_theme_read_service.py": """
        class EconomicThemeReader:
            def __init__(self, db): ...
    """,
    "services/readers.py": """
        from typing import Protocol
        from app.models.theme import ThemeCluster
        from app.services.economic_theme_read_service import EconomicThemeReader

        def read_clusters(db):
            return db.query(ThemeCluster).all()

        def read_raw(db):
            return db.execute("SELECT id FROM theme_clusters")

        def read_raw_upper(db):
            return db.execute("SELECT id FROM THEME_CLUSTERS")

        class ClusterPort(Protocol):
            def load(self): ...

        class SqlClusterReader:
            def __init__(self, db):
                self.db = db

            def load(self):
                return read_clusters(self.db)

        class UseCase:
            def __init__(self, *, reader):
                self.reader = reader

            def run(self):
                return self.reader.load()

        def build_use_case(db):
            return UseCase(reader=SqlClusterReader(db))

        class SafeReader:
            def load(self):
                return []

        class BindUseCase:
            def __init__(self, reader):
                self.reader = reader

            def run(self):
                return self.reader.load()

        class AliasUseCase:
            def __init__(self, *, reader):
                self.reader = reader

            def run(self):
                reader = self.reader  # a local copy must keep every injected type
                return reader.load()

        class ConditionalAttr:
            def __init__(self, db, flag):
                self.reader = SafeReader() if flag else SqlClusterReader(db)

            def run(self):
                return self.reader.load()

        class ReassignedAttr:
            def __init__(self, db, flag):
                reader = SafeReader()
                if flag:
                    reader = SqlClusterReader(db)
                self.reader = reader

            def run(self):
                return self.reader.load()

        class EconomicArmAttr:
            def __init__(self, db, reader: EconomicThemeReader):
                if reader.source_name == "economic":
                    self.impl = SqlClusterReader(db)  # selected under economic authority
                else:
                    self.impl = SafeReader()

            def run(self):
                return self.impl.load()

        class LegacyArmAttr:
            def __init__(self, db, reader: EconomicThemeReader):
                if reader.source_name == "economic":
                    self.impl = SafeReader()
                else:
                    self.impl = SqlClusterReader(db)  # legacy authority only

            def run(self):
                return self.impl.load()

        def pick_reader(db, flag):
            if flag:
                return SafeReader()
            return SqlClusterReader(db)
    """,
    "api/v1/themes_common.py": """
        def reject_legacy_theme_writes(db): ...
    """,
    "api/v1/themes_taxonomy.py": """
        def _reject_economic_mode(db): ...
    """,
    "services/legacy_theme_write_guard.py": """
        def ensure_legacy_theme_writes_allowed(db, force=False): ...

        def skip_in_economic_authority(task=None, *, on_skip=None): ...
    """,
    "entry.py": """
        from app.api.v1.themes_taxonomy import _reject_economic_mode
        from app.services.legacy_theme_write_guard import (
            ensure_legacy_theme_writes_allowed,
            skip_in_economic_authority,
        )
        from app.services.economic_theme_read_service import EconomicThemeReader
        from app.services.readers import build_use_case, read_clusters, read_raw

        def unrouted(db):
            return read_clusters(db)

        def raw_sql(db):
            return read_raw(db)

        def routed(db):
            if EconomicThemeReader(db):
                return []
            return read_clusters(db)

        def read_before_check(db):
            rows = read_clusters(db)
            if EconomicThemeReader(db):
                return []
            return rows

        def check_only_annotated(db, reader: EconomicThemeReader):
            return read_clusters(db)

        def check_in_other_branch(db, flag):
            if flag:
                EconomicThemeReader(db)
            return read_clusters(db)

        def reader_built_not_checked(db):
            reader = EconomicThemeReader(db)
            read_clusters(db)
            return reader

        def reader_parameter_branch(db, reader: EconomicThemeReader):
            if reader.source_name == "economic":
                return []
            return read_clusters(db)

        def read_in_economic_arm(db, reader: EconomicThemeReader):
            if reader.source_name == "economic":
                return read_clusters(db)
            return []

        def read_after_legacy_return(db, reader: EconomicThemeReader):
            if reader.source_name != "economic":
                return []
            return read_clusters(db)

        def read_in_legacy_arm(db, reader: EconomicThemeReader):
            if reader.source_name != "economic":
                return read_clusters(db)
            return []

        def read_in_else_of_mixed_check(db, reader: EconomicThemeReader, flag):
            if reader.source_name == "economic" and flag:
                return []
            return read_clusters(db)  # economic authority with flag False lands here

        def constructor_takes_every_candidate(db, flag):
            from app.services.readers import BindUseCase, SafeReader, SqlClusterReader
            reader = SafeReader()
            if flag:
                reader = SqlClusterReader(db)
            return BindUseCase(reader).run()

        def raising_check(db):
            _reject_economic_mode(db)
            return read_clusters(db)

        def caught_raising_check(db):
            try:
                _reject_economic_mode(db)
            except Exception:
                pass
            return read_clusters(db)

        def is_ready(reader):
            return True

        def reader_passed_to_helper(db, reader: EconomicThemeReader):
            if is_ready(reader):  # says nothing about authority
                return []
            return read_clusters(db)

        def forced_raising_check(db):
            ensure_legacy_theme_writes_allowed(db, force=True)  # bypasses the rejection
            return read_clusters(db)

        def unforced_raising_check(db):
            ensure_legacy_theme_writes_allowed(db, force=False)
            return read_clusters(db)

        def finally_after_raising_check(db):
            try:
                _reject_economic_mode(db)
            finally:
                read_clusters(db)  # runs while the economic-mode exception unwinds

        def reassigned_local(db, flag):
            from app.services.readers import SafeReader, SqlClusterReader
            reader = SqlClusterReader(db)
            if flag:
                reader = SafeReader()
            return reader.load()

        def raw_sql_upper(db):
            from app.services.readers import read_raw_upper
            return read_raw_upper(db)

        def alias_keeps_every_implementation(db):
            from app.services.readers import AliasUseCase, SafeReader, SqlClusterReader
            AliasUseCase(reader=SafeReader())
            AliasUseCase(reader=SqlClusterReader(db))
            return AliasUseCase(reader=SafeReader()).run()

        def every_injected_implementation(db):
            from app.services.readers import SafeReader, UseCase, SqlClusterReader
            UseCase(reader=SafeReader())
            UseCase(reader=SqlClusterReader(db))
            return UseCase(reader=SafeReader()).run()

        def injected(db):
            return build_use_case(db).run()

        def conditional_value(db, flag):
            from app.services.readers import SafeReader, SqlClusterReader
            reader = SafeReader() if flag else SqlClusterReader(db)
            return reader.load()

        def ready_from_reader(db):
            ready = is_ready(EconomicThemeReader(db))  # not an authority value
            if ready:
                return []
            return read_clusters(db)

        @skip_in_economic_authority
        def skipped_task_body(db):
            return read_clusters(db)

        @skip_in_economic_authority(on_skip=read_clusters)  # runs in economic mode
        def skip_callback(db):
            return []

        def raw_sql_fstring(db, schema):
            return db.execute(f"SELECT id FROM {schema}.theme_clusters")

        def raw_sql_concat(db, schema):
            return db.execute("SELECT id FROM " + schema + ".theme_clusters")

        def raw_sql_adjacent(db):
            return db.execute("SELECT id FROM theme_" + "clusters")

        def conditional_attr(db, flag):
            from app.services.readers import ConditionalAttr
            return ConditionalAttr(db, flag).run()

        def reassigned_attr(db, flag):
            from app.services.readers import ReassignedAttr
            return ReassignedAttr(db, flag).run()

        def economic_arm_attr(db, reader: EconomicThemeReader):
            from app.services.readers import EconomicArmAttr
            return EconomicArmAttr(db, reader).run()

        def legacy_arm_attr(db, reader: EconomicThemeReader):
            from app.services.readers import LegacyArmAttr
            return LegacyArmAttr(db, reader).run()

        def factory_branches(db, flag):
            from app.services.readers import pick_reader
            return pick_reader(db, flag).load()

        def task_body(db):
            return read_clusters(db)

        def dispatches(db):
            task_body.delay(db)
    """,
}


@pytest.fixture(scope="module")
def fixture_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate") / "app"
    for relative, source in _FIXTURE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    for package in (root, root / "models", root / "services", root / "api", root / "api" / "v1"):
        (package / "__init__.py").touch()
    return Index({"app.models.theme.ThemeCluster"}, {"theme_clusters"}, root=root)


def _reads(index, name):
    return index.legacy_reads(name, [index.funcs[f"app.entry.{name}"]])


def test_gate_reports_an_unrouted_read_with_its_call_path(fixture_index):
    [finding] = _reads(fixture_index, "unrouted")
    assert finding.model == "ThemeCluster"
    assert finding.path == ["app.entry.unrouted", "app.services.readers.read_clusters"]


@pytest.mark.parametrize("name", ["raw_sql", "raw_sql_upper", "raw_sql_fstring", "raw_sql_concat", "raw_sql_adjacent"])  # unquoted SQL names fold case
def test_gate_reports_raw_sql_naming_a_legacy_table(fixture_index, name):
    assert [f.model for f in _reads(fixture_index, name)] == ["table:theme_clusters"]


def test_gate_accepts_reads_routed_by_authority(fixture_index):
    assert _reads(fixture_index, "routed") == []


@pytest.mark.parametrize(
    "name",
    [
        "read_before_check",
        "check_only_annotated",
        "check_in_other_branch",
        "reader_built_not_checked",
        "every_injected_implementation",
        "read_in_economic_arm",  # the arm that runs under economic authority
        "read_after_legacy_return",  # past a legacy-arm return, only economic remains
        "alias_keeps_every_implementation",
        "reassigned_local",  # a later assignment must not hide the earlier type
        "read_in_else_of_mixed_check",  # the else arm of `economic and flag` is not legacy-only
        "constructor_takes_every_candidate",
        "caught_raising_check",  # a caught raise does not divert
        "reader_passed_to_helper",  # a reader in an unrelated call is not a check
        "forced_raising_check",  # force=True skips the rejection
        "finally_after_raising_check",  # finally runs as the exception unwinds
        "conditional_value",  # either branch of a conditional may be the value
        "ready_from_reader",  # a variable merely derived from a reader is not a check
        "skip_callback",  # the decorator's on_skip callback runs when skipped
        "conditional_attr",  # self.attr keeps every option
        "reassigned_attr",  # self.attr keeps every class its local held
        "economic_arm_attr",  # an implementation picked under economic authority
        "factory_branches",  # every class a factory's branches return
    ],
)
def test_gate_counts_reads_the_authority_check_does_not_cover(fixture_index, name):
    assert [f.model for f in _reads(fixture_index, name)] == ["ThemeCluster"]


@pytest.mark.parametrize(
    "name",
    [
        "reader_parameter_branch",
        "read_in_legacy_arm",
        "raising_check",
        "unforced_raising_check",
        "skipped_task_body",
        "legacy_arm_attr",  # picked only under legacy authority
    ],
)
def test_gate_accepts_a_branch_on_a_reader_parameter(fixture_index, name):
    assert _reads(fixture_index, name) == []


def test_gate_follows_injected_protocol_dependencies(fixture_index):
    [finding] = _reads(fixture_index, "injected")
    assert finding.path[-2:] == [
        "app.services.readers.SqlClusterReader.load",
        "app.services.readers.read_clusters",
    ]


def test_gate_leaves_celery_dispatch_to_the_task_entry_point(fixture_index):
    assert _reads(fixture_index, "dispatches") == []


def _named(module, qualname):
    """A live callable the gate maps to fixture source by module and qualname."""

    def call():
        return None

    call.__module__, call.__qualname__ = module, qualname
    return call


def test_route_guard_covers_only_what_runs_after_it(fixture_index):
    from fastapi import Depends, FastAPI

    read = _named("app.entry", "unrouted")  # reads ThemeCluster
    guard = _named("app.api.v1.themes_common", "reject_legacy_theme_writes")

    def before(_read=Depends(read), _guard=Depends(guard)):
        return None

    def after(_guard=Depends(guard), _read=Depends(read)):
        return None

    app = FastAPI()
    for path, endpoint in (("/before", before), ("/after", after)):
        endpoint.__module__, endpoint.__qualname__ = "app.entry", "unrouted"
        app.get(path)(endpoint)

    reads = {
        entry: [f.model for f in fixture_index.legacy_reads(entry, roots)]
        for entry, _endpoint, roots in api_entry_points(fixture_index, app)
    }

    # FastAPI resolves dependencies in declaration order and the endpoint last:
    # a read before the guard runs in economic mode; after it, only legacy.
    assert reads == {"GET /before": ["ThemeCluster"], "GET /after": []}
