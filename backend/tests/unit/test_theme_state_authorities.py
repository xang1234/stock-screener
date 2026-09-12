"""Review regressions, including opt-in real PostgreSQL lock checks.

Set THEME_REVIEW_TEST_DATABASE_URL to a disposable PostgreSQL database. Each test
uses and removes only its own random schema; no application tables are touched.
"""

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.services.theme_group_coordination import (
    lock_grouping_mutation,
    publication_scope,
)
from sqlalchemy.orm import Session


@pytest.fixture(params=["sqlite", "postgresql"])
def engine(request):
    if request.param == "sqlite":
        engine = sa.create_engine("sqlite:///:memory:")
        yield engine
        engine.dispose()
        return
    url = os.environ.get("THEME_REVIEW_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Disposable PostgreSQL URL not configured")
    admin = sa.create_engine(url)
    schema = "theme_review_" + uuid4().hex
    with admin.begin() as connection:
        connection.execute(sa.text(f"CREATE SCHEMA {schema}"))
    engine = sa.create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def test_populated_membership_migration_roundtrip(engine):
    path = (
        Path(__file__).parents[2]
        / "alembic/versions/20260912_0044_theme_state_authorities.py"
    )
    spec = importlib.util.spec_from_file_location("state_authorities_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = sa.MetaData()
    observations = sa.Table(
        "theme_development_observations",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("theme_ids", sa.JSON, nullable=False),
        sa.Column("facts", sa.JSON, nullable=False),
    )
    links = sa.Table(
        "theme_development_themes",
        metadata,
        sa.Column("observation_id", sa.Integer, primary_key=True),
        sa.Column("theme_id", sa.Integer, primary_key=True),
    )
    work = sa.Table(
        "theme_development_work",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_marker", sa.Integer, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            observations.insert().values(
                id=1,
                theme_ids=[2, 3],
                facts={"status": "confirmed", "theme_ids": [2, 3]},
            )
        )
        connection.execute(links.insert().values(observation_id=1, theme_id=2))
        connection.execute(work.insert().values(id=1, source_marker=7))
        module.op = Operations(MigrationContext.configure(connection))
        module.upgrade()
        assert "theme_ids" not in {
            col["name"] for col in sa.inspect(connection).get_columns(observations.name)
        }
        assert connection.execute(
            sa.select(links.c.theme_id).order_by(links.c.theme_id)
        ).scalars().all() == [2, 3]
        assert connection.execute(sa.select(observations.c.facts)).scalar_one() == {
            "status": "confirmed"
        }
        assert (
            connection.execute(
                sa.text("SELECT checked_at FROM theme_development_work")
            ).scalar_one()
            is not None
        )
        module.downgrade()
        assert connection.execute(sa.select(observations.c.theme_ids)).scalar_one() == [
            2,
            3,
        ]
        module.upgrade()
        assert (
            connection.execute(
                sa.select(sa.func.count()).select_from(links)
            ).scalar_one()
            == 2
        )


def test_publication_serializes_mutation_across_internal_commits(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Requires PostgreSQL advisory locks")
    started, completed = Event(), Event()

    def mutate():
        with Session(engine) as db:
            started.set()
            lock_grouping_mutation(db)
            db.commit()
            completed.set()

    with ThreadPoolExecutor(max_workers=1) as pool, Session(engine) as db:
        with publication_scope(db):
            future = pool.submit(mutate)
            assert started.wait(3)
            db.commit()
            with Session(engine) as nested, publication_scope(nested):
                nested.commit()
            assert not completed.wait(0.2)
        future.result(timeout=3)
        assert completed.is_set()


def test_publication_releases_lock_after_failure(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Requires PostgreSQL advisory locks")
    with Session(engine) as db:
        with pytest.raises(ValueError), publication_scope(db):
            raise ValueError("failed refresh")

    # New thread/connection proves this is not a same-context reentrant success.
    def publish():
        with Session(engine) as db, publication_scope(db):
            return True

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(publish).result(timeout=3)


def test_group_post_counts_handle_duplicate_posts_and_missing_tickers(engine):
    from app.services.theme_group_reads import _post_counts

    metadata = sa.MetaData()
    mentions = sa.Table(
        "theme_mentions",
        metadata,
        sa.Column("content_item_id", sa.Integer),
        sa.Column("theme_cluster_id", sa.Integer),
        sa.Column("pipeline", sa.String),
        sa.Column("social_work_id", sa.Integer),
        sa.Column("tickers", sa.JSON),
    )
    sources = sa.Table(
        "content_sources",
        metadata,
        sa.Column("id", sa.Integer),
        sa.Column("is_active", sa.Boolean),
    )
    eligibility = sa.Table(
        "content_pipeline_eligibility",
        metadata,
        sa.Column("content_item_id", sa.Integer),
        sa.Column("pipeline", sa.String),
        sa.Column("channel", sa.String),
        sa.Column("originating_source_id", sa.Integer),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(sources.insert(), [{"id": 1, "is_active": True}])
        connection.execute(
            eligibility.insert(),
            [
                {
                    "content_item_id": i,
                    "pipeline": "technical",
                    "channel": "legacy",
                    "originating_source_id": 1,
                }
                for i in [1, 2, 3]
            ],
        )
        connection.execute(
            mentions.insert(),
            [
                dict(
                    content_item_id=item,
                    theme_cluster_id=theme,
                    pipeline="technical",
                    social_work_id=None,
                    tickers=tickers,
                )
                for item, theme, tickers in [
                    (1, 1, ["MRVL"]),
                    (1, 2, ["MRVL"]),
                    (2, 1, None),
                    (3, 2, ["NVDA"]),
                ]
            ],
        )
    with Session(engine) as db:
        counts = _post_counts(db, [1, 2], "technical")
        rows = dict(db.execute(sa.select(counts.c.symbol, counts.c.count)).all())
        assert rows["MRVL"] == 1
        assert rows["NVDA"] == 1


def test_mutation_holds_publication_until_transaction_finishes(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Requires PostgreSQL advisory locks")
    started, completed = Event(), Event()

    def publish():
        with Session(engine) as db:
            started.set()
            with publication_scope(db):
                completed.set()

    with ThreadPoolExecutor(max_workers=1) as pool, Session(engine) as db:
        lock_grouping_mutation(db)
        future = pool.submit(publish)
        assert started.wait(3)
        db.flush()
        assert not completed.wait(0.2)
        db.commit()
        future.result(timeout=3)
        assert completed.is_set()
