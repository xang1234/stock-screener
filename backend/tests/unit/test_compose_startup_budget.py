"""Startup contract for the backend health check.

The backend runs its Alembic migrations inside the application ``lifespan``, blocking,
before uvicorn accepts HTTP traffic (``app.infra.db.migrations.migrate_database_to_head``
called from ``app.main``). Compose gives it a fixed start-up grace:

    start_period

Probe failures during that grace are not counted towards ``retries``. The grace does not
run to its end unconditionally, though: if a probe succeeds before it is spent, the
container counts as started and every later consecutive failure is counted, including
inside the remaining grace. Once ``retries`` consecutive failures have been counted, the
container is reported unhealthy.

``depends_on: condition: service_healthy`` gates the *start* of the containers declaring
it. While the backend is starting they are held back; if it never becomes healthy, their
start is abandoned and nothing retries it:

    dependency failed to start: container <backend> is unhealthy

A migration that legitimately outlives the grace therefore keeps the whole worker tier from
starting.

``start_period + interval * retries`` is deliberately not used here, in prose or in an
assertion. Docker does not define the unhealthy deadline as that sum -- probe scheduling
also depends on ``start_interval`` and on when prior checks complete -- and the sum
measures the window *after* the grace rather than the grace, so it can accept a migration
that does not fit inside ``start_period``.

Revision ``20260926_0058`` (an index over a 216 MB table) needed 519 s inside the
lifespan while the grace was 30 s -- eight Celery containers failed to start on two
consecutive nights. These tests pin the grace and the ordering the worker tier relies on,
so it cannot silently shrink again.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = ROOT / "docker-compose.yml"
PROD_COMPOSE = ROOT / "docker-compose.prod.yml"

# Measured on a QNAP TS-473A: revision 20260926_0058 built an index over a 216 MB table
# and needed 519 s in the backend lifespan before /readyz answered.
OBSERVED_MIGRATION_SECONDS = 519

# Floor for the start-up grace. A real migration has to fit with room to spare.
MIN_START_PERIOD_SECONDS = 600

# The grace redis was given for large persisted datasets (the BusyLoadingError fix).
MIN_REDIS_START_PERIOD_SECONDS = 120


def _yaml(path: Path) -> dict:
    """Parse a compose file into a mapping."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _duration_seconds(value: object) -> int:
    """Convert a compose duration literal (``30s``, ``15m``, ``30``) into seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", str(value))
    assert match, f"unrecognised duration literal: {value!r}"
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"": 1, "s": 1, "m": 60, "h": 3600}[unit]


def test_backend_start_period_covers_a_long_migration():
    """A running migration must never be reported as an unhealthy backend.

    ``start_period`` is the grace a bootstrapping container gets for free: probe failures
    during it are not counted towards ``retries``. A migration that fits inside the grace
    and succeeds on a probe afterwards is therefore never reported unhealthy, which is the
    property to assert.

    ``start_period + interval * retries`` deliberately is *not* asserted anywhere. Docker
    does not define the unhealthy deadline that way -- probe scheduling also depends on
    ``start_interval`` and on when prior checks complete -- so treating it as an exact
    budget either overstates or understates the real window. It also measures the wrong
    quantity: the window after the grace, not the grace itself.
    """
    check = _yaml(BASE_COMPOSE)["services"]["backend"]["healthcheck"]

    assert _duration_seconds(check["start_period"]) >= MIN_START_PERIOD_SECONDS, (
        "backend.healthcheck.start_period is shorter than the startup grace a long "
        "migration needs"
    )
    assert _duration_seconds(check["start_period"]) > OBSERVED_MIGRATION_SECONDS, (
        "the backend start_period does not cover the observed migration runtime"
    )


def test_backend_grace_is_not_smaller_than_the_redis_grace():
    """The backend grace must bound the deploy, not the redis grace.

    Redis got a long grace for large persisted datasets. The backend, which additionally
    has to migrate, carried a shorter one -- that asymmetry is what aborted the worker
    tier. Both gate the same deploy, so the backend may not be the tighter of the two.
    """
    redis_check = _yaml(BASE_COMPOSE)["services"]["redis"]["healthcheck"]

    assert _duration_seconds(redis_check["start_period"]) >= MIN_REDIS_START_PERIOD_SECONDS
    assert _duration_seconds(
        _yaml(BASE_COMPOSE)["services"]["backend"]["healthcheck"]["start_period"]
    ) >= _duration_seconds(redis_check["start_period"]), (
        "the backend grace is expected to bound the deploy, not the redis grace"
    )


def test_prod_backend_start_period_matches_the_base_file():
    """The production overlay may add a health check, but not a tighter grace."""
    base = _yaml(BASE_COMPOSE)["services"]["backend"]["healthcheck"]
    overlay = _yaml(PROD_COMPOSE)["services"]["backend"].get("healthcheck")

    if overlay is None or "start_period" not in overlay:
        return  # nothing to disagree with: the base value stands

    assert _duration_seconds(overlay["start_period"]) >= _duration_seconds(
        base["start_period"]
    ), "docker-compose.prod.yml shrinks the backend start_period of the base file"


def test_worker_tier_waits_for_a_healthy_backend():
    """Guard the topology the grace depends on.

    The grace only matters because the worker tier keys on ``service_healthy``: switch a
    worker to ``service_started`` and it would sail past the unhealthy window, hiding the
    bug instead of fixing it. ``frontend`` is a deliberate exception -- nginx serves the
    exported static site and does not need a healthy API.
    """
    conditions = {
        name: service["depends_on"]["backend"]["condition"]
        for name, service in _yaml(BASE_COMPOSE)["services"].items()
        if isinstance(service, dict)
        and isinstance(service.get("depends_on"), dict)
        and "backend" in service["depends_on"]
    }

    assert conditions, "no service declares a dependency on the backend any more"
    assert conditions.pop("frontend", None) == "service_started", (
        "frontend is expected to be the one dependent that does not wait for a healthy API"
    )
    assert all(value == "service_healthy" for value in conditions.values()), conditions
    assert len(conditions) > 1, "expected the whole worker tier to wait for the backend"
