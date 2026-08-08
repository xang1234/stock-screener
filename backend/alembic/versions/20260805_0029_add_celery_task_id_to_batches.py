"""Add celery_task_id to max_pain_batches and gex_batches.

The Operations dashboard's "Force Stop" button revokes the Celery task
(celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')) but
had no way to find the corresponding batch row -- batch_id is a UUID
generated *inside* the task function, never passed back out to the
operations layer. A SIGTERM kill doesn't reliably run the task's own
except/finally blocks (the only place that flips status to
'failed'/'completed'), so force-stopped runs left their batch row stuck at
status='running' forever ("zombie" rows -- four were found and manually
cleaned up before this fix).

celery_task_id lets the operations service look up "the batch row for this
task_id" directly and mark it failed at revoke time, instead of guessing
via "most recent running row" (unsafe if two runs ever overlap) or leaving
it orphaned.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260805_0029"
down_revision = "20260805_0028"
branch_labels = None
depends_on = None

_TABLES = ("max_pain_batches", "gex_batches")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("celery_task_id", sa.String(155), nullable=True))
        op.create_index(
            f"ix_{table}_celery_task_id",
            table,
            ["celery_task_id"],
            unique=False,
        )


def downgrade() -> None:
    for table in _TABLES:
        op.drop_index(f"ix_{table}_celery_task_id", table)
        op.drop_column(table, "celery_task_id")
