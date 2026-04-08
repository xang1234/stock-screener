"""Baseline schema for PostgreSQL deployments."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260408_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from app import models  # noqa: F401  # Ensure all ORM models are registered.
    from app.database import Base

    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    from app import models  # noqa: F401  # Ensure all ORM models are registered.
    from app.database import Base

    Base.metadata.drop_all(bind=op.get_bind())
