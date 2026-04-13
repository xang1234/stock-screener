"""Add content language and translation schema fields (T7.1).

Persists source_language plus translated title/content/excerpt and a
translation_metadata JSONB snapshot on content_items and theme_mentions.
The original text fields stay authoritative; translated_* and metadata
are the replay-preserving derivative that downstream multilingual stages
(T7.2 detection, T7.3 translation) populate.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260413_0010"
down_revision = "20260412_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- content_items: source language + translated variant + metadata ---
    op.add_column(
        "content_items",
        sa.Column("source_language", sa.String(length=8), nullable=True),
    )
    op.add_column(
        "content_items",
        sa.Column("translated_title", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "content_items",
        sa.Column("translated_content", sa.Text(), nullable=True),
    )
    op.add_column(
        "content_items",
        sa.Column(
            "translation_metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_content_items_source_language",
        "content_items",
        ["source_language"],
    )

    # --- theme_mentions: translated derivative of raw_theme / excerpt ---
    op.add_column(
        "theme_mentions",
        sa.Column("translated_raw_theme", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "theme_mentions",
        sa.Column("translated_excerpt", sa.Text(), nullable=True),
    )
    op.add_column(
        "theme_mentions",
        sa.Column(
            "translation_metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("theme_mentions", "translation_metadata")
    op.drop_column("theme_mentions", "translated_excerpt")
    op.drop_column("theme_mentions", "translated_raw_theme")

    op.drop_index("idx_content_items_source_language", table_name="content_items")
    op.drop_column("content_items", "translation_metadata")
    op.drop_column("content_items", "translated_content")
    op.drop_column("content_items", "translated_title")
    op.drop_column("content_items", "source_language")
