"""Persist live attachment evidence and the revision consumed by extraction."""

import sqlalchemy as sa
from alembic import op

revision = "20260911_0041"
down_revision = "20260911_0040"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "content_attachments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "content_item_id",
            sa.Integer(),
            sa.ForeignKey("content_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("reference_key", sa.String(64), nullable=False),
        sa.Column("policy_version", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prepared_at", sa.DateTime(timezone=True)),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("lease_token", sa.String(36)),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(100)),
        sa.Column("content_sha256", sa.String(64)),
        sa.Column("final_url", sa.Text()),
        sa.Column("original_text", sa.Text()),
        sa.Column("prepared_text", sa.Text()),
        sa.Column("provenance", sa.JSON()),
        sa.UniqueConstraint(
            "content_item_id", "reference_key", name="uix_attachment_parent_reference"
        ),
        sa.CheckConstraint("kind IN ('image', 'article')", name="ck_attachment_kind"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'complete', 'partial', 'failed')",
            name="ck_attachment_status",
        ),
    )
    op.create_index(
        "ix_content_attachments_content_item_id",
        "content_attachments",
        ["content_item_id"],
    )
    op.create_index(
        "idx_attachment_pending", "content_attachments", ["status", "next_attempt_at"]
    )
    op.add_column("content_items", sa.Column("attachment_revision", sa.String(64)))
    op.add_column(
        "content_item_pipeline_state", sa.Column("evidence_revision", sa.String(64))
    )


def downgrade():
    op.drop_column("content_items", "attachment_revision")
    op.drop_column("content_item_pipeline_state", "evidence_revision")
    op.drop_table("content_attachments")
