"""Shared social associations, immutable decisions and mention provenance."""
from alembic import op
import sqlalchemy as sa

revision = "20260907_0037"
down_revision = "20260907_0036"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("social_theme_associations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("theme_cluster_id", sa.Integer, sa.ForeignKey("theme_clusters.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("company_key", sa.Text),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("canonical_symbol", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("origin", sa.Text, nullable=False),
        sa.Column("decision_owner", sa.Text, nullable=False),
        sa.Column("evidence_work_ids", sa.JSON, nullable=False),
        sa.Column("policy_version", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("theme_cluster_id", "market", "canonical_symbol", name="uq_social_theme_listing"),
        sa.CheckConstraint("state IN ('proposed','accepted','rejected')", name="ck_social_theme_state"),
        sa.CheckConstraint("origin IN ('social','legacy')", name="ck_social_theme_origin"),
        sa.CheckConstraint("decision_owner IN ('system','admin')", name="ck_social_theme_owner"),
        sa.CheckConstraint("market IN ('US','HK','CN','JP','TW')", name="ck_social_theme_market"),
        sa.CheckConstraint("version >= 1", name="ck_social_theme_version"))
    op.create_table("social_theme_decisions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("association_id", sa.Integer, sa.ForeignKey("social_theme_associations.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("run_id", sa.Text, sa.ForeignKey("social_signal_runs.id", ondelete="RESTRICT")),
        sa.Column("actor", sa.Text, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("before_state", sa.Text, nullable=False),
        sa.Column("after_state", sa.Text, nullable=False),
        sa.Column("policy_version", sa.Text, nullable=False),
        sa.Column("evidence_work_ids", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    with op.batch_alter_table("theme_mentions") as batch:
        batch.add_column(sa.Column("social_work_id", sa.Integer))
        batch.add_column(sa.Column("social_run_id", sa.Text))
        batch.create_foreign_key("fk_theme_mention_social_work", "social_extraction_work", ["social_work_id"], ["id"], ondelete="RESTRICT")
        batch.create_foreign_key("fk_theme_mention_social_run", "social_signal_runs", ["social_run_id"], ["id"], ondelete="RESTRICT")
        batch.create_unique_constraint("uq_social_work_theme_mention", ["social_work_id", "theme_cluster_id"])
    # This is provenance, not an issuer inference. Unresolved legacy constituents
    # remain untouched in their original table and require later resolution.
    op.execute(sa.text("""INSERT INTO social_theme_associations
        (theme_cluster_id, company_key, market, canonical_symbol, state, origin,
         decision_owner, evidence_work_ids, policy_version, version, first_seen_at, accepted_at, updated_at)
        SELECT c.theme_cluster_id, NULL, s.market, c.symbol, 'accepted', 'legacy',
               'system', '[]', 'legacy-preserved-v1', 1,
               COALESCE(c.first_mentioned_at, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        FROM theme_constituents c JOIN stock_universe s ON s.symbol = c.symbol
        JOIN theme_clusters t ON t.id = c.theme_cluster_id
        WHERE c.is_active = true AND s.market IN ('US','HK','CN','JP','TW')"""))


def downgrade():
    with op.batch_alter_table("theme_mentions") as batch:
        batch.drop_constraint("uq_social_work_theme_mention", type_="unique")
        batch.drop_constraint("fk_theme_mention_social_work", type_="foreignkey")
        batch.drop_constraint("fk_theme_mention_social_run", type_="foreignkey")
        batch.drop_column("social_work_id")
        batch.drop_column("social_run_id")
    op.drop_table("social_theme_decisions")
    op.drop_table("social_theme_associations")
